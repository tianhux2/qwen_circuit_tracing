import random
from PIL import Image
from datasets import load_from_disk

# ================= 配置常量 =================
MAX_LONG_SIDE = 1500
# Qwen2-VL 视觉占位符
QWEN_VISION_TAG = "<|vision_start|><|image_pad|><|vision_end|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

def process_image(image_obj):
    """
    检查图片对象，确保其为 RGB 且长边不超过 1500。
    """
    if image_obj is None:
        return None
        
    # 确保转换为 RGB (防止 RGBA/P 模式报错)
    if image_obj.mode != "RGB":
        image_obj = image_obj.convert("RGB")
        
    w, h = image_obj.size
    if max(w, h) > MAX_LONG_SIDE:
        scale = MAX_LONG_SIDE / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        # 使用 LANCZOS 进行高质量缩放
        image_obj = image_obj.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return image_obj

def build_qwen_prompt(conversations_dict, has_image):
    """
    处理列式存储的对话格式：{'from': ['human', ...], 'value': ['<image>...', ...]}
    将其转换为 Qwen 格式字符串。
    """
    final_text = ""
    tag_found = False
    
    # 1. 获取列表
    roles = conversations_dict.get('from', [])
    values = conversations_dict.get('value', [])
    
    # 2. 校验长度一致性
    if len(roles) != len(values):
        raise ValueError(f"对话数据损坏：角色列表长度({len(roles)})与内容列表长度({len(values)})不一致")

    # 3. 使用 zip 并行遍历
    for role, content in zip(roles, values):
        # 角色映射
        if role == 'human':
            role_display = 'user'
        elif role == 'gpt':
            role_display = 'assistant'
        else:
            role_display = role # fallback

        # 标签替换与检查
        if "<image>" in content:
            if not has_image:
                raise ValueError(f"文本包含 <image> 标签，但 Image 对象为空。内容: {content[:30]}...")
            
            content = content.replace("<image>", QWEN_VISION_TAG)
            tag_found = True
        
        # 格式化拼接
        final_text += f"{IM_START}{role_display}\n{content}{IM_END}\n"

    # 4. 全局校验：有图必须有标签
    if has_image and not tag_found:
        raise ValueError("存在图片对象，但在对话文本中未找到 <image> 标签。")

    return final_text

def get_qwen_dataset_iterator(dataset_path, split="train", seed=42, log_interval=100):
    """
    输入：数据集磁盘路径
    输出：生成器，yield (formatted_text, processed_image_obj)
    """
    print(f"正在加载数据集: {dataset_path} ...")
    ds = load_from_disk(dataset_path)[split]
    
    print(f"正在打乱数据 (Seed={seed})...")
    ds = ds.shuffle(seed=seed)

    total_samples = len(ds)

    print("开始迭代数据...")
    for idx, item in enumerate(ds):
        if idx % log_interval == 0:
            current_processed = idx + 1
            usage_ratio = current_processed / total_samples
            print(f"current processed{current_processed}; usage_ratio: {usage_ratio:.2%}")

        try:
            # === 1. 图片处理 ===
            image_raw = item.get("image")
            processed_image = process_image(image_raw)
            has_image = (processed_image is not None)

            # === 2. 文本处理 ===
            conversations = item.get("conversations")
            raw_text = item.get("text")
            formatted_text = ""

            # 分支 A: 图文对话 (Conversations 存在且非空)
            # 注意：Datasets 的 Sequence 读取出来可能是一个包含空列表的字典，如 {'from': [], 'value': []}
            if conversations and len(conversations.get('from', [])) > 0:
                formatted_text = build_qwen_prompt(conversations, has_image)
            
            # 分支 B: 纯文本 (Text 存在)
            elif raw_text:
                # 纯文本通常没有 image，如果有 image 需要手动处理标签
                # 这里假设纯文本就是 user 的一段输入
                content = raw_text
                if has_image:
                    if "<image>" not in content:
                        # 强行插入
                        content = "<image>\n" + content
                    content = content.replace("<image>", QWEN_VISION_TAG)
                
                # 包装为 Qwen user 格式
                formatted_text = f"{content}"
                
                # 如果做预训练不需要 assistant 结尾；如果做 SFT 需要补充
                # formatted_text += f"{IM_START}assistant\n" 
            
            else:
                # 数据无效，跳过
                continue

            # === 3. 返回结果 ===
            yield formatted_text, processed_image

        except ValueError as ve:
            # 捕获标签校验错误 (例如：有图没标签，或有标签没图)
            # print(f"Skipping invalid data ID {item.get('id', '?')}: {ve}")
            continue
        except Exception as e:
            print(f"Unexpected Error processing item: {e}")
            continue

# ================= 测试代码 =================
if __name__ == "__main__":
    # 替换为你的数据集路径
    DATASET_PATH = "/work/nvme/bfga/tianhux2/llava/final_mixed_dataset_v2"
    
    iterator = get_qwen_dataset_iterator(DATASET_PATH)
    
    print("\n--- 检查前 2 条数据 ---")
    for i, (txt, img) in enumerate(iterator):
        if i >= 2: break
        print(f"\n[Sample {i}]")
        print(f"Image: {img.size if img else 'None'}")
        print(f"Text:\n{txt}")
        print("-" * 30)