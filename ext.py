import os
import shutil
import re

# ================= 配置区域 =================
SOURCE_FOLDER = r'./checkpoints'      # 源文件夹
DEST_FOLDER = r'./extracted_layers'   # 目标文件夹
# ===========================================

def get_file_score(filename):
    """
    给文件打分，分数越高越好。
    ae.pt = 无穷大 (最高优先级)
    ae_180000.pt = 180000
    """
    if filename == 'ae.pt':
        return float('inf')
    
    match = re.match(r'ae_(\d+)\.pt', filename)
    if match:
        return int(match.group(1))
    return -1

def process_folders():
    if not os.path.exists(DEST_FOLDER):
        os.makedirs(DEST_FOLDER)

    # 字典结构： { '15': {'path': '...', 'score': 180000, 'name': 'ae_180000.pt'}, ... }
    best_candidates = {} 

    print("正在扫描并筛选最佳模型...")

    # --- 第一步：扫描并筛选 ---
    for root, dirs, files in os.walk(SOURCE_FOLDER):
        # 1. 找层号
        path_parts = os.path.normpath(root).split(os.sep)
        layer_id = None
        for part in reversed(path_parts):
            if part.isdigit():
                layer_id = part
                break
        
        if not layer_id:
            continue

        # 2. 找当前文件夹里最好的文件
        pt_files = [f for f in files if f.endswith('.pt') and 'ae' in f]
        for f in pt_files:
            score = get_file_score(f)
            if score == -1: continue # 忽略不符合格式的文件

            # 3. 擂台赛：如果这个层号还没记录，或者新文件分数更高，就更新记录
            if layer_id not in best_candidates or score > best_candidates[layer_id]['score']:
                best_candidates[layer_id] = {
                    'path': os.path.join(root, f),
                    'score': score,
                    'orig_name': f
                }

    # --- 第二步：执行复制 ---
    print(f"筛选完毕，准备提取 {len(best_candidates)} 个文件...\n")
    
    for layer_id, info in sorted(best_candidates.items(), key=lambda x: int(x[0])):
        src_path = info['path']
        new_filename = f"layer_{layer_id}.pt"
        dest_path = os.path.join(DEST_FOLDER, new_filename)

        # 打印信息用于确认
        score_display = "最新 (ae.pt)" if info['score'] == float('inf') else f"步数 {info['score']}"
        print(f"层 {layer_id}: 选中 [{info['orig_name']}] ({score_display}) -> {new_filename}")
        
        shutil.copy2(src_path, dest_path)

    print(f"\n全部完成！")

if __name__ == '__main__':
    process_folders()