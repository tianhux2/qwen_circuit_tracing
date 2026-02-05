import torch as t
import argparse
import json
import heapq
import math
from tqdm import tqdm
from typing import List, Dict
import gc

# 引入你的项目依赖
from transformers import Qwen3VLForConditionalGeneration
from VisionLanguageModel import VisionLanguageModel
from dataset import get_qwen_dataset_iterator
from dictionary_learning.trainers.jumprelu_transcoder import JumpReluTranscoderAutoEncoder
# 假设 BetterActivationBuffer 在 buffer.py 中，这里我们只需要参考其 trace 逻辑，实际上我们会重写一个轻量级的 Tracker

class NeuronActivationTracker:
    def __init__(self, 
                 model: VisionLanguageModel, 
                 sae: JumpReluTranscoderAutoEncoder, 
                 submodule,
                 tokenizer,
                 k: int = 20,
                 context_window: int = 10):
        self.model = model
        self.sae = sae
        self.submodule = submodule
        self.tokenizer = tokenizer
        self.k = k
        self.context_window = context_window
        self.device = model.device

        self.id = 0
        
        # 视觉 Token 的特殊 ID，Qwen3-VL 通常有 vision_start/end
        # 这里需要根据实际 tokenizer 确认，以下是常见默认值
        try:
            self.vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
            self.vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        except:
            print("Warning: Vision tokens not found in tokenizer, using generic detection.")
            self.vision_start_id = -1

    def _process_batch_activations(self, 
                                 activations: t.Tensor, 
                                 input_ids: t.Tensor, 
                                 neuron_chunk: List[int],
                                 batch_text_data: List[str]):
        """
        activations: [Batch*Seq, D_model] (Flat input activations)
        input_ids: [Batch, Seq]
        neuron_chunk: List of neuron indices to track in this pass
        """
        # 1. SAE Encode (计算特征激活)
        # 此时 activations 已经是 [Total_Tokens, Dim]
        with t.no_grad():
            # Transcoder 的 encode 通常只需要输入侧的激活
            # features: [Total_Tokens, Dict_Size]
            features = self.sae.encode(activations) 
            
        # 2. 获取目标 Neurons 的激活值
        # target_feats: [Total_Tokens, len(neuron_chunk)]
        neuron_chunk_tensor = t.tensor(neuron_chunk, device=self.device)
        target_feats = features[:, neuron_chunk_tensor]
        
        # 3. 找到大于 0 的激活位置
        # 为了效率，我们先找到所有非零元素的索引
        # non_zero_indices: (token_global_idx, neuron_local_idx)
        non_zero_mask = target_feats > 0
        active_indices = non_zero_mask.nonzero()
        
        if active_indices.shape[0] == 0:
            return {}

        batch_size, seq_len = input_ids.shape
        
        # 结果缓存: {neuron_global_idx: [records]}
        chunk_results = {n: [] for n in neuron_chunk}
        
        # 提取值
        active_values = target_feats[non_zero_mask]
        
        # 转换为 CPU 处理元数据 (避免 GPU 同步开销过大，先批量转)
        active_indices_cpu = active_indices.cpu().numpy()
        active_values_cpu = active_values.cpu().tolist()
        input_ids_cpu = input_ids.cpu().numpy()
        
        for idx, (flat_idx, local_n_idx) in enumerate(active_indices_cpu):
            val = active_values_cpu[idx]
            global_n_idx = neuron_chunk[local_n_idx]
            
            # 反算 (Batch, Seq) 坐标
            # 注意：这里的 activations 是经过 mask 后的 flat tensor
            # 如果直接用 flat_idx 对应 input_ids 可能会错位（因为 padding 被移除了）
            # 所以我们需要在外部维护一个 map 或者不移除 padding
            # **策略调整**：为了对齐准确，建议在 trace 时不要移除 padding，或者记录 valid_indices
            
            # 这里假设外部传入的 activations 是对应 input_ids[mask] 的
            # 这种反查比较复杂。更简单的做法是：
            # 让 activations 保持 [Batch, Seq, Dim] 的形状传入，不 flatten
            pass 

        return chunk_results

    def track(self, data_iterator, neuron_indices: List[int], num_batches: int = 10, batch_size: int = 8):
        """
        主循环：遍历数据，统计 Top-K
        """
        # 初始化堆: {neuron_id: [(activation, metadata_dict), ...]}
        # 使用 Min-Heap，当 len > k 时，pushpop 掉最小的
        top_k_heaps = {n: [] for n in neuron_indices}
        
        # 自动分块处理 Neurons，避免一次 encode 输出太大
        # 假设最大显存允许同时追踪 4096 个 neuron 的激活
        chunk_size = 4096
        neuron_chunks = [neuron_indices[i:i + chunk_size] for i in range(0, len(neuron_indices), chunk_size)]
        
        print(f"Tracking {len(neuron_indices)} neurons in {len(neuron_chunks)} chunks...")
        
        # 外层循环：数据 Batch
        for batch_idx in tqdm(range(num_batches), desc="Scanning Batches"):
            try:
                batch_texts = [next(data_iterator) for _ in range(batch_size)]
            except StopIteration:
                break
                
            # 1. 获取模型激活 (Reference: BetterActivationBuffer logic)
            with t.no_grad():
                with self.model.trace(batch_texts, truncation=True, max_length=1500) as tracer:
                    # 获取输入 Input (Transcoder 需要 Input 来 encode)
                    # 注意：如果你的 Transcoder 是 output-only，请改用 .output
                    input_data = self.model.inputs.save()
                    act_raw = self.submodule.inputs[0].save()
                    tracer.stop()
                
                # act_raw: [Batch, Seq, Dim] (包含 Padding)
                if isinstance(act_raw, tuple): act_raw = act_raw[0]
                
                input_ids = input_data[1]["input_ids"] # [Batch, Seq]
                attention_mask = input_data[1]["attention_mask"]
                
                # 2. 遍历每个 Neuron Chunk 进行计算
                for chunk in neuron_chunks:
                    chunk_tensor = t.tensor(chunk, device=self.device)
                    
                    # 展平以便批量 Encode，但记录 shape
                    b, s, d = act_raw.shape
                    flat_acts = act_raw.view(-1, d) # [B*S, D]
                    
                    # Encode
                    features = self.sae.encode(flat_acts) # [B*S, Dict_Size]
                    
                    # 取出当前 Chunk 的特征
                    target_features = features[:, chunk_tensor] # [B*S, Chunk_Size]
                    target_features = target_features.view(b, s, -1) # [B, S, Chunk_Size]
                    
                    # 应用 Attention Mask (将 Padding 处的激活置为 0)
                    mask_expanded = attention_mask.unsqueeze(-1).expand_as(target_features)
                    target_features = target_features * mask_expanded
                    
                    # 3. 提取 Top-K 候选
                    # 策略：对于每个 Neuron，先在当前 Batch 内部找到 Top-K，再与全局 Heap 比较
                    # 这样可以大幅减少 Python 循环次数
                    
                    # [Chunk_Size, B*S] -> TopK
                    # 转置为 [Chunk_Size, Batch, Seq] 以便处理
                    vals_per_neuron = target_features.permute(2, 0, 1) # [Chunk, B, S]
                    
                    for i, local_idx in enumerate(range(len(chunk))):
                        n_id = chunk[local_idx]
                        neuron_vals = vals_per_neuron[i] # [B, S]
                        
                        # 快速筛选：只看当前 Batch 中最大的 K 个值
                        # 如果当前最大值都比堆里最小值小，就跳过
                        curr_topk_vals, curr_topk_indices = t.topk(neuron_vals.flatten(), k=min(self.k, neuron_vals.numel()))
                        
                        # 检查是否有资格进入堆
                        if len(top_k_heaps[n_id]) == self.k:
                            if curr_topk_vals[0].item() < top_k_heaps[n_id][0][0]:
                                continue # 连最大的都没资格，跳过
                        
                        # 处理这 K 个候选
                        curr_topk_vals = curr_topk_vals.cpu().tolist()
                        curr_topk_indices = curr_topk_indices.cpu().tolist()
                        
                        for val, flat_idx in zip(curr_topk_vals, curr_topk_indices):
                            if val <= 1e-6: continue # 忽略零激活
                            
                            batch_i = flat_idx // s
                            seq_i = flat_idx % s
                            
                            # 构建元数据
                            ctx_start = max(0, seq_i - self.context_window)
                            ctx_end = min(s, seq_i + self.context_window + 1)
                            
                            raw_ids = input_ids[batch_i]
                            ctx_ids = raw_ids[ctx_start:ctx_end]
                            token_str = self.tokenizer.decode(raw_ids[seq_i])
                            context_str = self.tokenizer.decode(ctx_ids)
                            
                            # 图像定位逻辑
                            prefix_ids = raw_ids[:seq_i].tolist()
                            img_count = prefix_ids.count(self.vision_start_id)
                            # 判断当前 token 是否在 vision 标签内 (简单逻辑)
                            # 如果 tokenizer 没有 vision_start，这里需要自定义逻辑
                            is_image_token = False
                            if self.vision_start_id != -1:
                                # 找到最后一个 start 和 end
                                try:
                                    last_start = len(prefix_ids) - 1 - prefix_ids[::-1].index(self.vision_start_id)
                                    # 检查在该 start 之后是否有 end
                                    if self.vision_end_id not in prefix_ids[last_start:]:
                                        is_image_token = True
                                except ValueError:
                                    pass # 没有 start
                            
                            record = {
                                "value": val,
                                "token": token_str,
                                "context": context_str,
                                "is_image": is_image_token,
                                "image_idx": img_count if is_image_token else -1,
                                "token_pos": int(seq_i)
                            }
                            
                            # 更新堆
                            heap = top_k_heaps[n_id]
                            if len(heap) < self.k:
                                heapq.heappush(heap, (val, self.id, record))
                            else:
                                heapq.heappushpop(heap, (val, self.id, record))
                            self.id += 1
                                
            # 清理缓存
            del act_raw, input_data, features, target_features
            t.cuda.empty_cache()

        # 整理输出
        final_report = {}
        for n_id, heap in top_k_heaps.items():
            # 堆是无序的，且我们要降序输出
            sorted_records = sorted([item[2] for item in heap], key=lambda x: x['value'], reverse=True)
            final_report[str(n_id)] = sorted_records
            
        return final_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--neurons", type=str, default="0-100", help="Range (0-100) or list (1,5,10)")
    parser.add_argument("--output_file", type=str, default="top_k_activations.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--scan_batches", type=int, default=50, help="How many batches to scan")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # 1. 解析 Neurons
    if "-" in args.neurons:
        start, end = map(int, args.neurons.split("-"))
        neuron_indices = list(range(start, end))
    else:
        neuron_indices = list(map(int, args.neurons.split(",")))

    # 2. 模型初始化
    print(f"Loading Model on {args.device}...")
    model = VisionLanguageModel(
        "Qwen/Qwen3-VL-2B-Instruct",
        automodel=Qwen3VLForConditionalGeneration,
        device_map=args.device,
        dispatch=True,
        dtype=t.bfloat16,
    )
    
    # 3. Transcoder 加载
    print("Loading Transcoder...")
    sae = JumpReluTranscoderAutoEncoder.from_pretrained(args.checkpoint, device=args.device)
    
    # 确定 submodule
    submodule = model.language_model.layers[args.layer].mlp

    # 4. 数据迭代器
    data_iterator = get_qwen_dataset_iterator(args.dataset_path, split="test")

    # 5. 执行追踪
    tracker = NeuronActivationTracker(
        model=model,
        sae=sae,
        submodule=submodule,
        tokenizer=model.tokenizer,
        k=20, # Top-K
        context_window=10
    )
    
    results = tracker.track(
        data_iterator=data_iterator,
        neuron_indices=neuron_indices,
        num_batches=args.scan_batches,
        batch_size=args.batch_size
    )

    # 6. 保存
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    main()