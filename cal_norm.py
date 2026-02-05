import torch as t
from transformers import Qwen3VLForConditionalGeneration
from VisionLanguageModel import VisionLanguageModel
from dataset import get_qwen_dataset_iterator
import argparse
from tqdm import tqdm

def calculate_layer_norms_and_lambdas():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_layer", type=int, default=1, help="Layer index used as reference (where L0 is good)")
    parser.add_argument("--base_lambda", type=float, default=2, help="The lambda value that works well for the base layer")
    parser.add_argument("--num_layers", type=int, default=28, help="Total number of layers in the model")
    parser.add_argument("--n_batches", type=int, default=10, help="Number of batches to average over")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for measurement")
    parser.add_argument("--dataset_path", type=str, default="/work/nvme/bfga/tianhux2/llava/final_mixed_dataset_v3")
    args = parser.parse_args()

    device = "cuda:0"
    model_name = "Qwen/Qwen3-VL-2B-Instruct"

    print(f"Loading model: {model_name}...")
    model = VisionLanguageModel(
        model_name,
        automodel=Qwen3VLForConditionalGeneration,
        device_map=device,
        dispatch=True,
        dtype=t.bfloat16,
    )

    # 准备数据
    data_iterator = get_qwen_dataset_iterator(args.dataset_path)
    
    # 存储每一层的 Norm 累加值
    layer_norms_sum = t.zeros(args.num_layers, device=device)
    count = 0

    print(f"Measuring activations over {args.n_batches} batches...")
    
    with t.no_grad():
        for i in tqdm(range(args.n_batches)):
            batch_text = []
            try:
                for _ in range(args.batch_size):
                    batch_text.append(next(data_iterator))
            except StopIteration:
                break
            
            # 使用 nnsight trace 一次性获取所有层的输出
            with model.trace(batch_text, truncation=True, max_length=1500) as tracer:
                # 注册所有层的输出
                layer_outputs = list().save()
                for layer_idx in range(args.num_layers):
                    # 注意：根据具体模型架构调整，这里假设是 Qwen/Llama 结构
                    # Qwen2/3 通常是 model.model.layers 或 model.language_model.layers
                    submodule = model.language_model.layers[layer_idx].mlp
                    # 我们需要 input 还是 output？Transcoder 通常用 Input 做归一化参考，
                    # 但 MSE 是基于 Output 算的，所以这里测量 MLP Output 的 Norm 更能反映 MSE 的量级。
                    layer_outputs.append(submodule.output)
            
            # 计算并累加 Norm
            for layer_idx, output in enumerate(layer_outputs):
                # output shape: (Batch, Seq, Dim)
                # 计算平均 L2 Norm
                activations = output
                # 排除 padding (如果全是 0 的话) 或者直接算 mean
                avg_norm = activations.norm(p=2, dim=-1).mean()
                layer_norms_sum[layer_idx] += avg_norm
            
            count += 1

    # 计算平均 Norm
    avg_norms = layer_norms_sum / count
    
    # 获取基准层的 Norm
    base_norm = avg_norms[args.base_layer].item()
    print(f"\nBase Layer: {args.base_layer}, Base Norm: {base_norm:.4f}, Base Lambda: {args.base_lambda}")
    print("-" * 60)
    print(f"{'Layer':<10} | {'Avg Norm':<15} | {'Scale Factor':<15} | {'Recommended Lambda':<20}")
    print("-" * 60)

    results = {}

    for layer_idx in range(args.num_layers):
        current_norm = avg_norms[layer_idx].item()
        
        # 核心公式：Lambda 缩放比例 = (Current_Norm / Base_Norm) ^ 2
        # 为什么要平方？因为 MSE Loss 与 范数的平方 成正比。
        scale_factor = (current_norm / base_norm)
        
        rec_lambda = args.base_lambda * scale_factor
        
        results[layer_idx] = rec_lambda
        
        print(f"{layer_idx:<10} | {current_norm:<15.4f} | {scale_factor:<15.4f} | {rec_lambda:<20.6f}")

    print("-" * 60)
    print("\nCopy paste dictionary for your config script:")
    print(results)

if __name__ == "__main__":
    calculate_layer_norms_and_lambdas()