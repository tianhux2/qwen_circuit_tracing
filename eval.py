import torch as t
import argparse
import os
from transformers import Qwen3VLForConditionalGeneration
from VisionLanguageModel import VisionLanguageModel
from dataset import get_qwen_dataset_iterator
from buffer import BetterActivationBuffer

from dictionary_learning.evaluation import evaluate

# 引入模型定义以加载权重
from dictionary_learning.trainers.jumprelu_transcoder import JumpReluTranscoderAutoEncoder

def main():
    parser = argparse.ArgumentParser()
    # 必须提供 checkpoint 路径
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained SAE/Transcoder checkpoint (.pt file)")
    parser.add_argument("--layer", type=int, default=15, help="Layer index to target")
    parser.add_argument("--dataset_path", type=str, default="/work/nvme/bfga/tianhux2/llava/final_mixed_dataset_v3")
    parser.add_argument("--device", type=str, default="cuda:0")
    # 可选：评估参数
    parser.add_argument("--n_batches", type=int, default=100, help="Number of batches to evaluate")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for loss recovery (smaller to save VRAM)")
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device}")

    # =====================================================
    # 1. 模型初始化 (与训练代码保持一致)
    # =====================================================
    model_name = "Qwen/Qwen3-VL-2B-Instruct"
    model = VisionLanguageModel(
        model_name,
        automodel=Qwen3VLForConditionalGeneration,
        device_map=device,
        dispatch=True,
        dtype=t.bfloat16,
    )

    layer = args.layer
    submodule = model.language_model.layers[layer].mlp
    activation_dim = 2048
    dictionary_size = 32 * activation_dim

    # =====================================================
    # 2. 加载训练好的 Transcoder
    # =====================================================
    print(f"Loading Transcoder from {args.checkpoint}...")
        
    sae = JumpReluTranscoderAutoEncoder.from_pretrained(args.checkpoint, device=device)

    # =====================================================
    # 3. 准备数据 Buffer
    # =====================================================
    # 使用较小的 batch size 避免 OOM，尤其是 loss_recovered 需要运行大模型
    llm_batch_size = 32 
    sae_batch_size = 1024 # 这里的 batch size 是给 SAE forward 用的
    
    data = get_qwen_dataset_iterator(args.dataset_path, split="test")

    buffer = BetterActivationBuffer(
        data=data,
        model=model,
        submodule=submodule,
        d_submodule=activation_dim,
        n_ctxs=15,
        io='in_and_out',  # 关键：必须与训练时一致，产生 (Input, Output) 配对数据
        ctx_len=1500,
        device=device,
        refresh_batch_size=llm_batch_size,
        out_batch_size=sae_batch_size,
    )

    # =====================================================
    # 4. 执行评估
    # =====================================================
    print("Starting evaluation...")
    
    metrics = evaluate(
        dictionary=sae,
        activations=buffer,
        max_len=1500,          # 控制 loss recovered 的上下文长度，太长会 OOM
        batch_size=args.eval_batch_size, # Loss Recovered 用的 batch size
        io="in_and_out",      # 明确告知 evaluate 这是 Transcoder 任务
        normalize_batch=False,# 除非你训练时使用了归一化，否则 Transcoder 建议设为 False
        device=device,
        n_batches=args.n_batches
    )

    # =====================================================
    # 5. 输出结果
    # =====================================================
    print(f"\nEvaluation Results (averaged over {args.n_batches} batches):")
    print("-" * 40)
    for key, value in metrics.items():
        print(f"{key:<30} : {value:.6f}")
    print("-" * 40)

if __name__ == "__main__":
    main()