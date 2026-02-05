import torch

# 加载文件
data = torch.load("/work/nvme/bfga/tianhux2/vt2/checkpoints/2026-01-28 00:06:43/15/trainer_0/checkpoints/ae_15000.pt", map_location="cpu")

# 判断类型
if isinstance(data, dict):
    print("=== File contains a dictionary ===")
    for key in data:
        if isinstance(data[key], torch.Tensor):
            print(f"{key}: {data[key].shape} ({data[key].dtype})")
        else:
            print(f"{key}: {type(data[key])}")
elif isinstance(data, torch.Tensor):
    print(f"=== Single tensor: {data.shape} ({data.dtype}) ===")
else:
    print(f"=== Unknown type: {type(data)} ===")

print(data['threshold'].min())
print(data['threshold'].max())
print(data['threshold'].mean())