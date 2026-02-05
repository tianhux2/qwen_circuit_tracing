from VisionLanguageModel import VisionLanguageModel
from dataset import get_qwen_dataset_iterator
from buffer import BetterActivationBuffer
from dictionary_learning.trainers.jumprelu_transcoder import JumpReluTranscoderTrainer, JumpReluTranscoderAutoEncoder
from dictionary_learning.training_transcoder import trainTranscoder
from transformers import Qwen3VLForConditionalGeneration
import torch as t
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--layer", type=int, default=19, help="Layer index to target for training")
parser.add_argument("--name", type=str, default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
args = parser.parse_args()

device = "cuda:0"
model_name = "Qwen/Qwen3-VL-2B-Instruct"  # can be any Huggingface model

model = VisionLanguageModel(
    model_name,
    automodel=Qwen3VLForConditionalGeneration,
    device_map=device,
    dispatch=True,
    dtype=t.bfloat16,
)

layer = args.layer
run_name = args.name

submodule = model.language_model.layers[layer].mlp  # layer 1 MLP
activation_dim = 2048  # output dimension of the MLP
dictionary_size = 32 * activation_dim
llm_batch_size = 128
sae_batch_size = 1024
training_steps = 200000

# data must be an iterator that outputs strings
DATASET_PATH = "/work/nvme/bfga/tianhux2/llava/final_mixed_dataset_v3"
data = get_qwen_dataset_iterator(DATASET_PATH)

buffer = BetterActivationBuffer(
    data=data,
    model=model,
    submodule=submodule,
    d_submodule=activation_dim,  # output dimension of the model component
    n_ctxs=512,  # you can set this higher or lower depending on your available memory
    io='in_and_out',
    ctx_len=1500,
    device=device,
    refresh_batch_size=llm_batch_size,
    out_batch_size=sae_batch_size,
)  # buffer will yield batches of tensors of dimension = submodule's output dimension

trainer_cfg = {
    "trainer": JumpReluTranscoderTrainer,
    "dict_class": JumpReluTranscoderAutoEncoder,
    "activation_dim": activation_dim,
    "dict_size": dictionary_size,
    "lr": 5e-5,
    "device": device,
    "steps": training_steps,
    "layer": layer,
    "lm_name": model_name,
    "target_l0": 30,
    "wandb_name": f"Jumprelu_neo_{run_name}_{layer}",
    "initial_bandwidth": 0.01,
    "target_bandwidth": 0.001,
    "coefficient_growth_start_pct": 0.1,
    "sparsity_penalty": 2,
}

# train the sparse autoencoder (SAE)
ae = trainTranscoder(
    data=buffer,  # you could also use another (i.e. pytorch dataloader) here instead of buffer
    trainer_configs=[trainer_cfg],
    steps=training_steps,  # The number of training steps. Total trained tokens = steps * batch_size
    use_wandb=True,
    wandb_entity=None,
    wandb_project="qwen3-transcoder",
    save_steps=[i for i in range(20000, training_steps + 1, 20000)],
    save_dir=f"checkpoints/{run_name}/{layer}",
    log_steps=100,
    transcoder=True,
    device=device,
    autocast_dtype=t.bfloat16,
    normalize_activations=True
)
