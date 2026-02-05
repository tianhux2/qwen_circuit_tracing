from dictionary_learning.buffer import *
from VisionLanguageModel import VisionLanguageModel

class BetterActivationBuffer:
    """
    Implements a buffer of activations. The buffer stores activations from a model,
    yields them in batches, and refreshes them when the buffer is less than half full.
    """
    def __init__(self, 
                 data, # generator which yields text data
                 model : VisionLanguageModel | LanguageModel, # LanguageModel from which to extract activations
                 submodule, # submodule of the model from which to extract activations
                 d_submodule=None, # submodule dimension; if None, try to detect automatically
                 io='out', # can be 'in', 'out', or 'in_and_out'
                 n_ctxs=3e4, # approximate number of contexts to store in the buffer
                 ctx_len=128, # length of each context
                 refresh_batch_size=512, # size of batches in which to process the data when adding to buffer
                 out_batch_size=8192, # size of batches in which to yield activations
                 device='cpu', # device on which to store the activations
                 remove_bos: bool = False,
                 ):
        
        # [修改] 允许 'in_and_out'
        if io not in ['in', 'out', 'in_and_out']:
            raise ValueError("io must be either 'in', 'out' or 'in_and_out'")

        if d_submodule is None:
            try:
                if io == 'in':
                    d_submodule = submodule.in_features
                else:
                    # 对于 'out' 或 'in_and_out' (通常输入输出维度相同，取out即可，如有不同需额外处理)
                    d_submodule = submodule.out_features
            except:
                raise ValueError("d_submodule cannot be inferred and must be specified directly")
        
        self.io = io
        self.d_submodule = d_submodule
        self.model = model
        
        # [修改] 根据 io 类型初始化不同形状的 tensor
        if self.io == 'in_and_out':
            # Transcoder 模式: [Buffer_Size, 2, Dim]
            self.activations = t.empty(0, 2, d_submodule, device=device, dtype=model.dtype)
        else:
            # SAE 模式: [Buffer_Size, Dim]
            self.activations = t.empty(0, d_submodule, device=device, dtype=model.dtype)
            
        self.read = t.zeros(0).bool()

        self.data = data
        self.submodule = submodule
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.activation_buffer_size = int(n_ctxs * ctx_len)
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.device = device
        self.remove_bos = remove_bos
    
    def __iter__(self):
        return self

    def __next__(self):
        """
        Return a batch of activations
        """
        with t.no_grad():
            # if buffer is less than half full, refresh
            if (~self.read).sum() < self.activation_buffer_size // 2:
                self.refresh()

            # return a batch
            unreads = (~self.read).nonzero().squeeze()
            # 确保 unreads 是 1D tensor，防止在只有1个元素时出错
            if unreads.ndim == 0:
                unreads = unreads.unsqueeze(0)
                
            idxs = unreads[t.randperm(len(unreads), device=unreads.device)[:self.out_batch_size]]
            self.read[idxs] = True
            return self.activations[idxs]
    
    def text_batch(self, batch_size=None):
        """
        Return a list of text
        """
        if batch_size is None:
            batch_size = self.refresh_batch_size
        try:
            return [
                next(self.data) for _ in range(batch_size)
            ]
        except StopIteration:
            raise StopIteration("End of data stream reached")
    
    def tokenized_batch(self, batch_size=None):
        """
        Return a batch of tokenized inputs.
        """
        raise ValueError("tokenized_batch not implemented")
        texts = self.text_batch(batch_size=batch_size)
        return self.model.tokenizer(
            texts,
            return_tensors='pt',
            max_length=self.ctx_len,
            padding=True,
            truncation=True
        )

    def refresh(self):
        gc.collect()
        t.cuda.empty_cache()
        self.activations = self.activations[~self.read]

        current_idx = len(self.activations)
        
        # [修改] 根据模式重新分配 buffer 空间
        if self.io == 'in_and_out':
            new_activations = t.empty(self.activation_buffer_size, 2, self.d_submodule, device=self.device, dtype=self.model.dtype)
        else:
            new_activations = t.empty(self.activation_buffer_size, self.d_submodule, device=self.device, dtype=self.model.dtype)

        new_activations[: len(self.activations)] = self.activations
        self.activations = new_activations

        # Optional progress bar
        # pbar = tqdm(total=self.activation_buffer_size, initial=current_idx, desc="Refreshing activations")
        while current_idx < self.activation_buffer_size:
            input_data = None
            hidden_states_in = None
            hidden_states_out = None

            try:
                with t.no_grad():
                    with self.model.trace(
                        self.text_batch(),
                        **tracer_kwargs,
                        truncation=True,
                        max_length=self.ctx_len,
                    ) as tracer:
                        input_data = self.model.inputs.save()

                        # [修改] 分别捕获 Input 和 Output
                        if self.io in ["in", "in_and_out"]:
                            hidden_states_in = self.submodule.inputs[0].save()
                        
                        if self.io in ["out", "in_and_out"]:
                            hidden_states_out = self.submodule.output.save()
                        
                        tracer.stop()
            except Exception as e:
                print("trace error", e)
                continue
            
            # 获取 Attention Mask
            attn_mask = input_data[1]["attention_mask"]
            
            # 提取 Tensor 值并处理 Tuple 情况
            if hidden_states_in is not None:
                if isinstance(hidden_states_in, tuple): hidden_states_in = hidden_states_in[0]
            
            if hidden_states_out is not None:
                if isinstance(hidden_states_out, tuple): hidden_states_out = hidden_states_out[0]

            # print(hidden_states_in.mean(), hidden_states_out.mean())
            # print(hidden_states_in.std(dim=0).mean(), hidden_states_out.std(dim=0).mean())
            # print()
            # hidden_states_in = hidden_states_in * 0.5
            # hidden_states_out = (hidden_states_out + 1.5) * 0.125

            # [修改] 处理 remove_bos
            if self.remove_bos:
                attn_mask = attn_mask[:, 1:]
                if hidden_states_in is not None: hidden_states_in = hidden_states_in[:, 1:, :]
                if hidden_states_out is not None: hidden_states_out = hidden_states_out[:, 1:, :]
            
            # [修改] 应用 Masking 并处理不同模式
            valid_indices = attn_mask != 0 # Boolean mask [Batch, Seq]

            if self.io == 'in_and_out':
                # 关键：先分别 Mask 展平，再 Stack
                # 这样可以丢弃 Padding 对应的无效对
                flat_in = hidden_states_in[valid_indices]   # [Total_Valid, Dim]
                flat_out = hidden_states_out[valid_indices] # [Total_Valid, Dim]
                hidden_states = t.stack([flat_in, flat_out], dim=1) # [Total_Valid, 2, Dim]
            
            elif self.io == 'in':
                hidden_states = hidden_states_in[valid_indices] # [Total_Valid, Dim]
            
            else: # out
                hidden_states = hidden_states_out[valid_indices] # [Total_Valid, Dim]

            # 填充 Buffer
            remaining_space = self.activation_buffer_size - current_idx
            assert remaining_space > 0
            
            # 边界情况：如果这一批全是 padding (极少见但可能)，hidden_states 为空
            if len(hidden_states) > 0:
                hidden_states = hidden_states[:remaining_space]
                self.activations[current_idx : current_idx + len(hidden_states)] = hidden_states.to(self.device)
                current_idx += len(hidden_states)

            # pbar.update(len(hidden_states))

        # pbar.close()
        self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)

    @property
    def config(self):
        return {
            'd_submodule' : self.d_submodule,
            'io' : self.io,
            'n_ctxs' : self.n_ctxs,
            'ctx_len' : self.ctx_len,
            'refresh_batch_size' : self.refresh_batch_size,
            'out_batch_size' : self.out_batch_size,
            'device' : self.device
        }

    def close(self):
        """
        Close the text stream and the underlying compressed file.
        """
        self.text_stream.close()