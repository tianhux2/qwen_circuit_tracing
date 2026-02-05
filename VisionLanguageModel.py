from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from transformers import AutoProcessor, BatchEncoding
from nnsight import LanguageModel

class VisionLanguageModel(LanguageModel):
    def __init__(
        self,
        *args,
        processor: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.processor = processor
        super().__init__(*args, **kwargs)

    def _load_processor(self, repo_id: str, **kwargs):
        if self.processor is None:
            try:
                self.processor = AutoProcessor.from_pretrained(repo_id, **kwargs)
                if self.tokenizer is None and hasattr(self.processor, "tokenizer"):
                    self.tokenizer = self.processor.tokenizer
            except Exception:
                pass

    def _load_meta(self, repo_id: str, **kwargs):
        self._load_processor(repo_id, **kwargs)
        return super()._load_meta(repo_id, **kwargs)

    def _load(self, repo_id: str, **kwargs):
        self._load_processor(repo_id, **kwargs)
        return super()._load(repo_id, **kwargs)

    def _tokenize(
        self,
        inputs: Union[str, List[str]],
        images: Optional[List[Any]] = None,
        **kwargs,
    ):
        """
        Tokenize logic that handles mixed inputs.
        """
        if self.processor is not None:
            # images should be a list of images with None values filtered out
            # If the list is empty (i.e., text-only input), set it to None so the processor handles it correctly
            if images is not None and len(images) == 0:
                images = None

            # 1. Text-only case
            if images is None:
                return self.processor(text=inputs, return_tensors="pt", padding=True, **kwargs)
            
            # 2. Mixed text and image case
            # Note: Most VLM processors (like LLaVA) require images to be a flattened list
            # They match images based on the number or order of <image> tokens in the text
            return self.processor(text=inputs, images=images, return_tensors="pt", padding=True, **kwargs)

        return super()._tokenize(inputs, **kwargs)

    def _prepare_input(
        self,
        *inputs,
        images: Any = None,
        **kwargs,
    ) -> Tuple[Tuple[()], Dict[str, Any]]:
        """
        Parse inputs, supporting mixed formats:
        1. model([("prompt", img), "text only", ("prompt", img2)])
        2. model("prompt", img)
        3. model(["p1", "p2"], images=[i1, i2]) - Legacy support
        """
        input_texts = []
        input_images = []

        # --- 解析逻辑 ---
        
        # 检查是否是单个 Batch 列表输入: model([item1, item2, ...])
        if len(inputs) == 1 and isinstance(inputs[0], list):
            raw_batch = inputs[0]
            for item in raw_batch:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    # 格式: ("Text", Image)
                    text_part, img_part = item
                    input_texts.append(text_part)
                    if img_part is not None:
                        input_images.append(img_part)
                elif isinstance(item, str):
                    # 格式: "Text only"
                    input_texts.append(item)
                    # 纯文本不需要向 input_images 添加 None，大多数 processor 会根据 <image> token 自动匹配
                else:
                    # 异常情况，作为纯文本处理或报错
                    input_texts.append(str(item))

        # 检查是否是经典的 model("text", image) 位置参数调用
        elif len(inputs) >= 1 and isinstance(inputs[0], str):
            input_texts.append(inputs[0])
            # 检查第二个位置参数是否是图片
            if len(inputs) > 1:
                input_images.append(inputs[1])
            # 检查 images 关键字参数 (legacy support)
            elif images is not None:
                if isinstance(images, list):
                    input_images.extend(images)
                else:
                    input_images.append(images)
        
        # Fallback: 如果上面的逻辑都没命中，尝试从 kwargs 获取 text
        if not input_texts:
            text_arg = kwargs.pop("text", None)
            if isinstance(text_arg, str):
                input_texts.append(text_arg)
            elif isinstance(text_arg, list):
                input_texts.extend(text_arg)
            
            # 如果 kwargs 里有 images
            if images is not None:
                 if isinstance(images, list):
                    input_images.extend(images)
                 else:
                    input_images.append(images)

        # --- 调用 Tokenize ---
        
        # 确保传递给 processor 的是纯净的列表
        # 如果 input_texts 只有一个元素，解包成字符串（某些 processor 对单个字符串和列表行为不同）
        final_texts = input_texts if len(input_texts) > 1 else input_texts[0] if input_texts else ""
        
        # 调用 _tokenize
        encodings = self._tokenize(final_texts, images=input_images, **kwargs)

        # Extract labels if present
        labels = kwargs.get("labels", None)
        
        return tuple(), {**encodings, "labels": labels}

    def _batch(
        self,
        batched_inputs: Optional[Tuple[Tuple[BatchEncoding], Dict[str, Any]]],
        **prepared_kwargs,
    ) -> Tuple[Dict[str, Any]]:
        raise ValueError("VisionLanguageModel does not support batching now")
        # _batch 逻辑保持之前的实现不变，用于处理 pixel_values 的拼接
        (args, new_batched_data), count = super()._batch(batched_inputs, **prepared_kwargs)

        new_pixel_values = prepared_kwargs.get("pixel_values", None)
        
        if new_pixel_values is not None:
            if batched_inputs is None:
                new_batched_data["pixel_values"] = new_pixel_values
            else:
                prev_data = batched_inputs[1]
                existing_pixels = prev_data.get("pixel_values", None)
                if existing_pixels is not None:
                    concatenated_pixels = torch.cat((existing_pixels, new_pixel_values), dim=0)
                    new_batched_data["pixel_values"] = concatenated_pixels
                else:
                    new_batched_data["pixel_values"] = new_pixel_values

        for key, val in prepared_kwargs.items():
            if key not in ["input_ids", "attention_mask", "labels", "pixel_values"] and isinstance(val, torch.Tensor):
                if batched_inputs is None:
                     new_batched_data[key] = val
                else:
                    prev_data = batched_inputs[1]
                    if key in prev_data:
                        new_batched_data[key] = torch.cat((prev_data[key], val), dim=0)

        return (args, new_batched_data), count