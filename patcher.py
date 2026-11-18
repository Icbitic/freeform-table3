"""The original cached forward path, retaining only head-output capture."""
from dataclasses import dataclass
import torch


@dataclass
class Captured:
    head: list


class HeadPatcher:
    def __init__(self, model, device, tokenizer):
        self.model, self.device, self.tokenizer = model, device, tokenizer
        self.config = getattr(model.config, "text_config", model.config)
        decoder = model.model.language_model if model.config.model_type == "gemma3" else model.model
        self.layers = list(decoder.layers)
        self.n_heads = self.config.num_attention_heads
        self.head_dim = self.layers[0].self_attn.head_dim
        if any(layer.self_attn.o_proj.in_features != self.n_heads * self.head_dim for layer in self.layers):
            raise ValueError("Unexpected attention head concatenation shape")
        self._capture = None
        self._handles = [layer.self_attn.o_proj.register_forward_pre_hook(self._head_hook(i))
                         for i, layer in enumerate(self.layers)]

    def _head_hook(self, layer):
        def hook(module, args):
            if self._capture is not None:
                self._capture.head[layer] = args[0][:, -1:, :].detach().clone()
        return hook

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def encode(self, messages):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 2 or len(ids) > self.config.max_position_embeddings:
            raise ValueError("Prompt is too short or exceeds the context window")
        return ids

    @torch.inference_mode()
    def prefill(self, ids):
        inputs = torch.tensor([ids[:-1]], device=self.device)
        output = self.model(input_ids=inputs, use_cache=True, logits_to_keep=1, return_dict=True)
        return output.past_key_values

    @torch.inference_mode()
    def last(self, ids, cache, capture=False):
        length = len(ids) - 1
        if cache.get_seq_length() != length:
            raise ValueError("Replay cache does not match this prompt's prefix length")
        inputs = torch.full((1, 1), ids[-1], dtype=torch.long, device=self.device)
        positions = torch.full((1, 1), length, dtype=torch.long, device=self.device)
        attention = torch.ones((1, length + 1), dtype=torch.long, device=self.device)
        if capture:
            self._capture = Captured([None] * len(self.layers))
        try:
            output = self.model(input_ids=inputs, attention_mask=attention, position_ids=positions,
                                cache_position=positions[0], past_key_values=cache, use_cache=True,
                                logits_to_keep=1, return_dict=True)
            scores = output.logits[:, -1, :].float()
            if not torch.isfinite(scores).all():
                raise ValueError("Non-finite logits")
            return scores, self._capture
        finally:
            self._capture = None
