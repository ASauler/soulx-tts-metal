import numpy as np
import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.models.cache import make_prompt_cache

from soulxpodcast.config import Config, SamplingParams
from dataclasses import fields


class MLXLLMEngine:
    """MLX-based LLM engine for Apple Silicon acceleration."""

    def __init__(self, model, mlx_model_path=None, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        config.eos = config.hf_config.eos_token_id

        if mlx_model_path is None:
            mlx_model_path = model + "-mlx"

        print(f"[INFO] MLX engine loading: {mlx_model_path}")
        self.mlx_model, self.tokenizer = mlx_load(mlx_model_path)
        print(f"[INFO] MLX model loaded")

        self.config = config
        self.pad_token_id = self.tokenizer.eos_token_id

    def make_cache(self):
        """Create a fresh KV cache for generate_step."""
        return make_prompt_cache(self.mlx_model)

    def generate(
        self,
        prompt: list[int],
        sampling_param: SamplingParams,
        past_key_values=None,
    ) -> dict:
        """Generate speech tokens using MLX.

        Args:
            prompt: list of token ids (input context)
            sampling_param: sampling parameters
            past_key_values: MLX prompt_cache (from make_cache()), updated in-place

        Returns:
            dict with 'text' and 'token_ids'
        """
        eos_token_id = self.config.hf_config.eos_token_id
        prompt_array = mx.array(prompt)

        generated_tokens = []
        speech_token_offset = self.config.hf_config.speech_token_offset

        mx.eval(mx.zeros(1))

        # Use passed cache for multi-turn KV reuse, or create fresh one
        cache = past_key_values if past_key_values is not None else make_prompt_cache(self.mlx_model)

        sampler = make_sampler(
            temp=sampling_param.temperature,
            top_p=sampling_param.top_p,
            top_k=sampling_param.top_k,
        )

        # Repetition penalty (matches HF engine's RepetitionPenaltyLogitsProcessor)
        rep_processors = make_logits_processors(
            repetition_penalty=sampling_param.repetition_penalty,
            repetition_context_size=50,
        )

        # Count speech tokens to detect runaway generation
        # 25Hz speech tokens, cap at 60% of max_tokens as safety limit
        max_speech_tokens = max(500, int(sampling_param.max_tokens * 0.6))
        speech_token_count = 0

        for (token, logprobs), _ in zip(
            generate_step(
                prompt=prompt_array,
                model=self.mlx_model,
                sampler=sampler,
                max_tokens=sampling_param.max_tokens,
                prompt_cache=cache,
                logits_processors=rep_processors,
            ),
            range(sampling_param.max_tokens),
        ):
            token_id = token.item() if hasattr(token, 'item') else int(token)

            # RAS (Repetition Aware Sampling) - re-sample if too repetitive
            if sampling_param.use_ras and len(generated_tokens) > 0:
                window = generated_tokens[-sampling_param.win_size:]
                rep_count = window.count(token_id) + 1
                if rep_count >= sampling_param.win_size * sampling_param.tau_r:
                    logits_np = np.array(logprobs)
                    logits_np = logits_np / sampling_param.temperature
                    probs = _softmax(logits_np)
                    sorted_indices = np.argsort(-probs)
                    sorted_probs = probs[sorted_indices]
                    cumsum = np.cumsum(sorted_probs)
                    mask = cumsum - sorted_probs > sampling_param.top_p
                    sorted_probs[mask] = 0
                    sorted_probs /= sorted_probs.sum()
                    token_id = np.random.choice(sorted_indices, p=sorted_probs)

            if token_id == eos_token_id:
                generated_tokens.append(token_id)
                break

            # Track speech tokens and cap runaway generation
            if token_id >= speech_token_offset:
                speech_token_count += 1
                if speech_token_count >= max_speech_tokens:
                    generated_tokens.append(eos_token_id)
                    print(f"[WARN] MLX: speech token limit reached ({max_speech_tokens}), forcing EOS")
                    break

            generated_tokens.append(token_id)

        # Ensure all MLX ops are complete before returning to PyTorch
        mx.eval(mx.zeros(1))

        output = {
            "text": self.tokenizer.decode(generated_tokens),
            "token_ids": generated_tokens,
        }
        return output


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
