from scripts.eval.sample_prefix_suffix_trajectory_cfg import (
    apply_generated_state,
    decode_generated_text,
    parse_args,
)


class _FakeModel:
    def __init__(self):
        self.calls = []

    def blend_into_cache(self, cache, states, blend):
        self.calls.append(("blend", cache, states, blend))
        return "blended"

    def inject_into_cache(self, cache, states):
        self.calls.append(("inject", cache, states))
        return "injected"


def test_independent_generation_honors_partial_state_blend():
    model = _FakeModel()
    result = apply_generated_state(model, "cache", ["state"], blend=0.5)

    assert result == "blended"
    assert model.calls == [("blend", "cache", ["state"], 0.5)]


def test_generation_defaults_to_ddpm_and_full_text_length():
    args = parse_args(["--ckpt_dir", "x", "--prompt", "x", "--output", "x"])

    assert args.diffusion_sampler == "ddpm"
    assert args.max_new_tokens == 512


def test_generation_decode_skips_special_tokens_and_replacement_chars():
    class _FakeTokenizer:
        def decode(self, ids, **kwargs):
            assert kwargs["skip_special_tokens"] is True
            return "clean\ufffd<|rwkv_tokenizer_end_of_text|>text"

    assert decode_generated_text(_FakeTokenizer(), [1, 2]) == "clean text"
