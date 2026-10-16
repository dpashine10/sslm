import torch
from torch.nn import functional as F

from models.model import CausalLMOutput, ModelConfig, SSMLanguageModel
from models.ssm import SelectiveSSM


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=29,
        d_model=12,
        n_layers=2,
        d_state=4,
        d_conv=3,
        d_inner=24,
        dt_rank=4,
        dropout=0.0,
    )


def test_logits_shape_loss_alignment_and_tied_embeddings() -> None:
    torch.manual_seed(0)
    model = SSMLanguageModel(tiny_config())
    input_ids = torch.randint(0, model.config.vocab_size, (3, 7))
    targets = torch.randint(0, model.config.vocab_size, (3, 7))

    output = model(input_ids, targets)

    assert isinstance(output, CausalLMOutput)
    assert output.logits.shape == (3, 7, model.config.vocab_size)
    expected = F.cross_entropy(
        output.logits[:, :-1].reshape(-1, model.config.vocab_size),
        targets[:, 1:].reshape(-1),
    )
    torch.testing.assert_close(output.loss, expected)
    assert model.lm_head.weight is model.tok_embeddings.weight
    assert model.parameter_count == model.num_parameters()
    assert model.parameter_count > 0


def test_backward_produces_finite_gradients() -> None:
    torch.manual_seed(1)
    model = SSMLanguageModel(tiny_config())
    input_ids = torch.randint(0, model.config.vocab_size, (2, 6))
    output = model(input_ids, input_ids)
    assert output.loss is not None

    output.loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_ssm_full_scan_matches_incremental_steps() -> None:
    torch.manual_seed(2)
    mixer = SelectiveSSM(
        d_model=8,
        d_state=3,
        d_inner=16,
        dt_rank=3,
        d_conv=4,
    )
    x = torch.randn(2, 9, 8)

    full_output, full_state = mixer.full_scan(x)
    state = mixer.initial_state(x.size(0), device=x.device, dtype=x.dtype)
    incremental_outputs = []
    for token in x.unbind(dim=1):
        output, state = mixer.step(token, state)
        incremental_outputs.append(output)
    incremental_output = torch.stack(incremental_outputs, dim=1)

    torch.testing.assert_close(full_output, incremental_output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(full_state.conv_state, state.conv_state, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(full_state.ssm_state, state.ssm_state, rtol=1e-5, atol=1e-6)


def test_model_full_logits_match_incremental_cache() -> None:
    torch.manual_seed(3)
    model = SSMLanguageModel(tiny_config()).eval()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))

    full = model(input_ids)
    cache = model.init_cache(input_ids.size(0), device=input_ids.device)
    incremental = []
    for token in input_ids.unbind(dim=1):
        logits, cache = model.step(token, cache)
        incremental.append(logits)
    incremental = torch.stack(incremental, dim=1)

    torch.testing.assert_close(full, incremental, rtol=2e-5, atol=2e-6)


def test_greedy_generation_is_deterministic_and_sampling_filters_work() -> None:
    torch.manual_seed(4)
    model = SSMLanguageModel(tiny_config()).eval()
    prompt = torch.randint(0, model.config.vocab_size, (2, 4))

    greedy_a = model.generate(prompt, max_new_tokens=5, temperature=0.0)
    greedy_b = model.generate(prompt, max_new_tokens=5, do_sample=False)
    assert greedy_a.shape == (2, 9)
    torch.testing.assert_close(greedy_a, greedy_b)

    torch.manual_seed(5)
    sampled = model.generate(
        prompt,
        max_new_tokens=4,
        temperature=0.8,
        top_k=5,
        top_p=0.9,
        do_sample=True,
    )
    assert sampled.shape == (2, 8)
    assert int(sampled.min()) >= 0
    assert int(sampled.max()) < model.config.vocab_size
