"""CPU inference loading and generation tests."""

from __future__ import annotations

import torch

from src.inference.common import load_model_and_tokenizer
from src.models import ModelConfig, SSMLanguageModel
from src.training.checkpointing import save_inference_export


def test_cpu_inference_loads_checkpoint_and_generates(tmp_path) -> None:
    tokenizers = __import__("tokenizers")
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.train_from_iterator(
        ["hello world", "<|user|> hello <|assistant|> hi", "small model"],
        trainer=BpeTrainer(vocab_size=32, special_tokens=["<unk>", "<eos>"]),
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    config = ModelConfig(vocab_size=tokenizer.get_vocab_size(), d_model=16, n_layers=2, d_state=4, max_seq_len=32)
    model = SSMLanguageModel(config)
    checkpoint = tmp_path / "model.pt"
    save_inference_export(model, checkpoint, model_config=config.__dict__, step=0)
    loaded, loaded_tokenizer, device, _ = load_model_and_tokenizer(checkpoint, tokenizer_path, device="cpu")
    ids = torch.tensor([loaded_tokenizer.encode("hello").ids])
    output = loaded.generate(ids, 3)
    assert device.type == "cpu"
    assert output.shape == (1, ids.size(1) + 3)
