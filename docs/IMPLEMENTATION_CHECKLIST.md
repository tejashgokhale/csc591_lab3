# Implementation Checklist

This document is the **student-facing execution map** for the lab.

Use it together with:
- `docs/LAB_GUIDE.md` for the overall workflow and scope
- `docs/LAB_CONCEPTS.md` for theory and math

This file answers a narrower question:

> **Exactly which code should I implement for the baseline, how do I test it, and what should I expect to see?**

Important interpretation:
- this checklist focuses on the **baseline required path**
- optional features such as GQA, RoPE, RMSNorm, SwiGLU, MoE, beam search, bucket batching, quantization, and SFT are **not** required for baseline completion
- training your own tokenizer is **not** required for the default English baseline because the repository already provides one

---

## 1. Recommended order

Follow this order unless you intentionally choose a different experimental path:

1. Part 1 core model
2. Part 2 data + loss + training smoke path
3. Part 2 generation path
4. one real baseline run
5. controlled experiments

Do **not** try to implement every `TODO` in the repository before your first end-to-end run works.

---

## 2. Baseline implementation checklist

For each row:
- **Required?** tells you whether the item is on the baseline path
- **Validate with** tells you the most relevant released test or command
- **Expected outcome** tells you what success should look like

### 2.1 Core model path

| File | Symbol / implementation target | Required? | Validate with | Expected outcome |
|---|---|---:|---|---|
| `src/components/activation.py` | activation(s) needed by your chosen baseline, usually `GELU` and `get_activation()` | Yes | `pytest -m part1 -v` | FFN path can construct and run |
| `src/components/attention.py` | `ScaledDotProductAttention.forward` | Yes | `pytest -m part1 -v` | attention weights have the right shape, masking works, weights sum to 1 |
| `src/components/attention.py` | `MultiHeadAttention.__init__`, `_split_heads`, `_combine_heads`, `forward` | Yes | `pytest -m part1 -v` | MHA output shape is correct and gradients flow |
| `src/components/attention.py` | `create_causal_mask`, `create_padding_mask` | Yes | `pytest -m part1 -v` | causal mask is lower triangular; padding mask marks real vs padding tokens correctly |
| `src/components/positional.py` | `SinusoidalPositionalEncoding.__init__` and `forward` | Yes | `pytest -m part1 -v` | position information can be added to token embeddings without shape errors |
| `src/components/normalization.py` | one working normalization path, usually `LayerNorm`; plus whichever residual-norm composition your decoder uses | Yes | `pytest -m part1 -v` | model forward is numerically stable and gradients exist |
| `src/components/feedforward.py` | one working FFN path, usually `PositionWiseFeedForward` | Yes | `pytest -m part1 -v` | decoder block can run end to end |
| `src/components/transformer.py` | `TransformerDecoderLayer` baseline path | Yes | `pytest -m part1 -v` | decoder layer preserves `(B, T, d_model)` shape |
| `src/model/language_model.py` | `__init__` baseline modules | Yes | `pytest -m part1 -v` | model constructs successfully from `ModelConfig` |
| `src/model/language_model.py` | `forward` | Yes | `pytest -m part1 -v` | logits have shape `(B, T, V)` |
| `src/model/language_model.py` | `generate` baseline path | Yes | `pytest -m part4_generation -v` | generated sequence becomes longer without crashing |

### 2.2 Tokenizer path

For the default English baseline, use:

```text
assets/tokenizers/english_bytebpe_8k.json
```

That means the baseline requirement is:

| File | Symbol / implementation target | Required? | Validate with | Expected outcome |
|---|---|---:|---|---|
| `src/tokenizer/loading.py` | tokenizer loading behavior for provided tokenizer | Yes | `python scripts/test_tokenizer.py --tokenizer_path assets/tokenizers/english_bytebpe_8k.json --text "Once upon a time,"` | tokenizer loads, encodes, and decodes successfully |
| `src/tokenizer/bpe.py`, `scripts/train_tokenizer.py` | train-your-own-tokenizer path | No for baseline | optional: `pytest -m part2_tokenizer -v` | only needed if you intentionally explore tokenizer training |

### 2.3 Dataset and batching path

| File | Symbol / implementation target | Required? | Validate with | Expected outcome |
|---|---|---:|---|---|
| `src/data/dataset.py` | `LanguageModelingDataset._load_and_tokenize` | Yes | small local batch inspection; later `pytest -m part3_training -v` | examples shorter than 2 tokens are filtered; long examples are skipped |
| `src/data/dataset.py` | `LanguageModelingDataset.__getitem__` | Yes | small local batch inspection | returns shifted `(input_ids, target_ids)` |
| `src/data/dataloader.py` | `collate_fn` | Yes | `pytest -m part3_training -v` | sequences are padded to common length and attention mask uses `1` for real tokens, `0` for padding |
| `src/data/dataloader.py` | `create_dataloader` | Yes | `pytest -m part3_training -v` | dataloader returns `(input_ids, target_ids, attention_mask)` batches |
| `scripts/prepare_packed_dataset.py` | packed-data preprocessing workflow | Use for recommended English baseline | run the command from `docs/LAB_GUIDE.md` | output directory contains `tokens.bin`, `offsets.npy`, `lengths.npy`, split indices, and `metadata.json` |

### 2.4 Loss, metrics, and training path

| File | Symbol / implementation target | Required? | Validate with | Expected outcome |
|---|---|---:|---|---|
| `src/training/loss.py` | `LanguageModelingLoss.forward` | Yes | `pytest -m part3_training -v` | cross-entropy runs and ignores padding tokens |
| `src/training/loss.py` | `compute_perplexity` | Yes | `pytest -m part3_training -v` | perplexity is finite; `loss = 0` gives perplexity `1` |
| `src/training/loss.py` | `compute_accuracy` | Yes in released baseline path | `pytest -m part3_training -v` and real training run | token accuracy can be logged without crashing |
| `src/training/loss.py` | `compute_top_k_accuracy` | Yes in released baseline path | real training run | top-k helper metric can be computed without crashing |
| `src/training/loss.py` | `MetricsTracker.update` and `MetricsTracker.compute` | Yes | `pytest -m part3_training -v` | running averages are correct |
| `src/training/trainer.py` | read and understand the provided baseline trainer | Yes conceptually | real training run | you can explain gradient accumulation, scheduler stepping, clipping, and checkpoint contents |
| `src/training/scheduler.py` | baseline scheduler path is already provided | No major coding needed for baseline | real training run | LR changes across training without blocking progress |
| `scripts/train_model.py` | config-driven training workflow | Yes | baseline training command | training runs, prints finite metrics, and saves checkpoints |

### 2.5 Generation path

| File | Symbol / implementation target | Required? | Validate with | Expected outcome |
|---|---|---:|---|---|
| `src/generation/generator.py` | `TextGenerator.generate` | Yes | `pytest -m part4_generation -v` | prompt can be encoded, generated, and decoded |
| `src/generation/generator.py` | `_generate_tokens` | Yes | `pytest -m part4_generation -v` | autoregressive token generation works |
| `src/generation/generator.py` | greedy decoding path | Yes | `pytest -m part4_generation -v` | at least one deterministic decode path works |
| `src/generation/generator.py` | temperature + one filtering strategy (`top-k` or `top-p`) | Yes | real generation command | you can show one stochastic sample in the report |
| `scripts/generate_text.py` | end-to-end checkpoint loading and generation | Yes | generation command in `docs/LAB_GUIDE.md` | generation runs from your own checkpoint without crashing |

---

## 3. Stage-by-stage done conditions

### Part 1 done condition

You are ready to leave Part 1 when:
- `pytest -m part1 -v` passes
- model forward returns logits of shape `(B, T, V)`
- gradients exist for model parameters
- you can explain causal masking

### Part 2 data/training done condition

You are ready to move on when:
- `pytest -m part3_training -v` passes
- one baseline training command runs without crashing
- loss is finite
- validation metrics are finite
- checkpoint files appear in the expected directory

### Part 2 generation done condition

You are ready to write the report baseline section when:
- `pytest -m part4_generation -v` passes
- `src/model/language_model.py::generate` works
- `src/generation/generator.py` works
- `python scripts/generate_text.py ...` runs from your own checkpoint

---

## 4. Recommended validation flow

For the **default baseline path**, use:

```bash
pytest -m part1 -v
pytest -m part3_training -v
pytest -m part4_generation -v
pytest -m "not stretch" -v
```

Then confirm with real commands:

```bash
python scripts/test_tokenizer.py \
  --tokenizer_path assets/tokenizers/english_bytebpe_8k.json \
  --text "Once upon a time,"

python scripts/train_model.py --config configs/tiny.yaml --num_epochs 1

python scripts/generate_text.py \
  --checkpoint checkpoints/tiny/best_model.pt \
  --tokenizer assets/tokenizers/english_bytebpe_8k.json \
  --prompt "Once upon a time," \
  --max_new_tokens 100
```

Optional tokenizer-training validation, only if you intentionally train your own tokenizer:

```bash
pytest -m part2_tokenizer -v
```

---

## 5. Common “I am stuck” diagnosis map

If `part1` fails:
- start with `src/components/attention.py`
- then check positional encoding, normalization, FFN, decoder layer, and LM forward wiring

If `part3_training` fails:
- first check `collate_fn`
- then check shifted input/target pairs
- then check `LanguageModelingLoss.forward`
- then check metrics helpers in `src/training/loss.py`

If `part4_generation` fails:
- check both `src/model/language_model.py::generate` and `src/generation/generator.py`
- make sure logits are taken from the **last position**
- make sure sampled / greedy next tokens are appended correctly

If the real training command fails after tests pass:
- check whether the packed dataset directory exists
- check whether the tokenizer path matches the config
- check whether the checkpoint directory is the one you expect

---

## 6. What you do not need for the baseline

You do **not** need the following before your first successful end-to-end run:
- GQA
- RoPE
- RMSNorm
- SwiGLU / GeGLU / MoE
- beam search
- bucket batching
- AMP optimization
- quantization
- SFT
- training your own tokenizer

Get the baseline working first, then use optional features as clean experiment variables.
