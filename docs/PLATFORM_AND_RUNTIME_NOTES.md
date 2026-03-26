# Platform and Runtime Notes

This note is for students using either a **local machine** or **Google Colab**.
All numbers here are **reference only**, not grading targets.

## 1. Local machine quick start

```bash
git clone <your-private-repo-url>
cd <your-repo-name>
pip install -e ".[dev]"

python scripts/download_english_dataset.py \
  --dataset tinystories \
  --max_examples 50000 \
  --output_path output/english_data/tinystories_train_50k.jsonl

python scripts/prepare_packed_dataset.py \
  --input_path output/english_data/tinystories_train_50k.jsonl \
  --tokenizer_path assets/tokenizers/english_bytebpe_8k.json \
  --output_dir output/packed/tinystories_tiny \
  --max_seq_len 512 \
  --max_examples 50000 \
  --no_add_special_tokens

python scripts/train_model.py --config configs/tiny.yaml --num_epochs 1
```

If CUDA is available, training will use your GPU automatically. Otherwise it
will fall back to CPU.

Very small smoke-test variant if you only want to check that the pipeline runs:

```bash
python scripts/download_english_dataset.py \
  --dataset tinystories \
  --max_examples 500 \
  --output_path output/english_data/tinystories_train_500.jsonl

python scripts/prepare_packed_dataset.py \
  --input_path output/english_data/tinystories_train_500.jsonl \
  --tokenizer_path assets/tokenizers/english_bytebpe_8k.json \
  --output_dir output/packed/tinystories_tiny_smoke \
  --max_seq_len 512 \
  --max_examples 500 \
  --no_add_special_tokens

python scripts/train_model.py \
  --config configs/tiny.yaml \
  --data_path output/packed/tinystories_tiny_smoke \
  --checkpoint_dir checkpoints/tiny_smoke \
  --batch_size 8 \
  --num_epochs 1
```

On such a tiny run, the validation split may contain only a few examples. That
is fine for smoke testing, but not for your main report evidence.

## 2. Google Colab quick start

In Colab:
1. open **Runtime -> Change runtime type**
2. select **GPU**
3. run:

```python
!git clone <your-private-repo-url>
%cd <your-repo-name>
!pip install -e ".[dev]"

!python scripts/download_english_dataset.py \
  --dataset tinystories \
  --max_examples 50000 \
  --output_path output/english_data/tinystories_train_50k.jsonl

!python scripts/prepare_packed_dataset.py \
  --input_path output/english_data/tinystories_train_50k.jsonl \
  --tokenizer_path assets/tokenizers/english_bytebpe_8k.json \
  --output_dir output/packed/tinystories_tiny \
  --max_seq_len 512 \
  --max_examples 50000 \
  --no_add_special_tokens

!python scripts/train_model.py --config configs/tiny.yaml --num_epochs 1
```

Colab reminder:
- save checkpoints and reports before the session ends
- for longer runs, save artifacts to Drive or download them

## 3. Reference 1-GPU runtimes

Measured on one machine with:
- 1x NVIDIA RTX PRO 6000 Blackwell Max-Q
- PyTorch 2.9.1+cu128

Reference timings:
- **English smaller baseline**  
  `configs/tiny.yaml` on TinyStories 50k, 1 epoch:  
  about **44.7 s**
- **English medium baseline**  
  `configs/small.yaml` on TinyStories 50k, 1 epoch:  
  about **91.9 s**
- **English stronger baseline that can already produce readable story-like text**  
  `configs/small_plus.yaml` on TinyStories 50k, 1 epoch:  
  about **163.3 s**
- **English optimized run**  
  `configs/optional/experiment_configs.yaml --experiment modern --num_epochs 4`
  on TinyStories 50k: about **106.8 s**

Your hardware may be slower or faster.

## 4. TinyStories 50k examples

### 4.1 Baseline English example

Setup:
- dataset: **TinyStories 50k**
- config: `configs/small_plus.yaml`
- tokenizer: `assets/tokenizers/english_bytebpe_8k.json`
- training: **1 GPU, 1 epoch**

Prompt:

```text
The little girl said
```

Sample output:

```text
The little girl said. She was very excited to go to the park. She was so excited to go to the park.

The girl was so excited to see what was inside. She asked her mom if she could do.

The girl said, "I'm going to the park. I will be
```

### 4.2 4-epoch optimized English example

Setup:
- dataset: **TinyStories 50k**
- model tweaks: **RMSNorm + RoPE + GQA + SwiGLU** (`modern`)
- tokenizer: `assets/tokenizers/english_bytebpe_8k.json`
- training: **1 GPU, 4 epochs via `--num_epochs 4` override on top of the optional experiment config**
- validation perplexity after the run: about **9.48**

Prompt:

```text
The little girl said
```

Sample output:

```text
The little girl said She was very excited. She wanted to go to the park. She asked her mom if she could go. Her mom said yes.

The little girl was so excited. She ran to the park and saw lots of other kids playing. She was so excited! She ran up to them and said, "Look, Mommy! I found a big, red ball!"
```

This second sample is still not perfect, but it is clearly more readable and
story-like than the shorter baseline run.

## 5. SFT note

The released starter repo does **not** currently include a runnable SFT trainer.
At the moment:
- `scripts/train_sft.py` exits with a message explaining that SFT is still a
  design/report extension
- so we do **not** include an honest SFT runtime or SFT generation benchmark
  here yet

## 6. Practical advice

- Start with a **1-epoch smoke test**
- If your hardware is limited, do the baseline first and keep experiments small
- Do not assume weak generation means your implementation is broken
