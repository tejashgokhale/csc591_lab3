# Lab Guide

## 1. Lab goals and pacing

In this lab, you will build and study a **small decoder-only transformer
language model**. The main goals are to understand:
- how the transformer components fit together
- how autoregressive next-token training works
- how tokenizer, architecture, and training choices create trade-offs
- how to support claims with controlled experiments

Suggested pacing:
- **Week 1:** core model implementation
- **Week 2:** tokenizer, data, training, and generation baseline
- **Week 3:** controlled experiments, analysis, and report

Use these companion documents when needed:
- theory and formulas: `docs/LAB_CONCEPTS.md`
- exact baseline code targets + validation map: `docs/IMPLEMENTATION_CHECKLIST.md`
- local / Colab startup and runtime notes: `docs/PLATFORM_AND_RUNTIME_NOTES.md`
- short-answer questions: `docs/LAB_QUESTIONS.md`
- grading criteria: `docs/EVALUATION_RUBRIC.md`
- deliverables: `docs/WHAT_TO_SUBMIT.md`

## 1.1 Practical prerequisites

Before you start, make sure you understand which steps require extra tooling.

- Some helper scripts download public datasets the first time you run them, so
  they require internet access.
- The repository dependencies are listed in `pyproject.toml`; install them
  before running scripts.
- The docs sometimes use `rg` (`ripgrep`) to search for `TODO`s. If you do not
  have `rg`, use `grep -RIn "TODO" ...` instead. If you are in vscode, you can simply search for `TODO` in the file explorer.

## 2. Required vs optional

### Required
You must complete all of the following:
- one working baseline transformer path
- one working tokenizer/data/training/generation path
- **two** controlled experiments
- a report in `report/main.md`
- short answers in `report/answers.md`

### Optional / bonus
These can strengthen your lab, but they should **not** block the baseline:
- GQA
- RoPE
- RMSNorm
- gated FFN variants such as SwiGLU
- bucket batching
- AMP / throughput optimization
- beam search
- bonus: quantization
- bonus: SFT concept / design extension

Recommended order:
1. get one baseline path working
2. pass the released baseline-path tests
3. run two clean experiments
4. write the report and short answers

Scope advice:
- start with **one** baseline path only
- do not implement optional features before your first baseline smoke test works
- treat optional features as ablation variables, not as prerequisites
- for optional tooling such as `wandb`, do not assume it is preinstalled on your machine; only use it if you explicitly choose to install/configure it

## 3. Part 1: core model

### 3.1 Required baseline scope

| Scope | What you need for the baseline | Main file(s) |
|---|---|---|
| Attention core | scaled dot-product attention and multi-head attention | `src/components/attention.py` |
| Masking | causal mask and padding mask | `src/components/attention.py` |
| Position information | sinusoidal positional encoding | `src/components/positional.py` |
| Normalization | one working normalization path, usually LayerNorm | `src/components/normalization.py` |
| Feed-forward | one working FFN path, usually standard FFN | `src/components/feedforward.py` |
| Activations | the activation(s) needed by your chosen baseline | `src/components/activation.py` |
| Decoder block | residual connections and decoder-layer composition | `src/components/transformer.py` |
| LM forward path | embeddings, decoder stack, LM head, forward pass | `src/model/language_model.py` |

### 3.2 Optional architecture extensions

| Optional scope | Main file(s) |
|---|---|
| GQA | `src/components/attention.py` |
| RoPE | `src/components/positional.py` |
| RMSNorm | `src/components/normalization.py` |
| gated FFN variants such as SwiGLU / GeGLU | `src/components/feedforward.py` |
| MoE | `src/components/feedforward.py` |
| extra decoder/model variants | `src/components/transformer.py`, `src/model/language_model.py` |

### 3.3 Find implementation work

Most of the starter code that you need to fill in is marked with `TODO`. To
locate likely implementation points, run:

```bash
rg -n "TODO" src/components src/model src/data src/training src/generation scripts
```

Fallback if `rg` is not installed:

```bash
grep -RIn "TODO" src/components src/model src/data src/training src/generation scripts
```

Use the scope tables in this guide to decide which TODOs are required for your
baseline and which are optional.

For a more precise function-level execution map, use:
- `docs/IMPLEMENTATION_CHECKLIST.md`

Practical interpretation:
- if a class / section is labeled **OPTIONAL EXTENSION**, its TODOs are not
  required for the baseline
- the baseline scope table above is the safest source of truth
- the baseline scheduler / trainer infrastructure is now mostly prefilled so
  students can focus first on the model/data path instead of boilerplate

### 3.4 Validation

```bash
pytest -m part1 -v
```

The released `part1` tests are meant to cover the baseline path:
- attention math
- causal/padding masks
- baseline LM wiring and forward path

Optional architecture variants such as GQA, RoPE, and RMSNorm belong in
`stretch`, not in the baseline `part1` expectation.

### 3.5 Done condition for Part 1

Before moving on, make sure:
- Part 1 tests pass
- your model forward returns logits of shape `(B, T, V)`
- gradients flow without obvious NaNs
- you can explain why the causal mask is needed in a decoder-only LM

## 4. Part 2: tokenizer, data, training, and generation

### 4.1 Configs you will probably use

Most students only need the files below.

| Config file | Typical use |
|---|---|
| `configs/tiny.yaml` | recommended English baseline |
| `configs/small.yaml` | stronger English baseline; expects `output/packed/tinystories_small/` |
| `configs/small_plus.yaml` | optional stronger run; expects `output/packed/tinystories_small_plus/` |
| `configs/optional/experiment_configs.yaml` | optional architecture ablations after the baseline works |
| `configs/optional/sft_*.yaml` | optional reference configs for SFT discussion only |

Recommended sequence:
1. use `configs/tiny.yaml` to get the baseline working
2. use `configs/optional/experiment_configs.yaml` only after your baseline works and you want to turn on an optional feature

Workload note:
- on our reference GPU, smaller English configs are roughly on the order of
  about 1 minute and medium ones around 2 minutes
- the default lab path is intentionally lightweight; if you want, you can try
  larger configs or more advanced implementations on top of the baseline

If you move from `tiny.yaml` to `small.yaml` or `small_plus.yaml`, you must
prepare a packed dataset in the matching output directory first:

```bash
python scripts/prepare_packed_dataset.py \
  --input_path output/english_data/tinystories_train_50k.jsonl \
  --tokenizer_path assets/tokenizers/english_bytebpe_8k.json \
  --output_dir output/packed/tinystories_small \
  --max_seq_len 512 \
  --max_examples 50000 \
  --no_add_special_tokens

python scripts/prepare_packed_dataset.py \
  --input_path output/english_data/tinystories_train_50k.jsonl \
  --tokenizer_path assets/tokenizers/english_bytebpe_8k.json \
  --output_dir output/packed/tinystories_small_plus \
  --max_seq_len 512 \
  --max_examples 50000 \
  --no_add_special_tokens
```

### 4.2 Tokenizer workflow

For the default classroom path, use the provided tokenizer.

Use the tokenizer already included in the repository:

```text
assets/tokenizers/english_bytebpe_8k.json
```

This path does **not** require you to train your own tokenizer before you can
start the LM baseline.

#### Scope map

| Scope | Required work | Main file(s) |
|---|---|---|
| One usable tokenizer path | use the provided tokenizer correctly for the English baseline | `src/tokenizer/*` |
| Verification | sanity-check encode/decode behavior | `scripts/test_tokenizer.py` |
| Trade-off discussion | explain one or two concrete tokenization trade-offs | `report/main.md`, `report/answers.md` |

Important scope note:
- for the default English baseline, **training your own tokenizer is not required**
- the repository already includes a usable tokenizer at
  `assets/tokenizers/english_bytebpe_8k.json`
- released tokenizer-training tests such as `part2_tokenizer` are therefore
  **optional/reference** unless you intentionally choose a tokenizer-training path

#### Done condition

- one tokenizer path works end to end
- encode/decode behavior is sanity-checked
- you can explain at least one trade-off of that tokenizer choice

Recommended sanity-check command for the default English path:

```bash
python scripts/test_tokenizer.py \
  --tokenizer_path assets/tokenizers/english_bytebpe_8k.json \
  --text "Once upon a time,"
```

Expected outcome:
- the tokenizer loads without error
- encode/decode runs end to end
- the decoded text is broadly consistent with the input text
- you can see whether BOS/EOS or other special tokens were added

Common failure modes to check first:
- the tokenizer file path is wrong
- you pointed training/generation at the wrong tokenizer file
- you skipped `scripts/test_tokenizer.py` and only discovered the mismatch later

### 4.3 Dataset and dataloader

| Scope | Required work | Main file(s) |
|---|---|---|
| LM examples | shifted input/target pairs | `src/data/dataset.py` |
| Batch collation | padding to common length | `src/data/dataloader.py` |
| Attention mask | `1` for real tokens and `0` for padding | `src/data/dataloader.py` |

Minimum checks:
- examples shorter than 2 tokens are filtered
- shifted input/target pairs are correct
- padded batch tensors have consistent shapes
- attention mask semantics are correct

Success artifacts:
- raw JSONL or packed dataset directory exists
- a batch from your dataloader has shape `(B, T)` for inputs/targets
- `attention_mask` has `1` for real tokens and `0` for padding

### 4.4 Training baseline

| Scope | Required work | Main file(s) |
|---|---|---|
| LM loss | cross-entropy for next-token prediction | `src/training/loss.py` |
| Scheduler | baseline implementation is mostly provided; understand how it is stepped | `src/training/scheduler.py` |
| Training loop | baseline implementation is mostly provided; understand the flow of one epoch and validation | `src/training/trainer.py` |
| Checkpointing | baseline implementation is mostly provided; understand what gets saved / restored | `src/training/trainer.py` |
| Scripted baseline path | config-driven training | `scripts/train_model.py`, `configs/*.yaml` |

Not required for the baseline:
- wandb
- AMP
- beam search
- bucket batching

Validation note:
- `pytest -m part3_training -v` is a released smoke check
- you still need a real training command to confirm the full pipeline works

Baseline infrastructure note:
- `src/training/scheduler.py` and `src/training/trainer.py` are intentionally
  much more filled in than the model files
- this is so students do not get stuck for hours on training boilerplate before
  they can test their transformer path
- you should still read these files and understand:
  1. why loss is divided by `gradient_accumulation_steps`
  2. where gradient clipping happens
  3. when the scheduler steps
  4. what is stored in a checkpoint

Minimal expected effect after `trainer.py` is working:
- training runs without crashing
- loss stays finite
- validation returns finite metrics
- checkpoint files appear under the configured checkpoint directory

#### Baseline: English packed-data path

```bash
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

python scripts/train_model.py --config configs/tiny.yaml
```

Expected outputs:
- packed dataset directory `output/packed/tinystories_tiny/`
- checkpoints under `checkpoints/tiny/`
- finite printed loss values during training

Recommended first smoke test:

```bash
python scripts/train_model.py --config configs/tiny.yaml --num_epochs 1
```

If you want an even smaller **pipeline smoke test** before the 50k run:

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

Small-smoke interpretation note:
- with only a few hundred examples and the default split ratio, validation may
  have only a handful of examples
- that is acceptable for a **pipeline smoke test**
- do not use that tiny run as your main experiment evidence

What success looks like:
- `output/packed/tinystories_tiny/` exists and contains metadata plus token files
- `best_model.pt` appears in `checkpoints/tiny/`
- the loss stays finite
- the generation script runs from that checkpoint

#### Baseline success criteria

A baseline run counts as successful if:
- training runs without crashing
- loss is finite
- checkpoint save/load works
- you can generate text from the resulting checkpoint

Important grading interpretation:
- for the baseline, generation is mainly an end-to-end correctness check
- you are **not** expected to get highly polished text from a 1-epoch smoke test
- weaker hardware or shorter training may reduce sample quality without meaning the baseline is wrong

For a report-quality baseline, also record:
- exact command
- config path
- tokenizer path
- checkpoint path
- device / hardware context

### 4.5 Generation

| Scope | Required work | Main file(s) |
|---|---|---|
| Baseline decoding | greedy decoding | `src/generation/generator.py` |
| Stochastic decoding | temperature sampling plus at least one filtering strategy (`top-k` or `top-p`) | `src/generation/generator.py` |
| Model-side helper | baseline `generate()` path used by released tests | `src/model/language_model.py` |
| End-to-end usage | generate from your own checkpoint | `scripts/generate_text.py` |

Example command:

```bash
python scripts/generate_text.py \
  --checkpoint checkpoints/tiny/best_model.pt \
  --tokenizer assets/tokenizers/english_bytebpe_8k.json \
  --prompt "Once upon a time," \
  --max_new_tokens 100
```

Success artifacts:
- generation script runs from your own checkpoint
- you can show at least one greedy sample
- you can show at least one stochastic sample
- if you implement both `top-k` and `top-p`, label clearly which one you used in the report

Common generation mistakes:
- using a checkpoint from one tokenizer with a different tokenizer at generation time
- pointing to `checkpoint_epoch_*.pt` or `best_model.pt` in the wrong directory
- expecting high-quality text from a 1-epoch smoke test

Optional script hooks if you choose to use them:

```bash
# beam search generation
python scripts/generate_text.py \
  --checkpoint checkpoints/tiny/best_model.pt \
  --tokenizer assets/tokenizers/english_bytebpe_8k.json \
  --prompt "Once upon a time," \
  --max_new_tokens 80 \
  --beam_width 3 \
  --length_penalty 1.0

# bucketed batching during training
python scripts/train_model.py \
  --config configs/tiny.yaml \
  --num_epochs 1 \
  --bucketed
```

### 4.6 Metric quick reference

Use metrics that match the claim you want to make.

- **Validation loss**: the default quality metric for most baseline and ablation comparisons
- **Perplexity**: `exp(loss)`; only compare it directly when the tokenizer is the same
- **Token accuracy**: fraction of non-padding target tokens predicted exactly; useful as a helper metric, but not required as the main metric
- **Top-k accuracy**: optional helper metric if you want a softer notion of correctness
- **Generation samples**: qualitative evidence to support analysis, not a substitute for quantitative evidence

Practical advice:
- if tokenization changes, avoid claiming one tokenizer is better just because its raw token-level perplexity is lower
- if hardware differs, be careful with runtime or throughput claims
- if training budgets differ, say so explicitly before interpreting quality differences

## 5. Controlled experiments

This is the most important graduate-level part of the lab.

Choose **two** experiments. Each experiment should change **one main
variable** and keep the rest as fixed as possible.

Important flexibility note:
- you do **not** have to choose from a fixed official menu
- you may propose your own ablation if it is still a meaningful transformer /
  tokenizer / training trade-off study
- the goal is not to copy one preset comparison; the goal is to make a clear,
  fair, evidence-backed claim

### 5.1 Possible experiment menu

You do **not** need to choose the most advanced options. A simple, fair,
well-analyzed experiment is better than an ambitious but messy one.

Possible choices include:
- LayerNorm vs RMSNorm
- sinusoidal vs RoPE
- MHA vs GQA
- standard FFN vs SwiGLU
- pre-norm vs post-norm
- tokenizer comparison
- model size comparison with tokenizer and data fixed

You may also choose another well-motivated ablation of your own if:
- one main variable is clearly identified
- the rest of the setup is controlled as much as possible
- you can report both a quality metric and a cost metric
- you can explain the trade-off you expected to study

### 5.1.1 Suggested trade-off menu

This table is a **planning aid**, not a restriction. You may use these ideas,
adapt them, or propose your own.

| Experiment family | Main variable | What might improve | What might get worse / cost more | Good quality metric(s) | Good cost metric(s) | Keep fixed when possible |
|---|---|---|---|---|---|---|
| Attention design | MHA vs GQA | lower parameter count, lower KV/inference memory | possible quality drop from shared KV heads | validation loss | parameter count, checkpoint size, generation-time memory/context | tokenizer, data, model width/depth, epochs, decoding |
| Positional encoding | sinusoidal vs RoPE | possibly better position handling | more implementation complexity; harder fairness if other changes sneak in | validation loss | usually same parameter count; runtime if relevant | tokenizer, data, norm, FFN, epochs |
| Normalization | LayerNorm vs RMSNorm | possibly cheaper / smoother training | quality may not always improve | validation loss | runtime, tokens/sec | tokenizer, data, attention, FFN, epochs |
| FFN design | standard FFN vs SwiGLU | possibly better quality | more parameters / more compute | validation loss | parameter count, epoch time | tokenizer, data, attention, norm, epochs |
| Norm placement | pre-norm vs post-norm | possible optimization differences | stability can change | validation loss, training stability notes | epoch time if relevant | tokenizer, data, attention, FFN, LR schedule |
| Tokenization | provided tokenizer vs a tokenizer you trained yourself | shorter sequences or different segmentation behavior | fairness is harder; perplexity is not directly comparable across tokenizers | controlled qualitative comparison, context efficiency, task-specific comparison | tokens/example, throughput, sequence efficiency | model size, dataset, training budget, decoding |
| Model size | tiny vs small vs small_plus with tokenizer/data fixed | lower loss or better generation | more compute, memory, artifact size | validation loss | parameter count, runtime, memory, checkpoint size | tokenizer, data, epochs, decoding |
| Your own proposal | clearly define one variable | state your hypothesis explicitly | state the expected cost or downside explicitly | choose metrics that match the claim | choose a cost metric that matches the trade-off | document what you kept fixed |

### 5.1.2 Suggested planning worksheet

Before you run an experiment, try to write down answers to these questions.
You do **not** need to use these exact headings in your final report.

1. What is the **claim** or hypothesis?
2. What is the **one main variable** that changed?
3. What did you intentionally keep **fixed**?
4. Which metric is your main **quality** metric?
5. Which metric is your main **cost** metric?
6. Why do those two metrics actually match the claim you want to make?
7. What result would count as evidence **for** your hypothesis?
8. What confound or fairness problem would make the conclusion weaker?

If you cannot answer these clearly before running, the experiment design is
usually still too vague.

### 5.2 What to keep fixed when possible

For each experiment, try to keep fixed:
- dataset and split
- tokenizer, unless tokenizer is the variable
- training budget (epochs or steps)
- batch size, unless batch size is the variable
- optimizer and scheduler settings
- decoding settings, if you compare generations
- random seed when practical

If something important was not kept fixed, say so clearly and explain why.

### 5.3 What each experiment must report

For **each** experiment, include:
- research question or hypothesis
- changed variable
- controlled variables
- quality metric(s)
- cost metric(s)
- hardware/device context if any cost metric depends on runtime or memory
- one result table or figure
- conclusion
- at least one limitation

Examples of quality metrics:
- validation loss
- perplexity, only when tokenization is the same
- token-level accuracy on non-padding targets, if you want a helper metric
- controlled qualitative generation comparison

Examples of cost metrics:
- parameter count
- checkpoint size
- peak memory
- time per epoch
- tokens/sec

Trade-off reminder:
- a strong experiment usually says not only **which run was better**
- it also says **better in what sense, at what cost, and under what fairness assumptions**

Good examples:
- "GQA reduced parameter count and checkpoint size, with a small loss increase."
- "A larger model lowered validation loss, but training time and artifact size increased."
- "RoPE improved loss slightly under the same setup, but the main practical cost was implementation complexity rather than parameter count."

Weaker examples:
- "Model A looked better in one sample."
- "Model B had lower perplexity" when the tokenizer also changed.
- "Run C was faster" without reporting hardware or batch size.

### 5.4 Hardware note for cost metrics

Runtime and memory numbers depend strongly on hardware. If you use metrics such
as time per epoch, tokens/sec, or peak memory, also report:
- CPU or GPU
- GPU model if applicable
- precision used
- batch size
- sequence length

If two runs used different hardware, say so explicitly and interpret the result
carefully.

### 5.5 What counts as weak experimental design

Avoid:
- changing many variables at once
- comparing different datasets without explanation
- using only cherry-picked text samples with no quantitative evidence
- making runtime claims without reporting hardware
- drawing strong conclusions from noisy or weakly controlled evidence

Also avoid this common mistake:
- choosing a cost metric that does **not** actually match the variable you changed

For example:
- if you compare model size, parameter count and runtime are natural cost metrics
- if you compare tokenizers, raw token perplexity across tokenizers is usually not a fair primary quality metric
- if you compare decoding strategies, training loss is not the main thing that changed, so generation-side evidence matters more

### 5.6 If you implement an optional feature, how to turn it on

Features such as RMSNorm, RoPE, GQA, and SwiGLU are **not** the default
baseline. If you implement one of them, you must also change the config so the
model actually uses it.

Relevant model config fields:
- `norm_type`: `layernorm` or `rmsnorm`
- `pos_encoding_type`: `sinusoidal` or `rope`
- `attention_type`: `mha` or `gqa`
- `num_kv_heads`: used when `attention_type: gqa`
- `ffn_type`: `standard`, `glu`, `swiglu`, `geglu`, or `moe`
- `activation`: for example `gelu` or `silu`
- `norm_position`: `pre` or `post`

#### Option A: use the provided experiment config

Important interpretation note:
- `configs/optional/experiment_configs.yaml` is mainly for **pairwise optional-feature comparisons**
- its base setup is intentionally different from the packed-data English baseline
  (`max_seq_len`, split ratio, raw-vs-packed data path, and default epoch count)
- so those runs are usually comparable **to each other**, but not automatically a
  fair apples-to-apples comparison against your main baseline unless you align
  the rest of the setup yourself

```bash
# RMSNorm instead of LayerNorm
python scripts/train_model.py --config configs/optional/experiment_configs.yaml --experiment rmsnorm

# RoPE instead of sinusoidal
python scripts/train_model.py --config configs/optional/experiment_configs.yaml --experiment rope

# GQA instead of MHA
python scripts/train_model.py --config configs/optional/experiment_configs.yaml --experiment gqa

# SwiGLU instead of standard FFN
python scripts/train_model.py --config configs/optional/experiment_configs.yaml --experiment swiglu_ffn

# post-norm instead of pre-norm
python scripts/train_model.py --config configs/optional/experiment_configs.yaml --experiment post_norm
```

#### Option B: copy a baseline config and edit the model section directly

```yaml
model:
  norm_type: "rmsnorm"
  pos_encoding_type: "rope"
  attention_type: "gqa"
  num_kv_heads: 2
  ffn_type: "swiglu"
  activation: "silu"
  norm_position: "post"
```

For a clean single-variable ablation, change only the field(s) needed for that
one experiment.

If a single-variable run is unstable, you may run a second **exploratory**
follow-up with tuned hyperparameters, but do **not** present that tuned run as a
clean single-variable ablation.

Good reporting pattern:
- first show the clean controlled comparison
- then, if useful, separately label any exploratory follow-up as exploratory

## 6. Bonus options

### 6.1 Bonus: quantization

A good hardware-aware bonus is to compare the **same trained checkpoint** under
different numerical formats at inference time. This is a deployment trade-off
study, not a second training project.

Recommended comparisons:
- FP32 vs FP16 or BF16
- FP32 vs INT8
- FP32 vs FP16/BF16 vs INT8

Keep fixed when possible:
- checkpoint
- prompts or evaluation set
- `max_new_tokens`
- decoding strategy and parameters
- hardware, if you want to make timing claims

What to report:
- which checkpoint you used
- which formats you compared
- hardware/device context
- at least one quality observation
- at least one efficiency metric

Example command:

```bash
python scripts/quantization_bonus.py \
  --checkpoint checkpoints/tiny/best_model.pt \
  --tokenizer assets/tokenizers/english_bytebpe_8k.json \
  --prompt "Once upon a time," \
  --formats fp32 int8 \
  --output_json output/quantization/tiny_fp32_int8.json
```

### 6.2 Bonus: SFT concept and design extension

Treat SFT as a **bonus design/report extension**. It is not part of the
required coding path.

Relevant files:
- `scripts/download_sft_dataset.py`
- `src/data/sft_dataset.py`
- `scripts/train_sft.py`
- `configs/optional/sft_*.yaml`

In the released starter repo:
- `scripts/download_sft_dataset.py` lets you inspect a small SFT-style dataset
- `src/data/sft_dataset.py` contains minimal formatting helpers
- `scripts/train_sft.py` is a placeholder entry point, not a full released pipeline

If you include this bonus, focus on:
- how a decoder-only SFT example should be formatted
- which tokens should contribute to the loss
- how prompt-token masking changes the objective
- what would count as a fair before/after evaluation

If you also choose to implement code for this bonus, the natural work is:
- load JSONL instruction-response examples
- build one autoregressive sequence per example
- mask prompt-region labels with `ignore_index`
- reuse the Part 2 model and trainer for a small before/after comparison

Do **not** let this bonus delay your baseline or required experiments.

## 7. Report, answers, and validation

Use the repository-local report folder:
- `report/main.md`
- `report/answers.md`
- `report/figures/`

For the exact deliverables and the repository-vs-Moodle split, see
`docs/WHAT_TO_SUBMIT.md`.

Validation commands:

```bash
pytest -m part1 -v
pytest -m part3_training -v
pytest -m part4_generation -v
pytest -v
```

Interpret the markers carefully:
- `part2` is an umbrella marker for later-stage tests; it is **not** tokenizer-only
- `part2_tokenizer` isolates tokenizer training/save-load behavior and is
  optional for the default English baseline path
- `part3_training` is the released training/checkpoint smoke path
- `part4_generation` is the released generation smoke path

For the most explicit student-facing checklist of which files/functions belong
to the baseline path and how to validate them, see:
- `docs/IMPLEMENTATION_CHECKLIST.md`

## 8. Quick troubleshooting and help requests

Before asking for help, check:
- does the file path in the command actually exist?
- does the tokenizer path match the path used for training?
- if you use an English config, did you prepare the packed dataset directory?
- are you still on the baseline path, or did you accidentally enable an optional feature?

When asking for help, include:
- the exact command
- the config path
- the exact error message or traceback
- what artifact you expected to appear
- whether the failure is in data prep, tokenizer training, model training, checkpoint loading, or generation
