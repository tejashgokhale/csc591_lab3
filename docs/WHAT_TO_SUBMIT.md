# What to Submit

## 1. Submission split

### Put these in your repository
- your source code
- the config file(s) you actually used
- small helper scripts you wrote for experiments
- `report/main.md`
- `report/answers.md`
- figures under `report/figures/`

### Put this in Moodle
Submit **one standalone inference package archive** as either:
- `<your_unity_id>.zip`, or
- `<your_unity_id>.tar.gz`

After unpacking, the archive should contain a small standalone folder that has:
- `standalone_inference.py`
- your trained weight file(s)
- tokenizer / config / helper files needed to load the model
- anything else required if you changed the model architecture

Important packaging rule:
- the unpacked package must run **outside your repository**
- assume we will untar/unzip it in a clean directory that does **not** contain
  your full repo
- therefore, you must include **every file needed for inference** inside the
  packaged folder
- do **not** assume relative imports or file paths that only work from your
  repo checkout

Before uploading, you should test the packaged folder yourself after copying or
unpacking it into a separate location.

Recommended starting point:
- `submission/standalone_inference_template.py`

Minimal baseline-style package shape:

```text
your_unity_id_package/
├── standalone_inference.py
├── best_model.pt
├── tokenizer.json
└── model_config.json        # only if your script needs it
```

If your architecture differs from the baseline, include any extra helper files
your own script imports, for example:

```text
your_unity_id_package/
├── standalone_inference.py
├── best_model.pt
├── tokenizer.json
├── model_config.json
└── my_model_runtime.py
```

Recommended local validation:

```bash
python scripts/validate_submission_inference.py \
  --script path/to/standalone_inference.py \
  --prompt "Once upon a time,"
```

## 2. Mandatory items

You must submit all of the following:
- one working baseline path
- two controlled experiments
- a report in `report/main.md`
- short answers in `report/answers.md`
- one runnable standalone inference package archive in Moodle, named with your
  own unity id

## 3. Standalone inference package contract

Your Moodle package must support a command like:

```bash
python standalone_inference.py \
  --prompt "Once upon a time," \
  --max_new_tokens 64 \
  --device auto
```

The script must print **one JSON object** to stdout.

Required JSON fields:
- `submission_name`
- `prompt`
- `generated_text`
- `response_text`
- `num_generated_tokens`
- `wall_time_sec`
- `seconds_per_generated_token`
- `tokens_per_second`
- `parameter_count`
- `artifact_size_bytes`
- `device`

Optional fields may also be included, for example:
- `dtype`
- `extra`

Why this interface exists:
- you can self-check that your package really runs standalone
- we can evaluate common metrics without rewriting your code

Important rule if you changed the architecture:
- that is completely fine
- you just need to make sure **your own script** can still load **your own
  packaged weights** correctly
- we will evaluate through your packaged interface without modifying your code

## 4. What the report must cover

Your report does **not** need to follow a rigid template, but it should make
these items easy to find.

Important flexibility note:
- you do **not** need to use the exact section titles below
- you do **not** need to choose only from a fixed ablation menu
- you may organize the report in your own style as long as the baseline path,
  the two experiments, the evidence, and the conclusions are easy to locate

### Baseline
Include:
- a clear description of your baseline
- the metrics (parameter count, training epoches etc.) you used to define the baseline
- evidence that the baseline ran

### Experiment 1
Include:
- research question or hypothesis
- changed variable
- controlled variables
- quality metric(s)
- cost metric(s)
- result table or figure
- conclusion
- limitation

Useful guiding prompts:
- what improvement did you expect?
- what cost or downside did you expect?
- why do your chosen metrics actually match that trade-off?
- what should a reader be careful not to over-interpret?

### Experiment 2
Include:
- research question or hypothesis
- changed variable
- controlled variables
- quality metric(s)
- cost metric(s)
- result table or figure
- conclusion
- limitation

Useful guiding prompts:
- what improvement did you expect?
- what cost or downside did you expect?
- why do your chosen metrics actually match that trade-off?
- what should a reader be careful not to over-interpret?

### Note on hardware context (if needed)

If you use hardware-dependent metrics such as runtime, throughput, or memory,
also report:
- CPU or GPU
- GPU model if applicable
- precision used
- batch size
- sequence length

### Note on report length

There is no strict page limit, but a concise report is more likely to be read carefully. Focus on clarity and quality of presentation rather than quantity. We won't give extra credit for longer reports, but we may give extra credit for especially clear and well-presented reports. We encourage you to write short and to the point, and to use figures and tables effectively to communicate your results.

## 5. Short answers

Put your answers to `docs/LAB_QUESTIONS.md` in:
- `report/answers.md`

## 6. Optional and bonus work

Optional work may include:
- GQA
- RoPE
- RMSNorm
- gated FFN variants such as SwiGLU
- bucket batching
- AMP / throughput optimization
- beam search
- optional SFT design discussion
- any extra efforts you made to make your report especially clear and well-presented

Bonus work may include:
- extra controlled experiments beyond the required two
- hardware-aware extension such as quantization
- especially strong analysis and presentation

If you include optional or bonus work, label it clearly in the report.

### Quantization bonus
If you do the quantization bonus, include:
- which checkpoint you used
- which formats you compared
- hardware context
- efficiency metric(s)
- quality evidence
- the trade-off you observed

If you use the helper script, you may include outputs from `scripts/quantization_bonus.py`. However, this is not the only way you show the effectiveness of your quantization approach, and you are free to design your own experiments and presentation style.

## 7. Final check before submission

Before submitting, make sure:
- your baseline path is clear
- the two experiments are easy to identify
- each experiment makes it clear what got better, what got worse or cost more,
  and what was kept fixed
- the report includes both quality and cost evidence
- runtime or memory comparisons include hardware context
- your archive filename is exactly your own unity id, for example
  `<your_unity_id>.zip` or `<your_unity_id>.tar.gz`
- your standalone inference package runs locally **after being unpacked outside
  the repository**
- your packaged script can load your packaged weights without extra edits
- all required files are included inside the archive
- optional or bonus work is clearly labeled
