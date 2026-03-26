# Submission package interface

For the Moodle submission, package this folder as either:
- `<your_unity_id>.zip`, or
- `<your_unity_id>.tar.gz`

The unpacked package should contain a small standalone folder that has:
- `standalone_inference.py`
- your trained weight file(s)
- tokenizer / config / helper files needed to load the model

The unpacked package must run:
- from its own directory
- **outside your repository**
- without us editing the code after submission

That means you must include every file needed for inference inside the packaged
folder.

Recommended starting point:
- `submission/standalone_inference_template.py`

Minimal baseline-style package example:

```text
submission_package/
├── standalone_inference.py
├── best_model.pt
├── tokenizer.json
└── model_config.json        # only if your script needs it
```

If your script imports extra runtime helpers, package them too:

```text
submission_package/
├── standalone_inference.py
├── best_model.pt
├── tokenizer.json
├── model_config.json
└── my_model_runtime.py
```

Recommended first step:
1. copy `submission/standalone_inference_template.py` to
   `submission/standalone_inference.py`
2. implement your own loading and generation logic there
3. validate that student-facing file locally before packaging

Local validation command:

```bash
python scripts/validate_submission_inference.py \
  --script submission/standalone_inference.py \
  --prompt "Once upon a time,"
```

Recommended final self-check:
1. create the zip or tar.gz
2. unpack it into a separate temporary directory outside the repo
3. run the validator on the unpacked `standalone_inference.py`

The standalone script must print **one JSON object** to stdout. The validator
checks the required keys and makes sure the script can run end to end.
