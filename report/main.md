# Lab 3 Report

Write your report here in your own structure.

Your structure is flexible. You do **not** need to follow a rigid template.
However, a reader should still be able to quickly find:
- the baseline path
- experiment 1
- experiment 2
- the evidence supporting each claim
- what trade-off each experiment is studying

Your report must clearly cover at least:
- baseline setup
- experiment 1
- experiment 2
- metrics / evidence
- trade-off discussion
- limitations / next steps
- generation comparison

Recommended: start with a compact summary block so a grader can find the core
run information quickly.

## Suggested summary block

- baseline command:
- baseline config:
- tokenizer path:
- checkpoint path:
- device / hardware:
- experiment 1 changed variable:
- experiment 2 changed variable:

If you report hardware-dependent cost metrics such as runtime, throughput, or
memory, also report the relevant run context, such as:
- CPU or GPU
- GPU model if applicable
- precision used
- batch size
- sequence length

Helpful experiment-writing reminder:
- for each experiment, try to make it easy to answer:
  - what changed?
  - what stayed fixed?
  - what got better?
  - what got worse or cost more?
  - how strong is the evidence?

Put figures and tables under `report/figures/` if helpful.
