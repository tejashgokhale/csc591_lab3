# CSC591 Lab 3: Evaluation Rubric

**Total: 100 points + optional bonus**

## A. Correctness and baseline functionality (35 pts)

### A1. Core implementation correctness (20 pts)
- **18-20:** released core functionality works; Part 1 implementation is largely correct; no major conceptual bugs in the baseline path
- **14-17:** mostly correct, with minor issues that do not break the main workflow
- **8-13:** partial implementation; important pieces are missing or fragile
- **0-7:** major parts missing or incorrect

### A2. Tokenizer/data/training/generation baseline path (15 pts)
- **13-15:** tokenizer, dataset/dataloader, training, checkpoint, and generation all work together
- **10-12:** one minor stage is weak but the main baseline works
- **5-9:** partial end-to-end functionality only
- **0-4:** no working baseline path

Important interpretation note:
- for the baseline path, generation is graded primarily as an **end-to-end functionality check**
- students are **not** required to produce highly fluent text from a short run or a small hardware budget
- absolute sample quality should be interpreted relative to training budget, hardware, and dataset choice

## B. Reproducible baseline run (20 pts)

### B1. Successful run evidence (10 pts)
- **9-10:** baseline run completed and includes clear logs/metrics evidence plus a runnable standalone inference package
- **7-8:** mostly clear evidence, minor omissions
- **4-6:** partial evidence
- **0-3:** little or no evidence

Strong evidence usually makes it easy to find:
- the exact command
- the config path
- the tokenizer path
- the standalone inference entry point and packaged artifact name
- at least one concrete run artifact or logged metric

Do not grade baseline success by absolute sample quality alone. A short smoke
test on weaker hardware may still deserve full baseline credit if the pipeline
is correct, documented, and produces finite training metrics plus a usable
checkpoint.

### B2. Reproducibility quality (10 pts)
- **9-10:** config, commands, tokenizer path, standalone inference entry point, and run context are clearly documented in the report
- **7-8:** mostly reproducible, minor omissions
- **4-6:** some details missing
- **0-3:** difficult to reproduce

## C. Two controlled ablations (25 pts)

### C1. Experiment design quality (10 pts)
- **9-10:** two clearly controlled experiments; hypothesis, changed variable, and controlled variables are explicit
- **7-8:** two experiments, but one has a design weakness
- **4-6:** only one strong experiment or two weak ones
- **0-3:** experiments are missing or not meaningfully controlled

### C2. Evidence quality (8 pts)
- **7-8:** each experiment reports both a quality metric and a cost metric with a clear table/figure; hardware-dependent metrics are contextualized properly; evidence is not only cherry-picked samples
- **5-6:** good evidence with minor gaps
- **3-4:** mostly qualitative or incomplete evidence
- **0-2:** weak or missing evidence

Metric fairness notes:
- validation loss is usually the safest primary quality metric for the baseline
- perplexity is only a fair comparison when tokenization is held fixed
- token-level accuracy may be reported as a helper metric, but it is **not** required to be the main experiment metric
- qualitative samples are useful supporting evidence, but they should not be the only evidence

### C3. Analysis quality (7 pts)
- **6-7:** conclusions are evidence-based and discuss trade-offs, fairness, confounds, and limitations
- **4-5:** mostly good analysis with limited depth
- **2-3:** superficial commentary
- **0-1:** no meaningful analysis

## D. Short-answer questions (10 pts)

Score the answers in `report/answers.md` mainly for:
- conceptual correctness
- clarity
- directness
- evidence of understanding rather than copied phrasing

- **9-10:** consistently strong and technically sound
- **7-8:** mostly correct with minor gaps
- **4-6:** mixed quality
- **0-3:** weak, incomplete, or largely incorrect

## E. Final report quality (10 pts)

The report should live in `report/main.md`. Students may organize it freely, but it must still be clear and analyzable.

### E1. Structure and clarity (5 pts)
- **5:** clear, concise, well organized
- **4:** good overall structure
- **3:** adequate but uneven
- **1-2:** hard to follow
- **0:** missing

### E2. Technical communication (5 pts)
- **5:** good tables/figures, clear experiment framing, hardware/run context reported when relevant, limitations acknowledged, and the standalone submission package is easy to trace back to commands/configs
- **4:** good overall quality with minor issues
- **3:** adequate communication
- **1-2:** weak presentation
- **0:** missing

## Optional bonus (up to +10 pts)

These are not required. Use them to distinguish unusually strong work.

- **+2 to +4:** additional well-designed experiments beyond the required two
- **+2 to +4:** hardware-aware extension such as quantization with clear trade-off analysis
- **+1 to +2:** particularly strong reproducibility/presentation quality
- **+1 to +2:** thoughtful optional SFT design analysis

Bonus should reward **quality of exploration**, not just quantity of extra code.
For example, stronger bonus cases often include:
- unusually careful fairness discussion
- a negative result analyzed thoughtfully
- especially clean evidence that the student explored rather than only copied a preset
