# Lab Concept Notes

This document explains the main ideas behind the lab. It is not a list of step-
by-step instructions. Use it together with `docs/LAB_GUIDE.md`.

## 1. Autoregressive language modeling

A decoder-only language model assigns probability to a token sequence
$x_1, x_2, \dots, x_T$ by factorizing it from left to right:

$$
p_\theta(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t}).
$$

This means the model predicts the next token using only the previous tokens.
That is why the training data is built from **shifted pairs**:
- input: $x_1, x_2, \dots, x_{T-1}$
- target: $x_2, x_3, \dots, x_T$

In code, this is the usual pattern:
- `input_ids = tokens[:-1]`
- `target_ids = tokens[1:]`

If input and target were identical, the model would be asked to copy the current
token rather than predict the next one. That would be the wrong training
objective.

The token-level training objective is cross-entropy:

$$
\mathcal{L} = - \sum_{t=1}^{T-1} \log p_\theta(x_{t+1} \mid x_{\le t}).
$$

## 2. Scaled dot-product attention

Given query, key, and value tensors $Q$, $K$, and $V$, scaled dot-product
attention is:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

Here $d_k$ is the key dimension. The division by $\sqrt{d_k}$ matters because
without it, the dot products can become large as the hidden dimension grows.
Large logits make the softmax too sharp, which often hurts optimization.

### Why masking happens before softmax

In a decoder-only LM, position $t$ must not look at future positions $> t$.
So the model applies a **causal mask** to the attention logits before softmax.
A common implementation is to replace masked logits by a very negative number.

If a masked logit is close to $-\infty$, then after softmax its probability is
approximately $0$. This is the right behavior: the masked position contributes
no attention mass.

If you tried to mask **after** softmax, the probabilities would no longer be
properly renormalized. That changes the attention distribution in the wrong way.

## 3. Multi-head attention and grouped-query attention

### Multi-head attention (MHA)

In multi-head attention, the model projects the input into multiple query, key,
and value subspaces:

$$
Q_h = XW_h^Q, \qquad K_h = XW_h^K, \qquad V_h = XW_h^V.
$$

Each head computes its own attention pattern. The head outputs are then
concatenated and projected back to model dimension.

Why do this instead of using one large head? The main idea is that different
heads can specialize in different interaction patterns. For example, some heads
may focus on short-range structure while others capture longer-range
relationships.

### Grouped-query attention (GQA)

Grouped-query attention keeps many query heads but uses fewer key/value heads.
So several query heads share the same key/value representation.

Conceptually:
- MHA: number of query heads = number of key/value heads
- GQA: number of query heads > number of key/value heads

This usually reduces:
- parameter count in the key/value projections
- KV-cache size during autoregressive inference
- memory bandwidth during generation

The main trade-off is representational flexibility: sharing key/value heads may
slightly reduce quality, depending on model size and task.

For this lab, GQA is a good **controlled experiment variable** because it can be
connected to both quality and efficiency.

## 4. Positional information

Self-attention by itself is permutation-invariant. Without additional position
information, the model cannot distinguish between sequences that contain the same
tokens in different orders.

### Sinusoidal positional encoding

The original Transformer uses fixed sinusoidal features:

$$
PE(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d}}\right),
$$

$$
PE(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d}}\right).
$$

These vectors are added to token embeddings before entering the transformer.
Important properties:
- no learned position parameters are needed
- the encoding is deterministic
- the frequencies form a geometric progression

### Rotary positional embedding (RoPE)

RoPE does not add a position vector to the embedding. Instead, it rotates the
query and key vectors as a function of position. In practice, the important lab
intuition is:
- sinusoidal encoding injects position by **addition**
- RoPE injects position directly inside the **attention computation**

This often makes RoPE a good comparison point against sinusoidal encoding.

## 5. Normalization choices

### LayerNorm

LayerNorm normalizes by subtracting the mean and dividing by the standard
deviation over the feature dimension:

$$
\operatorname{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta.
$$

It has two learned affine parameters:
- $\gamma$ for scale
- $\beta$ for shift

### RMSNorm

RMSNorm removes mean subtraction and normalizes only by the root mean square:

$$
\operatorname{RMSNorm}(x) = \frac{x}{\sqrt{\operatorname{mean}(x^2) + \epsilon}} \odot \gamma.
$$

Compared with LayerNorm, RMSNorm is simpler and often a bit cheaper. It also
matches modern decoder-only architectures more closely. That makes it a good
optional extension, but it should not be required for the baseline path.

### Pre-norm vs post-norm

A decoder block has residual connections around the attention and FFN sublayers.
Normalization can happen either before or after those sublayers.

- **pre-norm:** normalize first, then apply the sublayer
- **post-norm:** apply the sublayer first, then normalize after the residual add

In practice, pre-norm is often easier to optimize in deeper transformers. That
is why many modern architectures use it as the default.

## 6. Feed-forward networks and gated variants

A standard transformer FFN is a position-wise MLP:

$$
\operatorname{FFN}(x) = W_2 \, \phi(W_1 x).
$$

The same FFN is applied independently at every sequence position.

### Why gated variants matter

Variants such as GLU, GeGLU, and SwiGLU introduce multiplicative gating. A
simple schematic form is:

$$
\operatorname{GLU\text{-}style}(x) = (W_a x) \odot \phi(W_b x).
$$

This can improve quality, but it may also change parameter count and compute
cost. That makes it a strong candidate for ablation.

For this lab, the most relevant comparison is usually:
- standard FFN vs SwiGLU

If you run that comparison, you should remember that changing FFN type may also
require changing the activation field in the config.

## 7. Tokenization trade-offs

### Character tokenization

Character tokenization is simple and transparent. It is often a good debugging
path because:
- there is very little tokenizer machinery
- encode/decode behavior is easy to inspect
- punctuation and formatting are preserved naturally

But it also has a major downside: sequences become much longer. If your context
window is fixed, then longer token sequences mean fewer characters of useful
context fit into the model.

### BPE and byte-level BPE

Subword tokenizers such as BPE usually reduce sequence length by merging common
substrings. Byte-level BPE also avoids many Unicode edge cases because it starts
from bytes rather than language-specific characters.

Potential advantages:
- shorter sequences
- better context efficiency
- often better behavior on natural-language corpora

Potential disadvantages:
- more machinery
- more implementation details to get wrong
- token boundaries are less interpretable than character boundaries

### Why token-level perplexity is not directly comparable across tokenizers

Perplexity is defined over a tokenization. If one tokenizer splits text into
many small units and another splits it into larger units, then the token-level
prediction task is different. So raw token-level perplexity is not a fair
cross-tokenizer comparison.

If tokenizers differ, more comparable evidence may include:
- validation loss only within the same tokenizer family
- average tokens per example
- qualitative generation under controlled settings
- throughput at a fixed sequence length
- context efficiency for a fixed maximum token budget

### Token-level accuracy as a helper metric

Another metric you may see in the training code is **token-level accuracy**.
For a language model batch, this usually means:

$$
\text{accuracy} =
\frac{\#\{\text{predicted token} = \text{target token on non-padding positions}\}}
     {\#\{\text{non-padding target positions}\}}
$$

This can be useful as a sanity-check metric, but it is usually **not** the best
primary metric for LM comparisons:
- it is sensitive to tokenization
- exact-match token prediction is a strict criterion
- it often looks numerically low even when the model is improving

So for this lab, treat token accuracy as optional supporting evidence and treat
validation loss as the default primary baseline metric.

## 8. Controlled ablation design

A strong ablation changes one main variable and holds the rest fixed as much as
possible. The point is not just to produce two runs. The point is to support a
claim with evidence.

A good experiment usually answers these questions:
1. What is the hypothesis?
2. What changed?
3. What stayed fixed?
4. Which quality metric is reported?
5. Which cost metric is reported?
6. What conclusion is actually supported by the evidence?

Weak experiments often change too many things at once. For example, if you
change tokenizer, model size, optimizer, and dataset in a single comparison,
then you usually cannot say which factor caused the result.

## 9. Bonus concept: quantization

A good optional extension is to compare the **same checkpoint** under different
numerical formats, for example:
- FP32
- FP16 / BF16
- INT8 or weight-only quantized inference

This is appealing because it connects model behavior to deployment constraints
without requiring a full systems project.

Useful metrics include:
- model size on disk
- peak memory
- latency or tokens/sec
- qualitative degradation in outputs
- change in validation loss if evaluated consistently

Important fairness notes:
- runtime comparisons should ideally be on the same hardware
- if hardware differs, timing claims should be interpreted carefully
- quantization should usually be compared on the same checkpoint, not a different trained model

## 10. Bonus concept: SFT

This section is about the **idea** of SFT.
For the practical starter-vs-implementation expectations in this lab, see the
bonus section in `docs/LAB_GUIDE.md`.

In decoder-only supervised fine-tuning (SFT), the architecture can stay mostly
the same. The key change is in the **loss mask**.

Suppose an example is structured as:
- prompt or instruction
- response

The full sequence may be fed into the model, but only the response tokens should
contribute to the supervised loss. Conceptually:
- prompt tokens are part of the input context
- response tokens are the supervised target region

This is the main conceptual difference from plain next-token pretraining.

### 10.1 A concrete example

Suppose we format one example as:

```text
Instruction: Answer briefly.
Input: What do bees make?
Response: Bees make honey.
```

In a decoder-only model, we still build one long token sequence and still train
with the usual left-shifted language-model objective.

So conceptually we still form:
- `input_ids = tokens[:-1]`
- `target_ids = tokens[1:]`

The difference is that we do **not** want the model to be rewarded for
predicting the prompt text itself. We only want to supervise the answer region.

### 10.2 What the mask does

For SFT, the labels for the prompt part are replaced by an ignore value such as
`-100`.

So the target looks like:

```text
Instruction: Answer briefly.   -> ignored in the loss
Input: What do bees make?      -> ignored in the loss
Response: Bees make honey.     -> counted in the loss
```

That means:
- the prompt is still visible to the model
- but only the response tokens contribute gradients

This is the core teaching idea of SFT in this lab.

### 10.3 How this differs from pretraining

In plain pretraining, every next token is a training target.

For example, if the model sees:

```text
Once upon a time there was a fox ...
```

then each next token in that sequence contributes to the loss.

In SFT, the model sees the whole prompt-plus-answer sequence, but the loss is
restricted to the answer region. So:

- **pretraining** teaches general continuation behavior
- **SFT** teaches how to respond when a prompt is given in a specific format

### 10.4 What improvement you should expect

A small teaching SFT run does **not** usually turn a tiny base model into a
strong assistant. The expected improvement is narrower:

- better adherence to the `Instruction / Input / Response` format
- less drifting into unrelated continuation
- more task-conditioned answers on prompts that look like the SFT data

That is already enough to demonstrate the main idea.

### 10.5 Why a tiny SFT dataset can still be useful

Even a very small instruction dataset can be pedagogically useful because it
makes the training target visibly different from pretraining:

- the examples have a fixed prompt structure
- the response region is explicit
- the masking rule can be inspected directly in code

So the main goal of this optional part is **understanding the pipeline**, not
maximizing benchmark performance.

### 10.6 What students should be able to explain after this optional SFT section

If you understand this part well, you should be able to explain:

1. why prompt tokens are kept in the input
2. why prompt tokens should usually be masked out of the loss
3. why SFT can improve instruction-following even if the architecture does not change
4. why a fair SFT comparison should use the same base checkpoint before and after fine-tuning

## 11. Decoding choices for generation

Training and generation are related, but they are not the same procedure.
During training, the model sees the ground-truth prefix. During generation, it
must consume its own previous outputs.

### Greedy decoding

Greedy decoding always chooses the highest-probability next token:

$$
x_{t+1} = \arg\max_v \; p_\theta(v \mid x_{\le t}).
$$

This is simple and deterministic, but it can become repetitive because the model
never explores lower-probability alternatives.

### Temperature sampling

Temperature rescales logits before softmax:

$$
\tilde{z}_i = \frac{z_i}{T}.
$$

- $T < 1$ makes the distribution sharper
- $T > 1$ makes the distribution flatter

So temperature changes randomness without changing the underlying ranking of the
logits.

### Top-k and top-p filtering

These methods reduce the candidate set before sampling.

- **top-k:** keep only the $k$ highest-logit tokens
- **top-p / nucleus:** keep the smallest set of tokens whose cumulative
  probability is at least $p$

Why this matters for the lab:
- greedy vs stochastic decoding can produce visibly different outputs
- decoding settings must be held fixed if you want a fair generation comparison

### Practical decoding trade-offs

For this lab, a useful mental model is:
- **greedy**: most stable and reproducible, but often more repetitive
- **temperature only**: simple randomness control, but can still sample from a
  very long low-probability tail
- **top-k**: caps the candidate set to a fixed size; easy to reason about
- **top-p**: adapts the candidate set to the confidence of the distribution

There is no universally best decoding setting. Different settings change output
style even when the underlying checkpoint is identical. That is why decoding is
part of the evaluation procedure, not just a cosmetic script option.

### Why generation can fail differently from training

During training, the model always conditions on the true previous tokens. During
generation, one mistake can become part of the next input prefix. This is one
reason short generation samples can look worse than the validation loss alone
might suggest.

For students, the key takeaway is:
- a model can have a working training pipeline but still generate repetitive or
  weak text under a poor decoding setup
- if you compare generations across models, keep prompt and decoding settings
  fixed

## 12. Padding masks, causal masks, and ignored labels

Several different masks appear in this lab, and they solve different problems.

### Causal mask

The causal mask enforces autoregressive structure inside self-attention:
- earlier tokens can be attended to
- future tokens cannot

This mask is about **information flow inside the model**.

### Padding mask

When examples in a batch have different lengths, shorter examples are padded.
Those padding positions are not real data, so the model should not attend to
them.

This mask is also about **information flow inside the model**, but for padding
rather than future tokens.

### Ignoring padding in the loss

Even if attention masking is correct, padded targets should usually be excluded
from the loss:

$$
\mathcal{L} = \text{CrossEntropy}(\text{logits}, \text{targets}, \text{ignore\_index}=\text{pad})
$$

Otherwise the model would be penalized or rewarded for predicting artificial
padding symbols.

### Why this distinction matters

- causal mask: blocks future attention
- padding mask: blocks attention to padded positions
- ignore index in the loss: removes padded labels from optimization

Students often mix these up. They are related, but they are not interchangeable.

## 13. Learning-rate warmup and scheduler intuition

The lab asks you to implement one working scheduler path because optimization
behavior is part of modern LM training.

### Why warmup is common

Early in training, random initialization can produce unstable updates. Warmup
starts with a smaller learning rate and gradually increases it:

$$
\eta_t = \eta_{\text{base}} \cdot \frac{t}{T_{\text{warmup}}}
\qquad \text{for } t < T_{\text{warmup}}.
$$

This often makes training more stable, especially for transformers.

### Why cosine decay is common

After warmup, a cosine schedule gradually reduces the learning rate:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\text{base}} - \eta_{\min})
\left(1 + \cos\left(\pi \cdot \text{progress}\right)\right).
$$

This gives a smooth decay rather than a sudden drop.

### What students should understand

For this lab, you do not need a research-level optimizer study. But you should
understand:
- why warmup can stabilize early training
- why a scheduler changes optimization even when the model is unchanged
- why scheduler settings must be held fixed in a fair ablation unless the
  scheduler itself is the variable

### Connection to the implementation

In this lab, the scheduler is stepped inside the training loop, not inside the
model. So it changes optimization behavior without changing the forward pass.

This separation is important conceptually:
- **architecture choices** change what the model can represent
- **optimizer / scheduler choices** change how training moves through parameter
  space

If two runs differ in both architecture and scheduler, then it is harder to say
which one caused the result.

## 14. Packed datasets and why the lab uses them

The lab supports two data paths:
- **raw JSONL path**: read text, tokenize examples, and build training pairs at
  runtime
- **packed dataset path**: tokenize once ahead of time and save compact binary
  arrays for repeated training runs

### Why pack the dataset at all

Repeatedly tokenizing a large text file can waste time and memory. A packed
dataset shifts that work into a preprocessing step:

1. read each JSONL example once
2. tokenize it once
3. save token IDs into compact arrays on disk
4. memory-map those arrays during training

This is useful for the lab because students often rerun training many times
while debugging or doing ablations.

### What is stored in the packed format

Conceptually, the packed dataset contains:
- one flat token array with all token IDs
- one offset array saying where each example starts
- one length array saying how long each example is
- one saved train/val/test split
- metadata describing tokenizer, dtype, and preprocessing choices

So instead of storing `list[list[int]]` in Python, the code can recover example
$i$ by slicing into the flat token array with the saved offset and length.

### Why this helps memory and reproducibility

This usually helps in two ways:

1. **lower memory overhead**  
   Python lists of Python integers are expensive. Flat numeric arrays are much
   more compact.

2. **more reproducible reruns**  
   once tokenization and splitting are saved, later runs can reuse the same
   processed data rather than rebuilding it slightly differently by accident

### What packed preprocessing does not change

Packed preprocessing is a data-engineering optimization. It does **not** change
the language-model objective. After loading one packed example, the training
pair is still:
- `input_ids = tokens[:-1]`
- `target_ids = tokens[1:]`

So packed datasets change **how data is stored and loaded**, not **what the
model is trained to predict**.

### Why fair comparisons still matter

If one run uses a packed dataset and another uses a raw JSONL dataset that was
tokenized differently, filtered differently, or split differently, then the two
runs are not really using the same data path.

For a fair comparison, try to keep fixed:
- tokenizer
- max sequence length
- special-token policy
- train/val/test split
- number of examples kept after filtering
