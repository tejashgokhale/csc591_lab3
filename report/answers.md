# Lab 3 Short Answers

Answer the required questions from `docs/LAB_QUESTIONS.md` here.
Focus on conceptual or methodological reasoning. Put full empirical detail in
`report/main.md`.

## Q1
Masking attention logits before softmax is important because of how softmax behaves. Softmax converts logits into probabilities by exponentiating them and normalizing. If we mask after softmax, the masked positions would already have received some probability mass, which would then need to be redistributed incorrectly.

By instead adding a very large negative value to the logits before softmax, we effectively force those positions to have near-zero probability after softmax, so masked tokens contribute nothing to the final attention distribution.

If masking were applied after softmax, it would break normalization: the remaining probabilities would no longer sum to 1 unless explicitly renormalized, which introduces extra computation and potential numerical issues. Pre-softmax masking ensures correctness, efficiency, and numerical stability.

## Q2
Moving from Multi-Head Attention (MHA) to Grouped Query Attention (GQA) mainly reduces memory and computation at inference time. In MHA, each head has its own query, key, and value projections. In GQA, multiple query heads share the same key and value projections, which reduces parameter count and especially KV-cache size during decoding.

This matters most for inference: the KV cache dominates memory usage in autoregressive generation, so sharing keys/values significantly reduces memory bandwidth and latency. This is why modern LLMs often prefer GQA.

The trade-off is that GQA slightly reduces representational flexibility because fewer independent key/value projections are available. In practice, however, models tend to retain similar quality if grouping is chosen carefully (e.g., moderate grouping rather than extreme sharing).

So overall: GQA improves efficiency (memory + speed) with minimal quality loss, making it a practical engineering trade-off.

## Q3
The key difference between sinusoidal positional encoding and RoPE is how position information is incorporated. Sinusoidal encoding adds a fixed positional vector to token embeddings, while RoPE applies a rotation directly to the query and key vectors inside attention.

Implementation-wise, sinusoidal encoding is simple: precompute position vectors and add them to embeddings. RoPE is more involved, since it modifies attention computation by rotating Q and K vectors based on position indices.

Practically, RoPE tends to perform better for extrapolation to longer sequences because it encodes relative position information implicitly through rotations. This allows attention scores to generalize more naturally beyond training lengths.

In experiments, the expected difference is that RoPE may slightly improve validation loss or generation coherence, especially for longer contexts, while sinusoidal encoding works well as a simple and stable baseline.

## Q4
Token-level perplexity is not directly comparable across tokenizers because different tokenizers produce different numbers of tokens for the same text. A tokenizer with smaller tokens (e.g., character-level) will produce longer sequences, which affects the average log-probability per token and artificially changes perplexity.

This means a model might appear better simply because it uses fewer or larger tokens, not because it actually models language better.

To compare fairly, metrics should be normalized at the text level. Common alternatives include:
    (a) bits per character (BPC)
    (b) bits per byte (BPB)
    (c) perplexity normalized by sequence length in characters

In practice, I would report validation loss alongside a normalized metric like BPC to ensure comparisons reflect actual modeling quality rather than tokenization artifacts.

## Q5
The dataset uses shifted input/target pairs so that the model learns next-token prediction. Specifically, the input is a sequence of tokens, and the target is the same sequence shifted one position to the left.

If input and target were identical, the model would simply learn to copy tokens instead of predicting the next token. This would make training trivial and meaningless: the model could achieve very low loss without learning any language structure.

This bug would show up as extremely low training loss but completely useless generation, since the model never learned causal dependencies.

Shifting ensures the model learns the conditional probability, which is the core objective of language modeling.

## Q6
A single cherry-picked generation example is not reliable evidence because it can be misleading or unrepresentative. Language models are stochastic, and outputs vary depending on sampling, prompts, and random seeds. One good-looking sample does not prove consistent improvement.

To make a strong claim, we need systematic evidence:
    (a) Quantitative metrics (e.g., validation loss, perplexity)
    (b) Multiple samples across prompts
    (c) Controlled experiments (only one variable changed)
    (d) Statistical consistency (similar trends across runs)

Additionally, reporting both quality and cost metrics (e.g., runtime, memory) strengthens the argument.

In short, conclusions should be based on reproducible trends, not isolated examples. Text samples are useful for illustration, but not sufficient as primary evidence.

## Optional Q7

## Optional Q8
