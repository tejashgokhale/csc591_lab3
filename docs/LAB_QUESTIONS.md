# Lab Questions

Answer these in `report/answers.md`.
Recommended length is about 150-250 words per question.

These short answers should focus on **conceptual or methodological reasoning**.
Use the report for your full empirical evidence and experiment write-up.

## Required questions

1. **Attention masking**  
   Why do we mask attention logits with a very negative value before softmax instead of masking after softmax?

2. **MHA vs GQA**  
   What changes when moving from MHA to GQA? Discuss parameter count, inference-time memory, and any possible quality trade-off.

3. **Positional encoding**  
   Compare sinusoidal positional encoding and RoPE. What is the main implementation difference, and what practical difference did you expect or observe?

4. **Tokenizer fairness**  
   Why is token-level perplexity not directly comparable across different tokenizers? If you compared tokenizers, what metrics did you use instead?

5. **Pretraining pipeline**  
   Why does the dataset use shifted input/target pairs? What bug would appear if input and target were identical?

6. **Evidence quality**  
   Why is one cherry-picked generation sample not enough to support a strong conclusion about a model change? What additional evidence would make the claim more trustworthy?

## Optional bonus questions

7. **Hardware-aware extension**  
   If you explored quantization or another efficiency technique, what did it save and what trade-off did it create?

8. **Optional SFT design**  
   If you include the SFT extension conceptually, which tokens should contribute to the loss and why?
