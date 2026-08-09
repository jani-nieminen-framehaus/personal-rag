---
topic: ml
tags: [llm, context-length, attention]
---

# YaRN — Yet another RoPE extensioN

YaRN is a method to extend the context length of an existing LLM without
expensive continued pre-training. The paper is "YaRN: Efficient
Context-Window Extension of LLMs" (Peng et al., 2023).

## The problem

Rotary Position Embeddings (RoPE) generalize poorly to sequence lengths
beyond what the model saw in pre-training. Naive position interpolation
solves the "out of distribution" issue but hurts perplexity on short
contexts.

## The idea

YaRN combines two ideas:

1. **NTK-aware scaling** — instead of uniformly scaling all frequency
   components, treat high frequencies (which carry short-range info) and
   low frequencies (which carry long-range info) differently. Keep high
   frequencies close to original; stretch only the low ones.
2. **Attention scaling** — apply a temperature-like factor to the attention
   logits so the softmax doesn't saturate when the context grows.

In practice this means the same pretrained model can be extended from 4k
to 64k–128k context with relatively little fine-tuning (a few hundred
steps) and minimal quality loss on standard benchmarks.

## Why I care

For document-grounded tasks (RAG, long-form QA) the context window is the
binding constraint. YaRN-class tricks let you reuse a strong open model
without paying for a full re-pretrain.
