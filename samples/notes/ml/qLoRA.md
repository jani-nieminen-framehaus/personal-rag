---
topic: ml
tags: [finetuning, quantization, peft]
---

# QLoRA

QLoRA (Dettmers et al., 2023) is a parameter-efficient finetuning method
that combines 4-bit quantization (NF4) of a frozen base model with low-rank
adapters (LoRA) on top. The result: a 65B model can be finetuned on a
single 48GB GPU.

## The trick

- Base model weights are stored in 4-bit (NF4) and never updated.
- LoRA adapters are kept in bfloat16, so gradients are clean.
- Paged optimizers move optimizer state to CPU RAM when VRAM is tight.
- A small set of "gate" ops runs in float32 to keep numerics stable.

The end-to-end pipeline:

1. Quantize base to NF4 once (saves to disk).
2. Attach LoRA adapters (rank 16 is a good default).
3. Train in mixed precision; only the adapters' weights update.
4. Merge adapters back into the base for inference (or keep separate).

## When to use it

QLoRA is the right tool when you have a strong base model and a few
thousand to a few hundred thousand domain-specific examples. For
instruct-tuning a Llama-3-8B on a single 24GB card, it's the default
choice.
