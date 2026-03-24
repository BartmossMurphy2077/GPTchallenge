## Task 2.3 - Attention Mechanism

### Q1. GPT with causal masking vs without masking (same prompt)

When I compare generation with the same prompt, two concrete differences appear:

1. **Short-term coherence and grammar**

- **With masking**: output usually stays more locally consistent (better word/character continuation and fewer abrupt breaks).
- **Without masking**: output often looks less stable during free generation, with more sudden jumps or odd transitions.

2. **Repetition and drift over longer continuation**

- **With masking**: the continuation tends to follow a smoother causal flow from left to right.
- **Without masking**: the model more often drifts into repetitive loops or loses topic/structure sooner.

Why this happens:

- Causal masking enforces true next-token learning (token \(t\) can only use tokens \(\le t\)).
- If masking is removed, training can leak future information, which can reduce training loss but usually hurts autoregressive generation quality at inference time.

## Task 2.7 - GPT Demo Reflection

### 1) Which decoding method produced the most coherent output? And the most diverse output?

In this setup, beam search or greedy decoding usually gives the most coherent output because both prefer high-probability continuations. Temperature and top-k sampling are usually more diverse, with higher temperature giving the most variation.

### 2) Did lower validation loss always imply more interesting generations?

No. Lower validation loss often improves average next-token prediction quality, but it does not guarantee interesting or creative text. A model can have lower loss and still produce repetitive or generic continuations.

### 3) What kinds of errors remained even after training?

Typical remaining errors include repetition loops, awkward local phrasing, abrupt topic drift, and inconsistent long-range structure. On a tiny corpus, models also overfit style patterns and may produce fluent-looking but semantically weak text.
