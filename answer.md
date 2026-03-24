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
