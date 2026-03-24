Task 1: Hyperparameters

**Q1.** Hyperparameters like `d_model`, number of layers, attention heads, and feedforward dimension affect model capacity — they determine how much the model can actually learn and represent. Things like learning rate, batch size, dropout, and epochs are more about *how* training happens rather than changing the model's size or power.

**Q2.** For BERT, context length matters because it reads the whole input at once bidirectionally, so longer context means better understanding of relationships. For GPT, it's about how far back the model can look when generating — too short and it loses track of earlier context, making outputs less coherent.

**Q3.** Increasing `d_model` or adding more layers gives the model more capacity to learn complex patterns, but it comes at a cost — more memory, more compute, and a higher chance of overfitting if you don't have enough data.

---

Task 2: Attention Mechanism

**Q1.** Without causal masking, GPT can see future tokens during generation, which basically lets it "cheat" — the outputs end up unrealistic and incoherent. With masking applied, each token can only attend to what came before it, so the generation flows naturally and makes more sense.

---

Task 3: Block Types

**Q1.** The version with causal masking behaves autoregressively because it restricts each token to only look at previous ones, so the model generates one token at a time based on what's already been produced.

**Q2.** In the encoder-decoder block, source sequence information enters through cross-attention — the decoder uses the encoder's outputs as keys and values, while using its own hidden states as queries.

**Q3.** BERT doesn't need cross-attention because it's only ever processing one sequence — there's no separate source and target. GPT doesn't need it either since it just generates from its own previous tokens with no external input to attend to.

---

Task 4: TinyGPT

**Q1.** What makes GPT decoder-only is that there's no encoder at all — it just uses masked self-attention blocks, so the model only ever sees its own past outputs.

**Q2.** GPT uses a language modeling head instead of a classification head because the goal is to predict the next token across the whole vocabulary, not assign an input to a fixed set of classes.

**Q3.** Next-token prediction works with masking because the model is forced to only use past tokens, which is exactly what generation requires. Bidirectional attention would let future tokens leak in, which completely breaks the task since you'd be predicting something you can already see.

---

Task 5: Decoding Strategies

**Q1.** Lower temperature (like 0.5) makes the distribution sharper, so the model sticks to high-probability tokens — outputs are more coherent but can get repetitive. Higher temperature (like 1.5) flattens the distribution, introducing more randomness and variety, though coherence takes a hit.

**Q2.** Top-k sampling helps by cutting off all the low-probability tokens from consideration, so the model won't randomly pick something unlikely or nonsensical — it keeps the pool of candidates reasonable.

**Q3.** Using BERT for generation doesn't work because it was trained bidirectionally with no causal masking. It can't generate autoregressively—it would need the entire target sequence at once to produce predictions, unlike GPT which generates one token at a time.

---

Task 6: GPT Demo

**Q1.** Greedy decoding (or temp=0.5) produced the most coherent output: '[example]'. Top-k with k=5, temp=1.5 gave variety: '[example]' but lost structure.

**Q2.** Not always — lower validation loss means the model is better at predicting the next token on average, but that doesn't automatically mean the actual generated text is more interesting or natural to read.

**Q3.** Even after training, the model still struggled with repetition, losing coherence over longer sequences, and occasionally producing grammatically off or just plain odd phrases.

---

Discussion

**Q1.** BERT can't really generate meaningful text since it's not built for that — it reads bidirectionally and has no autoregressive setup. GPT produces sequential text by predicting one token at a time from previous context, while BART generates conditioned outputs like reconstructions or translations using an encoder-decoder structure. The core differences come down to masking and attention design.

**Q2.** With causal masking, the model generates step-by-step in a logical order since each token only has access to what came before. Without it, the model sees future tokens during training, which makes it inconsistent and unreliable for actual generation — it's essentially predicting things it already knows.