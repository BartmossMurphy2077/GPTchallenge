# ============================================================
# 05_training_utils_and_demos.py
# Full helpers and demos for BERT and BART.
# GPT demo intentionally left empty.
# ============================================================
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


@torch.no_grad()
def estimate_bert_loss(model, eval_iters=20):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        accs = []
        for _ in range(eval_iters):
            x, y = get_classification_batch(split)
            logits, loss = model(x, y)
            preds = logits.argmax(dim=-1)
            acc = (preds == y).float().mean().item()
            losses.append(loss.item())
            accs.append(acc)
        out[split] = {"loss": sum(losses) / len(losses), "acc": sum(accs) / len(accs)}
    model.train()
    return out


@torch.no_grad()
def estimate_bart_loss(model, eval_iters=20):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            src, tgt_in, tgt_out = get_seq2seq_batch(split)
            _, loss = model(src, tgt_in, tgt_out)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


@torch.no_grad()
def estimate_gpt_loss(model, eval_iters=20):
    """
    Students may use this function once TinyGPT is complete.
    """
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            x, y = get_lm_batch(split)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


@torch.no_grad()
def generate_beam_search(model, idx, max_new_tokens=100, beam_width=3):
    """
    Simple beam search for batch size 1.
    """
    beam_width = max(1, int(beam_width))
    beams = [(idx, 0.0)]

    for _ in range(max_new_tokens):
        candidates = []
        for seq, score in beams:
            idx_cond = seq[:, -model.context_length :]
            logits, _ = model(idx_cond)
            logits_last = logits[:, -1, :]
            log_probs = F.log_softmax(logits_last, dim=-1)

            k = min(beam_width, log_probs.size(-1))
            topk_log_probs, topk_idx = torch.topk(log_probs, k=k, dim=-1)

            for j in range(k):
                next_token = topk_idx[:, j : j + 1]
                token_log_prob = topk_log_probs[:, j].item()
                new_seq = torch.cat([seq, next_token], dim=1)
                candidates.append((new_seq, score + token_log_prob))

        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

    return beams[0][0]


# ============================================================
# BERT demo (full)
# ============================================================

print("\n" + "=" * 60)
print("BERT-LIKE DEMO")
print("=" * 60)

bert_model = TinyBERT(
    vocab_size=vocab_size,
    d_model=d_model,
    context_length=context_length,
    n_layers=n_layers,
    n_classes=2,
).to(device)

bert_optimizer = torch.optim.Adam(bert_model.parameters(), lr=learning_rate)

for step in range(101):
    if step % 50 == 0:
        stats = estimate_bert_loss(bert_model, eval_iters=10)
        print(
            f"Step {step:3d} | "
            f"train loss: {stats['train']['loss']:.4f} | "
            f"train acc: {stats['train']['acc']:.4f} | "
            f"val loss: {stats['val']['loss']:.4f} | "
            f"val acc: {stats['val']['acc']:.4f}"
        )

    xb, yb = get_classification_batch("train")
    logits, loss = bert_model(xb, yb)

    bert_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    bert_optimizer.step()


# ============================================================
# BART demo (full)
# ============================================================

print("\n" + "=" * 60)
print("BART-LIKE DEMO")
print("=" * 60)

bart_model = TinyBART(
    vocab_size=vocab_size,
    d_model=d_model,
    context_length=context_length,
    n_layers=n_layers,
).to(device)

bart_optimizer = torch.optim.Adam(bart_model.parameters(), lr=learning_rate)

for step in range(101):
    if step % 50 == 0:
        losses = estimate_bart_loss(bart_model, eval_iters=10)
        print(
            f"Step {step:3d} | "
            f"train loss: {losses['train']:.4f} | "
            f"val loss: {losses['val']:.4f}"
        )

    src, tgt_in, tgt_out = get_seq2seq_batch("train")
    logits, loss = bart_model(src, tgt_in, tgt_out)

    bart_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    bart_optimizer.step()


# ============================================================
# GPT demo (empty on purpose)
# ============================================================

print("\n" + "=" * 60)
print("GPT-LIKE DEMO")
print("=" * 60)

gpt_model = TinyGPT(
    vocab_size=vocab_size,
    d_model=d_model,
    context_length=context_length,
    n_layers=n_layers,
).to(device)

gpt_optimizer = torch.optim.Adam(gpt_model.parameters(), lr=learning_rate)

for step in range(101):
    if step % 50 == 0:
        losses = estimate_gpt_loss(gpt_model, eval_iters=10)
        print(
            f"Step {step:3d} | "
            f"train loss: {losses['train']:.4f} | "
            f"val loss: {losses['val']:.4f}"
        )

    xb, yb = get_lm_batch("train")
    logits, loss = gpt_model(xb, yb)

    gpt_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gpt_optimizer.step()

prompt = "The model "
start_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

gpt_model.eval()
with torch.no_grad():
    greedy_ids = gpt_model.generate_greedy(start_ids.clone(), max_new_tokens=120)
    temp_ids = gpt_model.generate_temperature(
        start_ids.clone(), max_new_tokens=120, temperature=0.8
    )
    topk_ids = gpt_model.generate_top_k(
        start_ids.clone(), max_new_tokens=120, temperature=1.0, k=5
    )
    beam_ids = generate_beam_search(
        gpt_model, start_ids.clone(), max_new_tokens=120, beam_width=3
    )
gpt_model.train()

print("\nPrompt:")
print(prompt)

print("\nGreedy:")
print(decode(greedy_ids[0].tolist()))

print("\nTemperature (0.8):")
print(decode(temp_ids[0].tolist()))

print("\nTop-k (k=5):")
print(decode(topk_ids[0].tolist()))

print("\nBeam search (width=3):")
print(decode(beam_ids[0].tolist()))
