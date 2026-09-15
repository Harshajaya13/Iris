# Generation

So in generation we are gonna generate text from the model. From the model we get the logits and loss — but here we only care about the logits, not the loss, because we are not training.

---

## Loading the Checkpoint

We instantiate the model object first — same architecture as training, but with `p=0.0` (dropout off, because we are evaluating not training).

```python
model = GPT(vocab_size, seq_len, num_dims, num_heads, num_layers, p=0.0).to(device)
checkpoint = torch.load("best_checkpoint.pt", map_location=device, weights_only=True)
```

Now the checkpoint we saved earlier is a **dict** — it has `model`, `optimizer`, `iter`, `best_val_loss` inside it. But someone might also pass just raw model weights (old saves, different format). So we check:

```python
if isinstance(checkpoint, dict) and "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)
```

If it is a dict and has `"model"` key — we fetch from that. If not, we treat the whole thing as raw weights and load directly. Either way works, no crash.

Then `model.eval()` — stops the training mode, disables dropout.

---

## Encoding the Prompt

We use tiktoken again here because we need to encode the user's text into tokens before giving it to the model.

```python
prompt_tokens = enc.encode(prompt)
idx = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
```

We convert the list to a tensor and then `unsqueeze(0)` adds a batch dimension. So the shape goes from `(T,)` to `(1, T)` — that is `(B, T)` where `B=1`. The model expects that batch dimension.

And inside the model, when computing loss, the logits get reshaped from `(B, T, vocab_size)` → `(B*T, vocab_size)`. That flattens all batch and time positions into one big row so cross-entropy loss can be computed over every single position at once.

---

## No Grad

```python
with torch.no_grad():
```

We are not training here, just using the model to generate. So we don't need to compute or store gradients. `torch.no_grad()` skips all that — faster and less memory.

---

## Context Cropping — `max_new_tokens` vs `seq_len`

These two are completely different things.

- `seq_len = 128` — this is the max number of tokens the model can look at **at once**. It is the context window, the model's memory. It was fixed during training and cannot change.
- `max_new_tokens = 50` — this is how many **new** tokens we want the model to generate on top of the prompt.

So if our prompt is 10 tokens and we generate 50 new ones, the full sequence becomes 60 tokens — still under 128, no problem. But if we keep generating past 128, we have to crop:

```python
idx_cond = idx if idx.size(1) <= seq_len else idx[:, -seq_len:]
```

We only feed the last `seq_len` tokens into the model. The earlier ones get dropped — the model just can't see that far back.

---

## Getting Logits

```python
logits, _ = model(idx_cond)
```

We are not passing targets so we don't get a loss — only logits. We ignore the second return value with `_`.

The model processes all positions at the same time. So if we give it `["The", "danger"]`, it outputs predictions for every position:

- Position 1 — what comes after `"The"`?
- Position 2 — what comes after `"danger"`?

The shape of logits coming out is `(1, 2, 50257)` — 1 batch, 2 words, 50257 scores for every token in the vocabulary.

We only care about what comes after the last word:

```python
logits = logits[:, -1, :]
```

`[:, -1, :]` means — take all batches, take the very last time step, keep all 50257 scores. Shape goes from `(1, T, 50257)` → `(1, 50257)`. One row of raw scores for the next token.

---

## Temperature

```python
logits = logits / temperature
```

Temperature controls how confident or random the model is. If `temperature = 1.0` the distribution is unchanged. If it is low like `0.1` the model gets very confident — always picks the most probable token, output is repetitive. If it is high like `1.5` the model gets more random and creative.

We are using `0.8` — slightly more focused than default, but still some variation.

---

## Top-K Truncation

```python
v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
logits[logits < v[:, [-1]]] = float("-inf")
```

Out of 50257 possible next tokens, most of them have extremely low probability and are basically noise. Top-K cuts everything except the `top_k` most probable tokens by setting everything below the threshold to `-inf`. After softmax, those `-inf` values become zero probability — they can never be sampled.

We are using `top_k = 40` — only the top 40 candidates survive.

---

## Softmax → Sample

```python
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```

Softmax converts the raw logit scores to a proper probability distribution that sums to 1.

`torch.multinomial` samples one token from that distribution — not always the top one, but weighted by probability. This is what makes the output feel natural instead of deterministic.

---

## Append and Loop

```python
idx = torch.cat((idx, idx_next), dim=1)
```

We append the newly sampled token back to the sequence and loop again. Each iteration generates one token. We do this for `max_new_tokens = 50` times.

---

## Decode

```python
generated_tokens = idx[0].tolist()
output_text = enc.decode(generated_tokens)
```

`idx[0]` takes the first (only) batch. `.tolist()` converts to a plain Python list. Then `enc.decode()` converts token IDs back to the actual text string.

> **Pipeline:** load checkpoint → encode prompt → crop context → forward → slice last logit → temperature → top-k → softmax → sample → append → decode
