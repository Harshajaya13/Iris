import torch
from dataset import DataLoader
from model import GPT

vocab_size = 50257
seq_len = 64
num_dims = 128
num_heads = 4
num_layers = 4

block_size = 64
batch_size = 16
device = "cuda" if torch.cuda.is_available() else "cpu"

learning_rate = 1e-3
max_iters = 1000
eval_interval = 100

dataset = DataLoader(block_size, batch_size, device)
model = GPT(vocab_size, seq_len, num_dims, num_heads, num_layers, p=0.1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


@torch.no_grad()
def estimate_loss(model, dataset, eval_iters=20):
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = dataset.get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


for i in range(max_iters):

    if i%eval_interval == 0 or i == max_iters-1 :
        losses = estimate_loss(model,dataset)
        print(f"Step {i:4d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")

    xb,yb = dataset.get_batch("train")

    optimizer.zero_grad(set_to_none=True)

    logits,loss = model(xb,yb)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

torch.save(model.state_dict(), "checkpoint.pt")
print("Training complete. Weights saved to checkpoint.pt")