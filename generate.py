import torch
import torch.nn.functional as F
import tiktoken
from model import GPT

device = "cuda" if torch.cuda.is_available() else "cpu"

vocab_size = 50257
seq_len = 128
num_dims = 256
num_heads = 4
num_layers = 6

# Initialize the architecture with 0% dropout for evaluation/inference
model = GPT(vocab_size, seq_len, num_dims, num_heads, num_layers, p=0.0).to(device)

# Load the saved model parameters
checkpoint = torch.load("best_checkpoint.pt", map_location=device, weights_only=True)

if isinstance(checkpoint, dict) and "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)

model.eval()

enc = tiktoken.get_encoding("gpt2")
prompt = input("enter the prompt: ")
prompt_tokens = enc.encode(prompt)
idx = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

max_new_tokens = 50
temperature = 0.8
top_k = 40

print(f"\nPrompt: {prompt}")
print("--- Generating ---")

with torch.no_grad():
    for _ in range(max_new_tokens):

        # Crop context to maximum sequence length if it gets too long
        idx_cond = idx if idx.size(1) <= seq_len else idx[:, -seq_len:]

        # When we don't pass the targets, we will be returned with the logits only
        logits, _ = model(idx_cond)

        logits = logits[:, -1, :]
        """
        You hand the model: ["The", "danger"].
        
        Because Transformers process all positions simultaneously, model(idx_cond) outputs guesses for every position:
            Position 1: "What comes after 'The'?"
            Position 2: "What comes after 'danger'?"
            
        Shape of logits coming out is: (1, 2, 50257).
            1 batch
            2 words
            50257 raw scores for every word in the English dictionary.
            
        We only care about what comes after "danger" (the very last word!).
        logits[:, -1, :] means: take batch 0, take the very last time position (-1), and keep all 50,257 vocabulary scores.
        Now the shape is just (1, 50257): one row of raw scores for the next upcoming word.
        """

        # Apply temperature scaling
        logits = logits / temperature

        # Top-k truncation: keep only the top K most probable tokens and mask out the rest
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        # Converts the logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # Sample the next token from the probability distribution
        idx_next = torch.multinomial(probs, num_samples=1)

        # Append the sampled token to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)

generated_tokens = idx[0].tolist()
output_text = enc.decode(generated_tokens)
print(output_text)