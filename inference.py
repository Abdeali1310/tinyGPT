import torch
import time
from gpt import BigramLanguageModel, block_size, device, decode, encode
# initialize model
model = BigramLanguageModel()

# load trained weights
model.load_state_dict(torch.load("tinygpt_model.pth", map_location=device))

model = model.to(device)
model.eval()

# prompt
prompt = input("Enter prompt: ")

# number of characters to generate
max_new_tokens = input("Number of characters to generate (default 1100): ")

# default value
if max_new_tokens.strip() == "":
    max_new_tokens = 1100
else:
    max_new_tokens = int(max_new_tokens)

context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)


print(prompt, end="", flush=True)

# generate text
for _ in range(max_new_tokens):

    idx_cond = context[:, -block_size:]

    logits, loss = model(idx_cond)

    logits = logits[:, -1, :]

    probs = torch.softmax(logits, dim=-1)

    idx_next = torch.multinomial(probs, num_samples=1)

    context = torch.cat((context, idx_next), dim=1)

    next_char = decode(idx_next[0].tolist())

    print(next_char, end="", flush=True)

    time.sleep(0.02)