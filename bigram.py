import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 10000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(1337)

with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()

#tokenizer (encode and decode) -> character tokenizer

chars = sorted(list(set(text)))
vocab_size = len(chars
                 )
stoi = {}
for i, ch in enumerate(chars):
    stoi[ch] = i

# number -> char
itos = {}
for i, ch in enumerate(chars):
    itos[i] = ch


def encode(s):
    result = []

    for ch in s:
        result.append(stoi[ch])

    return result


def decode(nums):
    text = ""

    for n in nums:
        text += itos[n]

    return text


encoded = encode("hi")
print(encoded)

decoded = decode(encoded)
print(decoded)

#encoding entire dataset 

data = torch.tensor(encode(text),dtype=torch.long)
print(data[:1000])

# split data into train and validation sets
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
  if split == 'train':
    data = train_data
  else:
    data = val_data
  
  indexes = torch.randint(len(data) - block_size,(batch_size,))
  x = torch.stack([data[index:index+block_size] for index in indexes])
  y = torch.stack([data[index+1:index+block_size+1] for index in indexes])
  x,y = x.to(device), y.to(device)
  return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
model = BigramLanguageModel()
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))