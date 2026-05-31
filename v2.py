import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32

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

#attention head class
class Head(nn.Module):
    #one head of self attention
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        v = self.value(x) # (B,T,head_size)
        # Compute attention scores between every token pair
        # (B,T,16) @ (B,16,T) -> (B,T,T)
        weights = q @ k.transpose(-2, -1) * C**-0.5

        # Replace future-token positions with -inf
        # After softmax they become 0 attention
        weights = weights.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1) # (B,T,T)
        out = weights @ v
        return out


#multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    
    
#computation of the token
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )


    def forward(self, x):
        return self.net(x)

#transformer block
class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x

#bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.block = nn.Sequential(
            Block(n_embd, num_heads=4),
            Block(n_embd, num_heads=4),
            Block(n_embd, num_heads=4)
        )

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape

        token_embeddings = self.token_embedding_table(idx)  # (B,T,C)
        position_embeddings = self.position_embedding_table(torch.arange(T,device=device))
        x = token_embeddings + position_embeddings  # (B,T,C)
        #block of transformer
        x = self.block(x) # (B,T,C)

        logits = self.lm_head(x)  # (B,T,C)

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
            idx_cond = idx[:, -block_size:]  # crop context to block_size
            logits, loss = self(idx_cond)
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