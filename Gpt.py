import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
# ------------------------------

with open('input.txt', 'r', encoding='utf-8') as F:
    text = F.read()


char=sorted(list(set(text)))
vocab_size=len(char)

# This is tokenizer that give us string to integer
stoi={ ch:i for i,ch in enumerate(char)}
itos={ i:ch for i,ch in enumerate(char)}
encode= lambda s:[stoi[c] for c in s]
decoder= lambda l: ''.join([itos[i] for i in l])

# Converted whole dataset into interger for prediction

data=torch.tensor(encode(text),dtype=torch.long)
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]


def get_batch(split):
  data=train_data if split=='train' else val_data
  ix=torch.randint(len(data)-block_size,(batch_size,))
  x=torch.stack([data[i:i+block_size] for i in ix])
  y=torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x,y

@torch.no_grad()
def estimate_loss():
   out={}
   model.eval()
   for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
   model.train()
   return out


class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
    self.positional_embedding=nn.Embedding(block_size,n_embd)
    self.ln_head=nn.Linear(n_embd,vocab_size)
  def forward(self,idx,target=None):
    token_embd=self.token_embedding_table(idx) #(B,T,C)
    pos_embd=self.positional_embedding(torch.arange(T,device=idx.device)) #(T,C)
    x=token_embd + pos_embd #(B,T,C)
    logits=self.ln_head(token_embd)
    if target==None :
      loss=None
    else:
      B,T,C= logits.shape
      logits=logits.view(B*T,C)
      target=target.view(B*T)
      loss=f.cross_entropy(logits,target)
    return logits,loss
  def generate(self,idx,max_new_tokens):
    for _ in range(max_new_tokens):
      logits,loss=self(idx)
      logits=logits[:,-1,:]
      probs=f.softmax(logits,dim=-1)
      idx_next=torch.multinomial(probs,num_samples=1)
      idx=torch.cat((idx,idx_next),dim=1)
    return idx

model=BigramLanguageModel()
m=model.to(device)
optimizer=optim.AdamW(model.parameters(),lr=learning_rate)
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")

    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decoder(model.generate(context, max_new_tokens=500)[0].tolist()))
