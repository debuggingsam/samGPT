# 1. TRAINING DATA 
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# 2. Figure out what characters are we are working with.

# What are all the UNIQUE characters in the training data?
unique_chars = set(text)

# Turn that into a list, and sort it
unique_chars_list = sorted(list(unique_chars))
count = (''.join(unique_chars))
# print(f"What are all the unique characters? {count} ")


vocab_size = len(unique_chars_list)
# print(f'How many characters does the model know about? --> {vocab_size}') 














# 3. Tokenizer 
stoi = { ch:i for i, ch in enumerate(unique_chars_list) }
itos = { i:ch for i, ch in enumerate(unique_chars_list) } 

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# print(encode("bejohn121llllll!$%")) # Just testing to see if the encoder works 
# print(decode([5, 6, 9, 9, 11, 0])) # Decoding 




# 4. Turning our sample data into data pytorch can use. 
import torch

# Encode everything from our original text 
encoded = encode(text) 

pytorch_data = torch.tensor(encoded, dtype=torch.long)
# print(f"Type of integers used: {pytorch_data.dtype}")
# print(f"Shape of data: {pytorch_data.shape}")
# print(pytorch_data[:1000]) 



















# 5. Split up our initial data into a 90/10 split.
len_training_data = int(0.9 * len(pytorch_data))

# The first 90% is what we're going to train our model on.
train_data = pytorch_data[:len_training_data] 

# The last 10% is going to basically be used as a test to check how good we are doing. 
# We're hiding this last 10% from the model so that we have something to benchmark against.
# There IS a right answer 
val_data=pytorch_data[len_training_data:]







# 6. Blocking 
block_size = 8

x = train_data[:block_size]
y = train_data[1:block_size + 1] 

for t in range(block_size):
    context = x[:t+1] 
    target = y[t] 

    # print(f"""
    #     All samGPT sees right now is: {context}
    #     samGPT needs to be able to guess {target} 
    # """)





# 7. Parallel
torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
# print('inputs:')
# print(xb.shape)
# print(xb)
# print('targets:')
# print(yb.shape)
# print(yb)

# print('----')

# for b in range(batch_size): # batch dimension
#     for t in range(block_size): # time dimension
#         context = xb[b, :t+1]
#         target = yb[b,t]
#         print(f"when input is {context.tolist()} the target: {target}")
















# 8. Guessing game!

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
# print(logits.shape)
# print(loss)



print("--------------------")
print("Before optimizations") 
print("--------------------")
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))






#9. Optimize it

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
batch_size = 32
for steps in range(10000): # increase number of steps for good results...

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# print(loss.item())



# 10. PRAYGE!

print("--------------------")
print("After optimizations") 
print("--------------------")
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))