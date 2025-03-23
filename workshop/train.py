# 1. TRAINING DATA 
text = "bejohn121llllll!$%"




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
validation_data=pytorch_data[len_training_data:]








# 6. Blocking 
block_size = 8

x = train_data[:block_size]
y = train_data[1:block_size + 1] 

for t in range(block_size):
    context = x[:t+1] 
    target = y[t] 

    print(f"""
        All samGPT sees right now is: {context}
        samGPT needs to be able to guess {target} 
    """)
