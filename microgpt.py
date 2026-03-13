"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy
"""

import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
random.seed(42) # Let there be order among chaos

from numpy import array
from micrograd import Value, Args, concatenate, vstack
from micrograd.optim import SGD

# Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for a special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")


# Initialize the parameters, to store the knowledge of the model
n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd = 16     # width of the network (embedding dimension)
block_size = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
n_head = 4      # number of attention heads
head_dim = n_embd // n_head # derived dimension of each head
matrix = lambda nout, nin, std=0.08: Value(array([[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)]))
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(n_embd, vocab_size)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(n_embd, 4 * n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(4 * n_embd, n_embd)
params = list(state_dict.values())
print(f"num params: {len(params)}")

# Define the model architecture: a function mapping tokens and parameters to logits over what comes next
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU

def rmsnorm(x):
    ms = (x * x).mean()
    scale = (ms + 1e-5) ** -0.5
    return x * scale

def gpt(token_id, pos_id, target_id, keys, values, logits_lst):
    tok_emb = state_dict['wte'].attend(token_id) # token embedding
    pos_emb = state_dict['wpe'].attend(pos_id) # position embedding
    x = tok_emb + pos_emb
    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection

    for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)
        q = x @ state_dict[f'layer{li}.attn_wq']
        k = x @ state_dict[f'layer{li}.attn_wk']
        v = x @ state_dict[f'layer{li}.attn_wv']
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            args = slice(hs, hs + head_dim)
            attn_logits = (q.attend(args)
                           @ vstack([ki.attend(args)
                                     for ki in keys[li]]).T / head_dim ** .5)
            attn_weights = attn_logits.softmax()
            head_out = (attn_weights
                        @ vstack([vi.attend(args)
                                  for vi in values[li]]))
            x_attn.append(head_out)

        x = concatenate(x_attn, axis=0) @ state_dict[f'layer{li}.attn_wo']
        x += x_residual
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = x @ state_dict[f'layer{li}.mlp_fc1']
        x = x.relu()
        x = x @ state_dict[f'layer{li}.mlp_fc2']
        x += x_residual

    logits = x @ state_dict['lm_head']
    logits_lst.append(logits)
    probs = logits.softmax()
    loss = - probs.attend(target_id).log()
    return loss

losses = []
keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
logits_lst = []
avg_loss = []
for j in range(block_size):
    losses.append(gpt(Args(0, name=f'token{j}'),
                      Args(0, name=f'pos{j}'),
                      Args(0, name=f'target{j}'),
                      keys, values, logits_lst))
    avg_loss.append(concatenate(losses, axis=0).mean())

def sgd_learning_rate():
    r = .002
    while True:
        yield r
        r *= .998

sgd = SGD(list(state_dict.values()), learning_rate=sgd_learning_rate(),
          momentum=.99)

# Repeat in sequence
num_steps = 1000 # number of training steps
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # Forward the token sequence through the model, building up the computation graph all the way to the loss
    io_dict = {}
    for j in range(n):
        io_dict[f'token{j}'] = tokens[j]
        io_dict[f'target{j}'] = tokens[j + 1]
        io_dict[f'pos{j}'] = j

    # Backward the loss, calculating the gradients with respect to all model parameters
    avg_loss[n - 1].forward(**io_dict)
    avg_loss[n - 1].backward()
    sgd.step()

    print(f"step {step+1:4d} / {num_steps:4d}"
          f" | loss {avg_loss[n - 1].data:.4f}", end='\r')

# Inference: may the model babble back to us
temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    token_id = BOS
    sample = []
    io_dict = {}
    for pos_id in range(block_size):
        io_dict[f'token{pos_id}'] = token_id
        io_dict[f'pos{pos_id}'] = pos_id
        logits_lst[pos_id].forward(**io_dict)
        probs = (logits_lst[pos_id] / temperature).softmax()
        token_id = random.choices(range(vocab_size), weights=probs.data)[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
