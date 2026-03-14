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

from numpy import array, zeros
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
n_state = 32
block_size = 16 # maximum context length of the attention window (note: the longest name is 15 characters)

matrix = lambda nout, nin, std=0.08: Value(array([[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)]))
state_dict = {'wte': matrix(vocab_size, n_embd),
              'wpe': matrix(block_size, n_embd),
              'm': matrix(n_state, n_state),
              'token_proj': matrix(n_embd, n_state),
              'pos_proj': matrix(n_embd, n_state),
              'lm_head': matrix(n_state, vocab_size)}
params = list(state_dict.values())
print(f"num params: {len(params)}")

h = Value(zeros(n_state,))
logits_lst = []
losses = []
avg_loss = []
for j in range(block_size):
    token_id = Args(0, name=f'token{j}')
    pos_id = Args(0, name=f'pos{j}')
    target_id = Args(0, name=f'target{j}')

    h = (h @ state_dict['m']
         + state_dict['wte'].attend(token_id) @ state_dict['token_proj']
         + state_dict['wpe'].attend(pos_id) @ state_dict['pos_proj']).relu()

    logits = h @ state_dict['lm_head']
    logits_lst.append(logits)
    losses.append(- logits.softmax().attend(target_id).log())
    avg_loss.append(concatenate(losses, axis=0).mean())


def sgd_learning_rate():
    r = .01
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
