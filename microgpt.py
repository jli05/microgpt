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
from numpy.random import normal
from micrograd import Value, Args
from micrograd.optim import SGD, ADAM

from dataloader import loader
from tokenizer import get_tokenizer

tokenizer = get_tokenizer()
# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
BOS = tokenizer.encode_special("<|bos|>")
vocab_size = tokenizer.get_vocab_size()
print(f"vocab size: {vocab_size}")


# Initialize the parameters, to store the knowledge of the model
n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd = 64     # width of the network (embedding dimension)
n_state = 256
n_att = 128
block_size = 512 # maximum context length of the attention window (note: the longest name is 15 characters)

matrix = lambda nout, nin, std=0.01: Value(array([[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)]))
state_dict = {'wte': matrix(vocab_size, n_embd),
              'wpe': matrix(block_size, n_embd),
              'm': Value(normal(0, 0.01, ( n_state, n_state))),
              'm2': Value(normal(0, 0.01, ( n_state, n_state))),
              'm3': Value(normal(0, 0.01, ( n_state, n_state))),
              'm4': Value(normal(0, 0.01, ( n_state, n_state))),
              'token_proj': matrix(n_embd, n_state),
              'pos_proj': matrix(n_embd, n_state),
              #'mode': matrix(n_state, n_mode),
              'lm_head': matrix(n_state, vocab_size)}
              #'outdict': matrix(vocab_size, n_embd)}
params = list(state_dict.values())
print(f"num params: {len(params)}")

h = Value(zeros(n_state,))
b = Value(zeros(n_state,))
logits_lst = []
loss_acc = Value(0)
avg_loss = []
avg_pen_loss = []
args_lst = []
for j in range(block_size):
    token_id = Args(0, name=f'token{j}')
    pos_id = Args(0, name=f'pos{j}')
    target_id = Args(0, name=f'target{j}')

    args = (h - b).topk(n_att)
    args_lst.append(args)
    inc_b = (h.attend(args) @ state_dict['m3'].attend(args)
             + b.attend(args) @ state_dict['m4'].attend(args))
    inc_h = (h.attend(args) @ state_dict['m'].attend(args)
             + b.attend(args) @ state_dict['m2'].attend(args)
             + state_dict['wte'].attend(token_id) @ state_dict['token_proj']
             + state_dict['wpe'].attend(pos_id) @ state_dict['pos_proj'])
    b += inc_b
    h += inc_h
    b = b.relu().log1p()
    h = h.relu().log1p()

    logits = (h - b) @ state_dict['lm_head']
    logits_lst.append(logits)

    curr_loss = - logits.softmax().attend(target_id).log()
    avg_loss.append(loss_acc + (curr_loss - loss_acc) / (j + 1))
    loss_acc = avg_loss[-1]
    avg_pen_loss.append(avg_loss[-1])


num_steps = 5000 # number of training steps

def learning_rate(lr0, num_steps):
    d = lr0 / num_steps
    lr = lr0
    while True:
        yield lr
        lr -= d


optimizer = ADAM(list(state_dict.values()),
                 learning_rate=learning_rate(.01, num_steps),
                 beta1=.85, beta2=.99, eps_adam=1e-8)

# Repeat in sequence
text_iterator = loader()

for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    text = next(text_iterator)
    tokens = tokenizer.encode(text, prepend='<|bos|>', append='<|bos|>')
    n = min(block_size, len(tokens) - 1)

    # Forward the token sequence through the model, building up the computation graph all the way to the loss
    io_dict = {}
    for j in range(n):
        io_dict[f'token{j}'] = tokens[j]
        io_dict[f'target{j}'] = tokens[j + 1]
        io_dict[f'pos{j}'] = j

    # Backward the loss, calculating the gradients with respect to all model parameters
    if n:
        avg_pen_loss[n - 1].forward(**io_dict)
        avg_pen_loss[n - 1].backward()
        optimizer.step()

    print(f"step {step+1:4d} / {num_steps:4d}"
          f" | loss {avg_loss[n - 1].data:.4f}"
          f" | pen_loss {avg_pen_loss[n - 1].data:.4f}", end='\r')


# Inference: may the model babble back to us
temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (new, hallucinated names) ---")
from numpy import sort
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
    for j in range(pos_id):
        #print(f'args{j}', sort(args_lst[j].data))
        print(mode_lst[j].data)
