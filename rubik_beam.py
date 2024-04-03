import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

def color(v):
    return v.argmax().item()


def make_color(color):
    return torch.tensor([1 if i == color else 0 for i in range(6)], dtype=torch.int8)


def make_face(color):
    return torch.cat([make_color(color) for _ in range(8)])


def show_face(face, center):
    return [
        [color(face[0:6]), color(face[6:12]), color(face[12:18])],
        [color(face[18:24]), center, color(face[24:30])],
        [color(face[30:36]), color(face[36:42]), color(face[42:48])],
    ]


def show_state(state):
    faces = [show_face(state[48 * i : 48 * i + 48], i) for i in range(6)]
    strs = [["|" + " ".join(map(str, row)) for row in face] for face in faces]
    print("+-----+")
    for row in range(3):
        print(strs[0][row] + "|")
    print("+-----+-----+-----+-----+")
    for row in range(3):
        print(strs[1][row] + strs[2][row] + strs[3][row] + strs[4][row] + "|")
    print("+-----+-----+-----+-----+")
    for row in range(3):
        print(strs[5][row] + "|")
    print("+-----+")


def solved_state():
    return torch.cat([make_face(i) for i in range(6)])


def cycle(state, indices):
    result = state.clone()
    for i in range(len(indices)):
        old = indices[i]
        new = indices[(i + 1) % len(indices)]
        result[6 * old : 6 * old + 6] = state[6 * new : 6 * new + 6]
    return result


def cycles(state, cycles):
    result = state.clone()
    for indices in cycles:
        result = cycle(result, indices)
    return result


def clock(center):
    return [
        [0 + 8 * center, 5 + 8 * center, 7 + 8 * center, 2 + 8 * center],
        [1 + 8 * center, 3 + 8 * center, 6 + 8 * center, 4 + 8 * center],
    ]

def counter(center):
    return [
        [0 + 8 * center, 2 + 8 * center, 7 + 8 * center, 5 + 8 * center],
        [1 + 8 * center, 4 + 8 * center, 6 + 8 * center, 3 + 8 * center],
    ]


def u(state):
    return cycles(
        state,
        clock(0) + [[8 * j + i for j in [1, 2, 3, 4]] for i in range(3)],
    )


def d(state):
    return cycles(
        state,
        counter(5) + [[8 * j + i for j in [1, 2, 3, 4]] for i in [5, 6, 7]],
    )


def l(state):
    return cycles(
        state,
        counter(4) + [[0, 8, 40, 31], [3, 11, 43, 28], [5, 13, 45, 26]],
    )

def r(state):
    return cycles(
        state,
        clock(2) + [[2, 10, 42, 29], [4, 12, 44, 27], [7, 15, 47, 24]],
    )
    
def f(state):
    return cycles(
        state,
        clock(1) + [[7, 34, 40, 21], [6, 36, 41, 19], [5, 39, 42, 16]],
    )

def b(state):
    return cycles(
        state,
        counter(3) + [[0, 37, 47, 18], [1, 35, 46, 20], [2, 32, 45, 23]],
    )

_SOLVED_STATE = solved_state()

def step(depth, prev=None):
    if depth == 0:
        return torch.stack([_SOLVED_STATE])
    states = []
    for move in [u, d, l, r, f, b]:
        if prev is None or move != prev:
            old = step(depth - 1, move)
            for state in old:
                for _ in range(3):
                    state = move(state)
                    states.append(state)
    return torch.stack(states)

_MOVES = list(map(torch.vmap, [u, d, l, r, f, b]))

_STEPS = 20
data = []

_HASH = torch.randint(0, 1<<54, (288,), dtype=torch.int64).cpu()

def hash_state(state):
    return torch.dot(_HASH, state.long()).item()

all_states = torch.stack([_SOLVED_STATE]).cuda()
sample_size = 300000

if True:
    data_tensor = torch.load("data.pt").cuda()
else:
    for step_count in range(_STEPS):
        start = time.time()
        #print(start)
        new_states = None
        for move in _MOVES:
            news = []
            last_new = all_states
            for _ in range(3):
                last_new = move(last_new)
                news.append(last_new)
            all_new = torch.cat(news)
            if new_states is None:
                new_states = all_new
            else:
                new_states = torch.cat([new_states, all_new])
        all_states = new_states
        if step_count < 5:
            all_states = torch.unique(all_states, dim=0, sorted=False)
        else:
            if step_count == 5:
                five_states = all_states.cpu()
            all_states = all_states[torch.randperm(all_states.size(0))[:sample_size]]
        data.append(all_states)
        print(step_count, time.time() - start, all_states.shape)
        data_tensor = torch.cat(data)
        torch.save(data_tensor, "data.pt")

if True:
    five_hashes_tensor = torch.load("five_hashes.pt")
    five_states = torch.load("five_states.pt")
    five_hash_set = set(five_hashes_tensor.cpu().numpy())
else:
    start = time.time()
    five_hashes = []
    five_hash_set = set()
    for i, state in enumerate(five_states):
        h = hash_state(state)
        five_hashes.append(h)
        five_hash_set.add(h)
        if i % 100000 == 0:
            print(i, time.time() - start, len(five_hash_set), len(five_hashes))
    print("Hashing", time.time() - start, len(five_hash_set), len(five_hashes))
    five_hashes_tensor = torch.tensor(five_hashes, dtype=torch.int64).cpu()
    torch.save(five_states, "five_states.pt")
    torch.save(five_hashes_tensor, "five_hashes.pt")

def is_solved(state):
    h = hash_state(state)
    if h not in five_hash_set:
        return False
    for c in five_states[h == five_hashes_tensor]:
        if c.equal(state):
            return True
    return False

task = [u, f, f, r, d, l, u, b, b, u]
state = _SOLVED_STATE
for move in task:
    state = move(state)
show_state(state)
print(is_solved(state))

exit()

_HIDDEN = 256

_LOWLOW = 14
_LOW = 16
_HIGH = 16
_HIGHHIGH = 18
_DROPOUT = 0.25
simple = nn.Sequential(
    nn.Linear(288, _HIDDEN),
    nn.ReLU(),
    nn.Dropout(_DROPOUT),
    nn.Linear(_HIDDEN, _HIDDEN),
    nn.ReLU(),
    nn.Dropout(_DROPOUT),
    nn.Linear(_HIDDEN, _HIDDEN),
    nn.ReLU(),
    nn.Dropout(_DROPOUT),
    nn.Linear(_HIDDEN, _HIDDEN),
    nn.ReLU(),
    nn.Dropout(_DROPOUT),
    nn.Linear(_HIDDEN, _HIDDEN),
    nn.ReLU(),
    nn.Dropout(_DROPOUT),
    nn.Linear(_HIDDEN, 1),
    nn.Sigmoid(),
).cuda()

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(simple.parameters(), lr=0.0001)

train_size = sample_size
test_size = sample_size // 10

small_data = torch.cat(data[_LOWLOW:_LOW])
small_data = small_data[torch.randperm(small_data.size(0))]
small_train = small_data[:train_size]
small_test = small_data[train_size:train_size + test_size]

large_data = torch.cat(data[_HIGH:_HIGHHIGH])
large_data = large_data[torch.randperm(large_data.size(0))]
large_train = large_data[:train_size]
large_test = large_data[train_size:train_size + test_size]

train_input = torch.cat([small_train, large_train])
train_output = torch.cat([torch.ones(train_size), torch.zeros(train_size)]).unsqueeze(1).cuda()
# Shuffle the training data
train_perm = torch.randperm(train_input.size(0))
train_input = train_input[train_perm].float()
train_output = train_output[train_perm].float()

test_input = torch.cat([small_test, large_test])
test_output = torch.cat([torch.ones(test_size), torch.zeros(test_size)]).unsqueeze(1).cuda()
# Shuffle the test data
test_perm = torch.randperm(test_input.size(0))
test_input = test_input[test_perm].float()
test_output = test_output[test_perm].float()

_BATCH_SIZE = 128
for epoch in range(1000):
    print(f"Epoch {epoch}")
    for i in range(0, train_size, _BATCH_SIZE):
        optimizer.zero_grad()
        output = simple(train_input[i:i + _BATCH_SIZE])
        loss = loss_fn(output, train_output[i:i + _BATCH_SIZE])
        loss.backward()
        optimizer.step()
    if epoch % 1 == 0:
        with torch.no_grad():
            output = simple(test_input)
            loss = loss_fn(output, test_output)
            print(epoch, loss.item())
        with torch.no_grad():
            output = simple(train_input[:10000])
            loss = loss_fn(output, train_output[:10000])
            print(epoch, loss.item())
