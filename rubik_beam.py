import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import random


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

reload = True

if reload:
    _HASH = torch.load("hash.pt")
else:
    _HASH = torch.randint(0, 1 << 54, (288,), dtype=torch.int64).cpu()
    torch.save(_HASH, "hash.pt")


def hash_state(state):
    return torch.dot(_HASH, state.long()).item()


sample_size = 300000


def find_moves(start, target, depth):
    if depth == 0:
        return None
    for move in [u, d, l, r, f, b]:
        for rep in [1, 2, 3]:
            next_state = start
            for _ in range(rep):
                next_state = move(next_state)
            if next_state.equal(target):
                return [move.__name__ * rep]
            moves = find_moves(next_state, target, depth - 1)
            if moves is not None:
                return [move.__name__ * rep] + moves
    return None


def run_steps(first_state, num_steps):
    all_states = torch.stack([first_state]).cuda()
    five_states = None
    data = []
    for step_count in range(num_steps):
        start = time.time()
        # print(start)
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
        # print(step_count, time.time() - start, all_states.shape)
    return data, five_states


if reload:
    data = torch.load("data.pt")
else:
    data, five_states = run_steps(_SOLVED_STATE, _STEPS)
    torch.save(data, "data.pt")

if reload:
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


def candidates(state, depth=4):
    all_states, _ = run_steps(state, depth)
    return torch.cat(all_states).unique(dim=0, sorted=False)


def thing():
    show_state(five_states[42])
    print(is_solved(five_states[42]))

    task = [u, l, r, r, f, d, d, d, l]
    state = _SOLVED_STATE
    for move in task:
        state = move(state)
    show_state(state)
    print(is_solved(state))

    for move in [f, r, l, b]:
        state = move(state)
    show_state(state)
    print(is_solved(state))

    guesses = candidates(state)
    print(guesses.shape)
    for i, guess in enumerate(guesses):
        if is_solved(guess.cpu()):
            print(i, "Solved")
            show_state(guess)
            break
    else:
        print("Not solved")

    exit()


# _LOWLOW = 14
# _LOW = 16
# _HIGH = 16
# _HIGHHIGH = 18

# simple = nn.Sequential(
#     nn.Linear(288, _HIDDEN),
#     nn.ReLU(),
#     nn.Dropout(_DROPOUT),
#     nn.Linear(_HIDDEN, _HIDDEN),
#     nn.ReLU(),
#     nn.Dropout(_DROPOUT),
#     nn.Linear(_HIDDEN, _HIDDEN),
#     nn.ReLU(),
#     nn.Dropout(_DROPOUT),
#     nn.Linear(_HIDDEN, _HIDDEN),
#     nn.ReLU(),
#     nn.Dropout(_DROPOUT),
#     nn.Linear(_HIDDEN, _HIDDEN),
#     nn.ReLU(),
#     nn.Dropout(_DROPOUT),
#     nn.Linear(_HIDDEN, 1),
#     nn.Sigmoid(),
# ).cuda()
# loss_fn = nn.MSELoss()


def nice(steps):
    if len(steps) == 0:
        return []
    first = steps[0]
    count = 1
    while count < len(steps) and steps[count] == first:
        count += 1
    modifier = (
        "" if count == 1 else "2" if count == 2 else "'" if count == 3 else str(count)
    )
    result = [first.upper() + modifier] + nice(steps[count:])
    return result


def simplify_repeats(steps):
    if len(steps) < 4:
        return steps
    first = steps[0]
    count = 1
    while count < len(steps) and steps[count] == first:
        count += 1
    if count >= 4:
        return steps[4:]
    return steps[:count] + simplify_repeats(steps[count:])


def simplify_interleaves(steps):
    if len(steps) == 0:
        return steps
    first = steps[0]
    opposite = {'u': 'd', 'd': 'u', 'l': 'r', 'r': 'l', 'f': 'b', 'b': 'f'}
    other = opposite[first]
    count = 1
    while count < len(steps) and steps[count] in [first, other]:
        count += 1
    slice = steps[:count]
    return sorted(slice) + simplify_interleaves(steps[count:])

def simplify(steps):
    rep = simplify_repeats(steps)
    if rep != steps:
        return rep
    inter = simplify_interleaves(steps)
    if inter != steps:
        return inter
    return steps


def pretty_steps(steps):
    result = []
    for step in steps:
        for c in step:
            if c in "ruf":
                result.append(c)
            else:
                for _ in range(3):
                    result.append(c)
    while True:
        simple = simplify(result)
        if simple == result:
            return " ".join(nice(result))
        result = simple


def solve6(state):
    three = candidates(state.cuda(), 3)
    three_hashes = {hash_state(s.cpu()): s for s in three}
    first3 = candidates(_SOLVED_STATE, 3)
    first3_hashes = {hash_state(s.cpu()): s for s in first3}
    meet = set(three_hashes.keys()) & set(first3_hashes.keys())
    if len(meet) == 0:
        return []
    meet = list(meet)[0]
    return find_moves(state.cpu(), first3_hashes[meet].cpu(), 3) + find_moves(
        first3_hashes[meet].cpu(), _SOLVED_STATE.cpu(), 3
    )


_HIDDEN = 512
_DROPOUT = 0.25

_CLASSIFIER_DEPTH = 8
_LOW_CLASS = 8
_HIGH_CLASS = 18

simple_classifier_components = (
    [nn.Linear(288, _HIDDEN), nn.ReLU(), nn.Dropout(_DROPOUT)]
    + [nn.Linear(_HIDDEN, _HIDDEN), nn.ReLU(), nn.Dropout(_DROPOUT)] * _CLASSIFIER_DEPTH
    + [nn.Linear(_HIDDEN, 1)]
)

simple_classifier = nn.Sequential(*simple_classifier_components).cuda()

_TRAIN_MODEL = False

if not _TRAIN_MODEL:
    simple_classifier.load_state_dict(torch.load("classifier00005.pt"))
    state = _SOLVED_STATE
    task = []
    for _ in range(35):
        move = random.choice([u, d, l, r, f, b])
        state = move(state)
        task.append(move.__name__)
    print(task)
    print(pretty_steps(task))
    beam = torch.stack([state]).cuda()
    beam_size = 20
    iter = 0
    beam_depth = 3
    cpu_beams = [beam.cpu()]
    comes_from = [None]
    while True:
        print(f"Iteration {iter}")
        iter += 1
        next_states = []
        next_scores = []
        scores_come_from = []
        for i, state in enumerate(beam):
            print(".", end="")
            news = candidates(state, depth=beam_depth)
            for j, new in enumerate(news):
                if is_solved(new.cpu()):
                    print("Solved")
                    show_state(new)
                    print(task)
                    beam_idx = i
                    curr_state = new.cpu()
                    solve = []
                    s6 = solve6(new.cpu())
                    while beam_idx is not None:
                        iter -= 1
                        prev_state = cpu_beams[iter][beam_idx]
                        beam_idx = (
                            comes_from[iter][beam_idx].item() if iter > 0 else None
                        )
                        last_moves = find_moves(prev_state, curr_state, beam_depth)
                        print(last_moves)
                        solve = last_moves + solve
                        show_state(prev_state)
                        curr_state = prev_state
                    print(pretty_steps(task))
                    print("-" * 40)
                    print(pretty_steps(solve))
                    print(s6)
                    print(pretty_steps(s6))
                    result = pretty_steps(solve + s6)
                    print(result)
                    print(len(result.split(" ")))
                    exit()
            scores = simple_classifier(news.float())
            # print(torch.min(scores), torch.max(scores), torch.mean(scores))
            min_ind = torch.topk(scores.flatten(), beam_size, largest=False).indices
            next_states.append(news[min_ind])
            next_scores.append(scores[min_ind])
            scores_come_from.append(torch.tensor([i] * min_ind.size(0)).cuda())
        beam = torch.cat(next_states)
        beam_scores = torch.cat(next_scores)
        min_ind = torch.topk(beam_scores.flatten(), beam_size, largest=False).indices
        beam_scores = beam_scores[min_ind]
        beam = beam[min_ind]
        origins = torch.cat(scores_come_from)[min_ind]
        comes_from.append(origins)
        cpu_beams.append(beam.cpu())
        print(
            torch.min(beam_scores).item(),
            torch.mean(beam_scores).item(),
            torch.max(beam_scores).item(),
        )
    exit()

loss_classifier = nn.MSELoss()

optimizer = torch.optim.Adam(simple_classifier.parameters(), lr=0.0001)


def sample(tensor, n):
    return tensor[torch.randperm(tensor.size(0))[:n]]


train_sample_size = 300000

low = sample(torch.cat(data[:_LOW_CLASS]), train_sample_size)
high = sample(torch.cat(data[_HIGH_CLASS:]), train_sample_size)
middle = [sample(data[i], train_sample_size) for i in range(_LOW_CLASS, _HIGH_CLASS)]

inputs = torch.cat([low, high] + middle)
output_labels = torch.cat(
    [
        torch.tensor([_LOW_CLASS - 1]).expand(len(low), 1),
        torch.tensor([_HIGH_CLASS]).expand(len(high), 1),
    ]
    + [torch.tensor([_LOW_CLASS + i]).expand(len(m), 1) for i, m in enumerate(middle)]
)

perm = torch.randperm(inputs.size(0))
shuffled_inputs = inputs[perm].float()
shuffled_output_labels = output_labels[perm].float()

num_examples = shuffled_inputs.size(0)
train_size = num_examples // 10 * 9
train_input = shuffled_inputs[:train_size].cuda()
train_output = shuffled_output_labels[:train_size].cuda()
test_input = shuffled_inputs[train_size:].cuda()
test_output = shuffled_output_labels[train_size:].cuda()


# train_size = sample_size
# test_size = sample_size // 10

# small_data = torch.cat(data[_LOWLOW:_LOW])
# small_data = small_data[torch.randperm(small_data.size(0))]
# small_train = small_data[:train_size]
# small_test = small_data[train_size : train_size + test_size]

# print(small_data.shape, small_train.shape, small_test.shape)

# large_data = torch.cat(data[_HIGH:_HIGHHIGH])
# large_data = large_data[torch.randperm(large_data.size(0))]
# large_train = large_data[:train_size]
# large_test = large_data[train_size : train_size + test_size]

# print(large_data.shape, large_train.shape, large_test.shape)

# train_input = torch.cat([small_train, large_train])
# train_output = (
#     torch.cat([torch.ones(train_size), torch.zeros(train_size)]).unsqueeze(1).cuda()
# )
# # Shuffle the training data
# train_perm = torch.randperm(train_input.size(0))
# train_input = train_input[train_perm].float()
# train_output = train_output[train_perm].float()

# test_input = torch.cat([small_test, large_test])
# test_output = (
#     torch.cat([torch.ones(test_size), torch.zeros(test_size)]).unsqueeze(1).cuda()
# )
# # Shuffle the test data
# test_perm = torch.randperm(test_input.size(0))
# test_input = test_input[test_perm].float()
# test_output = test_output[test_perm].float()

_BATCH_SIZE = 128
for epoch in range(10):
    print(f"Epoch {epoch}")
    for i in range(0, train_size, _BATCH_SIZE):
        optimizer.zero_grad()
        output = simple_classifier(train_input[i : i + _BATCH_SIZE])
        loss = loss_classifier(output, train_output[i : i + _BATCH_SIZE])
        loss.backward()
        optimizer.step()
    if epoch % 1 == 0:
        with torch.no_grad():
            output = simple_classifier(test_input)
            loss = loss_classifier(output, test_output)
            print(epoch, loss.item())
        with torch.no_grad():
            output = simple_classifier(train_input[:10000])
            loss = loss_classifier(output, train_output[:10000])
            print(epoch, loss.item())
    if epoch % 5 == 0:
        torch.save(simple_classifier.state_dict(), f"classifier{epoch:05}.pt")
