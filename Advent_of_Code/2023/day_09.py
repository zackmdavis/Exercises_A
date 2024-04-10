import itertools
from collections import Counter

with open('input.txt') as f:
    content = f.read()
    lines = [line for line in content.split('\n') if line]

sequences = [[int(n) for n in line.split()] for line in lines]


def take_differences(sequence):
    return [sequence[i+1] - sequence[i] for i in range(len(sequence) - 1)]


def predict_next(sequence):
    derivatives = [sequence]
    while not all(s == 0 for s in derivatives[-1]):
        derivatives.append(take_differences(derivatives[-1]))

    # We don't actually need the terminating all-zeros sequence
    derivatives = derivatives[:-1]

    n = len(derivatives)
    for k in range(n):
        delta = 0 if k == 0 else derivatives[n-k][-1]
        derivatives[n-1-k].append(derivatives[n-1-k][-1] + delta)

    return derivatives[0][-1]


# Same thing, with some negatives and taking from 0 instead of -1
def predict_prior(sequence):
    derivatives = [sequence]

    while not all(s == 0 for s in derivatives[-1]):
        derivatives.append(take_differences(derivatives[-1]))

    derivatives = derivatives[:-1]

    n = len(derivatives)
    for k in range(n):
        delta = 0 if k == 0 else derivatives[n-k][0]
        derivatives[n-1-k].insert(0, derivatives[n-1-k][0] - delta)

    return derivatives[0][0]



def the_first_star():
    return sum(predict_next(s) for s in sequences)

def the_second_star():
    return sum(predict_prior(s) for s in sequences)


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
