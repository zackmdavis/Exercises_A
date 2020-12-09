import itertools
from collections import Counter

with open('data.txt') as f:
    in_content = f.read()
    in_lines = in_content.split('\n')
    seq = [int(n) for n in in_lines if n]


def possible_sums_from(n):
    possible_sums = set()
    for i in range(n, n+25):
        for j in range(n, n+25):
            if i != j:
                possible_sums.add(seq[i]+seq[j])
    return possible_sums


def the_first_star():
    for k in range(26, len(seq)):
        possible_sums = possible_sums_from(k-25)
        if seq[k] not in possible_sums:
            return seq[k]

def the_second_star():
    target = 104054607
    for i in range(0, len(seq)):
        for j in range(i, len(seq)):
            span = [seq[k] for k in range(i, j)]
            if sum(span) == target:
                s = sorted(span)
                return s[0] + s[-1]


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
