import itertools
from collections import Counter

with open('data.txt') as f:
    in_content = f.read()
    in_lines = in_content.split('\n')


def the_first_star():
    counts = []
    group = set()
    for line in in_lines:
        group = group.union(set(line))
        if not line:
            counts.append(len(group))
            group = set()
    return sum(counts)


def inter(lines):
    counts = []
    group = set()
    for line in lines:
        if not group:
            group = set(line)
        if not line:
            counts.append(len(group))
            print(group)
            group = set()
        group = group.intersection(set(line))
    return sum(counts)


def the_second_star():
    # Not sure what I'm doing wrong on part 2?!
    return inter(in_lines)


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
#     print(inter(['abc',
# '',
# 'a',
# 'b',
# 'c',
# '',
# 'ab',
# 'ac',
# '',
# 'a',
# 'a',
# 'a',
# 'a',
# '',
# 'b']))
