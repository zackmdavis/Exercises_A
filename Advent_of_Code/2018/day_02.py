import itertools
from collections import Counter

with open('data.txt') as f:
    in_content = f.read()
    in_lines = in_content.split('\n')

def exactly_two(s):
    c = Counter(s)
    for key, count in c.items():
        if count == 2:
            return True
    return False

def exactly_three(s):
    c = Counter(s)
    for key, count in c.items():
        if count == 3:
            return True
    return False

def one_diff(s, t):
    d = 0
    p = ('', '')
    for c1, c2 in zip(s, t):
        if c1 != c2:
            d += 1
            p = (c1, c2)
    if d == 1:
        return True, p
    return False


def the_first_star():
    tw = 0
    th = 0
    for s in in_lines:
        if exactly_two(s):
            tw += 1
        if exactly_three(s):
            th += 1
    return tw * th


def the_second_star():
    for s in in_lines:
        for t in in_lines:
            d = one_diff(s, t)
            if d:
                print(d)
                return s, t


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
