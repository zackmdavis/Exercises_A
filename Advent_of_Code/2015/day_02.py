import itertools
from collections import Counter

with open('input') as f:
    in_content = f.read().rstrip()
    in_lines = in_content.split('\n')

gifts = []
for line in in_lines:
    d1, d2, d3 = [int(d) for d in line.split('x')]
    gifts.append(tuple(sorted([d1, d2, d3])))


def the_first_star():
    total = 0
    for gift in gifts:
        d1, d2, d3 = gift
        base_wrap = 2*d1*d2 + 2*d2*d3 + 2*d1*d3
        slack = d1*d2
        total += base_wrap + slack
    return total

def the_second_star():
    total = 0
    for gift in gifts:
        d1, d2, d3 = gift
        base_ribbon = 2*d1 + 2*d2
        bow = d1 * d2 * d3
        total += base_ribbon + bow
    return total


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
