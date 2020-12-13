import itertools
from collections import Counter

with open('data.txt') as f:
    in_content = f.read()
    in_lines = [l for l in in_content.split('\n') if l]


def the_first_star():
    start_stamp = int(in_lines[0])
    buses = [int(n) for n in in_lines[1].split(',') if n != 'x']
    min_wait = 100000000
    min_bus = None
    for bus in buses:
        wait = bus - start_stamp % bus
        if wait < min_wait:
            min_wait = wait
            min_bus = bus

    return min_bus * min_wait


def the_second_star():
    # don't care enough to finish this one right now
    buses = [(i, int(n)) for i, n in enumerate(in_lines[1].split(',')) if n != 'x']
    constraints = [
        lambda x: x%n == i
        for i, n in buses
    ]
    c = 19
    for _ in range(100000000000):
        if all(constraint(c) for constraint in constraints):
            return c
        c += 19

if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
