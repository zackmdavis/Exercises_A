import itertools
from collections import Counter

with open('data.txt') as f:
    in_content = f.read()
    in_lines = in_content.split('\n')

def parse_claim(claim_line):
    if not claim_line:
        return None
    n, claim_line = claim_line.split('@')
    start, size = claim_line.split(': ')
    start_x, start_y = map(int, start.strip().split(','))
    size_x, size_y = map(int, size.split('x'))
    return [int(n.strip()[1:]), (start_x, start_y), (size_x, size_y)]

claims = [parse_claim(l) for l in in_lines if parse_claim(l) is not None]

def the_first_star():
    grid = [[0 for i in range(1200)] for i in range(1200)]
    for claim in claims:
        _, start, size = claim
        start_x, start_y = start
        size_x, size_y = size
        for x in range(size_x):
            for y in range(size_y):
                grid[start_y + y][start_x + x] += 1

    dup = 0
    for i in range(1200):
        for j in range(1200):
            if grid[i][j] > 1:
                dup += 1
    return dup


def the_second_star():
    # ... I don't see the bug
    grid = [[set() for i in range(1200)] for i in range(1200)]
    for claim in claims:
        n, start, size = claim
        start_x, start_y = start
        size_x, size_y = size
        for x in range(size_x):
            for y in range(size_y):
                grid[start_y + y][start_x + x].add(n)

    all_claims = set(range(1, 1410))
    for i in range(1200):
        print(len(all_claims))
        for j in range(1200):
            spot = grid[start_y + y][start_x + x]
            # print(spot)
            if len(spot) > 1:
                for entry in spot:
                    try:
                        all_claims.remove(entry)
                    except KeyError:
                        pass
    return all_claims


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
