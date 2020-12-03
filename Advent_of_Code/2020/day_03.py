import itertools
from collections import Counter

with open('data.txt') as f:
    in_content = f.read()
    in_lines = in_content.split('\n')


def grid():
    our_map = []
    for line in in_lines:
        if line:
            our_map.append([c for c in line])
    print(our_map)
    return our_map


def the_first_star():
    our_map = grid()
    pointer = [0, 0]
    trees = 0
    for line in our_map:
        try:
            if our_map[pointer[0]][pointer[1]] == '#':
                trees += 1
        except IndexError:
            print(pointer)
        pointer[0] += 1
        pointer[1] += 3
        pointer[1] = pointer[1] % len(our_map[0])
    return trees


def slope_finder(cols, rows):
    our_map = grid()
    pointer = [0, 0]
    trees = 0
    for line in our_map:
        try:
            if our_map[pointer[0]][pointer[1]] == '#':
                trees += 1
        except IndexError:
            break
        pointer[0] += rows
        pointer[1] += cols
        pointer[1] = pointer[1] % len(our_map[0])
    return trees


def the_second_star():
    result = 1
    for (c, r) in [(1, 1), (3, 1), (5, 1), (7, 1), (1, 2)]:
        result *= slope_finder(c, r)
    return result


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
