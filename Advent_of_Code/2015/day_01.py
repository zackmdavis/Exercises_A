import itertools
from collections import Counter

with open('input') as f:
    in_content = f.read().rstrip()


def the_first_star():
    floor = 0
    for char in in_content:
        match char:
            case '(':
                floor += 1
            case ')':
                floor -= 1
    return floor


def the_second_star():
    floor = 0
    for i, char in enumerate(in_content):
        match char:
            case '(':
                floor += 1
            case ')':
                floor -= 1
        if floor < 0:
            return i+1



if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
