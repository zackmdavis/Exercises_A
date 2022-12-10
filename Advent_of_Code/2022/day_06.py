import itertools
from collections import Counter

with open('input') as f:
    in_content = f.read()
    in_lines = in_content.split('\n')

sequence, *_ = in_lines

def the_first_star():
    for i in range(len(sequence)):
        if len(Counter(sequence[i:i+4])) == 4:
            return i+4


def the_second_star():
    for i in range(len(sequence)):
        if len(Counter(sequence[i:i+14])) == 14:
            return i+14

if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
