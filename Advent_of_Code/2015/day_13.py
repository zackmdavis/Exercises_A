import itertools
from collections import Counter

import re

constraints = []
constraint_regex = re.compile(r"(\w+) would (gain|lose) (\d+) happiness units by sitting next to (\w+).")

with open('input') as f:
# with open('test_input') as f:
    in_content = f.read().rstrip()
    in_lines = in_content.split('\n')

for line in in_lines:
    m = constraint_regex.match(line)
    match m.group(2):
        case 'gain':
            sign = 1
        case 'lose':
            sign = -1
    constraints.append((m.group(1), m.group(4), sign * int(m.group(3))))


def score_seating(arrangement, debug=False):
    utils = 0
    for i in range(len(arrangement)):
        guest1 = arrangement[i]
        guest2 = arrangement[(i+1) % len(arrangement)]
        for constraint in constraints:
            if (guest1 == constraint[0] and guest2 == constraint[1]) or (guest1 == constraint[1] and guest2 == constraint[0]):
                utils += constraint[2]
                if debug:
                    print(constraint)
    return utils


def the_first_star():
    best_seating = None
    max_utility = float("-inf")
    for arrangement in itertools.permutations(["Alice", "Bob", "Carol", "David", "Eric", "Frank", "George", "Mallory"]):
        utility = score_seating(arrangement)
        if utility > max_utility:
            best_seating = arrangement
            max_utility = utility
    print(best_seating)
    score_seating(best_seating, debug=True)
    return max_utility


def the_second_star():
    best_seating = None
    max_utility = float("-inf")
    for arrangement in itertools.permutations(["Alice", "Bob", "Carol", "David", "Eric", "Frank", "George", "Mallory", "You"]):
        utility = score_seating(arrangement)
        if utility > max_utility:
            best_seating = arrangement
            max_utility = utility
    print(best_seating)
    score_seating(best_seating, debug=True)
    return max_utility


if __name__ == "__main__":
    print(constraints)
    print(the_first_star())
    print(the_second_star())
