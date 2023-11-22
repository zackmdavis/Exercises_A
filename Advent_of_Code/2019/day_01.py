import itertools
from collections import Counter

from math import floor

with open('input') as f:
    in_content = f.read()
    in_lines = in_content.split('\n')

module_masses = [int(n) for n in in_lines if n]

def the_first_star():
    total = 0
    for mass in module_masses:
        total += floor(mass / 3) - 2
    return total


def the_second_star():
    total = 0
    for mass in module_masses:
        fuel_costs = [floor(mass / 3) - 2]
        while True:
            next_cost = floor(fuel_costs[-1] / 3) - 2
            if next_cost > 0:
                fuel_costs.append(next_cost)
            else:
                break
        total += sum(fuel_costs)
    return total



if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
