import itertools
from collections import Counter

with open('input.txt') as f:
    content = f.read()
    grid = [list(line) for line in content.split('\n') if line]

def decompile_grid(grid):
    number_locations = []
    symbol_locations = []

    number_assembly = []
    number_location_assembly = []

    for i, line in enumerate(grid):
        for j, char in enumerate(line):
            # If it's a digit, either start building a number or extend the number being built
            if char.isdigit():
                number_assembly.append(char)
                number_location_assembly.append((i, j))
            else:
                # If it's not a digit, and we've already been building a number, close it out
                if number_assembly:
                    number_locations.append((number_location_assembly, int(''.join(number_assembly))))
                    number_assembly = []
                    number_location_assembly = []

                # But if it's a symbol, record that
                if char != '.':
                    symbol_locations.append(((i, j), char))
    return number_locations, symbol_locations


def in_neighborhood(address1, address2):
    return abs(address1[0] - address2[0]) <= 1 and abs(address1[1] - address2[1]) <= 1

def the_first_star():
    total = 0
    number_locations, symbol_locations = decompile_grid(grid)
    for locations, number in number_locations:
        include_number = False
        for location in locations:
            for slocation, symbol in symbol_locations:
                if in_neighborhood(location, slocation):
                    include_number = True
                    break
        if include_number:
            total += number
    return total


def the_second_star():
    total = 0
    number_locations, symbol_locations = decompile_grid(grid)
    for symbol_location, symbol in symbol_locations:
        if symbol == "*":
            adjacent_numbers = []
            for this_number_locations, number in number_locations:
                for this_number_location in this_number_locations:
                    if in_neighborhood(this_number_location, symbol_location):
                        adjacent_numbers.append(number)
                        break
            if len(adjacent_numbers) == 2:
                total += adjacent_numbers[0] * adjacent_numbers[1]
    return total


if __name__ == "__main__":
    print([s[1] for s in decompile_grid(grid)[0]])
    print(the_first_star())
    print(the_second_star())
