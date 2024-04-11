import itertools
from collections import Counter

with open('test_input.txt') as f:
    content = f.read()
    lines = [line for line in content.split('\n') if line]
    grid = [list(line) for line in lines]


def insert_row(grid, position):
    grid.insert(position, ['.'] * len(grid[0]))


def insert_col(grid, position):
    for row in grid:
        row.insert(position, '.')


def expand(grid):
    empty_rows = []
    empty_cols = []

    for row in range(len(grid)):
        empty = True
        for col in range(len(grid[0])):
            if grid[row][col] == '#':
                empty = False
                break
        if empty:
            empty_rows.append(row)

    for col in range(len(grid[0])):
        empty = True
        for row in range(len(grid)):
            if grid[row][col] == '#':
                empty = False
                break
        if empty:
            empty_cols.append(col)

    # need to offset the indices, which will be different after a row/col has
    # been inserted, probably?

    for offset, empty_row in enumerate(empty_rows):
        insert_row(grid, row + offset)

    for offset, empty_col in enumerate(empty_cols):
        insert_col(grid, col + offset)

    return grid


def locate_galaxies(grid):
    galaxies = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '#':
                galaxies.append((i, j))
    return galaxies


def taxicab_distance(address1, address2):
    return abs(address1[0] - address2[0]) + abs(address1[1] - address2[1])


def the_first_star():
    starchart = expand(grid)
    galaxies = locate_galaxies(starchart)
    total = 0
    for i in range(1, len(galaxies)):
        for j in range(i):
            total += taxicab_distance(galaxies[i], galaxies[j])
    return total


def the_second_star():
    ...


# XXX—this is wrong. Where is the bug?!

if __name__ == "__main__":
    g = expand(grid)
    for row in g:
        print(''.join(row))
    print(the_first_star())
    print(the_second_star())
