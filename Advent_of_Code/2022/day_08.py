import itertools
from collections import Counter

with open('input') as f:
    in_content = f.read()
    in_lines = in_content.split('\n')
    grid = []
    for line in in_lines:
        grid.append([int(c) for c in line])


def select_row(grid, i):
    return grid[i]


def select_col(grid, j):
    return [grid[i][j] for i in range(len(grid))]


def visible(grid, i, j):
    tree = grid[i][j]
    row = select_row(grid, i)
    col = select_col(grid, j)
    row_before = [t < tree for t in row[:j]]
    row_after = [t < tree for t in row[j+1:]]
    col_before = [t < tree for t in col[:i]]
    col_after = [t < tree for t in col[i+1:]]
    return all(row_before) or all(row_after) or all(col_before) or all(col_after)

def the_first_star():
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if visible(grid, i, j):
                count += 1
    return count


import math

def scenic_score(grid, i, j):
    tree = grid[i][j]
    row = select_row(grid, i)
    col = select_col(grid, j)
    row_before = [t < tree for t in row[:j]]
    row_after = [t < tree for t in row[j+1:]]
    col_before = [t < tree for t in col[:i]]
    col_after = [t < tree for t in col[i+1:]]

    directions = [reversed(row_before), reversed(col_before), row_after, col_after]
    subscores = []
    for direction in directions:
        subscore = 0
        for tree in direction:
            subscore += 1
            if not tree:
                break
        subscores.append(subscore)

    product = 1
    for subscore in subscores:
        product *= subscore
    return product



test_grid = [[3,0,3,7,3],
             [2,5,5,1,2],
             [6,5,3,3,2],
             [3,3,5,4,9],
             [3,5,3,9,0]]

def the_second_star():
    spot = (0, 0)
    hi_score = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            score = scenic_score(grid, i, j)
            if score > hi_score:
                spot = (i, j)
                hi_score = score
    print(spot)
    return hi_score


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
