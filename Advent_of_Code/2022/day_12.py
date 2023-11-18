import itertools
from collections import Counter

with open('input') as f:
    in_content = f.read()

in_lines = in_content.split('\n')
grid = [list(line) for line in in_lines]

start = None
end = None
rows = len(grid)
cols = len(grid[0])
for i in range(rows):
    for j in range(cols):
        print(i, j)
        if grid[i][j] == 'S':
            start = (i, j)
        if grid[i][j] == 'E':
            end = (i, j)


def the_first_star():
    # TODO: this one is going to be an A* search
    # https://en.wikipedia.org/wiki/A*_search_algorithm

def the_second_star():
    ...


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
