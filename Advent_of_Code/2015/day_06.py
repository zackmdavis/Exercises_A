import itertools
from collections import Counter
import re

with open('input') as f:
    in_content = f.read().rstrip()
    in_lines = in_content.split('\n')

def parse_command(line):
    a, b = line.split(" through ")
    bottomright = tuple(int(x) for x in b.split(","))
    m = re.match("(toggle|turn on|turn off) (\d+),(\d+)", a)
    topleft = (int(m.group(2)), int(m.group(3)))
    action_spec = m.group(1)
    return action_spec, topleft, bottomright


def command(grid, action_spec, topleft, bottomright):
    row_start, col_start = topleft
    row_end, col_end = bottomright
    match action_spec:
        case 'turn on':
            action = lambda x: True
        case 'turn off':
            action = lambda x: False
        case 'toggle':
            action = lambda x: not x

    for i in range(row_start, row_end+1):
        for j in range(col_start, col_end+1):
            grid[i][j] = action(grid[i][j])


def the_first_star():
    grid = [[False for _ in range(1000)] for _ in range(1000)]
    for line in in_lines:
        action_spec, topleft, bottomright = parse_command(line)
        command(grid, action_spec, topleft, bottomright)
    count = 0
    for i in range(1000):
        for j in range(1000):
            if grid[i][j]:
                count += 1
    return count


def dimmer_command(grid, action_spec, topleft, bottomright):
    row_start, col_start = topleft
    row_end, col_end = bottomright
    match action_spec:
        case 'turn on':
            action = lambda x: x+1
        case 'turn off':
            action = lambda x: x-1 if x-1 >= 0 else 0
        case 'toggle':
            action = lambda x: x+2

    for i in range(row_start, row_end+1):
        for j in range(col_start, col_end+1):
            grid[i][j] = action(grid[i][j])


def the_second_star():
    grid = [[0 for _ in range(1000)] for _ in range(1000)]
    for line in in_lines:
        action_spec, topleft, bottomright = parse_command(line)
        dimmer_command(grid, action_spec, topleft, bottomright)
    count = 0
    for i in range(1000):
        for j in range(1000):
            count += grid[i][j]
    return count



if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
