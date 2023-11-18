import itertools
from collections import defaultdict


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
        if grid[i][j] == 'S':
            start = (i, j)
        if grid[i][j] == 'E':
            end = (i, j)


def manhattan_distance(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])

def height(indicator):
    if indicator == 'S':
        return ord('a')
    elif indicator == 'E':
        return ord('z')
    else:
        return ord(indicator)

def lookup(grid, address):
    try:
        return grid[address[0]][address[1]]
    except IndexError:
        return '}'  # out of bounds, might as well be a tall wall


def reconstruct_path(came_from, destination):
    total_path = []
    current = destination
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return list(reversed(total_path))


def the_first_star():
    # https://en.wikipedia.org/wiki/A*_search_algorithm
    open_set = {start}

    came_from = {}

    g = defaultdict(lambda: float("inf"))
    g[start] = 0

    f = defaultdict(lambda: float("inf"))
    f[start] = manhattan_distance(start, end)

    while open_set:
        frontier = sorted([(manhattan_distance(n, end), n) for n in open_set])
        current = frontier[0][1]
        if current == end:
            return reconstruct_path(came_from, end)
        open_set.remove(current)
        for diff in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            scouting = (current[0] + diff[0], current[1] + diff[1])
            if height(lookup(grid, scouting)) > height(lookup(grid, current)) + 1:
                continue

            scouting_g_score = g[current] + 1
            if scouting_g_score < g[scouting]:
                came_from[scouting] = current
                g[scouting] = scouting_g_score
                f[scouting] = manhattan_distance(scouting, end)
                if scouting not in open_set:
                    open_set.add(scouting)


def the_second_star():
    ...


if __name__ == "__main__":
    print('\n'.join(''.join(c for c in line) for line in grid))
    print("\n\n\n")
    path = the_first_star()
    for step in path:
        grid[step[0]][step[1]] = '•'
    print('\n'.join(''.join(c for c in line) for line in grid))
    print(len(path))  # XXX: solution checker thinks 482 is too high?!
    print(the_second_star())
