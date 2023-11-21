import itertools
from collections import Counter

with open('input') as f:
    in_content = f.read()
    in_lines = in_content.split('\n')
    grid = [[int(c) for c in line] for line in in_lines]


def neighborhood(i, j):
    neighbors = []
    for patch in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
        neighbors.append([i+patch[0], j+patch[1]])
    return neighbors

def lookup(grid, address):
    return grid[address[0]][address[1]]

def the_first_star():
    lowpoint_risk_levels = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            neighbor_values = [lookup(grid, address) for address in neighborhood(i, j)
                               if 0 <= address[0] < len(grid)
                               if 0 <= address[1] < len(grid[0])]
            if lookup(grid, [i, j]) < min(neighbor_values):
                lowpoint_risk_levels.append(lookup(grid, [i, j]) + 1)
    return sum(lowpoint_risk_levels)


def adjacent(addr1, addr2):
    return (addr1[0] - addr2[0], addr1[1] - addr2[1]) in [(0, 1), (0, -1), (1, 0), (-1, 0)]


# XXX: wrong
def the_second_star():
    basins = [set()]
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 9:
                continue
            place_at = None
            for basin in basins:
                for basin_element in basin:
                    if adjacent(basin_element, (i, j)):
                        place_at = basin
            if place_at is not None:
                place_at.add((i, j))
            else:
                basins.append({(i, j)})

    print("found {} basins".format(len(basins)))
    basin_sizes = sorted([len(basin) for basin in basins])
    return basin_sizes[-1] * basin_sizes[-2] * basin_sizes[-3]


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
