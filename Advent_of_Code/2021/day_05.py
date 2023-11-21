with open("test_input") as f:
    content = f.read()
    lines = content.split('\n')

## XXX: what's wrong with me?! I'm trying to practice "quickly solve a simple
## puzzle", and then I flub it, and then debugging seems daunting.

vents = []
for line in lines:
    raw_begin, raw_end = line.split(" -> ")
    x_begin, y_begin = [int(x) for x in raw_begin.split(",")]
    x_end, y_end = [int(x) for x in raw_end.split(",")]
    vents.append(((x_begin, y_begin), (x_end, y_end)))


def populate_grid():
    grid = [[0 for _ in range(10)] for _ in range(10)]
    for vent in vents:
        (x_begin, y_end), (x_end, y_end) = vent
        if x_begin == x_end:
            # vertical
            length = abs(y_end - y_begin) + 1
            start = min(y_begin, y_end)
            for i in range(length):
                grid[x_begin][start+i] += 1
        if y_begin == y_end:
            # horizontal
            length = abs(x_end - x_begin) + 1
            start = min(x_begin, x_end)
            for i in range(length):
                grid[start+i][y_begin] += 1
    return grid


def the_first_star():
    grid = populate_grid()
    for line in grid:
        print(line)
    total = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] > 1:
                total += grid[i][j]
    return total


if __name__ == "__main__":
    print(the_first_star())
