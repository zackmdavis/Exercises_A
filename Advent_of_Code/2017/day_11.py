with open("input.txt") as f:
    steps = f.read().strip().split(',')

compass = {'n': [0, 1], 's': [0, -1],
           'ne': [1, 0], 'sw': [-1, 0],
           'se': [1, -1],   'nw': [-1, 1]}

def from_origin(point):
    print("from origin, ", point)
    if point[0] == 0:
        return point[1]
    if point[1] == 0:
        return point[0]

    sign_0 = abs(point[0]) == point[0]
    sign_1 = abs(point[1]) == point[1]
    if sign_0 != sign_1:
        return max(abs(point[0]), abs(point[1]))
    else:
        return abs(point[0]) + abs(point[1])


def follow_directions(directions):
    at = [0, 0]
    max_dist = 0
    last_dist = 0
    for step_spec in directions:
        step = compass[step_spec]
        print("at: {}, stepping {}".format(at, step))
        at = [at[0] + step[0], at[1] + step[1]]
        new_dist = from_origin(at)
        if from_origin(at) > max_dist:
            max_dist = from_origin(at)

        last_dist = new_dist
    print(at)
    print(max_dist)

follow_directions(steps)
