import itertools
from collections import Counter

with open('data.txt') as f:
    in_content = f.read()
    in_lines = [l for l in in_content.split('\n') if l]


def the_first_star():
    orders = in_lines
    current_direction = 0
    location = [0, 0]
    for order in orders:
        cmd = order[0]
        val = int(order[1:])
        # Action N means to move north by the given value.
        if cmd == 'N':
            location[1] += val
        elif cmd == 'S':
            location[1] -= val
        elif cmd == 'W':
            location[0] -= val
        elif cmd == 'E':
            location[0] += val
        elif cmd == 'L':
            current_direction = (current_direction + val) % 360
        elif cmd == 'R':
            current_direction = (current_direction - val) % 360
        elif cmd == 'F':
            if current_direction == 0:
                location[0] += val
            elif current_direction == 90:
                location[1] += val
            elif current_direction == 180:
                location[0] -= val
            elif current_direction == 270:
                location[1] -= val
        elif cmd == 'B':
            if current_direction == 0:
                location[0] -= val
            elif current_direction == 90:
                location[1] -= val
            elif current_direction == 180:
                location[0] += val
            elif current_direction == 270:
                location[1] += val
    return location


def the_second_star():
    # not sure where the bug is ...
    orders = in_lines
    current_direction = 0
    location = [0, 0]
    waypoint = [10, 1]
    for order in orders:
        cmd = order[0]
        val = int(order[1:])
        # Action N means to move north by the given value.
        if cmd == 'N':
            waypoint[1] += val
        elif cmd == 'S':
            waypoint[1] -= val
        elif cmd == 'W':
            waypoint[0] -= val
        elif cmd == 'E':
            waypoint[0] += val
        elif cmd == 'L':
            if val == 0:
                ...
            elif val == 90:
                waypoint = [-waypoint[1], waypoint[0]]
            elif val == 180:
                waypoint = [-waypoint[0], -waypoint[1]]
            elif val == 270:
                waypoint = [waypoint[1], -waypoint[0]]
        elif cmd == 'R':
            if val == 0:
                ...
            elif val == 90:
                waypoint = [waypoint[1], -waypoint[0]]
            elif val == 180:
                waypoint = [-waypoint[0], -waypoint[1]]
            elif val == 270:
                waypoint = [-waypoint[1], waypoint[0]]
        elif cmd == 'F':
            location = [location[0] + waypoint[0], location[1] + waypoint[1]]
        elif cmd == 'B':
            location = [location[0] - waypoint[0], location[1] - waypoint[1]]
    return location


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
