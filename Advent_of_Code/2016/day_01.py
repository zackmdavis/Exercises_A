DIRECTIONS = "L1, L3, L5, L3, R1, L4, L5, R1, R3, L5, R1, L3, L2, L3, R2, R2, L3, L3, R1, L2, R1, L3, L2, R4, R2, L5, R4, L5, R4, L2, R3, L2, R4, R1, L5, L4, R1, L2, R3, R1, R2, L4, R1, L2, R3, L2, L3, R5, L192, R4, L5, R4, L1, R4, L4, R2, L5, R45, L2, L5, R4, R5, L3, R5, R77, R2, R5, L5, R1, R4, L4, L4, R2, L4, L1, R191, R1, L1, L2, L2, L4, L3, R1, L3, R1, R5, R3, L1, L4, L2, L3, L1, L1, R5, L4, R1, L3, R1, L2, R1, R4, R5, L4, L2, R4, R5, L1, L2, R3, L4, R2, R2, R3, L2, L3, L5, R3, R1, L4, L3, R4, R2, R2, R2, R1, L4, R4, R1, R2, R1, L2, L2, R4, L1, L2, R3, L3, L5, L4, R4, L3, L1, L5, L3, L5, R5, L5, L4, L2, R1, L2, L4, L2, L4, L1, R4, R4, R5, R1, L4, R2, L4, L2, L4, R2, L4, L1, L2, R1, R4, R3, R2, R2, R5, L1, L2"

COMPASS = [[0, 1], [1, 0], [0, -1], [-1, 0]]

def parse_step(step):
    return (step[0], int(step[1:]))

def parse_directions(directions):
    return [parse_step(step) for step in directions.split(", ")]

def follow_directions(steps):
    stops = set()
    compass_index = 0
    location = [0, 0]
    for turn, move in steps:
        if turn == "L":
            compass_index -= 1
        elif turn == "R":
            compass_index += 1
        compass_index %= 4
        for _ in range(move):
            location = list(map(lambda a, b: a+b, location, COMPASS[compass_index]))
            if tuple(location) in stops:
                return location
            stops.add(tuple(location))
    return location

if __name__ == "__main__":
    print(follow_directions(parse_directions(DIRECTIONS)))
