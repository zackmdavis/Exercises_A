with open("input.txt") as f:
    lines = f.read().split('\n')

pipes = {}

for line in lines:
    split = line.split('<->')
    print(split)
    try:
        left, right = split
    except:
        break
    pipes[int(left)] = [int(c) for c in right.split(', ')]

print(pipes)


def search_from(x):
    reachable = set()
    visited = set()
    to_visit = [x]

    while to_visit:
        next_to_visit = to_visit.pop()
        reachable.add(next_to_visit)
        visited.add(next_to_visit)
        for output in pipes[next_to_visit]:
            if output not in visited:
                to_visit.append(output)
            reachable.add(output)
    return reachable


def the_second_star():
    groups = 1
    unreached = {i for i in range(2000)}
    unreached = unreached - search_from(0)
    while unreached:
        new_seed = list(unreached).pop()
        newly_reached = search_from(new_seed)
        unreached = unreached - newly_reached
        groups += 1
        print(groups)
