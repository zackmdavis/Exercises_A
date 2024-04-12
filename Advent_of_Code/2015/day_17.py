import itertools
from collections import Counter

container_sizes = [43, 3, 4, 10, 21, 44, 4, 6, 47, 41, 34, 17, 17, 44, 36, 31, 46, 9, 27, 38]

def the_first_star():
    hits = 0
    for no_containers in range(len(container_sizes)):
        for chosen_containers in itertools.combinations(container_sizes, no_containers):
            if sum(chosen_containers) == 150:
                hits += 1
    return hits

def the_second_star():
    for no_containers in range(len(container_sizes)):
        hits = 0
        for chosen_containers in itertools.combinations(container_sizes, no_containers):
            if sum(chosen_containers) == 150:
                hits += 1
        if hits:
            return hits

    return hits



if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
