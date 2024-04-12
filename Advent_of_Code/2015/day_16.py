import itertools
from collections import Counter

with open('input') as f:
    content = f.read().rstrip()
    lines = [line for line in content.split('\n')]

# strings as variables to support dirty eval hack?!

children = "children"
cats = "cats"
samoyeds = "samoyeds"
pomeranians = "pomeranians"
akitas = "akitas"
vizslas = "vizslas"
goldfish = "goldfish"
trees = "trees"
cars = "cars"
perfumes = "perfumes"


def parse_roster(lines):
    candidates = []
    for line in lines:
        _, clues = line.split(': ', 1)
        candidates.append(eval("{{ {} }}".format(clues)))
    return candidates


candidates = parse_roster(lines)
target = {
    children: 3,
    cats: 7,
    samoyeds: 2,
    pomeranians: 3,
    akitas: 0,
    vizslas: 0,
    goldfish: 5,
    trees: 3,
    cars: 2,
    perfumes: 1,
}


def the_first_star():
    hits = []
    for i, candidate in enumerate(candidates):
        if all(target[key] == value for key, value in candidate.items()):
            hits.append(i + 1)
    return hits


def the_second_star():
    hits = []
    for i, candidate in enumerate(candidates):
        if all(
            candidate.get(key) is None or target[key] == candidate[key]
            for key in [children, samoyeds, akitas, vizslas, cars, perfumes]
        ):
            if all(
                candidate.get(key) is None or candidate[key] > target[key]
                for key in [cats, trees]
            ):
                if all(
                    candidate.get(key) is None or candidate[key] < target[key]
                    for key in [pomeranians, goldfish]
                ):
                    hits.append(i + 1)
    return hits


if __name__ == "__main__":
    print(parse_roster(lines))
    print(the_first_star())
    print(the_second_star())
