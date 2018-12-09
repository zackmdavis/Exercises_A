import itertools
from collections import Counter


def the_first_star(players, top_marble):
    scores = [0 for _ in range(players)]
    i = 0
    n = 1
    ring = [0]
    while n <= top_marble:
        print(ring)
        if n % 23 == 0:
            scores[(n-1) % len(scores)] += n
            scores[(n-1) % len(scores)] += ring.pop((i - 7) % len(ring))
            i = ((i - 7) % len(ring)) + 1
        else:
            ring.insert(((i + 2) % len(ring)) + 1, n)
            i = (i + 2) % len(ring)
        n += 1

    max_score = 0
    argmax = None
    for arg, score in enumerate(scores):
        if score > max_score:
            max_score = score
    return score


def the_second_star():
    ...


if __name__ == "__main__":
    print(the_first_star(9, 25))
    # print(the_first_star(10, 1618))

    # print(the_first_star(13, 7999))

    # print(the_first_star(424, 71482))
    print(the_second_star())
