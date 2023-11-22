import itertools
from collections import Counter

with open('input') as f:
    in_content = f.read().rstrip()

def the_first_star():
    cursor = (0, 0)
    gift_count = Counter({(0, 0): 1})
    for direction in in_content:
        match direction:
            case '^':
                cursor = (cursor[0], cursor[1] + 1)
                gift_count[cursor] += 1
            case 'v':
                cursor = (cursor[0], cursor[1] - 1)
                gift_count[cursor] += 1
            case '<':
                cursor = (cursor[0] - 1, cursor[1])
                gift_count[cursor] += 1
            case '>':
                cursor = (cursor[0] + 1, cursor[1])
                gift_count[cursor] += 1
    return len(gift_count)

def the_second_star():
    cursor1 = (0, 0)
    cursor2 = (0, 0)
    turn_toggle = True
    gift_count = Counter({(0, 0): 2})
    for direction in in_content:
        match direction:
            case '^':
                if turn_toggle:
                    cursor1 = (cursor1[0], cursor1[1] + 1)
                    gift_count[cursor1] += 1
                else:
                    cursor2 = (cursor2[0], cursor2[1] + 1)
                    gift_count[cursor2] += 1
            case 'v':
                if turn_toggle:
                    cursor1 = (cursor1[0], cursor1[1] - 1)
                    gift_count[cursor1] += 1
                else:
                    cursor2 = (cursor2[0], cursor2[1] - 1)
                    gift_count[cursor2] += 1
            case '<':
                if turn_toggle:
                    cursor1 = (cursor1[0] - 1, cursor1[1])
                    gift_count[cursor1] += 1
                else:
                    cursor2 = (cursor2[0] - 1, cursor2[1])
                    gift_count[cursor2] += 1
            case '>':
                if turn_toggle:
                    cursor1 = (cursor1[0] + 1, cursor1[1])
                    gift_count[cursor1] += 1
                else:
                    cursor2 = (cursor2[0] + 1, cursor2[1])
                    gift_count[cursor2] += 1
        turn_toggle = not turn_toggle
    return len(gift_count)


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
