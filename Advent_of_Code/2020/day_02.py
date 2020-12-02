import itertools
from collections import Counter

with open('data.txt') as f:
    in_content = f.read()
    in_lines = in_content.split('\n')


def the_first_star():
    valid = 0
    for line in in_lines:
        if not line:
            break
        first, sec = line.split(":")
        rep_spec, letter = first.split(" ")
        password = sec.strip()
        low, high = rep_spec.split('-')
        low = int(low)
        high = int(high)
        count = Counter(password)
        if low <= count[letter] <= high:
            valid += 1
    return valid

def the_second_star():
    valid = 0
    for line in in_lines:
        if not line:
            break
        first, sec = line.split(":")
        rep_spec, letter = first.split(" ")
        password = sec.strip()
        low, high = rep_spec.split('-')
        low = int(low)
        high = int(high)
        first = password[low-1] == letter
        last = password[high-1] == letter
        if first ^ last:
            valid += 1
    return valid


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
