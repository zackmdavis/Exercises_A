import itertools
from collections import Counter

with open('input') as f:
    in_content = f.read().rstrip()
    in_lines = in_content.split('\n')


naughty_substrings = ['ab', 'cd', 'pq', 'xy']
vowels = ['a', 'e', 'i', 'o', 'u']

def has_double(string):
    for i in range(len(string)-1):
        if string[i] == string[i+1]:
            return True
    return False

def has_three_vowels(string):
    return len([c for c in string if c in vowels]) >= 3

def has_naughty_substing(string):
    for n in naughty_substrings:
        if n in string:
            return True
    return False

def is_nice(string):
    if has_three_vowels(string) and has_double(string) and not has_naughty_substing(string):
        return True
    return False


def the_first_star():
    nice_count = 0
    for string in in_lines:
        if is_nice(string):
            nice_count += 1
    return nice_count


def pair_reappears(string):
    for i in range(len(string)-1):
        pair = string[i:i+2]
        for j in range(i+2, len(string)):
            if pair == string[j:j+2]:
                return True
    return False

def double_one_gap(string):
    for i in range(len(string)-2):
        if string[i] == string[i+2]:
            return True
    return False

def is_nice_2(string):
    if pair_reappears(string) and double_one_gap(string):
        return True
    return False

def the_second_star():
    nice_count = 0
    for string in in_lines:
        if is_nice_2(string):
            nice_count += 1
    return nice_count


if __name__ == "__main__":
    print(the_first_star())
    assert is_nice_2("qjhvhtzxzqqjkmpb")
    assert is_nice_2("xxyxx")
    assert not is_nice_2("uurcxstgmygtbstg")
    assert not is_nice_2("ieodomkazucvgmuy")
    print(the_second_star())
