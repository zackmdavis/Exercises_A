import re
import itertools
from collections import Counter

with open('data.txt') as f:
    in_content = f.read()
    in_lines = [l for l in in_content.split('\n') if l]


def expand_rule(outer_bag_color, ruleset, already_expanded):
    contained_colors = set(ruleset[outer_bag_color].keys())
    for contained_color in contained_colors:
        subcontained_colors = expand_rule(contained_color, ruleset, already_expanded.union({outer_bag_color}))
        contained_colors = contained_colors.union(subcontained_colors)
    return contained_colors


def expand_rule_counting(outer_bag_color, ruleset):
    contained = ruleset[outer_bag_color]
    if not contained:
        print("recursion leaf on", outer_bag_color)
        return 0
    else:
        counter = 0
        for color, number in contained.items():
            counter += number
            counter += number*expand_rule_counting(color, ruleset)
        return counter


def parse_rules():
    # rules = []
    rules = {}
    for line in in_lines:
        bag_spec, content_spec = line.split(" contain ")
        bag_color = bag_spec[:-5]
        content_bag_specs = content_spec.strip(".").split(", ")
        contents = {}
        for content_bag_spec in content_bag_specs:
            if content_bag_spec[0] == 'n':
                break
            n = int(content_bag_spec[0])
            content_bag_color = content_bag_spec.rstrip("s").strip()[2:-4]
            contents[content_bag_color] = n
        rules[bag_color] = contents
    return rules

def the_first_star():
    outer_bag_possibilities = 0
    ruleset = parse_rules()
    for outer_bag_color in ruleset:
        contained_colors = expand_rule(outer_bag_color, ruleset, set())
        if 'shiny gold' in contained_colors:
            outer_bag_possibilities += 1
    return outer_bag_possibilities


def the_second_star():
    ruleset = parse_rules()
    return expand_rule_counting('shiny gold', ruleset)


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
