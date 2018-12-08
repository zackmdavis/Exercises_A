import itertools
from collections import Counter

with open('data.txt') as f:
    in_content = f.read()
    in_lines = [int(n) for n in in_content.split(' ')]


i = 0
entries = in_lines

total_weight = 0


class Node:
    def __init__(self):
        self.children = []
        self.meta = []


def process_node():
    global i
    global total_weight
    here = Node()
    n_children = entries[i]
    i += 1
    n_meta = entries[i]
    i += 1
    for child_no in range(n_children):
        here.children.append(process_node())
    for meta_no in range(n_meta):
        here.meta.append(entries[i])
        total_weight += entries[i]
        i += 1
    return here


def valuation(node):
    if not node.children:
        return sum(node.meta)
    else:
        sub = 0
        for m in node.meta:
            try:
                n = node.children[m-1]
                sub += valuation(n)
            except IndexError:
                continue
        return sub


def stars():
    root = process_node()
    print(total_weight)
    return valuation(root)


if __name__ == "__main__":
    print(stars())
