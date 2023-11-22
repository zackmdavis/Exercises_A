import itertools
from collections import Counter
import json


with open('input') as f:
    in_content = f.read().rstrip()
    blob = json.loads(in_content)


def number_search(entity):
    total = 0
    if isinstance(entity, int):
        total += entity
    elif isinstance(entity, list):
        for element in entity:
            total += number_search(element)
    elif isinstance(entity, dict):
        for key, val in entity.items():
            total += number_search(val)
    elif isinstance(entity, str):
        ...

    return total


def the_first_star():
    return number_search(blob)


def nonred_number_search(entity):
    total = 0
    if isinstance(entity, int):
        total += entity
    elif isinstance(entity, list):
        for element in entity:
            total += nonred_number_search(element)
    elif isinstance(entity, dict):
        subtotal = 0
        for key, val in entity.items():
            if val == "red":
                break
            subtotal += nonred_number_search(val)
        else:
            total += subtotal
    elif isinstance(entity, str):
        ...

    return total



def the_second_star():
    return nonred_number_search(blob)


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
