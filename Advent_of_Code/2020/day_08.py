import copy
import itertools
from collections import Counter

with open('data.txt') as f:
    in_content = f.read()
    in_lines = [l for l in in_content.split('\n') if l]

def parse_instructions():
    instructions = []
    for line in in_lines:
        ins, val = line.split(' ')
        val = int(val)
        instructions.append([ins, val])
    return instructions

def the_first_star():
    accumulator = 0
    instructions = parse_instructions()
    pointer = 0
    visited = set()
    while True:
        if pointer in visited:
            return accumulator
        visited.add(pointer)
        ins, val = instructions[pointer]
        if ins == "nop":
            ...
            pointer += 1
        elif ins == "acc":
            accumulator += val
            pointer += 1
        elif ins == "jmp":
            pointer += val


def run_to_loop_or_end(instructions):
    accumulator = 0
    pointer = 0
    visited = set()
    target = len(instructions)
    while True:
        if pointer in visited:
            return "looped"
        if pointer == target:
            return accumulator
        visited.add(pointer)
        ins, val = instructions[pointer]
        if ins == "nop":
            pointer += 1
        elif ins == "acc":
            accumulator += val
            pointer += 1
        elif ins == "jmp":
            pointer += val


def the_second_star():
    instructions = parse_instructions()
    for i in range(len(instructions)):
        if instructions[i][0] == "nop":
            mutated = copy.deepcopy(instructions)
            mutated[i][0] = "jmp"
        elif instructions[i][0] == "jmp":
            mutated = copy.deepcopy(instructions)
            mutated[i][0] = "nop"
        else:
            continue
        v = run_to_loop_or_end(mutated)
        if v == "looped":
            ...
        else:
            return v

if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
