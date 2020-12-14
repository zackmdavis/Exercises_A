import itertools
from collections import Counter

with open('data.txt') as f:
    in_content = f.read()
    in_lines = [l for l in in_content.split('\n') if l]

# def parse_ins():

def apply_mask(mask, n):
    bin_n = bin(n)[2:].zfill(36)
    total = 0
    for i in range(36):
        if mask[-(i+1)] != 'X':
            if mask[-(i+1)] == '1':
                total += 2 ** i
        elif bin_n[-(i+1)] == '1':
                total += 2 ** i
    return total


def the_first_star():
    memory = {}
    for line in in_lines:
        if line.startswith("mask"):
            mask = line[7:]
            continue
        line = line[4:]
        mem_slot, val = line.split("] = ")
        mem_slot = int(mem_slot)
        val = int(val)
        memory[mem_slot] = apply_mask(mask, val)
    return sum(memory.values())


def add_to_set(s, n):
    return {e+n for e in s}

def maybe_add_to_set(s, n):
    return s | {e+n for e in s}

def apply_mem_mask(mask, n):
    bin_n = bin(n)[2:].zfill(36)
    possible_totals = set([0])
    for i in range(36):
        if mask[-(i+1)] == '1':
            possible_totals = add_to_set(possible_totals, 2**i)
        elif mask[-(i+1)] == '0':
            if bin_n[-(i+1)] == '1':
                possible_totals = add_to_set(possible_totals, 2**i)
        elif mask[-(i+1)] == 'X':
            possible_totals = maybe_add_to_set(possible_totals, 2**i)
    return possible_totals


def the_second_star():
    memory = {}
    for line in in_lines:
        if line.startswith("mask"):
            mask = line[7:]
            continue
        line = line[4:]
        mem_slot, val = line.split("] = ")
        val = int(val)
        mem_slot = int(mem_slot)
        mem_slots = apply_mem_mask(mask, mem_slot)
        for slot in mem_slots:
            memory[slot] = val
    return sum(memory.values())



if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
