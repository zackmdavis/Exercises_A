from collections import deque

with open("input.txt") as fi:
    raw_steps = fi.read().split(',')

def parse_step(raw_step):
    instruction, rest = raw_step[0], raw_step[1:]
    if instruction == "s":
        return ("spin", int(rest))
    elif instruction == "p":
        first, second = rest.split('/')
        return ("part", first, second)
    elif instruction == "x":
        first, second = rest.split('/')
        return ("exc", int(first), int(second))

def intepret_steps(our_steps):
    dance_line = deque(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'])
    for raw_step in raw_steps:
        step = parse_step(raw_step)
        if step[0] == "spin":
            dance_line.rotate(step[1])
        elif step[0] == "part":
            f = list(dance_line).index(step[1])
            g = list(dance_line).index(step[2])
            temp1 = dance_line[f]
            temp2 = dance_line[g]
            dance_line[f] = temp2
            dance_line[g] = temp1
        elif step[0] == "exc":
            temp1 = dance_line[step[1]]
            temp2 = dance_line[step[2]]
            dance_line[step[1]] = temp2
            dance_line[step[2]] = temp1
    return ''.join(dance_line)

print(intepret_steps(raw_steps))

# abcdefghijklmnop
# gkmndaholjbfcepi

pos_table = {'a': 0,
 'b': 1,
 'c': 2,
 'd': 3,
 'e': 4,
 'f': 5,
 'g': 6,
 'h': 7,
 'i': 8,
 'j': 9,
 'k': 10,
 'l': 11,
 'm': 12,
 'n': 13,
 'o': 14,
 'p': 15}


def the_second_star():
    p = {}
    for i, l in enumerate("gkmndaholjbfcepi"):
        p[i] = pos_table[l]
    line = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
    buf = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
    # this is pretty dumb and slow; looks like it's going to finish in about
    # ten minutes; what I should really do is batch up the permuations
    for iteration in range(1000000000):
        for i, j in p.items():
            buf[j] = line[i]
        old_line = line  # keep pointers to existing buffers
        old_buf = buf  # to avoid allocations
        line = old_buf
        buf = old_line
        if iteration % 1000000 == 0:
            print("done ", iteration, " iterations")
    print(line)

the_second_star()

# ... and it turns out to be wrong anyway :(
