import sys
sys.setrecursionlimit(10000)

from functools import reduce

key = "ugkiagan"

def knot_hash(hash_in):  # from Day 10
    our_list = list(range(256))
    current_position = 0
    skip_size = 0
    lengths = [ord(c) for c in hash_in] + [17, 31, 73, 47, 23]

    for _ in range(64):
        for length in lengths:
            if current_position + length < 256:
                our_list[current_position:current_position+length] = list(reversed(our_list[current_position:current_position+length]))
            else:
                tail = 256 - current_position
                to_reverse = our_list[current_position:256] + our_list[0:current_position+length-256]
                done = list(reversed(to_reverse))
                assert len(our_list) == 256, len(our_list)
                our_list[current_position:256] = done[:tail]
                our_list[0:len(done)-tail] = done[tail:]

            current_position = (current_position + length + skip_size) % 256
            skip_size += 1

    denser = []
    for i in range(16):
        denser.append(reduce(lambda a, b: a ^ b, [our_list[(16*i)+j] for j in range(16)]))
    return ''.join('0'*(2-len(hex(d)[2:]))+hex(d)[2:] # missed the padding on day 10!
                   for d in denser)


def the_first_star():
    r = []
    total = 0
    for i in range(128):
        h = knot_hash("{}-{}".format(key, i))
        bits = [bin(int(c, 16))[2:] for c in h]
        row = []
        for chunk in bits:
            for bit in chunk:
                if bit == '1':
                    total += 1
    return total


def the_second_star():
    r = []
    total = 0
    for i in range(128):
        h = knot_hash("{}-{}".format(key, i))
        bits = [bin(int(c, 16))[2:] for c in h]
        padded = ["0"*(4-len(chunk))+chunk for chunk in bits]
        row = []
        for chunk in padded:
            for bit in chunk:
                if bit == '1':
                    row.append(True)
                else:
                    row.append(False)
        r.append(row)

    assert len(r) == 128
    for i in range(128):
        assert len(r[i]) == 128

    regions = 0
    used_map = [[False for _ in range(128)] for _ in range(128)]
    label_map = [[None for _ in range(128)] for _ in range(128)]

    def explore(ri, ci, label):
        if ri < 0 or ci < 0:
            return
        used_map[ri][ci] = True
        label_map[ri][ci] = label
        for direction in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
            x, y = direction
            try:
                if r[ri+x][ci+y] and not used_map[ri+x][ci+y]:
                    explore(ri+x, ci+y, label)
            except IndexError:
                pass

    for row_index in range(128):
        for col_index in range(128):
            u = used_map[row_index][col_index]
            ree = r[row_index][col_index]
            if not u and ree:
                explore(row_index, col_index, regions)
                regions += 1

    return regions


print(the_first_star())
print(the_second_star())
