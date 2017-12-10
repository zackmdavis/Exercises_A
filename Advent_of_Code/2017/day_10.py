from functools import reduce

our_input = "88,88,211,106,141,1,78,254,2,111,77,255,90,0,54,205"

def the_first_star(our_input):
    our_input = [int(i) for i in our_input.split(',')]
    our_list = list(range(256))
    current_position = 0
    skip_size = 0
    lengths = our_input

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

    print(our_list[0] * our_list[1])


def the_second_star(our_input):
    our_list = list(range(256))
    current_position = 0
    skip_size = 0
    lengths = [ord(c) for c in our_input] + [17, 31, 73, 47, 23]

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
    print(''.join(hex(d)[2:] for d in denser))


the_first_star(our_input)
the_second_star(our_input)
