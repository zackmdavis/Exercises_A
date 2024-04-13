
first_code = 20151125

def next_code(last_code):
    return last_code * 252533 % 33554393

# I thought the number would be small enough to brute-force—do I need to find a
# closed-form expression?—no, I just have dumb bugs.

endpoint = (2978, 3083)

def build_sequence_table():
    table = {(1, 1): 0, (2, 1): 1, (1, 2): 2}
    i = 3
    segment = 3
    keep_going = True
    while keep_going:
        current = (segment, 1)
        for diagonal_entry in range(segment):
            table[current] = i

            if current == endpoint:
                keep_going = False
                break

            i += 1
            current = (current[0]-1, current[1]+1)

        segment += 1
    return table


table = build_sequence_table()

def the_first_star():
    iteration_no = table[endpoint]
    print("Need to iterate to code no.", iteration_no)
    code = first_code
    for i in range(iteration_no):
        code = next_code(code)
    return code


def the_second_star():
    # You need to 100% the rest of the problems for the last one to unlock.
    ...


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
