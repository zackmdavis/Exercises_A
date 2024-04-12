

puzzle_input = 29000000

def presents(house_no):
    total = 0
    for i in range(1, house_no + 1):
        if house_no % i == 0:
            total += 10 * i
    return total

# There must be something cleverer than brute-forcing it (which is too
# slow). ... can we work backwards—producing numbers with the biggest
# sums-of-factors by multiplying factors? Will the record-setter be a
# factorial?—maybe not, if higher factors don't contribute to the sum in the
# same way they contribute to the number itself?
