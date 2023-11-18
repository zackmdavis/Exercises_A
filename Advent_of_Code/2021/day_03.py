with open("input") as f:
    content = f.read().rstrip()
    lines = content.split("\n")

bitstrings = [[int(d) for d in line] for line in lines]

def bit_voting(bitstrings, majority):
    senate = []
    for position in range(12):
        ones = 0
        zeroes = 0
        for bitstring in bitstrings:
            if bitstring[position] == 1:
                ones += 1
            else:
                zeroes += 1
        if majority:
            if ones > zeroes:
                senate.append(1)
            else:
                senate.append(0)
        else:
            if ones > zeroes:
                senate.append(0)
            else:
                senate.append(1)
    total = 0
    for i, bit in enumerate(reversed(senate)):
        total += 2**i * bit
    return total


def bit_filtering(bitstrings, majority):
    for position in range(12):
        if len(bitstrings) == 1:
            break
        ones = 0
        zeroes = 0
        for bitstring in bitstrings:
            if bitstring[position] == 1:
                ones += 1
            else:
                zeroes += 1
        if majority:
            if ones > zeroes:
                keep = 1
            elif zeroes > ones:
                keep = 0
            else:
                keep = 1
        else:
            if ones > zeroes:
                keep = 0
            elif zeroes > ones:
                keep = 1
            else:
                keep = 0

        survivors = [bitstring for bitstring in bitstrings if bitstring[position] == keep]
        bitstrings = survivors
    total = 0
    for i, bit in enumerate(reversed(bitstrings[0])):
        total += 2**i * bit
    return total


def the_first_star():
    return bit_voting(bitstrings, True) * bit_voting(bitstrings, False)

def the_second_star():
    return bit_filtering(bitstrings, True) * bit_filtering(bitstrings, False)

if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
