import itertools
from collections import Counter

with open('data.txt') as f:
    in_content = f.read()
    in_lines = [l for l in in_content.split('\n') if l]


def seat_id(row, col):
    return 8*row + col

def read_spec(spec):
    row_num = 0
    col_num = 0
    for i in range(7):
        if spec[i] == 'B':
            row_num += 2**(7-i-1)
    for i in range(7, 10):
        if spec[i] == 'R':
            col_num += 2**(3-i+6)
    return (row_num, col_num)


def the_first_star():
    max_seat = 0
    for spec in in_lines:
        row, col = read_spec(spec)
        seat = seat_id(row, col)
        if seat > max_seat:
            max_seat = seat
    return max_seat


def the_second_star():
    all_seats = sorted([seat_id(read_spec(spec)[0], read_spec(spec)[1])
                        for spec in in_lines])
    for i in range(len(all_seats)):
        if all_seats[i+1] - all_seats[i] == 2:
            return all_seats[i]+1


if __name__ == "__main__":
    print(read_spec("FBFBBFFRLR"))
    print(the_first_star())
    print(the_second_star())
