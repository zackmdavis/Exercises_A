import itertools
from collections import Counter

with open('day_09_input.txt') as f:
    in_content = f.read()
    in_lines = in_content.split('\n')

class Rope:
    def __init__(self):
        self.head = (0, 0)
        self.tail = (0, 0)

    def update_tail(self):
        # match on x, update y
        if self.head[0] == self.tail[0]:
            if self.head[1] - self.tail[1] >= 2:
                self.tail
        # match on y, update x
        if self.head[1] == self.tail[1]:

    def up(self):
        self.head += 1

    def down(self):
        self.head -= 1

    def right(self):
        self.head += 1

    def left(self):
        self.head -= 1



def the_first_star():
    ...

def the_second_star():
    ...


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
