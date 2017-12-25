with open("input.txt") as f:
    lines = f.read().split('\n')


class TuringMachine:
    def __init__(self):
        self.state = 'A'
        self.left_tape = []
        self.right_tape = []
        self.cursor = 0

    def count(self):
        return sum(i for i in self.left_tape + self.right_tape if i == 1)

    def look(self):
        if self.cursor >= 0:
            try:
                return self.right_tape[self.cursor]
            except IndexError:
                self.right_tape.append(0)
                return 0
        else:
            try:
                return self.left_tape[-self.cursor-1]
            except IndexError:
                self.left_tape.append(0)
                return 0

    def write(self, v):
        if self.cursor >= 0:
            try:
                self.right_tape[self.cursor] = v
            except IndexError:
                self.right_tape.append(v)
        else:
            try:
                self.left_tape[-self.cursor-1] = v
            except IndexError:
                self.right_tape.append(v)

    def do(self):
        if self.state == 'A':
            val = self.look()
            if val == 0:
                self.write(1)
                self.cursor += 1
                self.state = 'B'
            elif val == 1:
                self.write(0)
                self.cursor -= 1
                self.state = 'C'

        elif self.state == 'B':
            val = self.look()
            if val == 0:
                self.write(1)
                self.cursor -= 1
                self.state = 'A'
            elif val == 1:
                self.write(1)
                self.cursor -= 1
                self.state = 'D'

        elif self.state == 'C':
            val = self.look()
            if val == 0:
                self.write(1)
                self.cursor += 1
                self.state = 'D'
            elif val == 1:
                self.write(0)
                self.cursor += 1
                self.state = 'C'

        elif self.state == 'D':
            val = self.look()
            if val == 0:
                self.write(0)
                self.cursor -= 1
                self.state = 'B'
            elif val == 1:
                self.write(0)
                self.cursor += 1
                self.state = 'E'

        elif self.state == 'E':
            val = self.look()
            if val == 0:
                self.write(1)
                self.cursor += 1
                self.state = 'C'
            elif val == 1:
                self.write(1)
                self.cursor -= 1
                self.state = 'F'

        elif self.state == 'F':
            val = self.look()
            if val == 0:
                self.write(1)
                self.cursor -= 1
                self.state = 'E'
            elif val == 1:
                self.write(1)
                self.cursor += 1
                self.state = 'A'


t = TuringMachine()
for i in range(12172063):
    t.do()

print(t.count())
