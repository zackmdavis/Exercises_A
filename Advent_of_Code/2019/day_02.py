import itertools
from collections import Counter

code = [1,0,0,3,1,1,2,3,1,3,4,3,1,5,0,3,2,9,1,19,1,19,5,23,1,9,23,27,2,27,6,31,1,5,31,35,2,9,35,39,2,6,39,43,2,43,13,47,2,13,47,51,1,10,51,55,1,9,55,59,1,6,59,63,2,63,9,67,1,67,6,71,1,71,13,75,1,6,75,79,1,9,79,83,2,9,83,87,1,87,6,91,1,91,13,95,2,6,95,99,1,10,99,103,2,103,9,107,1,6,107,111,1,10,111,115,2,6,115,119,1,5,119,123,1,123,13,127,1,127,5,131,1,6,131,135,2,135,13,139,1,139,2,143,1,143,10,0,99,2,0,14,0]

def the_first_star():
    program = code[:]
    pointer = 0
    program[1] = 12
    program[2] = 2
    while True:
        print(program)
        instruction = program[pointer]
        match instruction:
            case 1:
                print(program[pointer+1], program[pointer+2])
                print(program[pointer+1] + program[pointer+2], program[pointer+3])
                program[program[pointer+3]] = program[program[pointer+1]] + program[program[pointer+2]]
                pointer += 4
            case 2:
                program[program[pointer+3]] = program[program[pointer+1]] * program[program[pointer+2]]
                pointer += 4
            case 99:
                break
    return program[0]

class IntcodeComputer:
    def __init__(self, code):
        self.program = code
        self.pointer = 0

    def run(self):
        while True:
            instruction = self.program[self.pointer]
            match instruction:
                case 1:
                    print(self.program[self.pointer+1], self.program[self.pointer+2])
                    print(self.program[self.pointer+1] + self.program[self.pointer+2], self.program[self.pointer+3])
                    self.program[self.program[self.pointer+3]] = self.program[self.program[self.pointer+1]] + self.program[self.program[self.pointer+2]]
                    self.pointer += 4
                case 2:
                    self.program[self.program[self.pointer+3]] = self.program[self.program[self.pointer+1]] * self.program[self.program[self.pointer+2]]
                    self.pointer += 4
                case 99:
                    break
        return self.program[0]

def the_second_star():
    for i in range(100):
        for j in range(100):
            trial_code = code.copy()
            trial_code[1] = i
            trial_code[2] = j
            computer = IntcodeComputer(trial_code)
            if computer.run() == 19690720:
                return i, j

if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
