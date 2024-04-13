example_program = [
    ["inc", "a"],
    ["jio", "a", +2],
    ["tpl", "a"],
    ["inc", "a"],
]

program = [
    ["jio", "a", +16],
    ["inc", "a"],
    ["inc", "a"],
    ["tpl", "a"],
    ["tpl", "a"],
    ["tpl", "a"],
    ["inc", "a"],
    ["inc", "a"],
    ["tpl", "a"],
    ["inc", "a"],
    ["inc", "a"],
    ["tpl", "a"],
    ["tpl", "a"],
    ["tpl", "a"],
    ["inc", "a"],
    ["jmp", +23],
    ["tpl", "a"],
    ["inc", "a"],
    ["inc", "a"],
    ["tpl", "a"],
    ["inc", "a"],
    ["inc", "a"],
    ["tpl", "a"],
    ["tpl", "a"],
    ["inc", "a"],
    ["inc", "a"],
    ["tpl", "a"],
    ["inc", "a"],
    ["tpl", "a"],
    ["inc", "a"],
    ["tpl", "a"],
    ["inc", "a"],
    ["inc", "a"],
    ["tpl", "a"],
    ["inc", "a"],
    ["tpl", "a"],
    ["tpl", "a"],
    ["inc", "a"],
    ["jio", "a", +8],
    ["inc", "b"],
    ["jie", "a", +4],
    ["tpl", "a"],
    ["inc", "a"],
    ["jmp", +2],
    ["hlf", "a"],
    ["jmp", -7],
]


class Computer:
    def __init__(self, program):
        self.program = program
        self.instruction_pointer = 0
        self.registers = {"a": 0, "b": 0}

    def step(self, command):
        match command[0]:
            case "hlf":
                self.registers[command[1]] /= 2
                self.instruction_pointer += 1
            case "tpl":
                self.registers[command[1]] *= 3
                self.instruction_pointer += 1
            case "inc":
                self.registers[command[1]] += 1
                self.instruction_pointer += 1
            case "jmp":
                self.instruction_pointer += command[1]
            case "jie":
                if self.registers[command[1]] % 2 == 0:
                    self.instruction_pointer += command[2]
                else:
                    self.instruction_pointer += 1
            case "jio":
                if self.registers[command[1]] == 1:
                    self.instruction_pointer += command[2]
                else:
                    self.instruction_pointer += 1

    def execute(self):
        while True:
            try:
                instruction = self.program[self.instruction_pointer]
                # print(instruction, self.instruction_pointer)
            except IndexError:
                break
            self.step(instruction)


def the_example():
    computer = Computer(example_program)
    computer.execute()
    return computer.registers['a']

def the_first_star():
    computer = Computer(program)
    computer.execute()
    return computer.registers['b']


def the_second_star():
    computer = Computer(program)
    computer.registers['a'] = 1
    computer.execute()
    return computer.registers['b']



if __name__ == "__main__":
    print(the_example())
    print(the_first_star())
    print(the_second_star())
