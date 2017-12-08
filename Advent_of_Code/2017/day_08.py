# goddawful code while I was trying to be FAST and thought I could save time by
# leveraging Python's `eval` to do the condition-evaluation for me

# unclear how good of a strategy this was (e.g., I ended up needing to Emacs
# search-replace in the input file to rename the register `is` to `is_`)

# ended up a solid six minutes behind the point-getters

with open("input.txt") as my_file:
    lines = my_file.read().split('\n')

from collections import Counter

def work():
    maximum = -10000
    registers = Counter()
    for line in lines:
        try:
            register, *_ = line.split(' ')
            globals()[register] = 0
            registers[register] = 0
        except Exception as e:
            print(e)

    for line in lines:
        try:
            register, op, val, token_if, condition = line.split(' ', 4)
        except:
            break
        if register not in globals():
            globals()[register] = 0


        cond = eval(condition)

        if cond:
            if op == "inc":
                registers[register] += int(val)
                globals()[register] += int(val)
            elif op == "dec":
                registers[register] -= int(val)
                globals()[register] -= int(val)
        maximum = max(maximum, max(registers.values()))

    print("globals are {}".format(globals()))
    print("registers are {}".format(registers))
    print(max(registers.values()))
    print(maximum)

work()
