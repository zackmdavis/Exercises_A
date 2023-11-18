with open("input") as f:
    content = f.read().rstrip()
    lines = content.split('\n')

def follow_instructions(lines):
    position = 0
    depth = 0
    for line in lines:
        cmd, arg = line.split()
        match cmd:
            case "up":
                depth -= int(arg)
            case "down":
                depth += int(arg)
            case "forward":
                position += int(arg)
    return position, depth


def follow_correct_instructions(lines):
    position = 0
    depth = 0
    aim = 0
    for line in lines:
        cmd, arg = line.split()
        match cmd:
            case "up":
                aim -= int(arg)
            case "down":
                aim += int(arg)
            case "forward":
                position += int(arg)
                depth += int(arg) * aim
    return position, depth



def the_first_star():
    position, depth = follow_instructions(lines)
    return position * depth


def the_second_star():
    position, depth = follow_correct_instructions(lines)
    return position * depth


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
