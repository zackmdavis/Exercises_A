import re

with open('input.txt') as f:
    in_content = f.read()
    in_lines = in_content.split('\n')


def the_first_star():
    total = 0
    for line in in_lines:
        digits = ''.join(c for c in line if c.isdigit())
        if digits:
            total += int(digits[0] + digits[-1])
    return total

digit_map = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}

digit_regex = re.compile("[0-9]|one|two|three|four|five|six|seven|eight|nine")

def convert_result(result):
    if result.isdigit():
        return result
    else:
        return str(digit_map[result])

def the_second_star():
    total = 0
    for line in in_lines:
        results = digit_regex.findall(line)
        print(results)
        if results:
            total += int(convert_result(results[0]) + convert_result(results[-1]))
    return total


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())

# Confused at what could possibly be wrong with my Day 1 (!) code, I search for
# `advent of code 2023 day 1 bug`, and one of the Reddit commenters points out
# that `twone` counts as "two" and "one" (whereas the regex doesn't find
# overlapping matches). Charming. Count this as a dumb warm-up.
