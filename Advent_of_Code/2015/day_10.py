puzzle_input = [int(x) for x in "1321131112"]

def look_and_say(seq):
    say_seq = []
    current_run = 1
    current_element = seq[0]
    for element in seq[1:]:
        if element == current_element:
            current_run +=1
        else:
            say_seq.extend([current_run, current_element])
            current_run = 1
            current_element = element
    say_seq.extend([current_run, current_element])
    return say_seq

def the_first_star():
    seq = puzzle_input
    for _ in range(40):
        seq = look_and_say(seq)
    return len(seq)

def the_second_star():
    seq = puzzle_input
    for _ in range(50):
        seq = look_and_say(seq)
    return len(seq)


if __name__ == "__main__":
    assert look_and_say([1,1,1,2,2,1]) == [3,1,2,2,1,1]
    print(the_first_star())
    print(the_second_star())
