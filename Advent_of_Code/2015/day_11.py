
puzzle_input = "vzbxkghb"

def increment_character(c):
    if c == 'z':
        return 'a'
    else:
        return chr(ord(c) + 1)

def increment(password):
    tail = increment_character(password[-1])
    if tail == 'a':  # rolled over
        return increment(password[:-1]) + tail
    else:
        return password[:-1] + tail


def has_straight(password):
    straight_counter = 0
    for i in range(len(password)-1):
        if ord(password[i+1]) - ord(password[i]) == 1:
            straight_counter += 1
            if straight_counter == 3:
                return True
        else:
            straight_counter = 0
    return False


def no_confusables(password):
    return all(confusable not in password for confusable in ["i", 'o', 'l'])

def has_two_pair(password):
    pairs = []
    for i in range(len(password)-1):
        if password[i+1] == password[i]:
            if not pairs or pairs[-1][0] != i-1:
                pairs.append((i, password[i]))
    if len(pairs) >= 2 and not all(c == pairs[0][1] for i, c in pairs):
        return True
    return False

def valid(password):
    if has_straight(password) and no_confusables(password) and has_two_pair(password):
        return True
    return False

def find_next_password(password):
    while True:
        if valid(password):
            return password
        password = increment(password)

def the_first_star():
    # XXX: I get "vzccdeff", which doesn't validate ... not sure what I'm doing wrong?!?
    return find_next_password(puzzle_input)


if __name__ == "__main__":
    assert has_straight("hijklmmn")
    assert has_two_pair("abbceffg")
    assert not has_two_pair("abbcegjk")
    assert not has_straight("ghjaaabb")
    # assert find_next_password("abcdefgh") == "abcdffaa", find_next_password("abcdefgh")
    # assert find_next_password("ghijklmn") == "ghjaabcc", find_next_password("ghijklmn")

    print(the_first_star())
