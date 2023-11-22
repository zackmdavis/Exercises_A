
input_range = [264360, 746325]

def has_pair(password):
    for i in range(len(password)-1):
        if password[i] == password[i+1]:
            return True
    return False

def nondecreasing(password):
    for i in range(len(password)-1):
        if password[i+1] < password[i]:
            return False
    return True

def validate_password_candidate(password):
    if has_pair(password) and nondecreasing(password):
        return True
    return False

def the_first_star():
    count = 0
    for password_no in range(*input_range):
        password = [int(d) for d in str(password_no)]
        if validate_password_candidate(password):
            count += 1
    return count


def has_strict_pair(password):
    for i in range(len(password)-1):
        if password[i] == password[i+1]:
            # we have a pair, confirm that it's not part of a triple, quad, &c.
            if (i-1 < 0 or password[i-1] != password[i]) and (i+2 >= len(password) or password[i+2] != password[i+1]):
                return True
    return False


def validate_password_candidate_2(password):
    if has_strict_pair(password) and nondecreasing(password):
        return True
    return False

def the_second_star():
    count = 0
    for password_no in range(*input_range):
        password = [int(d) for d in str(password_no)]
        if validate_password_candidate_2(password):
            count += 1
    return count


if __name__ == "__main__":
    assert validate_password_candidate([1,1,1,1,1,1])
    assert not validate_password_candidate([2,2,3,4,5,0])
    assert not validate_password_candidate([1,2,3,7,8,9])

    assert not validate_password_candidate_2([1,2,3,4,4,4])
    print(the_first_star())
    print(the_second_star())
