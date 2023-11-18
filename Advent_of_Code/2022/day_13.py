with open('input') as f:
    in_content = f.read()
    in_pairs = in_content.split('\n\n')
    pairs = []
    for raw_pair in in_pairs:
        raw_left, raw_right = raw_pair.split('\n')
        left = eval(raw_left)
        right = eval(raw_right)
        pairs.append((left, right))

def comparison(left, right):
    if isinstance(left, int) and isinstance(right, int):
        return left < right
    if isinstance(left, list) and isinstance(right, list):
        for i in range(max(len(left), len(right))):
            if i >= len(left) and i < len(right):
                return True
            if i < len(left) and i >= len(right):
                return False
            if left[i] != right[i]:
                return comparison(left[i], right[i])
    if isinstance(left, int) and isinstance(right, list):
        return comparison([left], right)
    if isinstance(left, list) and isinstance(right, int):
        return comparison(left, [right])
    return None


def the_first_star():
    print(len(pairs))
    correct = []
    for i, (left, right) in enumerate(pairs):
        if comparison(left, right):
            correct.append(i+1)
    print(correct)
    # XXX: solution checker thinks 5602 is too high?!—but my code gets the
    # right answer on the example, which makes it harder to suss out what
    # edgey-case I'm getting wrong. :'(
    return sum(correct)

def the_second_star():
    ...


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
