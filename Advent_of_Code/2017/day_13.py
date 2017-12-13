with open("input.txt") as f:
    lines = f.read().split('\n')

s = {}

for line in lines:
    try:
        left, right = line.split(':')
    except:
        pass
    s[int(left)] = int(right)

print(s)

def env():
    l = [None] * 97
    for enemy in s:
        l[enemy] = [0, s[enemy], 'd']
    return l

def advance(e):
    for i, slot in enumerate(e):
        if slot is not None:
            b, depth, direction = slot
            if direction == 'd':
                if b == depth-1:
                    b -= 1
                    direction = 'u'
                else:
                    b += 1
            elif direction == 'u':
                if b == 0:
                    b += 1
                    direction = 'd'
                else:
                    b -= 1
            else:
                raise Exception
            e[i] = [b, depth, direction]
    return e

def journey(delay):
    e = env()
    for _ in range(delay):
        e = advance(e)
    severity = 0
    for j in range(97):
        if e[j] is not None:
            if e[j][0] == 0:
                severity += j * e[j][1]
        e = advance(e)
    return severity


def second_star_attempt():
    # this doesn't work
    #
    # my attempt to use Rust iterator magic to find the first delay that
    # satisfied all the constraints (delay + defense_layer % scanning-period !=
    # 0) didn't work, either (the full version wouldn't compile in a reasonable
    # amount of time, and an abbreviated test overflowed its integer)
    #
    # I'll take another look some other day
    our_delay = 1
    while True:
        sev = journey(our_delay)
        print(our_delay, sev)
        if sev == 0:
            print("WIN", our_delay)
            break
        our_delay += 1
