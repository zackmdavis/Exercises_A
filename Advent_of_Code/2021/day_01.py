with open("input") as f:
    content = f.read().rstrip()

measurements = [int(n) for n in content.split('\n')]

def the_first_star():
    depth_increases = 0
    for i in range(len(measurements)-1):
        if measurements[i+1] > measurements[i]:
            depth_increases += 1
    return depth_increases

def the_second_star():
    depth_increases = 0
    for i in range(1, len(measurements)-2):
        if sum(measurements[i:i+3]) > sum(measurements[i-1:i+2]):  # half-open ranges, dummy
            depth_increases += 1
    return depth_increases


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
