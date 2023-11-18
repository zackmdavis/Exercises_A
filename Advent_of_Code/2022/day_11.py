import itertools
from collections import Counter, deque


class Monkey:
    def __init__(self, items, operation, test, true_destination, false_destination):
        self.items = items
        self.operation = operation
        self.test = test
        self.true_destination = true_destination
        self.false_destination = false_destination
        self.inspection_count = 0


class MonkeyGroup:
    def __init__(self, monkeys):
        self.monkeys = monkeys

    def round(self):
        for monkey in self.monkeys:
            for item in monkey.items:
                monkey.inspection_count += 1
                item = monkey.operation(item)
                item //= 3
                test_result = monkey.test(item)
                if test_result:
                    self.monkeys[monkey.true_destination].items.append(item)
                else:
                    self.monkeys[monkey.false_destination].items.append(item)
            monkey.items = []


class AnxiousMonkeyGroup:
    def __init__(self, monkeys):
        self.monkeys = monkeys

    def round(self):
        for monkey in self.monkeys:
            for item in monkey.items:
                monkey.inspection_count += 1
                item = monkey.operation(item)
                test_result = monkey.test(item)
                if test_result:
                    self.monkeys[monkey.true_destination].items.append(item)
                else:
                    self.monkeys[monkey.false_destination].items.append(item)
            monkey.items = []


base_monkeys = [
    Monkey([52, 78, 79, 63, 51, 94], lambda x: x * 13, lambda x: x % 5 == 0, 1, 6),
    Monkey([77, 94, 70, 83, 53], lambda x: x + 3, lambda x: x % 7 == 0, 5, 3),
    Monkey([98, 50, 76], lambda x: x * x, lambda x: x % 13 == 0, 0, 6),
    Monkey([92, 91, 61, 75, 99, 63, 84, 69], lambda x: x + 5, lambda x: x % 11 == 0, 5, 7),
    Monkey([51, 53, 83, 52], lambda x: x + 7, lambda x: x % 3 == 0, 2, 0),
    Monkey([76, 76], lambda x: x + 4, lambda x: x % 2 == 0, 4, 7),
    Monkey([75, 59, 93, 69, 76, 96, 65], lambda x: x * 19, lambda x: x % 17 == 0, 1, 3),
    Monkey([89], lambda x: x + 2, lambda x: x % 19 == 0, 2, 4),
]

def the_first_star():
    monkey_group = MonkeyGroup(base_monkeys)
    for r in range(20):
        monkey_group.round()
    monkeys_by_activity = sorted(monkey_group.monkeys, key=lambda m: m.inspection_count)
    return monkeys_by_activity[-1].inspection_count * monkeys_by_activity[-2].inspection_count

def the_second_star():
    # There's got to be some algebra fact that determines what information we
    # need to keep in order to get all the divisibility tests right without
    # keeping the absurdly large numbers around, but I'm not sure what it is.
    monkey_group = AnxiousMonkeyGroup(base_monkeys)
    for r in range(10000):
        print(r, monkey_group.monkeys[0].items)
        monkey_group.round()
    monkeys_by_activity = sorted(monkey_group.monkeys, key=lambda m: m.inspection_count)
    return monkeys_by_activity[-1].inspection_count * monkeys_by_activity[-2].inspection_count


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
