import itertools
from collections import Counter

# TODO

class Character:
    def __init__(self, name, hp, damage, armor, mana):
        self.name = name
        self.hp = hp
        self.damage = damage
        self.armor = armor
        self.mana = mana

    def __repr__(self):
        return "<{} {}💓 {}🗡️ {}🛡️ {} ✨>".format(self.name, self.hp, self.damage, self.armor, self.mana)


class Effect:
    def __init__(timer):
        self.timer = timer

    def tick():
        ...


class Shield:
    ...

class Poison:
    ...

class Recharge:
    ...


def simulate_fight(player, boss):
    while True:
        ...


def the_first_star():
    ...

def the_second_star():
    ...


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
