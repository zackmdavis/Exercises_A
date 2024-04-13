import itertools
from collections import Counter

boss_hp = 100
boss_damage = 8
boss_armor = 2

# Is there supposed to be some principled solution here, or am I just to write
# code to simulate the fight, and try out equipment combinations? I guess the
# answer could be pretty subtle and can't be intuited (because expensive armor
# could make up for a cheap weapon).


class Character:
    def __init__(self, name, hp, damage, armor):
        self.name = name
        self.hp = hp
        self.damage = damage
        self.armor = armor

    def __repr__(self):
        return "<{} {}💓 {}🗡️ {}🛡️ >".format(self.name, self.hp, self.damage, self.armor)


def simulate_fight(player, boss):
    while True:
        # Important: you have to check health after every attack; a `while both
        # HP > 0` loop isn't checking often enough, dummy
        boss.hp -= max(player.damage - boss.armor, 1)
        if boss.hp <= 0:
            break
        player.hp -= max(boss.damage - player.armor, 1)
        if player.hp <= 0:
            break
    return player.hp > 0


# Since I have the same starting HP as him (coincidence??), then to win, I need
# to be dealing than receiving more damage per round.

class Item:
    def __init__(self, name, cost, damage, armor, class_):
        self.name = name
        self.cost = cost
        self.damage = damage
        self.armor = armor
        self.class_ = class_

    def __repr__(self):
        return "<{} ${} {}🗡️ {}🛡️ >".format(self.name, self.cost, self.damage, self.armor)

shop = {
    "weapon": [
        Item("dagger", 8, 4, 0, "weapon"),
        Item("shortsword", 10, 5, 0, "weapon"),
        Item("warhammer", 25, 6, 0, "weapon"),
        Item("longsword", 40, 7, 0, "weapon"),
        Item("greataxe", 74, 8, 0, "weapon"),
    ],
    "armor": [
        Item("the empty armor", 0, 0, 0, "armor"),
        Item("leather", 13, 0, 1, "armor"),
        Item("chainmail", 31, 0, 2, "armor"),
        Item("splintmail", 53, 0, 3, "armor"),
        Item("bandedmail", 75, 0, 4, "armor"),
        Item("platemail", 102, 0, 5, "armor"),
    ],
    "ring": [
        Item("the first empty ring", 0, 0, 0, "ring"), # the "empty ring"
        Item("the second empty ring", 0, 0, 0, "ring"),
        Item("D+1", 25, 1, 0, "ring"),
        Item("D+2", 50, 2, 0, "ring"),
        Item("D+3", 100, 3, 0, "ring"),
        Item("A+1", 20, 0, 1, "ring"),
        Item("A+2", 40, 0, 2, "ring"),
        Item("A+3", 80, 0, 3, "ring"),
    ],
}


def the_first_star():
    lowest_cost = float("inf")
    for weapon in shop["weapon"]:
        for armor in shop["armor"]:
            for ring1 in shop["ring"]:
                for ring2 in shop["ring"]:
                    cost = weapon.cost + armor.cost + ring1.cost + ring2.cost
                    pc = Character(
                        "Player Character",
                        100,
                        weapon.damage + ring1.damage + ring2.damage,
                        armor.armor + ring1.armor + ring2.armor,
                    )
                    boss = Character("Boss", boss_hp, boss_damage, boss_armor)
                    win = simulate_fight(pc, boss)
                    if win:
                        if cost < lowest_cost:
                            lowest_cost = cost
                            print(pc, "won with new lowest cost {}".format(cost))
                            print("inventory was", weapon, armor, ring1, ring2)
                            print()


    return lowest_cost


def the_second_star():
    highest_cost = float("-inf")
    for weapon in shop["weapon"]:
        for armor in shop["armor"]:
            for ring1 in shop["ring"]:
                for ring2 in shop["ring"]:
                    cost = weapon.cost + armor.cost + ring1.cost + ring2.cost
                    pc = Character(
                        "Player Character",
                        100,
                        weapon.damage + ring1.damage + ring2.damage,
                        armor.armor + ring1.armor + ring2.armor,
                    )
                    boss = Character("Boss", boss_hp, boss_damage, boss_armor)
                    loss = not simulate_fight(pc, boss)
                    if loss:
                        if cost > highest_cost:
                            highest_cost = cost
                            print(pc, "lost with new highest cost {}".format(cost))
                            print("inventory was", weapon, armor, ring1, ring2)
                            print()

    return highest_cost


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
