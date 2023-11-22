class Moon:
    def __init__(self, x, y, z):
        self.pos = [x, y, z]
        self.vel = [0, 0, 0]

    def potential_energy(self):
        return sum(abs(x) for x in self.pos)

    def kinetic_energy(self):
        return sum(abs(v) for v in self.vel)

def the_first_star():
    moons = [
        Moon(8, 0, 8),
        Moon(0, -5, -10),
        Moon(16, 10, -5),
        Moon(19, -10, -7),
    ]

    # moons = [
    #     Moon(-1, 0, 2),
    #     Moon(2, -10, -7),
    #     Moon(4, -8, 8),
    #     Moon(3, 5, -1),
    # ]

    # for moon in moons:
    #     print(moon.pos, moon.vel)
    # print("----")

    for step in range(1000):
        for i in range(len(moons)):
            for j in range(i+1, len(moons)):
                moon1 = moons[i]
                moon2 = moons[j]
                for coordinate in range(3):
                    if moon1.pos[coordinate] > moon2.pos[coordinate]:
                        moon1.vel[coordinate] -= 1
                        moon2.vel[coordinate] += 1
                    elif moon1.pos[coordinate] < moon2.pos[coordinate]:
                        moon1.vel[coordinate] += 1
                        moon2.vel[coordinate] -= 1
        for moon in moons:
            for coordinate in range(3):
                moon.pos[coordinate] += moon.vel[coordinate]

        # for moon in moons:
        #     print(moon.pos, moon.vel)
        # print("-----")

    energy = 0
    for moon in moons:
        energy += moon.potential_energy() * moon.kinetic_energy()
    return energy


if __name__ == "__main__":
    print(the_first_star())
