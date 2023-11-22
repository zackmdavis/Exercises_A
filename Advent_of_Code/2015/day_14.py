

class Reindeer:
    def __init__(self, name, sprint_speed, sprint_time, downtime):
        self.name = name
        self.sprint_speed = sprint_speed
        self.sprint_time = sprint_time
        self.downtime = downtime

        self.sprinting = True
        self.timer = self.sprint_speed
        self.distance = 0

    def tick(self):
        self.timer -= 1
        if self.sprinting:
            self.distance += self.sprint_speed
        if self.timer == 0:
            match self.sprinting:
                case True:
                    self.sprinting = False
                    self.timer = self.downtime
                case False:
                    self.sprinting = True
                    self.timer = self.sprint_time



def the_first_star():
    reindeer = [
        Reindeer("Dancer", 27, 5, 132),
        Reindeer("Cupid", 22, 2, 41),
        Reindeer("Rudolph", 11, 5, 48),
        Reindeer("Donner", 28, 5, 134),
        Reindeer("Dasher", 4, 16, 55),
        Reindeer("Blitzen", 14, 3, 38),
        Reindeer("Prancer", 3, 21, 40),
        Reindeer("Comet", 18, 6, 103),
        Reindeer("Vixen", 18, 5, 84),
    ]
    for _ in range(2503):
        for deer in reindeer:
            deer.tick()

    return max(deer.distance for deer in reindeer)


if __name__ == "__main__":
    print(the_first_star())  # XXX: wrong
