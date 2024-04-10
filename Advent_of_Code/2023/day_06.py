import itertools
from collections import Counter


times = [44, 70, 70, 80]
distances = [283, 1134, 1134, 1491]


def the_first_star():
    wins_per_race = []
    for race_no, (race_time, race_distance) in enumerate(zip(times, distances)):
        winning_charge_time_count = 0
        for charging_time in range(race_time+1):
            performance = (race_time - charging_time) * charging_time
            if performance > distances[race_no]:
                winning_charge_time_count += 1
        wins_per_race.append(winning_charge_time_count)
    product = 1
    for wins in wins_per_race:
        product *= wins
    return product



def the_second_star():
    true_time = int(''.join(str(t) for t in times))
    true_distance = int(''.join(str(d) for d in distances))

    wins = 0
    for charging_time in range(true_time+1):
        performance = (true_time - charging_time) * charging_time
        if performance > true_distance:
            wins += 1
    return wins



if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
