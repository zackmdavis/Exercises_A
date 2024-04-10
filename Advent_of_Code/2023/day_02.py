import re

with open('input.txt') as f:
    in_content = f.read()
    in_lines = [line for line in in_content.split('\n') if line]

def parse_game(game_line):
    game_prefix, description = game_line.split(': ')
    game_no = int(game_prefix[5:])
    game_content = []
    rounds = description.split(';')
    for round_ in rounds:
        round_content = {}
        cubes_shown = round_.split(', ')
        for showing in cubes_shown:
            showing = showing.lstrip().rstrip()
            no, color = showing.split(' ')
            round_content[color] = int(no)
            game_content.append(round_content)
    return (game_no, game_content)

games = []
for line in in_lines:
    games.append(parse_game(line))

def the_first_star():
    total = 0
    for game in games:
        game_no, rounds = game
        game_ok = True
        for round_ in rounds:
            if round_.get('red', 0) > 12 or round_.get('green', 0) > 13 or round_.get('blue', 0) > 14:
                game_ok = False
                break
        if game_ok:
            total += game_no
    return total


print(the_first_star())

def the_second_star():
    total = 0
    for game in games:
        _, rounds = game
        color_maxes = {'red': 0, 'green': 0, 'blue': 0}
        for round_ in rounds:
            for color in ['red', 'green', 'blue']:
                if round_.get(color, 0) > color_maxes[color]:
                    color_maxes[color] = round_[color]
        total += color_maxes['red'] * color_maxes['green'] * color_maxes['blue']
    return total

print(the_second_star())
