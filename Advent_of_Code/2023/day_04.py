import itertools
from collections import deque

with open('input.txt') as f:
    content = f.read()
    lines = [line for line in content.split('\n') if line]

def parse_card_description(line):
    prefix, description = line.split(': ')
    raw_winners, raw_players = description.split(' | ')
    winners = [int(n) for n in raw_winners.split()]
    players = [int(n) for n in raw_players.split()]
    return winners, players


def the_first_star():
    total = 0
    for line in lines:
        ticket_winners, ticket_players = parse_card_description(line)
        win_count = 0
        for ticket_player in ticket_players:
            if ticket_player in ticket_winners:
                win_count += 1
        if win_count:
            total += 2**(win_count-1)
    return total


def score_card(winning_numbers, held_numbers):
    win_count = 0
    for held_number in held_numbers:
        if held_number in winning_numbers:
            win_count += 1
    return win_count


def the_second_star():
    deck = [parse_card_description(line) for line in lines]
    scored_tickets = [score_card(*card) for card in deck]

    tickets_to_process = deque([(index, ticket) for index, ticket in enumerate(scored_tickets)])

    total = 0
    while tickets_to_process:
        i, wins = tickets_to_process.popleft()
        total += 1
        if wins:
            for bonus_ticket in range(1, wins+1):
                tickets_to_process.append((i+bonus_ticket, scored_tickets[i+bonus_ticket]))

    return total


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
