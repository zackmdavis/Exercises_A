# "Design an algorithm to figure out if someone has won a game of tic-tac-toe."

def lines(field):
    # rows
    for row in field:
        yield row
    # columns
    for i in range(len(field)):
        yield [field[j][i] for j in range(len(field))]
    # diagonals
    yield [field[i][i] for i in range(len(field))]
    yield [field[-i][i] for i in range(len(field))]


def victory_condition(field):
    for line in lines(field):
        if all([c == line[0] for c in line]):
            return line[0]
    return None


import unittest

class TicTacToeVerificationTestCase(unittest.TestCase):
    def test_victory_condition(self):
        cases = [([['X', ' ', ' '],
                   ['X', ' ', ' '],
                   ['X', ' ', ' ']], 'X'),
                 ([['X', ' ', ' '],
                   [' ', 'X', ' '],
                   [' ', ' ', 'X']], 'X'),
                 ([['X', ' ', ' '],
                   [' ', 'O', ' '],
                   [' ', ' ', 'X']], None)]
        for field, expected_outcome in cases:
            self.assertEqual(expected_outcome, victory_condition(field))

if __name__ == "__main__":
    unittest.main()
