# http://www.reddit.com/r/dailyprogrammer/comments/
# 2qxrtk/20141231_challenge_195_intermediate_math_dice/

# Arithmetic dice! (The prompt calls it "math dice", but this is
# surely a clerical error.)

# ... with type checking! (http://mypy.readthedocs.org/ on 3.4, trying
# to actually use tip-of-3.5 gave me `ImportError: cannot import name
# 'Undefined'` (probably the fault of
# https://github.com/JukkaL/mypy/issues/639 being open))

from typing import Any, Callable, List, Tuple, Iterable, Iterator, Sequence

import itertools
import random

from functools import reduce  # type: ignore
from operator import add, sub  # type: ignore


def roll_d_n(n: int) -> int:
    return random.randint(1, n)

def possible_signed_additions(numbers: Sequence[int]) -> Iterator[int]:
    ops_sequences = itertools.product(
        (add, sub),
        repeat=len(numbers)-1
    )  # type: Iterator[Sequence[Callable[[int, int], int]]]
    for opseq in ops_sequences:
        ops = iter(opseq)
        yield reduce(lambda a, b: next(ops)(a, b), numbers)


def roll_and_solve(target_d, summand_d, summand_count):
    target = roll_d_n(target_d)
    summands = [roll_d_n(summand_d) for _ in summand_count]
    # TODO finish ...


import unittest

class ArithmeticDiceTestCase(unittest.TestCase):

    def test_roll_d_n(self):
        results = [roll for roll in (roll_d_n(6),) * 20]  # type: List[int]
        for result in results:
            self.assertGreaterEqual(result, 1)
            self.assertLessEqual(result, 6)

    def test_possible_signed_additions(self):
        for numbers, expected in {
                # 2 + 1 = 3, but 2 - 1 = 1
                (2, 1): [3, 1],
                # 1 + 2 + 3 + 4 = 10; 1 + 2 + 3 - 4 = 2;
                # 1 + 2 - 3 + 4 = 4; 1 + 2 - 3 - 4 = -4;
                # 1 - 2 + 3 + 4 = 6; 1 - 2 + 3 - 4 = -2;
                # 1 - 2 - 3 + 4 = 0; 1 - 2 - 3 - 4 = -8
                (1, 2, 3, 4): [10, 2, 4, -4, 6, -2, 0, -8]
        }.items():
            observed = [total for total in possible_signed_additions(numbers)]
            self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
