# https://www.reddit.com/r/dailyprogrammer/comments/5llkbj/2017012_challenge_298_easy_too_many_parentheses/

# XXX: haven't finished or even run this

import unittest
# XXX CAUTION: needs Python 3.5 or typing backport from PyPI
from typing import List, Optional, Tuple

def match_parens(our_input: str) -> Optional[List[Tuple[int, int]]]:
    pairs = []
    stack = []
    for i, char in enumerate(our_input):
        if char == '(':
            stack.append((i, char))
        elif char == ')':
            if not stack:
                return None
            else:
                open_index, _ = stack.pop()
                pairs.append((open_index, i))
    if stack:
        return None
    else:
        return pairs


def prune_needless_parens(our_input: str) -> str:
    # So, the thought here is that `match_parens` could validate the input and
    # tell us where matching pairs of parens are, and then you could try
    # deleting a pair and see if it still validated. Every pair is either
    # deletable or not independently of previous deletions (right??), so this
    # should be linear in no. of pairs.
    #
    # ...
    #
    # "See if it still validated"—no, deleting a pair would always result in
    # another legal expression, what we care about is whether it represents the
    # _same_ grouping ... it's like an equivalence call on expression trees.
    pass


class TooManyParensTestCase(unittest.TestCase):
    def test_prune_needless_parens(self):
        for our_input, expectation in (("((a((bc)(de)))f)", "((a((bc)(de)))f)"),
                                       ("(((zbcd)(((e)fg))))", "((zbcd)((e)fg))"),
                                       ("ab((c))", "ab(c)")):
            self.assertEqual(expectation, prune_needless_parens(our_input))


if __name__ == "__main__":
    unittest.main()
