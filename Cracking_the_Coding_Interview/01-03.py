# Let me do this in Python, too, just to viscerally _feel_ the
# contrast in time and effort between "solving a trivial problem in a
# high-level programming language I know really well" and "solving a
# trivial problem in a comparatively-lowish-level (??) programming
# language that I barely know anything about at all"

from collections import Counter

def permuted(first, second):
    return Counter(first) == Counter(second)

import unittest

class TestPermutation(unittest.TestCase):
    def test_detect_accept_permutation(self):
        for pair in (("rah", "ahr"),
                     ("william shakespeare", "iamaweakish speller")):
            with self.subTest(the_pair=pair):  # rah Python 3.4 subtests!!
                self.assertTrue(permuted(*pair))
        for pair in (("dog", "america"), ("python", "pytho")):
            with self.subTest(the_pair=pair):
                self.assertFalse(permuted(*pair))

if __name__ == "__main__":
    unittest.main()
