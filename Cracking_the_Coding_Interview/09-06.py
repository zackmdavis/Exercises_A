# "Implement a program to point all valid [...] combinations of n-pairs of
# parentheses."

cache = {}

def parens(n):
    cached = cache.get(n)
    if cached is not None:
        return cached
    else:
        if n == 1:
            return {"()"}
        else:
            extension = {"(){}".format(ps) for ps in parens(n-1)}
            extension |= {"{}()".format(ps) for ps in parens(n-1)}
            extension |= {"({})".format(ps) for ps in parens(n-1)}
            cache[n] = extension
            return extension

# text solutions point out that building the string from remaining-paren counts
# is more efficient

import unittest

class ParentingTestCase(unittest.TestCase):
    def test_three(self):
        self.assertEqual({"((()))", "(()())", "(())()", "()(())", "()()()"},
                         parens(3))

if __name__ == "__main__":
    unittest.main()
