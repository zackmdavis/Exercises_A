# reddit.com
# /r/dailyprogrammer/comments
# /11par4/10182012_challenge_104_intermediate_bracket_racket/

def is_correctly_paired(expression):
    open_delimiters = ('(', '<', '[', '{')
    close_delimiters = (')', '>', ']', '}')
    close_to_match = dict(zip(close_delimiters, open_delimiters))
    unmatched = []
    for character in expression:
        if character in open_delimiters:
            unmatched.append(character)
        elif character in close_delimiters:
            if (len(unmatched) == 0 or
                close_to_match[character] != unmatched[-1]):
                return False
            else:
                unmatched.pop()
    return not unmatched

import unittest

class CorrectDelimiterPairingTestCase(unittest.TestCase):
    def test_the_light(self):
        light = (".format(\"{foo}\", {'foo': \"bar\"})",
                 "([{<()>}()])",)
        for photon in light:
            with self.subTest(i=photon):
                self.assertTrue(is_correctly_paired(photon))

    def test_the_darkness(self):
        darkness = ("foo(''.format(\"[no pun intended]\",)",
                    "foo(''.format(\"[no pun intended]\",)))",
                    "([{<()>}()))",
                    "(((",)
        for negative_luxagen in darkness:
            with self.subTest(i=negative_luxagen):
                self.assertFalse(is_correctly_paired(negative_luxagen))

if __name__ == "__main__":
    unittest.main()
