import itertools
import random
from string import ascii_lowercase
import unittest

from lexer_maker import (
    Automaton, AutomatonState, basic_pattern_to_automaton,
    pattern_disjunction_to_automaton
)

def recognizer_for_foo():
    s0, s1, s2, s3 = (AutomatonState(preaccepting_for=("foo",))
                      for _ in range(4))
    s0.active = True
    s0.transitions['f'] = s1
    s1.transitions['o'] = s2
    s2.transitions['o'] = s3
    s3.accepting_for = ["foo"]
    return Automaton(s0, s1, s2, s3)

def recognizer_for_cat_car_cart():
    s0, s1, s2, s3, s4, s5 = (AutomatonState() for _ in range(6))

    s0.active = True
    s0.preaccepting_for = ["cat", "car", "cart"]
    s0.transitions['c'] = s1

    s1.preaccepting_for = ["cat", "car", "cart"]
    s1.transitions['a'] = s2

    s2.preaccepting_for = ["cat", "car", "cart"]
    s2.transitions['t'] = s3
    s2.transitions['r'] = s4

    s3.preaccepting_for = ["cat"]
    s3.accepting_for = ["cat"]

    s4.preaccepting_for = ["car", "cart"]
    s4.accepting_for = ["car"]
    s4.transitions['t'] = s5

    s5.preaccepting_for = ["cart"]
    s5.accepting_for = ["cart"]

    return Automaton(s0, s1, s2, s3, s4, s5)


class NondeterministicFiniteAutomataTestCase(unittest.TestCase):

    def test_recognizer_for_foo(self):
        foo_noticer = recognizer_for_foo()
        for ch in "fo":
            foo_noticer.pump(ch)
            self.assertEqual(set(), foo_noticer.presently_accepting())
        foo_noticer.pump('o')
        self.assertEqual({"foo"}, foo_noticer.presently_accepting())

    def test_recognizer_for_car_cat_cart(self):
        for word in ("cat", "car", "cart"):
            with self.subTest(word=word):
                recognizer = recognizer_for_cat_car_cart()
                for ch in word:
                    recognizer.pump(ch)
                self.assertEqual({word}, recognizer.presently_accepting())

def my_basic_pattern():
    return [''.join(random.choice(ascii_lowercase)
                    for _c in range(random.randrange(1, 4)))
            for _p in range(random.randrange(10))]


class AutomatonConstructionTestCase(unittest.TestCase):

    def test_basic_pattern_to_automaton(self):
        for _ in range(3):
            basic_pattern = my_basic_pattern()
            matches = list(itertools.product(*basic_pattern))
            print(
                "testing {} matches for {}".format(len(matches), basic_pattern))
            for match in matches:
                with self.subTest(pattern=basic_pattern, expect_match=match):
                    automaton = basic_pattern_to_automaton(basic_pattern)
                    for ch in match:
                        self.assertFalse(automaton.presently_accepting())
                        automaton.pump(ch)
                    self.assertTrue(automaton.presently_accepting())

    @unittest.expectedFailure  # passes sometimes, but chokes on edge cases
    def test_pattern_disjunction_to_automaton(self):
        first_basic_pattern, second_basic_pattern = (my_basic_pattern()
                                                     for _ in range(2))
        first_matches = itertools.product(*first_basic_pattern)
        second_matches = itertools.product(*second_basic_pattern)
        for match in itertools.chain(first_matches, second_matches):
            with self.subTest(
                    first=first_basic_pattern, second=second_basic_pattern,
                    expected_match=match):
                automaton = pattern_disjunction_to_automaton(
                    first_basic_pattern, second_basic_pattern)
                for ch in match:
                    # self.assertFalse(automaton.presently_accepting())
                    automaton.pump(ch)
                self.assertTrue(automaton.presently_accepting())



if __name__ == "__main__":
    unittest.main()
