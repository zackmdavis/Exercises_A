# https://twitter.com/compiledwrong/status/1066882636189065216
#
# I promise to never try to use this interview question on anyone. Given a list
# of words Y spelled without letters in a particular word X, figure out word X
# from omissions in other words + dictionary lookup using as few as possible
# words from Y

import unittest

with open('/usr/share/dict/words') as dictionary:
    ALL_WORDS = set(
        word for word in dictionary.read().split('\n')
        if "'" not in word
        if word == word.lower()
        if len(word) > 2
    )

# def infer_secret_word(all_words, words_without_secret_letters):
#     candidates = all_words
#     winnowing = len(candidates)
#     for shadow_word in words_without_secret_letters:
#         candidates = {
#             word for word in candidates
#             if all(c not in word for c in shadow_word)
#         }
#         what_is_left = len(candidates)
#         print(what_is_left)
#         if what_is_left == winnowing:
#             return candidates
#     return candidates

# minimized for Twitter—

def infer_secret_word(x, y):
    ca = x
    wi = len(ca)
    for shd in y:
        ca = {
            w for w in ca
            if all(c not in w for c in shd)
        }
        wl = len(ca)
        if wl == wi:
            return ca
    return ca


class ShadowWordTestCase(unittest.TestCase):
    def test_infer_secret_word(self):
        secret_wordset = infer_secret_word(
            ALL_WORDS,
            ['the', 'quick', 'brown', 'flys']
        )
        print("secret wordset is", secret_wordset)
        self.assertEqual(
            secret_wordset,
            {
                'jazz', 'madam', 'mamma', 'magma', 'gad', 'zap', 'amp',
                'jam', 'gap', 'damp', 'gamma', 'dad', 'pap', 'vamp', 'adz',
                'gag', 'mama', 'jag', 'map', 'add', 'mad', 'dam', 'papa', 'pad'
            }
        )


if __name__ == "__main__":
    unittest.main()
