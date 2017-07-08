# https://www.reddit.com/r/dailyprogrammer/comments/611tqx/20170322_challenge_307_intermediate_scrabble/

# What is the longest word you can build in a game of Scrabble one letter at a
# time? That is, starting with a valid two-letter word, how long a word can you
# build by playing one letter at a time on either side to form a valid
# three-letter word, then a valid four-letter word, and so on? (For example, HE
# could become THE, then THEM, then THEME, then THEMES, for a six-letter result

import os
from string import ascii_lowercase

def load_words():
    if os.path.exists("/usr/share/dict/words"):
        path = "/usr/share/dict/words"
    else:
        # I'm at the Code Self Study meetup and /usr/share/dict/words doesn't
        # exist on my phone
        #
        # Why am I at the meetup doing this trivial exercise? I guess I wanted
        # to be social, and Cauzzle doesn't run on my phone?
        #
        # and maybe, for some reason, this particular "longest word such that"
        # problem is charming and I'm curious about the answer
        path = "/data/data/com.termux/files/home/Code/wordlist"
    with open(path) as f:
        return set((f.read().split('\n')))


WORDS = load_words()

def search(seed, trail):
    longest = seed
    farsight = trail
    for direction in ['←', '→']:
        for extension in ascii_lowercase:
            if direction == '←':
                step = seed + extension
            elif direction == '→':
                step = extension + seed
            if step in WORDS:
                beyond, vision = search(step, trail + [(direction, extension)])
                if len(beyond) > len(longest):
                    longest = beyond
                    farsight = vision
    return longest, farsight

if __name__ == "__main__":
    print(search("a", []))
    print(search("i", []))
    print(search("o", []))
