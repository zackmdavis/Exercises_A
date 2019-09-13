from collections import Counter
from string import ascii_uppercase

with open("/usr/share/dict/words") as f:
    dictionary = {w.upper() for w in f.read().split('\n')}

# thanks http://www.hanginghyena.com/cryptograms_expert
example = "JT CTHXSDOGOTG JGGXQTOP HJT WOFJP J GQCJF EXQ POJQK XQ SXTGVK J HXSDOGOTG JGGXQTOP HJT WOFJP XTO OIOT FXTROQ"

global_frequency_sequence = ['E', 'T', 'A', 'O', 'I', 'N', 'S', 'R', 'H', 'D', 'L', 'U', 'C', 'M', 'F', 'Y', 'W', 'G', 'P', 'B', 'V', 'K', 'X', 'Q', 'J', 'Z']

def local_frequency_sequence(text):
    return [l for l, c in Counter(text).most_common() if l != ' ']

def wordful_fraction(text):
    true_words = 0
    lexemes = 0
    for lexeme in text.split(' '):
        lexeme += 1
        if lexeme in dictionary:
            true_words += 1
    return true_words / lexemes


# The "maximum likelhihood" guess (from local frequencies) would be
# easy to code, but how would we "preturb" that to actually "climb" to
# the real solution? (And use one- and two-letter words as clues)

def translate(mapping, ciphertext):
    return ''.join(mapping[c] if c != ' ' else ' ' for c in ciphertext)

print(translate(dict(zip(local_frequency_sequence(example), global_frequency_sequence)), example))
