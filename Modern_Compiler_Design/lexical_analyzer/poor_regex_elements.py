# SCRATCH

# The text describes an algorithm for a linear-time lexer in great
# detail, but even so, there are a lot of implementation details left
# un- or under-specified if we're actually going to code the thing.

# A regex engine is itself already a kind of compiler—to avoid having
# to write my own regex engine, I might invent my own regex format??

# I thought I might get away with just having a record of character
# class, and minimum and maximum number of repetitions, but you can't
# express _disjunctions_ of regexes that way, which limitation is a
# genuine loss of power. Maybe it will work with the modification of
# character class to _either_ be a string of allowed characters (base
# case), _or_ a sequence of more PoorRegexElements. (For extra
# theoretically purity, you could make it be a single character or a
# sequence of PoorRegexElements, and express character classes as a
# bunch of single-character-matching-max-1-min-1 PREs, but nah.)

# re.compile("[abc]") ~= PoorRegexElement("abc", 1, 1)
# re.compile("[abc]{1,4}") ~= PoorRegexElement("abc", 1, 4)
# re.compile("[abc]*") ~= PoorRegexElement("abc", 0, float("inf"))
# re.compile("([ab]{2})|([cd]+)") ~=
#    PoorRegexElement([PoorRegexElement("ab", 2, 2),
#                      PoorRegexElement("cd", 1, float("inf"))],
#                     1, 1)

class PoorRegexElement:
    def __init__(self, characters, min_repetitions=1, max_repetitions=1):
        self.characters = characters
        self.min_repetitions = min_repetitions
        self.max_repetitions = max_repetitions

    @property
    def primitive(self):
        return isinstance(self.characters, str)

    def match(self, candidate):
        best_match = None
        if self.primitive:
            for match_length in range(min_repetitions, max_repetitions+1):
                # XXX quadratic time
                if all(c in self.characters for c in candidate[:match_length]):
                    best_match = candidate[:match_length]
        else:
            ...

# yeah, I'm not sure this is such a good approach; how do
# you handle the backtracking inherent in trying to match
# a disjunction of variable-match-length regexes?

# Is this entire approach doomed to failure because I'm trying to
# imitate surface properties of regexes in an ad hoc manner instead of
# actually building a finite automaton?

# Maybe it's better to not be so literal-minded about "dotted items"?
# Our ultimate _goal_ is going to be a finite automaton for
# recognizing tokens. Is there some better way to model that? The
# entire idea of "putting a dot in between elements of a regex only
# makes sense if _already_ have a tokenization of the regex, which is
# profoundly unhelpful when the entire thing we're trying to do is
# build a tokenizer from scratch!! (This might be one area where I
# actually end up preferring the approach in the less-favored Cooper
# and Torczon book?)

class Token:
    ...

class IntegerLiteral(Token):
    rec_spec = [PoorRegexElement("0123456789", 1, float("inf"))]


class FloatLiteral(Token):
    rec_spec = [PoorRegexElement("0123456789", 0, float("inf")),
                PoorRegexElement('.', 1, 1),
                PoorRegexElement("0123456789", 1, float("inf"))]

OUR_TOKENS = [IntegerLiteral, FloatLiteral]

class Item:
    def __init__(self, token, recognized, to_recognize):
        self.token = token
        self.recognized = recognized
        self.to_recognize = to_recognize


def item_closure(items):
    ...
