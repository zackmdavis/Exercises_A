import logging
import sys

def is_layout_character(ch):
    return ch in " \t\n"


class Token:
    def __init__(self, ch):
        self.classification = "digit" if ch in "0123456789" else ch
        self.representation = ch

    def __repr__(self):
        return "{}Token('{}')".format(
            "Digit" if self.classification == 'digit' else '',
            self.representation
        )


def lex(filename):
    with open(filename) as source_file:
        source = source_file.read()
    return (Token(ch) for ch in source if not is_layout_character(ch))


class Expression:
    # actual fields monkey-patched on instance after initialization

    def __setattr__(self, name, value):
        logging.info("in `Expression.__setattr__` for %s", self)
        super().__setattr__(name, value)

    def __str__(self):
        # digit case
        if ((getattr(self, 'kind', None) == "digit") and
            (getattr(self, 'value', None))):
            return str(self.value)
        else:  # parenthesized expression case
            return ' '.join(
                str(getattr(self, attr, "∅"))
                for attr in ('left', 'operator', 'right')
            )

    def __repr__(self):
        return str(self)

class DemoParsingException(Exception):
    pass


def parse_operator(under_construction, tokenstream):
    t = next(tokenstream)
    logging.info("drew %s from the tokenstream", t)

    if t.classification == '+':
        under_construction.operator = '+'
        return True
    elif t.classification == '*':
        under_construction.operator = '*'
        return True
    else:
        return False


def parse_expression(under_construction, tokenstream):
    t = next(tokenstream)
    logging.info("drew %s from the tokenstream", t)

    if t.classification == "digit":
        under_construction.kind = "digit"
        under_construction.value = int(t.representation)
        return True

    if t.classification == '(':
        under_construction.kind = "parenthesized"

        under_construction.left = Expression()
        if not parse_expression(under_construction.left, tokenstream):
            raise DemoParsingException("missing expression")

        under_construction.operator = None
        if not parse_operator(under_construction, tokenstream):
            raise DemoParsingException("missing operator")

        under_construction.right = Expression()
        if not parse_expression(under_construction.right, tokenstream):
            raise DemoParsingException("missing expression")

        return True

    return False


def parse_program(filename):
    logging.basicConfig(level=logging.INFO)
    ast_root = Expression()
    parse_expression(ast_root, lex(filename))
    print("parse result:", ast_root)


if __name__ == "__main__":
    parse_program(sys.argv[1])

# XXX: seems to only be able to handle leftmost derivations?!
