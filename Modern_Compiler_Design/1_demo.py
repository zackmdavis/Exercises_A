import sys

def is_layout_character(ch):
    return ch in " \t\n"


class Token:
    def __init__(self, ch):
        self.classification = "digit" if ch in "0123456789" else ch
        self.representation = ch

    def __repr__(self):
        return "{}Token({})".format(
            "Digit" if self.classification == 'digit' else '',
            self.representation
        )


def lex(filename):
    with open(filename) as source_file:
        source = source_file.read()
    return (Token(ch) for ch in source if not is_layout_character(ch))


class Expression:
    # # actual fields monkey-patched on instance after initialization
    # # kind = None
    # # value = None
    # # operator = None
    # # left = None
    # # right = None

    def __str__(self):
        # digit case
        if getattr(self, 'value'):
            print self.value
            return
        else:
            for attr in ('left', 'operator', 'right'):
                print getattr(self, attr, "__ none __")


class DemoParsingException(Exception):
    pass


def parse_operator(under_construction, tokenstream):
    t = next(tokenstream)
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
    ast_root = Expression()
    parse_expression(ast_root, lex(filename))


if __name__ == "__main__":
    parse_program(sys.argv[1])

# XXX doesn't work as written

# zmd@ExpectedReturn:~/Code/Textbook_Exercises_A/Modern_Compiler_Design$
# python 1_demo.py demo.demo
# Traceback (most recent call last):
#   File "1_demo.py", line 93, in <module>
#     parse_program(sys.argv[1])
#   File "1_demo.py", line 89, in parse_program
#     parse_expression(ast_root, lex(filename))
#   File "1_demo.py", line 76, in parse_expression
#     raise DemoParsingException("missing operator")
# __main__.DemoParsingException: missing operator
