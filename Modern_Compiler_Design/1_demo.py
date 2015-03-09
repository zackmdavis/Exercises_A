
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
    pass

class DigitExpression(Expression):
    type = "digit"

    def __init__(self, value):
        self.value = value

class ParenthesizedExpression(Expression):
    type = "parenthesized"

    def __init__(self, operator, left, right):
        self.operator = operator
        self.left = left
        self.right = right

# to be continued ...
