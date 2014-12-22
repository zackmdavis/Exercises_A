
SEGMENTS = {
    'top': ('-', lambda s: [(0, i) for i in range(1, s+1)]),
    'middle': ('-', lambda s: [(s+1, i) for i in range(1, s+1)]),
    'bottom': ('-', lambda s: [(-1, i) for i in range(1, s+1)]),
    'upper-left': ('|', lambda s: [(i, 0) for i in range(1, s+1)]),
    'upper-right': ('|', lambda s: [(i, -1) for i in range(1, s+1)]),
    'lower-left': ('|', lambda s: [(s+1+i, 0) for i in range(1, s+1)]),
    'lower-right': ('|', lambda s: [(s+1+i, -1) for i in range(1, s+1)]),
}

FIGURE_TO_SEGMENTS = {
    "1": ('upper-right', 'lower-right'),
    "2": ('top', 'middle', 'bottom', 'upper-right', 'lower-left'),
    "3": ('top', 'middle', 'bottom', 'upper-right', 'lower-right'),
    "4": ('middle', 'upper-left', 'upper-right', 'lower-right'),
    "5": ('top', 'middle', 'bottom', 'upper-left', 'lower-right'),
    "6": ('top', 'middle', 'bottom', 'upper-left', 'lower-left', 'lower-right'),
    "7": ('top', 'upper-right', 'lower-right'),
    "8": ('top', 'middle', 'bottom',
          'upper-left', 'upper-right', 'lower-left', 'lower-right'),
    "9": ('top', 'middle', 'bottom',
          'upper-left', 'upper-right', 'lower-right'),
    "0": ('top', 'bottom',
          'upper-left', 'upper-right', 'lower-left', 'lower-right')
}

class DigitCanvas:
    def __init__(self, size):
        self.size = size
        self.data = [[' ' for _j in range(size+2)] for _i in range(2*size+3)]

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __add__(self, other):
        for i, row in enumerate(self.data):
            # having addition mutate the first summand is
            # unequivocally awful, but it is written that YOLO
            row.append(' ')
            row.extend(other.data[i])
        return self

    def __str__(self):
        return '\n'.join(''.join(line) for line in self.data)

def apply_segment(canvas, segment):
    character, cell_generator = SEGMENTS[segment]
    cells = cell_generator(canvas.size)
    for row, col in cells:
        canvas[row][col] = character
    return canvas

def apply_figure(canvas, figure):
    segments = FIGURE_TO_SEGMENTS[figure]
    for segment in segments:
        apply_segment(canvas, segment)
    return canvas

def lcd_display(size, figures):
    first, *rest = [apply_figure(DigitCanvas(size), figure)
                    for figure in figures]
    return str(sum(rest, first))


import unittest

class LcdDigitTest(unittest.TestCase):

    def test_eight(self):
        self.assertEqual(
"""
 - $
| |
 - $
| |
 - """[1:].replace('$', ''),
# (the '$'s are a gross hack to prevent my usually-helpful
# delete-trailing-whitespace hook from eating spaces that actually
# should be part of the string)
            str(apply_figure(DigitCanvas(1), '8'))
        )

class LcdDisplayTest(unittest.TestCase):
    expected_one_two_three_four_five = """
      --   --        -- $
   |    |    | |  | |   $
   |    |    | |  | |   $
      --   --   --   -- $
   | |       |    |    |
   | |       |    |    |
      --   --        -- """[1:].replace('$', '')

    # but I think I have an extra space between the six and the seven,
    # not sure why??
    expected_six_seven_eight_nine_zero = """
 ---   ---   ---   ---   --- $
|         | |   | |   | |   |
|         | |   | |   | |   |
|         | |   | |   | |   |
 ---         ---   ---       $
|   |     | |   |     | |   |
|   |     | |   |     | |   |
|   |     | |   |     | |   |
 ---         ---   ---   --- """[1:].replace('$', '')

    def test_sample_output(self):
        self.assertEqual(
            self.expected_one_two_three_four_five,
            lcd_display(2, "12345")
        )
        self.assertEqual(
            self.expected_six_seven_eight_nine_zero,
            lcd_display(3, "67890")
        )

if __name__ == "__main__":
    unittest.main()
