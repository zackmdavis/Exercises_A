
# C style, imitate book sol'n

def swap(a, i, j):
    temp = a[i]
    a[i] = a[j]
    a[j] = temp

def reverse_a_c_style_string(c_string):
    found_end = False
    i = 0
    while not found_end:
        looking = c_string[i]
        if looking == 0:
            found_end = True
            terminator = i
        else:
            i += 1
    f = 0
    b = terminator - 1
    while f < b:
        swap(c_string, f, b)
        f += 1
        b -= 1


import unittest

class CStyleStringReversingTestCase(unittest.TestCase):

    def test_reversing(self):
        # both even- and odd-length strings
        test_strings = [bytearray(b) for b in (b"hello world\0", b"four\0")]
        for test_string in test_strings:
            reversed_string = test_string[:]
            reverse_a_c_style_string(reversed_string)
            self.assertEqual(
                bytes(reversed_string),
                bytes(reversed(test_string[:-1])) + b'\0'
            )

if __name__ == "__main__":
    unittest.main()
