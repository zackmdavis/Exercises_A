# Given an array of integers, write a method to find indices m and n such that
# if you sorted elements m though n, the entire array would be sorted. Minimize
# n−m (that is, find the small such sequence).

# Without really thinking about it carefully, I came up with this, which is
# wrong—

def stupid_subsort_bounds(array):
    for i in range(len(array)):
        if array[i] > array[i+1]:
            first_reversal = i+1
            break
    for i in reversed(range(len(array))):
        if array[i] < array[i-1]:
            last_reversal = i
            break
    return (first_reversal, last_reversal)

# Let's be serious.

def next_largest_element_index(array, i):
    # XXX: doesn't handle empty base case
    return min([(x, j) for j, x in list(enumerate(array))[i+1:]])[1]

def last_smallest_element_index(array, i):
    return max([(x, j) for j, x in list(enumerate(array))[:i]])[1]

def subsort_bounds(array):
    lower = 0
    upper = len(array)
    for i in range(len(array)):
        if next_largest_element_index(array, i) != i+1:
            lower = i+1
            break
    for i in reversed(range(len(array))):
        if last_smallest_element_index(array, i) != i-1:
            upper = i-1 # dislike the book's apparent closed-range convention here
            break
    return (lower, upper)


import unittest

class SubsequenceSorterTestCase(unittest.TestCase):
    def test_example(self):
        array = [1, 2, 4, 7, 10, 11, 7, 12, 6, 7, 16, 18, 19]
        self.assertEqual((3, 9), subsort_bounds(array))


if __name__ == "__main__":
    unittest.main()
