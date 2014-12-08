import itertools

def displacement(position, another):
    # if I had a dollar for every time I wrote this function
    return tuple(c[1] - c[0] for c in zip(position, another))

def attackable(position, another):
    gap = tuple(abs(c) for c in displacement(position, another))
    return gap[0] != 0 and gap[0] == gap[1]

def positions(arena_size):
    return tuple((i, j) for i, j in itertools.product(*[range(arena_size)]*2))

def allowable(world, entrant):
    return all(not attackable(extant, entrant) for extant in world)

def allowable_worlds(world, arena_size):
    return {world.union({entrant}) for entrant in positions(arena_size)
            if allowable(world, entrant)
            and max(world.union({entrant})) == entrant}  # impose order,
                                                         # avoid redundancy

def little_bishops(arena_size, bishop_count):
    possible_worlds = {frozenset()}
    for _bishop in range(bishop_count):
        next_possibilites = set({})
        for world in possible_worlds:
            next_possibilites |= allowable_worlds(world, arena_size)
        possible_worlds = next_possibilites
    return len(possible_worlds)


import unittest

class LittleBishopsTestCase(unittest.TestCase):

    def test_attackable(self):
        self.assertTrue(attackable((0, 0), (1, -1)))
        self.assertFalse(attackable((2, 2), (3, -4)))
        self.assertFalse(attackable((1, 1), (1, 1)))

    def test_positions(self):
        self.assertEqual(((0, 0), (0, 1), (1, 0), (1, 1)), positions(2))

    def test_sample_output(self):
        # XXX: AssertionError: 600 != 260
        self.assertEqual(little_bishops(4, 4), 260)

        # XXX: takes too long
        # self.assertEqual(little_bishops(8, 6), 5599888)

if __name__ == "__main__":
    unittest.main()
