import time
import itertools

def displacement(position, another):
    # if I had a dollar for every time I wrote this function
    return tuple(c[1] - c[0] for c in zip(position, another))

def attackable(position, another):
    gap = tuple(abs(c) for c in displacement(position, another))
    return gap[0] == gap[1]

def positions(arena_size):
    return tuple((i, j) for i, j in itertools.product(*[range(arena_size)]*2))

def allowable(world, entrant):
    return all(not attackable(extant, entrant) for extant in world)

def allowable_worlds(world, arena_size):
    return {world.union({entrant}) for entrant in positions(arena_size)
            if allowable(world, entrant)
            and max(world.union({entrant})) == entrant}  # impose order,
                                                         # avoid redundancy

def little_bishops(arena_size, bishop_count, time_debug=False):
    if time_debug:
        start = time.time()
        print("\nstart! arena_size={} bishop_count={}".format(
            arena_size, bishop_count)
        )
    possible_worlds = {frozenset()}
    for bishop in range(bishop_count):
        next_possibilites = set()
        for world in possible_worlds:
            next_possibilites |= allowable_worlds(world, arena_size)
        possible_worlds = next_possibilites
        if time_debug:
            print(
                "bishop #{} placed across possible worlds after "
                "{} seconds!".format(
                    bishop, time.time() - start
                )
            )
    return len(possible_worlds)


import unittest

class LittleBishopsTestCase(unittest.TestCase):

    def test_attackable(self):
        self.assertTrue(attackable((0, 0), (1, -1)))
        self.assertFalse(attackable((2, 2), (3, -4)))
        # Words can be used in many ways; saying that a bishop can
        # attack its own square appears to perhaps be the natural
        # extension of 'attackable' to this degenerate case for this
        # problem
        self.assertTrue(attackable((1, 1), (1, 1)))

    def test_positions(self):
        self.assertEqual(((0, 0), (0, 1), (1, 0), (1, 1)), positions(2))

    def test_sample_output(self):
        self.assertEqual(little_bishops(2, 2), 4)
        self.assertEqual(little_bishops(4, 4), 260)

    @unittest.skip("too slow")
    def test_sample_output_seriously(self):
        # start! arena_size=8 bishop_count=6
        # bishop #0 placed across possible worlds after
        # 0.00020170211791992188 seconds!
        # bishop #1 placed across possible worlds after
        # 0.02759242057800293 seconds!
        # bishop #2 placed across possible worlds after
        # 1.1087563037872314 seconds!
        # bishop #3 placed across possible worlds after
        # 22.26791214942932 seconds!
        #
        # TODO: time complexity analysis: what exponential thing is
        # the current code doing that (presumably) it doesn't have
        # to??
        self.assertEqual(little_bishops(8, 6, time_debug=True), 5599888)


if __name__ == "__main__":
    unittest.main()
