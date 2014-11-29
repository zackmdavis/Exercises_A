# I'll name a file starting with a numeral and including a hyphen if I
# feel like it, and you can't tell me what to do
backtracking_lib = __import__('8-backtracking')
globals()['backtrack'] = backtracking_lib.backtrack
globals()['FINISHED'] = backtracking_lib.FINISHED

FINISHED = False

def is_solution(candidate, step, data_input):
    return step >= (data_input - 1)

def construct_possible_nexts(candidate, step, data_input):
    if step >= len(candidate):
        return []
    else:
        return [True, False]

# and bizarre global state too; I am nobody's subordinate (but am also
# imitating the textbook)
SOLUTIONS = set()

def process_solution(candidate, step, data_input):
    global SOLUTIONS
    SOLUTIONS.add(
        frozenset({i for i, present in enumerate(candidate) if present})
    )

def powerset(size):
    backtrack([None]*size, -1, size,
              is_solution, process_solution, construct_possible_nexts)
    print(SOLUTIONS)
    print(len(SOLUTIONS))
    from math import log
    print(log(len(SOLUTIONS), 2))

if __name__ == "__main__":
    powerset(8)
