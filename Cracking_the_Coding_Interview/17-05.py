
def mastermind_guess_score(guess, solution):
    hits = 0
    pseudohits = 0
    for (guess_slot, solution_slot) in zip(guess, solution):
        if guess_slot == solution_slot:
            hits += 1
        else:
            if guess_slot in solution:
                pseudohits += 1
    return (hits, pseudohits)

assert((1, 1) == mastermind_guess_score('RGBY', 'GGRR'))

# XXX: solution differs w.r.t. pseudohit criterion; you could argue that this
# is a bug in the problem statement, but only if you were some kind of asshole
