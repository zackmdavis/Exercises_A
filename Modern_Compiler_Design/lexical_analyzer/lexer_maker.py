# Better explanation (with guidance from Cooper and Torczon): we can
# build the nondeterministic finite automaton associated with the
# regular expression for each token class (using Thompson's
# construction), combine them as alternatives into a big NFA, use the
# subset construction to turn that into a deterministic finite
# automaton (with the complication of keeping track of which of the
# keyword-component NFAs were accepting or on-track at that "time"),
# minimize that DFA, and, uh, serialize it into a table or something.


class Automaton:
    def __init__(self, *states):
        self.states = list(states)

    def pump(self, character):
        for state in self.states:
            state.pump(character)
        for state in self.states:
            state.active = state.activating
            state.activating = False

    def presently_accepting(self):
        accepting = set()
        for state in self.states:
            if state.active:
                accepting |= set(
                    map(lambda m: tuple(m) if not isinstance(m, str) else m,
                        state.accepting_for)
                )
        return accepting

    def presently_preaccepting(self):
        preaccepting = set()
        for state in self.states:
            if state.active:
                preaccepting |= set(map(tuple, state.preaccepting_for))
        return preaccepting

    def __repr__(self):
        return "<{}: {}>".format(
            self.__class__.__name__, self.states)


class SingleEntrySingleExitAutomaton(Automaton):
    def __init__(self, *states, start, finish):
        self.start_state = start
        self.finish_state = finish
        super().__init__(*states)

class ε:
    def __contains__(self, element):
        return False

class AutomatonState:
    def __init__(self, transitions=None,
                 *, active=False, accepting_for=None, preaccepting_for=None):
        if transitions is None:
            transitions = {}
        self.transitions = transitions

        # bookkeeping attribute for future pumpings
        self.activating = False

        self.active = active

        if accepting_for is None:
            accepting_for = []
        self.accepting_for = accepting_for

        # QUESTION: is this really necessary? Presumably "not the
        # prefix of any token" will be represented by the automaton
        # going off the rails ("transitioning to the empty state S_ω")
        if preaccepting_for is None:
            preaccepting_for = []
        self.preaccepting_for = preaccepting_for

    def pump(self, character):
        if not self.active:
            ...  # nothing to do
        else:
            for transition_condition, following in self.transitions.items():
                if character in transition_condition:
                    following.activating = True
                elif transition_condition.__class__ is ε:
                    # There can be multiple ε-transitions, so let's
                    # adopt the convention that the value for the
                    # `ε` key in the transition dictionary is
                    # actually an iterable of the next states on ε (no
                    # need to distinguish between them)
                    for subsequent_state in following:
                        subsequent_state.activating = True

    def __repr__(self):
        return "<{}: transitions on {}, active={}, accepting_for={}>".format(
            self.__class__.__name__,
            (', '.join(t for t in self.transitions.keys())
             if self.transitions else "∅"),
            self.active,
            self.accepting_for
        )


def basic_pattern_to_automaton(pattern, being_subpattern=False):
    start_state = AutomatonState(active=not being_subpattern)
    states = [start_state]
    for ch in pattern:
        next_state = AutomatonState()
        states[-1].transitions[ch] = next_state
        states.append(next_state)
    finish_state = states[-1]
    finish_state.accepting_for.append(pattern)
    return SingleEntrySingleExitAutomaton(
        *states, start=start_state, finish=states[-1])

# Above, we've been representing a "preparsed" regular expression as
# an iterable of strings (note that in Python, a string is itself an
# iterable of strings), each element of which is a character or
# character-class to match. This isn't sufficiently expressive for
# disjunctions and the Kleene star, but expanding to "an itertable of
# strings or tuples (with the first element of such tuples being a
# disjunction or repetition 'tag')" isn't so bad.

def pattern_disjunction_to_automaton(first_pattern, second_pattern,
                                          being_subpattern=False):
    start_state = AutomatonState(active=not being_subpattern)
    finish_state = AutomatonState(
        accepting_for=[(('|', first_pattern, second_pattern))])
    states = [start_state, finish_state]
    first_subatomaton = pattern_to_automaton(first_pattern)
    second_subatomation = pattern_to_automaton(second_pattern)
    start_state.transitions[ε()] = (
        first_subatomaton.start_state, second_subatomation.start_state)
    first_subatomaton.finish_state.transitions[ε()] = finish_state
    second_subatomation.finish_state.transitions[ε()] = finish_state
    union_states = (
        states + first_subatomaton.states + second_subatomation.states)
    return SingleEntrySingleExitAutomaton(
        *union_states, start=start_state, finish=finish_state)

def starred_pattern_to_automaton(pattern):
    ...


def pattern_to_automaton(pattern, being_subpattern=False):
    if all(isinstance(sp, str) for sp in pattern):
        return basic_pattern_to_automaton(pattern)
    ...  # TODO "mixed patterns"
