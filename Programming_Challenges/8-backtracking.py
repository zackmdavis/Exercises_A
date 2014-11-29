# The Skiena & Revilla book has a code sample for fully general
# backtracking to be customized for a particular problem by supplying
# the needed auxillary functions, which I think is an interesting
# approach (we <3 inheritance)! Here's the idea reexpressed in Python
# (sorry)

FINISHED = False

def backtrack(our_candidate, step, data_input,
              is_solution, process_solution, construct_possible_nexts):
    if is_solution(our_candidate, step, data_input):
        process_solution(our_candidate, step, data_input)
    else:
        step += 1
        possible_next_elements = construct_possible_nexts(
            our_candidate, step, data_input
        )
        for element in possible_next_elements:
            our_candidate[step] = element
            backtrack(our_candidate, step, data_input,
                      is_solution, process_solution, construct_possible_nexts)
            if FINISHED:
                return
