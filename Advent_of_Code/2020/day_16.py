import itertools
from collections import Counter

with open('data.txt') as f:
    in_content = f.read()
    in_lines = in_content.split('\n')

def parse_problem():
    fields = []
    for line in in_lines:
        if not line:
            break
        fields.append(parse_field(line))
    print(in_lines)
    my_ticket_header_index = in_lines.index("your ticket:")
    nearby_tickets_header_index = in_lines.index("nearby tickets:")
    my_ticket = [int(m) for m in in_lines[my_ticket_header_index+1].split(",")]
    nearby_tickets = []
    for line in in_lines[nearby_tickets_header_index+1:]:
        if not line:
            break
        nearby_tickets.append([int(m) for m in line.split(",")])
    return fields, my_ticket, nearby_tickets


def parse_field(line):
    field_name, val_ranges = line.split(": ")
    range1_spec, range2_spec = val_ranges.split(" or ")
    val_range1 = [int(s) for s in range1_spec.split("-")]
    val_range2 = [int(s) for s in range2_spec.split("-")]
    return (field_name, val_range1, val_range2)

def not_valid_for_any_field(n, fields):
    for field in fields:
        if field[1][0] < n < field[1][1] or field[2][0] < n < field[2][1]:
            return False
    return True

def the_first_star():
    fields, my_ticket, nearby_tickets = parse_problem()
    invalid_counter = 0
    for ticket in nearby_tickets:
        for val in ticket:
            if not_valid_for_any_field(val, fields):
                invalid_counter += val
    return invalid_counter


def the_second_star():
    # This is a tough one ...
    fields, my_ticket, nearby_tickets = parse_problem()
    valid_tickets = []
    for ticket in nearby_tickets:
        valid = True
        for val in ticket:
            if not_valid_for_any_field(val, fields):
                valid = False
                break
        if valid:
            valid_tickets.append(ticket)
    field_specs = {
        field_name: [spec1, spec2] for field_name, spec1, spec2 in fields
    }
    field_possibilities = {
        field_name: set(range(len(my_ticket))) for field_name in field_specs.keys()
    }
    print(field_specs, field_possibilities)
    for ticket in valid_tickets:
        for i, val in enumerate(ticket):
            for field_name, field_spec in field_specs.items():
                if not ((field_spec[0][0] < val < field_spec[0][1]) or
                        (field_spec[1][0] < val < field_spec[1][1])):
                    if i in field_possibilities[field_name]:
                        field_possibilities[field_name].remove(i)

    for _ in range(3):
        for field_name, poss in field_possibilities.items():
            if len(poss) == 1:
                found, = list(poss)
                for field_name2, poss2 in field_possibilities.items():
                    if field_name2 != field_name:
                        if found in poss2:
                            poss2.remove(found)

    print(field_possibilities)
    # assert all([len(v) == 1 for k, v in field_possibilities.items()])

    # e = sorted([(field_name, len(poss), poss) for field_name, poss in field_possibilities.items()], key=lambda it: it[1])
    # print(e)

    f = set()
    for k, v in field_possibilities.items():
        if k.startswith("departure"):
            print(k, v)
            f = f.union(v)


    m = 1
    for r in f:
        m *= my_ticket[r]
    return m

# not 697680?!


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
