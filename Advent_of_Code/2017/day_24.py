with open("input.txt") as f:
    lines = f.read().split('\n')

pieces = [tuple(map(int, s.split('/'))) for s in lines[:-1]]

print(pieces)

def bridge_search(b, end, pieces):
    next_bridges = []
    for piece in pieces:
        if end in piece:
            if piece[0] == end:
                new_end = piece[1]
            else:
                new_end = piece[0]
            new_pieces = pieces[:]
            new_pieces.remove(piece)
            next_bridges.extend(bridge_search(b[:] + [piece], new_end, new_pieces))
    if not next_bridges:
        return [b]
    else:
        return next_bridges

def bridge_strength(pieces):
    return sum(p[0] + p[1] for p in pieces)

test_pieces = [(0, 2), (2, 2), (2, 3), (3, 4), (3, 5), (0, 1), (10, 1), (9, 10)]
test_pieces.remove((0, 2))
print(bridge_search([(0, 2)], 2, test_pieces))

# the first star

for piece in pieces:
    if 0 in piece:
        if piece[0] == 0:
            new_end = piece[1]
        else:
            new_end = piece[0]
        new_pieces = pieces[:]
        new_pieces.remove(piece)
        bridges_from = bridge_search([piece], new_end, new_pieces)
        print(max(map(bridge_strength, bridges_from)))


# the second star

for piece in pieces:
    if 0 in piece:
        if piece[0] == 0:
            new_end = piece[1]
        else:
            new_end = piece[0]
        new_pieces = pieces[:]
        new_pieces.remove(piece)
        bridges_from = bridge_search([piece], new_end, new_pieces)
        print(max(map(lambda b: (len(b), bridge_strength(b)), bridges_from)))
