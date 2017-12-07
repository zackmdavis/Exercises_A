def the_first_star():
    parents = []
    children = []

    for line in lines:
        if '->' in line:
            pre, post = line.split('->')
            base = pre[:pre.find(' ')]
            parents.append(base)
            childs = [c.strip() for c in post.split(',')]
            children.extend(childs)

    print(set(parents) - set(children))

# second star code follows (plus some manual tree traversal at the end)

class Program:
    def __init__(self, name, own_weight):
        self.name = name
        self.own_weight = own_weight
        self.children = []

    def __repr__(self):
        return "{} ({})".format(self.name, self.own_weight)

    def set_children(self, children):
        self.children = children

    def bears(self):
        b = 0
        for child in self.children:
            b += programs[child].own_weight
            b += programs[child].bears()
        return b

    def balance(self):
        c = {}
        for child in self.children:
            c[child] = programs[child].bears() + programs[child].own_weight
        return c


programs = {}

for line in lines:
    if '->' in line:
        pre, post = line.split('->')
        sp = pre.find(' ')
        op = pre.find('(')
        cl = pre.find(')')
        parent_name = pre[:sp]
        parent_weight = int(pre[op+1:cl])

        parent = Program(parent_name, parent_weight)
        parent.set_children([c.strip() for c in post.split(',')])

        programs[parent_name] = parent
    else:
        sp = line.find(' ')
        op = line.find('(')
        cl = line.find(')')
        prog_name = line[:sp]
        prog_weight = int(line[op+1:cl])

        programs[prog_name] = Program(prog_name, prog_weight)
