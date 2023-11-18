import itertools
from collections import Counter

with open('input_07.txt') as f:
    in_content = f.read()
    in_lines = in_content.split('\n')


class Directory:
    def __init__(self, name, parent = None):
        print("creating {}".format(name))
        self.name = name
        self.files = {}
        self.subdirectories = {}
        self.parent = parent


# What is wrong with my head?! This is a mess and unacceptable for a professional!!
def the_first_star(lines):
    root = Directory("/")
    cwd = root
    lines = iter(lines)
    line = next(lines)
    while True:
        print("top of loop, cwd is {}".format(cwd.name))
        args = line.split()
        if args[0] == '$' and args[1] == "cd":
            subdir_name = args[2]
            # current directory: no-op
            if cwd.name == subdir_name:
                pass
            # ..
            elif subdir_name == "..":
                cwd = cwd.parent
            else:
                # existing subdirectory
                if subdir := cwd.subdirectories.get(subdir_name):
                    cwd = subdir
                # unknown subdirectory
                else:
                    subdir = Directory(subdir_name, parent=cwd)
                    cwd = subdir
            try:
                line = next(lines)
            except StopIteration:
                break
        elif args[0] == '$' and args[1] == "ls":
            listing = True
            while listing:
                try:
                    line = next(lines)
                except StopIteration:
                    break
                if line[0] == '$':
                    break
                entry = line.split()
                if entry[0] == "dir":
                    subdir_name = args[1]
                    cwd.subdirectories[subdir_name] = Directory(subdir_name, parent=cwd)
                else:
                    size, filename = entry
                    cwd.files[filename] = size


def the_second_star():
    ...


if __name__ == "__main__":
    print(the_first_star(in_lines))
    print(the_second_star())
