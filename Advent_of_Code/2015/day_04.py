import itertools
from collections import Counter

import hashlib

key = "iwrupvqb"

def the_first_star():
    i = 0
    while True:
        # I was trying to shell out with subprocess here, and was annoyed that
        # I couldn't figure out the right command invocation with just `help`,
        # but when I broke down and asked Claude, it points out that `hashlib`
        # has MD5
        h = hashlib.md5((key+str(i)).encode('utf-8')).hexdigest()
        if h[:5] == '00000':
            return i
        i += 1


def the_second_star():
    i = 0
    while True:
        # I was trying to shell out with subprocess here, and was annoyed that
        # I couldn't figure out the right command invocation with just `help`,
        # but when I broke down and asked Claude, it points out that `hashlib`
        # has MD5
        h = hashlib.md5((key+str(i)).encode('utf-8')).hexdigest()
        if h[:6] == '000000':
            return i
        i += 1



if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
