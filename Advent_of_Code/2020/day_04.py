import itertools
from collections import Counter

with open('data.txt') as f:
    in_content = f.read()
    in_lines = in_content.split('\n')


def parse_data():
    docs = []
    doc = {}
    for line in in_lines:
        if not line:
            docs.append(doc)
            doc = {}
            continue
        pairs = line.split(' ')
        for pair in pairs:
            k, v = pair.split(':')
            doc[k] = v
    return docs


def the_first_star():
    docs = parse_data()
    needed_fields = {'byr',
'iyr',
'eyr',
'hgt',
'hcl',
'ecl',
'pid',
'cid'}
    valid = 0
    for doc in docs:
        missing = needed_fields - set(doc.keys())
        if missing in [set(), {'cid'}]:
            valid += 1
    return valid


def ht_validator(h):
    if h.endswith("cm"):
        return 150 <= int(h[:-2]) <= 193
    elif h.endswith("in"):
        return 59 <= int(h[:-2]) <= 76

def hair_validator(c):
    return c.startswith("#") and all(h in "0123456789abcdef" for h in c[1:]) and len(c) == 7

def eye_validator(e):
    return e in ['amb', 'blu', 'brn', 'gry', 'grn', 'hzl', 'oth']

def pid_validator(p):
    return all(c in "0123456789" for c in p) and len(p) == 9

validators = {
    'byr': lambda v: len(v) == 4 and (1920 <= int(v) <= 2002),
    'iyr': lambda v: len(v) == 4 and (2010 <= int(v) <= 2020),
    'eyr': lambda v: len(v) == 4 and (2020 <= int(v) <= 2030),
    'hgt': ht_validator,
    'hcl': hair_validator,
    'ecl': eye_validator,
    'pid': pid_validator,
}

def validate_doc(doc):
    for key, validator in validators.items():
        if not validator(doc[key]):
            return False
    return True


valids = [
    {"pid":"087499704", "hgt":"74in", "ecl":"grn", "iyr":"2012", "eyr":"2030", "byr":"1980", "hcl":"#623a2f"}
]

invalids = [
    {"eyr":"1972", "cid":"100", "hcl":"#18171d", "ecl":"amb", "hgt":"170", "pid":"186cm", "iyr":"2018", "byr":"1926"},
    {'iyr':'2019', 'hcl':'#602927', 'eyr':'1967', 'hgt':'170cm', 'ecl':'grn', 'pid':'012533040', 'byr':'1946'},

    {'hgt':'59cm', 'ecl':'zzz', 'eyr':'2038', 'hcl':'74454a', 'iyr':'2023', 'pid':'3556412378', 'byr':'2007'}
]

assert all(validate_doc(vdoc) for vdoc in valids)
assert all(not validate_doc(idoc) for idoc in invalids)


def the_second_star():
    docs = parse_data()
    needed_fields = {'byr',
'iyr',
'eyr',
'hgt',
'hcl',
'ecl',
'pid',
'cid'}

    valid = 0

    for doc in docs:
        missing = needed_fields - set(doc.keys())
        if missing in [set(), {'cid'}]:
            if validate_doc(doc):
                valid += 1

    return valid


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
