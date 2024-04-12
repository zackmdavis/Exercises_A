
base = "CRnCaCaCaSiRnBPTiMgArSiRnSiRnMgArSiRnCaFArTiTiBSiThFYCaFArCaCaSiThCaPBSiThSiThCaCaPTiRnPBSiThRnFArArCaCaSiThCaSiThSiRnMgArCaPTiBPRnFArSiThCaSiRnFArBCaSiRnCaPRnFArPMgYCaFArCaPTiTiTiBPBSiThCaPTiBPBSiRnFArBPBSiRnCaFArBPRnSiRnFArRnSiRnBFArCaFArCaCaCaSiThSiThCaCaPBPTiTiRnFArCaPTiBSiAlArPBCaCaCaCaCaSiRnMgArCaSiThFArThCaSiThCaSiRnCaFYCaSiRnFYFArFArCaSiRnFYFArCaSiRnBPMgArSiThPRnFArCaSiRnFArTiRnSiRnFYFArCaSiRnBFArCaSiRnTiMgArSiThCaSiThCaFArPRnFArSiRnFArTiTiTiTiBCaCaSiRnCaCaFYFArSiThCaPTiBPTiBCaSiThSiRnMgArCaF"

replacements = {
    'Al': ['ThF', 'ThRnFAr'],
    'B': ['BCa', 'TiB', 'TiRnFAr'],
    'Ca': ['CaCa', 'PB', 'PRnFAr', 'SiRnFYFAr', 'SiRnMgAr', 'SiTh'],
    'F': ['CaF', 'PMg', 'SiAl'],
    'H': [
        'CRnAlAr',
        'CRnFYFYFAr',
        'CRnFYMgAr',
        'CRnMgYFAr',
        'HCa',
        'NRnFYFAr',
        'NRnMgAr',
        'NTh',
        'OB',
        'ORnFAr',
    ],
    'Mg': ['BF', 'TiMg'],
    'N': ['CRnFAr', 'HSi'],
    'O': ['CRnFYFAr', 'CRnMgAr', 'HP', 'NRnFAr', 'OTi'],
    'P': ['CaP', 'PTi', 'SiRnFAr'],
    'Si': ['CaSi'],
    'Th': ['ThCa'],
    'Ti': ['BP', 'TiTi'],
    'e': ['HF', 'NAl', 'OMg'],
}


def the_first_star():
    # sliding window of length 1 or 2
    products = set()
    for i in range(len(base)):
        if substitutions := replacements.get(base[i]):
            for substitution in substitutions:
                products.add(base[:i] + substitution + base[i+1:])
        if substitutions := replacements.get(base[i:i+2]):
            for substitution in substitutions:
                products.add(base[:i] + substitution + base[i+2:])
    return len(products)


def the_second_star():
    # TODO: this is nontrivial! It can be formulated as a graph search, though.
    ...


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
