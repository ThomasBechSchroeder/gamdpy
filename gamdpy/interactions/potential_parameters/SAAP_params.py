""" SAAP potential parameters """
SAAP_Deiters2019_params = {
    'comment':
        'SAAP potential parameters for He, Ne, Ar, Kr, and Xe. The as are in units of eps and sig.',
    'reference': {
        'title': 'Two-body interatomic potentials for He,  Ne,  Ar,  Kr, and Xe from ab initio data',
        'volume': 150,
        'ISSN': '1089-7690',
        'DOI': '10.1063/1.5085420',
        'url': 'http://dx.doi.org/10.1063/1.5085420',
        'number': 13,
        'journal': 'The Journal of Chemical Physics',
        'shortjournal': 'J. Chem. Phys.',
        'author': ['Ulrich K. Deiters', 'Richard J. Sadus'],
        'year': 2019,
        'month': 'apr'
    },
    'units': {
        'M': 'g/mol',
        'eps': 'K',
        'sig': 'nm',
    },
    'Ne': {
        'name': 'Neon',
        'Z': 10,  # Atomic number
        'M': 20.1797, #  g/mol
        'eps': 42.36165080,  # in K
        'sig': 0.2759124561,  # in nm
        'a0': 211781.8544,  # a's are in units of eps and sig
        'a1': -10.89769496,
        'a2': -20.94225988,
        'a3': -2.317079421,
        'a4': -1.854049559,
        'a5': 0.7454617542
    },
    'Ar': {
        'name': 'Argon',
        'Z': 18,
        'M': 39.948,
        'eps': 143.4899372,
        'sig': 0.3355134529,
        'a0': 65214.64725,
        'a1': -9.452343340,
        'a2': -19.42488828,
        'a3': -1.958381959,
        'a4': -2.379111084,
        'a5': 1.051490962
    },
    'Kr': {
        'name': 'Krypton',
        'Z': 36,
        'M': 83.798,
        'eps': 201.0821392,
        'sig': 0.357999364,
        'a0': 60249.13228,
        'a1': -9.456080572,
        'a2': -24.40996013,
        'a3': -2.18227961,
        'a4': -1.959180470,
        'a5': 0.874092399
    },
    'Xe': {
        'name': 'Xenon',
        'Z': 54,
        'M': 131.293,
        'eps': 280.1837503,
        'sig': 0.3901195551,
        'a0': 44977.3164,
        'a1': -9.121814449,
        'a2': -29.63636182,
        'a3': -2.278991444,
        'a4': -1.876430370,
        'a5': 0.8701531593
    }
}
