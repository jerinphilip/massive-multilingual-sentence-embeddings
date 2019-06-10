from numbers import Number
from collections import OrderedDict

def _format(val):
    default = lambda x: x
    cases = OrderedDict([
        (float, lambda x: '{:3f}'.format(x)),
        (int, lambda x: '{:.2E}'.format(x)),
    ])
        
    for key in cases:
        if isinstance(val, key):
            return cases[key](val)
    return default(val)



def prettified(progress_dict):
    entries = ['{}:{}'.format(key, _format(value)) \
            for key, value in progress_dict.items()]
    fmted = '|'.join(entries)
    return fmted
