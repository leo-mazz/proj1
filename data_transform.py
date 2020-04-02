import random
import copy

import pandas as pd

def split_train_test(records, test_perc=90, seed=None):
    """ Divide dataset into training and test sets """
    if seed:
        random.seed(seed)

    random.shuffle(records)

    n = len(records)
    n_train = int(n * (test_perc/100))

    return records[:n_train], records[n_train:]

def apply_gen(records, gen_state, gen_rules):
    """ Apply to some records the state of a generalization node """
    gen_records = copy.deepcopy(records)
    for r in gen_records:
        for col, gen_level in gen_state.items():
            r[col] = gen_rules[col].apply(r[col], gen_level)

    return gen_records