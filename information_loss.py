import numpy as np

class InformationLoss():
    def __init__(self, name, formula, need_generalization=True):
        self.name = name
        self.compute = formula
        self.need_generalization = need_generalization


def prec_formula(gen_rules, gen_status, other=None, stuff=None):
    loss = 0
    for qi_name, qi_gen in gen_status.items():
        loss += (qi_gen / gen_rules[qi_name].max_level)

    return loss

prec = InformationLoss('Prec', prec_formula, need_generalization=False)


def norm_prec_formula(gen_rules, gen_status, other=None, stuff=None):
    loss = 0
    for qi_name, qi_gen in gen_status.items():
        loss += (qi_gen / gen_rules[qi_name].max_level)

    return loss / len(gen_rules)

norm_prec = InformationLoss('NormPrec', norm_prec_formula, need_generalization=False)


def weighted_norm_prec_formula(gen_rules, gen_status, other=None, stuff=None, weights=None):
    if weights == None:
        weights = [1] * len(gen_rules)

    loss = 0
    for i, (qi_name, qi_gen) in enumerate(gen_status.items()):
        loss += (qi_gen / gen_rules[qi_name].max_level) * weights[i]

    return loss / sum(weights)

weighted_norm_prec = InformationLoss('WeightedNormPrec', weighted_norm_prec_formula, need_generalization=False)


def make_penalty_prec(penalty_factor, max_penalty):
    def penalty_prec_formula(gen_rules, gen_status, penalty):
        result = norm_prec_formula(gen_rules, gen_status)
        result += penalty_factor * penalty

        return result

    return InformationLoss('PenaltyPrec', penalty_prec_formula, need_generalization=False)


def dm_star_formula(gen_rules, gen_status, dataset, generalization):
    def group_generalization(records, qis):
        values = lambda record: tuple([record[idx] for idx in qis])

        eq_classes = {}
        for r in records:
            r_signature = values(r)
            if r_signature in eq_classes.keys():
                eq_classes[r_signature].append(r)
            else:
                eq_classes[r_signature] = [r]

        return eq_classes.values()

    grouped_generalization = group_generalization(generalization, gen_rules.keys())
    records_per_class = [len(ec) for ec in grouped_generalization]
    loss = 0
    for ec_size in records_per_class:
        loss += ec_size**2

    return loss

dm_star = InformationLoss('DM*', dm_star_formula)


def entropy_formula(gen_rules, gen_status, dataset, generalization):
    def update_freq(store, value):
        if value in store.keys():
            store[value] += 1
        else:
            store[value] = 1

    qis = gen_rules.keys()
    # Build frequency dictionaries
    freq_a = {}
    freq_b = {}
    for q in qis:
        freq_a[q] = {}
        freq_b[q] = {}
        for d in dataset:
            update_freq(freq_a[q], d[q])
        for r in generalization:
            update_freq(freq_b[q], r[q])

    summation = 0
    for q in qis:
        for i, d in enumerate(dataset):
            a = d[q]
            b = generalization[i][q]
            summation += np.log2(freq_a[q][a]/freq_b[q][b])


    return -summation

entropy = InformationLoss('Entropy', entropy_formula)