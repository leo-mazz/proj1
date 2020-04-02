from itertools import product

import experiment

def group_release(records, ks):
    """ Split records into equivalence classes based on a list of knowledge states """
    result = {}
    for state in ks:
        eq_classes = {}

        for r in records:
            r_signature = col_values(r, state)
            if r_signature in eq_classes.keys():
                eq_classes[r_signature].append(r)
            else:
                eq_classes[r_signature] = [r]
        result[state] = eq_classes

    return result

def col_values(r, ks):
    """ For a record, get only its values corresponding to known columns """
    known_col = [i for i, col in enumerate(ks) if col]
    vals = lambda record: tuple([record[idx] for idx in known_col])
    return vals(r)

def knowledge_states(no_attributes, qi, sa):
    """ Build all the possible combinations of known attributes. Assumes that all attributes not in 
    probs_knowing_sa are never known """
    states = list(product(*[[True, False]] * len(sa)))
    for i, _ in enumerate(states):
        known_sa = states[i]
        states[i] = [False] * no_attributes
        for q in qi:
            states[i][q] = True
        for j, known in enumerate(known_sa):
            states[i][sa[j]] = known
        
        states[i] = tuple(states[i])

    return states

def p_knowledge_state(k, probs_knowing):
    """ Probability of knowing a set of columns given the probability of knowing each of them """
    prod = 1
    for i in range(len(k)):
        if k[i]:
            prod *= probs_knowing[i]
        else:
            prod *= 1 - probs_knowing[i]

    return prod

def p_reid_in_state(record, p_inclusion, grouped_release, k):
    """ Compute the size of an equivalence class given a knowledge state """
    col_vals = col_values(record, k)
    crowd = len(grouped_release[col_vals])

    return (1 / crowd) * p_inclusion

def p_knowing_each_attribute(no_attributes, gen_rules, probs_knowing_sa):
    """ Turn the probability model of knowing sensitive attributes into an ordered list matching
    the schema of the database, including the probability of knowing quasi-identifiers, i.e.: 1"""
    result = [0] * no_attributes
    for qi in gen_rules:
        result[qi] = 1
    for sa, p in probs_knowing_sa.items():
        result[sa] = p

    return result

@experiment.step('m_concealing')
def prob_reid(release, prob_inclusion, probs_knowing_sa, gen_rules):
    """ Compute the m_concealing measure on a list of records, given a model of the probabiloty of knowing
    each sensitive column """
    no_attributes = len(release[0])
    k_states = knowledge_states(no_attributes, list(gen_rules.keys()), list(probs_knowing_sa.keys()))
    grouped_release = group_release(release, k_states)

    result = 0
    probs_knowing_col = p_knowing_each_attribute(no_attributes, gen_rules, probs_knowing_sa)

    for r in release:
        summation = 0
        terms = []
        for k in k_states:
            col_vals = col_values(r, k)
            crowd = len(grouped_release[k][col_vals])
            if crowd == 1:
                terms.append(grouped_release[k][col_vals])

            summation += p_knowledge_state(k, probs_knowing_col) * p_reid_in_state(r, prob_inclusion, grouped_release[k], k)
        if result < summation:
            result = summation
        
    result = 1/result

    return round(result, 4), {
        'params': {
            'prob_inclusion': prob_inclusion,
            'probs_knowing_sa': probs_knowing_sa,
        },
        'results': {
            'm': round(result, 4)
        }
    }