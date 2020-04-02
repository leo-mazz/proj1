import copy
import time
import math

from tqdm import tqdm

import quasi_identifiers
from quasi_identifiers import QuasiIdentifier
import experiment
import data_transform
import lattice

def add_k_minimal(node, k_min_set):
    """ Add node to k-minimal set, removing all higher nodes in path to leaf """
    to_remove = []

    for old_node in k_min_set:
        if node.has_descendant(old_node):
            to_remove.append(old_node)

    for doomed in to_remove:
        k_min_set.remove(doomed)

    k_min_set.add(node)
    

def compile_k_anonymous_check(logger=None):
    if logger is not None:
        info = logger.print
    else:
        info = print

    def are_k_anonymous_dict(records, node, k, max_sup):
        """ Check whether records are k-anonymous for some max suppression """
        # Using simple dict instead of 'collections' library: much better space requirements for Python > 3.6
        qi_values = lambda record: tuple([record[idx] for idx in node.root.rules.keys()])
        eq_classes = {}

        max_sup = int(len(records) * max_sup / 100)

        info('Making equivalence classes')
        for r in records:
            qi_signature = qi_values(r)
            if qi_signature in eq_classes.keys():
                eq_classes[qi_signature] +=1
            else:
                eq_classes[qi_signature] = 1

        info('Checking that all equivalence classes have size k')
        for val in eq_classes.values():
            if val < k:
                if max_sup < val:
                    info('--> was not k-anonymous')
                    return False
                else:
                    max_sup -= val

        info('--> was k-anonymous')
        return True
    
    return {'compute': are_k_anonymous_dict}


def k_min(b_node, t_node, k, max_sup, k_min_set=set()):
    """ Core of OLA's operation: build k-minimal set with binary search in generalization
    strategies of lattice """
    lattice_lvls = lattice.make(b_node, t_node)
    h = len(lattice_lvls)

    if h > 2:
        # look halfway between top and bottom node
        h = math.floor(h/2)
        for n in lattice_lvls[h]:
            if n.suitable_tag == True:
                k_min(b_node, n, k, max_sup, k_min_set)
            elif n.suitable_tag == False:
                k_min(n, t_node, k, max_sup, k_min_set)
            elif n.is_suitable(k, max_sup):
                n.set_suitable()
                k_min(b_node, n, k, max_sup, k_min_set)
            else:
                n.set_non_suitable()
                k_min(n, t_node, k, max_sup, k_min_set)

    else: # special case of a 2-node lattice
        if b_node.suitable_tag == False:
            n = t_node
        # It's not possible to know that b_node is k_anonymous. Otherwise it would have been selected as top
        # But it's possible across different strategies! (e.g.: if root is k-anonymous)
        elif b_node.suitable_tag == True:
            n = b_node
        elif b_node.is_suitable(k, max_sup):
            b_node.set_suitable()
            n = b_node
        else:
            b_node.set_non_suitable()
            n = t_node

        if n.suitable_tag == True:
            add_k_minimal(n, k_min_set)
        elif n.is_suitable(k, max_sup):
            n.set_suitable()
            add_k_minimal(n, k_min_set)

    return k_min_set


def make_release(records, qis, k):
    """ Finalize release by suppressing required records and producing some stats """
    original_size = len(records)
    qi_values = lambda record: tuple([record[idx] for idx in qis])

    eq_classes = {}
    release = []

    for r in records:
        qi_signature = qi_values(r)
        if qi_signature in eq_classes.keys():
            eq_classes[qi_signature].append(r)
        else:
            eq_classes[qi_signature] = [r]

    sup_ec = 0
    sup_rec = 0
    for val in eq_classes.values():
        if len(val) >= k:
            release += val
        else:
            sup_ec += 1
            sup_rec += len(val)

    stats = {
        'eq_classes_before_sup': len(eq_classes.keys()),
        'suppressed_classes': sup_ec,
        'suppressed_records': sup_rec,
        'perc_suppressed_records': round((sup_rec/original_size)*100, 2),
    }

    return release, stats

@experiment.step('ola')
def run(records, generalization_rules, k, max_sup, info_loss, logs=True):
    """ Execute OLA """
    l = experiment.Logger(active=logs)
    l.log_step('BUILDING LATTICE')

    k_anonymous_check = compile_k_anonymous_check(logger=l)
    b_node, t_node = lattice.Node.build_network(generalization_rules, records, k_anonymous_check, logger=l)

    l.log_step('SEARCHING LATTICE')
    k_min_nodes = k_min(b_node, t_node, k, max_sup)

    if len(k_min_nodes) == 0:
        # This cannot happen if, as they should, all generalization rules bring values to indistinguishability
        l.log_step('NO STRATEGY FOUND')
        return None

    l.print(f"visited {b_node.visited_nodes} nodes, checked {b_node.checked_nodes} nodes")
    l.print(f"num k {b_node.num_suitable} nodes, num not k {b_node.num_not_suitable} nodes")

    l.log_step('CHOOSING STRATEGY')

    if not info_loss.need_generalization:
        get_loss = lambda node: info_loss.compute(
            node.root.rules,
            node.gen_state,
        )
    else:
        get_loss = lambda node: info_loss.compute(
            node.root.rules,
            node.gen_state,
            records,
            node.apply_gen(),
        )

    losses = [(get_loss(node), node) for node in k_min_nodes]
    optimal_loss, optimal_node = min(losses, key=lambda x: x[0])

    l.log_step('GENERATING RELEASE with loss {}: {}'.format(optimal_loss, optimal_node))
    release, release_stats = make_release(optimal_node.apply_gen(), generalization_rules.keys(), k)



    stats = {
        'params': {
            'k': k,
            'max_sup': max_sup,
            'loss_metric': info_loss.name,
            'generalization_rules': {qi: rule.max_level for qi, rule in generalization_rules.items()},
        },
        'results': experiment.merge_stats(
            release_stats,
            {
                'info_loss': round(optimal_loss, 4),
                'node': str(optimal_node),
                'visited_nodes': b_node.visited_nodes,
                'checked_nodes': b_node.checked_nodes,
                'b_node.k_anonymous_nodes': b_node.num_suitable,
                'b_node.not_k_anonymous_nodes': b_node.num_not_suitable,
            }
        )
    }

    return [release, optimal_node.gen_state], stats