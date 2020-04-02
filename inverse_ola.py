import copy
import time
import math

from tqdm import tqdm

import quasi_identifiers
from quasi_identifiers import QuasiIdentifier
import experiment
import data_transform
import lattice

def add_info_loss_minimal(node, info_loss_min_set):
    to_remove = []

    for old_node in info_loss_min_set:
        if old_node.has_descendant(node):
            to_remove.append(old_node)

    for doomed in to_remove:
        info_loss_min_set.remove(doomed)

    info_loss_min_set.add(node)
    

def compile_info_loss_check(metric, weights=None, logger=None):
    if logger is not None:
        info = logger.print
    else:
        info = print

    def info_loss(release, node, max_loss):
        loss = metric.compute(
            node.root.rules,
            node.gen_state,
            node.root.records,
            release,
            weights=weights,
        )

        info(f'loss {loss}')

        if loss <= max_loss:
            return True
        else:
            return False
    
    return {'compute': info_loss, 'need_generalization': metric.need_generalization}

node_count = 0
def info_loss_min(b_node, t_node, info_loss, info_loss_min_set=set()):
    lattice_lvls = lattice.make(b_node, t_node)
    h = len(lattice_lvls)

    if h > 2:
        # look halfway between top and bottom node
        h = math.floor(h/2)
        for n in lattice_lvls[h]:
            if n.suitable_tag == True:
                info_loss_min(n, t_node, info_loss, info_loss_min_set)
            elif n.suitable_tag == False:
                info_loss_min(b_node, n, info_loss, info_loss_min_set)
            elif n.is_suitable(info_loss):
                n.set_suitable()
                info_loss_min(n, t_node, info_loss, info_loss_min_set)
            else:
                n.set_non_suitable()
                info_loss_min(b_node, n, info_loss, info_loss_min_set)

    else: # special case of a 2-node lattice
        if t_node.suitable_tag == False:
            n = b_node
        elif t_node.suitable_tag == True:
            n = t_node
        elif t_node.is_suitable(info_loss):
            t_node.set_suitable()
            n = t_node
        else:
            t_node.set_non_suitable()
            n = b_node

        if n.suitable_tag == True:
            add_info_loss_minimal(n, info_loss_min_set)
        elif n.is_suitable(info_loss):
            n.set_suitable()
            add_info_loss_minimal(n, info_loss_min_set)

    return info_loss_min_set


def make_release(records, qis, k):
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
    

@experiment.step('inverted-ola')
def run(records, generalization_rules, max_loss, max_sup, info_loss, weights=None, logs=True):
    l = experiment.Logger(active=logs)
    l.log_step('BUILDING LATTICE')

    info_loss_check = compile_info_loss_check(info_loss, weights=weights, logger=l)
    b_node, t_node = lattice.Node.build_network(generalization_rules, records, info_loss_check,
        suitable_upwards=False,  # Unlike OLA, predictively tag towards the bottom
        logger=l)

    l.log_step('SEARCHING LATTICE')
    info_loss_max = info_loss_min(b_node, t_node, max_loss)


    if len(info_loss_max) == 0:
        l.log_step('NO STRATEGY FOUND')
        return None

    l.print(f"visited {b_node.visited_nodes} nodes, checked {b_node.checked_nodes} nodes")
    l.print(f"{b_node.num_suitable} good nodes, {b_node.num_not_suitable} bad nodes")

    l.log_step('CHOOSING STRATEGY')


    def get_k(node, records, max_sup):
        max_sup = int(len(records) * max_sup / 100)

        qis = node.root.rules.keys()
        records = node.apply_gen()

        qi_values = lambda record: tuple([record[idx] for idx in qis])
        eq_classes = {}

        for r in records:
            qi_signature = qi_values(r)
            if qi_signature in eq_classes.keys():
                eq_classes[qi_signature] +=1
            else:
                eq_classes[qi_signature] = 1

        sizes = sorted(eq_classes.values())
        
        min_k = len(records)

        for s in sizes:
            if s > max_sup:
                min_k = s
                break
            else:
                max_sup -= s


        return min_k


    losses = [(get_k(node, records, max_sup), node) for node in info_loss_max]
    optimal_k, optimal_node = max(losses, key=lambda x: x[0])

    l.log_step('GENERATING RELEASE with k {}: {}'.format(optimal_k, optimal_node))
    release = optimal_node.apply_gen()
    release, release_stats = make_release(optimal_node.apply_gen(), generalization_rules.keys(), optimal_k)

    stats = {
        'params': {
            'max_loss': max_loss,
            'max_sup': max_sup,
            'loss_metric': info_loss.name,
            'generalization_rules': {qi: rule.max_level for qi, rule in generalization_rules.items()},
        },
        'results': experiment.merge_stats(release_stats, {
            'k': optimal_k,
            'node': str(optimal_node),
            'visited_nodes': b_node.visited_nodes,
            'checked_nodes': b_node.checked_nodes,
            'b_node.good_info_loss_nodes': b_node.num_suitable,
            'b_node.bad_info_loss_nodes': b_node.num_not_suitable,
        })
    }

    return [release, optimal_node.gen_state], stats
