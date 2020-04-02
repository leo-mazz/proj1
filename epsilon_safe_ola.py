import copy
import time
import math

import numpy as np
from tqdm import tqdm

import quasi_identifiers
from quasi_identifiers import QuasiIdentifier
import experiment
import data_transform
import information_loss
import lattice


def add_k_minimal(node, k_min_set):
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

    def are_k_anonymous_dict(records, qis, k, max_sup):
        qi_values = lambda record: tuple([record[idx] for idx in qis])
        eq_classes = {}

        k_anon = True
        suppression = 0

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
                    k_anon = False
                else:
                    max_sup -= val
                suppression += val

        return k_anon, suppression/len(records)*100
    
    return are_k_anonymous_dict


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

class LatticeDistribution():
    def __init__(self):
        self.nodes = {}

    def add_node(self, node):
        self.nodes[node] = {}
    
    def enrich_node(self, node, k_anon_tag, suppression, info_loss):
        self.nodes[node] = {
            'gen_state': node.gen_state,
            'k_anon_tag': k_anon_tag,
            'suppression': suppression,
            'info_loss': info_loss,
        }
    
    def list_nodes(self):
        return self.nodes.keys()

    def get(self):
        return list(self.nodes.values())


@experiment.step('epsilon-safe-ola')
def run(records, generalization_rules, k, max_sup, logs=True):
    prec = information_loss.norm_prec.compute

    l = experiment.Logger(active=logs)
    l.log_step('BUILDING LATTICE')

    k_anonymous_check = compile_k_anonymous_check(logger=l)
    b_node, t_node = lattice.Node.build_network(generalization_rules, records, k_anonymous_check, logger=l)

    l.print(f"visited {b_node.visited_nodes} nodes, checked {b_node.checked_nodes} nodes")
    l.print(f"num k {b_node.num_suitable} nodes, num not k {b_node.num_not_suitable} nodes")

    distribution = LatticeDistribution()
    def build_node_list(node):
        distribution.add_node(node)
        for c in node.children:
            build_node_list(c)
    build_node_list(b_node)

    for n in distribution.list_nodes():
        info_loss = prec(n.root.rules, n.gen_state)
        k_anon_tag, suppression = k_anonymous_check(n.apply_gen(), generalization_rules.keys(), k, max_sup)
        distribution.enrich_node(n, k_anon_tag, suppression, info_loss)



    return distribution.get(), {}