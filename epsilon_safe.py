import decimal

import numpy as np
from scipy.stats import binom

import experiment
import information_loss

def get_beta(epsilon_prime, epsilon):
    beta  = 1 - np.exp(epsilon_prime - epsilon)
    return beta

def parameters(k, beta, epsilon, size_d):
    """ Compute epsilon' AND the associated delta for a k """
    epsilon_prime = np.log(1-beta) + epsilon
    if epsilon_prime <= 0:
        return None, None # Try again, with higher epsilon or lower beta

    def big_d(k, beta, epsilon):
        gamma = (np.exp(epsilon) - 1 + beta) / (np.exp(epsilon))
        n_min = int(np.ceil((k/gamma) - 1))

        maxim = 0
        c = 0
        for n in range(n_min, size_d):
            c += 1
            summ = 0
            for j in range(int(np.ceil(gamma*n)), n+1):
                summ += binom.pmf(j, n, beta)

            if maxim < summ:
                maxim = summ
                        
            if c == 50: # estimate D to make evaluation tractable
                break

        return maxim
    
    return epsilon_prime, big_d(k, beta, epsilon-epsilon_prime)
        



def output_stats(nodes, max_sup=None, max_info_loss=None):
    """ Takes a distribution over output nodes and computes some probabilistic stats """
    average_info_loss = 0

    average_suppression = 0
    baseline_suppression = experiment.AverageMeter()
    min_suppression = 100
    max_suppression = 0

    good_nodes = 0
    baseline_good_nodes = 0

    for node in nodes:
        p = node['p']

        average_info_loss += p * node['info_loss']

        average_suppression += p * node['suppression']
        baseline_suppression.add(node['suppression'])
        min_suppression = min(min_suppression, node['suppression'])
        max_suppression = max(max_suppression, node['suppression'])
        
        if node['suppression'] <= max_sup and node['info_loss'] <= max_info_loss:
            average_suppression += p * node['suppression']
            baseline_good_nodes += 1
            good_nodes += node['p']

    baseline_good_nodes /= len(nodes)

    return average_info_loss, min_suppression, max_suppression, average_suppression, baseline_suppression.get(), good_nodes, baseline_good_nodes


def exp_mechanism(epsilon, utility, sensitivity):
    large = decimal.Decimal((epsilon * utility / (2 * sensitivity)))
    return np.exp(large)

def compute_utility(nodes, epsilon, penalty_factor, max_penalty, sensitivity, mode, gen_rules):
    """ Enrich list representing the nodes in a lattice with their utility and probability of being
    sampled by the exponential mechanism """
    penalty_prec = information_loss.make_penalty_prec(penalty_factor, max_penalty).compute

    total_p = 0
    total_utility = 0
    # Add penalty term to info loss
    for n in nodes:
        penalty = n['suppression'] if mode == 'suppression' else int(not n['k_anon_tag'])
        n['utility'] = penalty_prec(gen_rules, n['gen_state'], penalty)
        total_utility += n['utility']
    # Compute utility as the multiplicative inverse of penalized information loss
    for n in nodes:
        n['utility'] = 1 / n['utility']
    # Calculate unnormalized probabilities and compute norm. factor
    for n in nodes:
        n['p'] = exp_mechanism(epsilon, n['utility'], sensitivity)
        total_p += n['p']
    # Normalize probabilities
    for n in nodes:
        n['p'] /= total_p
        n['p'] = float(n['p'])
    

    return nodes
