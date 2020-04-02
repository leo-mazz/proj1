import time
import os
import pickle

import numpy as np

import epsilon_safe_ola as ela
import epsilon_safe as es
import quasi_identifiers
import information_loss
from dataset import adult
import experiment


experiment_root = os.path.expanduser('~/Documents/uni/dissertation/results/p3.2/')

gen_rules = {
    0: quasi_identifiers.generalize_age_rule,
    1: quasi_identifiers.adult_generalize_workclass_rule,
    3: quasi_identifiers.adult_generalize_education_rule,
    5: quasi_identifiers.adult_generalize_marital_status_rule,
    # 6: quasi_identifiers.adult_generalize_occupation_rule,
    # 7: quasi_identifiers.suppress_rule,
    # 8: quasi_identifiers.suppress_rule,
    # 9: quasi_identifiers.suppress_rule,
    # 13: quasi_identifiers.adult_generalize_country_rule
}

records = adult.retrieve()
draws_file = os.path.join(experiment_root, 'draws.pickle')
lattice_file = os.path.join(experiment_root, 'lattice.pickle')
betas = [0.05, 0.25, 0.45, 0.75]

def draw_from_adult():
    all_draws = {}
    for beta in betas:
        all_draws[beta] = []
        for i in range(10):
            experiments = np.random.binomial(1, beta, len(records))
            draw = [r for j, r in enumerate(records) if experiments[j] == 1]
            all_draws[beta].append(draw)


        with open(draws_file, 'wb') as handle:
            pickle.dump(all_draws, handle)



def build_rich_lattice():
    with open(draws_file, 'rb') as handle:
        all_draws = pickle.load(handle)

    k = 50
    max_sup = 5

    lattices = []

    beta = 0.75
    for input_draw in all_draws[beta]:
        lattice, _ = ela.run(input_draw, gen_rules, k=k, max_sup=max_sup, logs=False)
        lattices.append(lattice)


    with open(lattice_file, 'wb') as handle:
        pickle.dump(lattices, handle)


def make_data_points():
    with open(lattice_file, 'rb') as handle:
        lattices = pickle.load(handle)
    
    with open(draws_file, 'rb') as handle:
        draws = pickle.load(handle)[0.75]

    penalties = [0.002, 0.005, 0.01, 0.02, 0.025, 0.04, 0.05, 0.1, 0.2, 0.5]
    epsilons = [0.01, 0.05, 0.1, 0.5]

    n = 0

        
    for epsilon in epsilons:
        for penalty in penalties:
            n += 1
            avg_average_info_loss = experiment.AverageMeter()
            avg_average_suppression = experiment.AverageMeter()
            avg_baseline_suppression = experiment.AverageMeter()
            avg_good_nodes = experiment.AverageMeter()
            avg_baseline_good_nodes = experiment.AverageMeter()
            avg_min_suppression = experiment.AverageMeter()
            avg_max_suppression = experiment.AverageMeter()


            for l, d in zip(lattices, draws):
                sensitivity = penalty * (100 / len(d))
                nodes = es.compute_utility(l, epsilon, penalty, len(d), sensitivity, 'suppression', gen_rules)
                probs = [(n['p']) for n in nodes]
                average_info_loss, min_suppression, max_suppression, average_suppression, suppression_baseline, good_nodes, good_nodes_baseline = es.output_stats(nodes, max_sup=5, max_info_loss=0.5)
                avg_average_info_loss.add(average_info_loss)
                avg_average_suppression.add(average_suppression)
                avg_baseline_suppression.add(suppression_baseline)
                avg_good_nodes.add(good_nodes)
                avg_baseline_good_nodes.add(good_nodes_baseline)
                avg_min_suppression.add(min_suppression)
                avg_max_suppression.add(max_suppression)

            
            stats = {
                'params': {
                    'mode': 'suppression',
                    'penalty': penalty,
                    'epsilon\'': epsilon,
                },
                'results': {
                    'avg_info_loss': avg_average_info_loss.get(),
                    'min_suppression': avg_min_suppression.get(),
                    'max_suppression': avg_max_suppression.get(),
                    'avg_suppression': avg_average_suppression.get(),
                    'uniform_suppression': avg_baseline_suppression.get(),
                    'good_nodes': avg_good_nodes.get(),
                    'uniform_good_nodes': avg_baseline_good_nodes.get()
                }
            }

            experiment.publish_stats(stats, experiment_root, nonce=n)


    epsilons = [0.01, 0.05, 0.1, 0.5, 1, 2]
    for epsilon in epsilons:    
        penalties = [0.002, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9]
        for penalty in penalties:
            n += 1
            sensitivity = penalty

            avg_average_info_loss = experiment.AverageMeter()
            avg_average_suppression = experiment.AverageMeter()
            avg_baseline_suppression = experiment.AverageMeter()
            avg_good_nodes = experiment.AverageMeter()
            avg_baseline_good_nodes = experiment.AverageMeter()
            avg_min_suppression = experiment.AverageMeter()
            avg_max_suppression = experiment.AverageMeter()

            for l, d in zip(lattices, draws):
                nodes = es.compute_utility(l, epsilon, penalty, len(d), sensitivity, 'k_anon', gen_rules)
                probs = [(n['p']) for n in nodes]
                average_info_loss, min_suppression, max_suppression, average_suppression, suppression_baseline, good_nodes, good_nodes_baseline = es.output_stats(nodes, max_sup=5, max_info_loss=0.5)
                avg_average_info_loss.add(average_info_loss)
                avg_average_suppression.add(average_suppression)
                avg_baseline_suppression.add(suppression_baseline)
                avg_good_nodes.add(good_nodes)
                avg_baseline_good_nodes.add(good_nodes_baseline)
                avg_min_suppression.add(min_suppression)
                avg_max_suppression.add(max_suppression)

            
            stats = {
                'params': {
                    'mode': 'kanon',
                    'penalty': penalty,
                    'epsilon\'': epsilon,
                },
                'results': {
                    'avg_info_loss': avg_average_info_loss.get(),
                    'min_suppression': avg_min_suppression.get(),
                    'max_suppression': avg_max_suppression.get(),
                    'avg_suppression': avg_average_suppression.get(),
                    'uniform_suppression': avg_baseline_suppression.get(),
                    'good_nodes': avg_good_nodes.get(),
                    'uniform_good_nodes': avg_baseline_good_nodes.get()
                }
            }

            experiment.publish_stats(stats, experiment_root, nonce=n)

        


def make_graph():
    epsilons = {'suppression': [0.01, 0.05, 0.1, 0.5], 'kanon': [0.01, 0.05, 0.1, 0.5, 1, 2]}
    for mode, e_vals in epsilons.items():
        for feature, baseline in [('good_nodes', 'uniform_good_nodes'), ('avg_suppression', 'uniform_suppression')]:
            all_exp = {}
            for epsilon in e_vals:
                exp = experiment.look_for_stats(experiment_root, params={'epsilon\'': epsilon, 'mode': mode})
                exp = experiment.filter_many_stats(exp, params=['penalty'], results=[feature])
                exp = sorted([tuple(point.values()) for point in exp])
                all_exp[epsilon] = exp

            uniform = experiment.look_for_stats(experiment_root, params={'epsilon\'': epsilon, 'mode': mode})
            uniform = experiment.filter_many_stats(uniform, params=['penalty'], results=[baseline])
            uniform = sorted([tuple(point.values()) for point in uniform])


            
            filename = os.path.expanduser(f'~/Documents/uni/dissertation/results/p3.2/{mode}-{epsilon}-{feature}.pdf')

            experiment.graph_points({
                'uniform distribution': uniform,
                **all_exp,
            }, 'penalty', 'good nodes', baseline=0, filename=filename)



# draw_from_adult()
# build_rich_lattice()
# make_data_points()
# make_graph()