import time
import os

from tqdm import tqdm

import ola
import quasi_identifiers
import information_loss
import m_concealing
from dataset import adult
import experiment


experiment_root = os.path.expanduser('~/Documents/uni/dissertation/results/p2.1/')

gen_rules = {
    0: quasi_identifiers.generalize_age_rule,
    1: quasi_identifiers.adult_generalize_workclass_rule,
    3: quasi_identifiers.adult_generalize_education_rule,
    5: quasi_identifiers.adult_generalize_marital_status_rule,
    6: quasi_identifiers.adult_generalize_occupation_rule,
    7: quasi_identifiers.suppress_rule,
    8: quasi_identifiers.suppress_rule,
    9: quasi_identifiers.suppress_rule,
    13: quasi_identifiers.adult_generalize_country_rule
}

records = adult.retrieve()

def make_data_points():

    ks = [5, 25, 100]
    max_sups = [0, 1, 5, 10, 20, 50]
    losses = [information_loss.prec, information_loss.dm_star, information_loss.entropy]
    
    c = 0
    for k in tqdm(ks):
        for sup in tqdm(max_sups):
            for loss in tqdm(losses):
                [release, gen], ola_stats = ola.run(records, gen_rules, k=k, max_sup=sup, info_loss=loss, logs=False)
                experiment.publish_stats(ola_stats, experiment_root)


def make_graph():
    for loss_metric in ['Prec', 'DM*', 'Entropy']:
            
        k5 = experiment.look_for_stats(experiment_root, params={'k': 5, 'loss_metric': loss_metric})
        k5 = experiment.filter_many_stats(k5, params=['max_sup'], results=['info_loss'])
        k5 = sorted([tuple(point.values()) for point in k5])

        k25 = experiment.look_for_stats(experiment_root, params={'k': 25, 'loss_metric': loss_metric})
        k25 = experiment.filter_many_stats(k25, params=['max_sup'], results=['info_loss'])
        k25 = sorted([tuple(point.values()) for point in k25])

        k100 = experiment.look_for_stats(experiment_root, params={'k': 100, 'loss_metric': loss_metric})
        k100 = experiment.filter_many_stats(k100, params=['max_sup'], results=['info_loss'])
        k100 = sorted([tuple(point.values()) for point in k100])

        scales = ['linear', 'log']
        n = 1

        # Normalize Prec
        if loss_metric == 'Prec':
            k5 = [(sup, loss/9) for sup, loss in k5]
            k25 = [(sup, loss/9) for sup, loss in k25]
            k100 = [(sup, loss/9) for sup, loss in k100]
        # Consider the difference with the root node for DM*
        elif loss_metric == 'DM*':
            n = 2
            initial_dm_star = information_loss.dm_star_formula(gen_rules, None, records, records)
            k5 = [(sup, loss-initial_dm_star) for sup, loss in k5]
            k25 = [(sup, loss-initial_dm_star) for sup, loss in k25]
            k100 = [(sup, loss-initial_dm_star) for sup, loss in k100]

        
        for i in range(n):
            filename = os.path.expanduser(f'~/Documents/uni/dissertation/results/p2.1/{loss_metric}-{scales[i]}.pdf')

            experiment.graph_points({
                'k=5': k5,
                'k=25': k25,
                'k=100': k100,
            }, 'max suppression', 'information loss', scale=scales[i], filename=filename)



def make_heatmap():
    import seaborn as sns
    from matplotlib import pyplot as plt
    import numpy as np
    import pandas as pd

    fig = plt.figure()


    checked = experiment.look_for_stats(experiment_root, params={'loss_metric': 'Prec'})
    checked = experiment.filter_many_stats(checked, params=['k', 'max_sup'], results=['checked_nodes'])
    checked = {
        'k': [point['params.k'] for point in checked],
        'max sup': [point['params.max_sup'] for point in checked],
        'checked nodes': [point['results.checked_nodes'] for point in checked],
    }

    checked = pd.pandas.DataFrame.from_dict(checked)
    checked = checked.pivot('k', 'max sup', 'checked nodes')
    print(checked)
    r = sns.heatmap(checked, cmap='YlGnBu')
    plt.show()


# make_data_points()
# make_graph()
# make_heatmap()