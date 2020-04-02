import time
import os

from tqdm import tqdm

import ola
import quasi_identifiers
import information_loss
import m_concealing
from dataset import adult
import experiment
import data_transform
import predict_adult


experiment_root = os.path.expanduser('~/Documents/uni/dissertation/results/p2.2/')

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

def train_full():
    train, test = adult.get_frames()

    acc, adult_learning_stats = predict_adult.train_and_get_accuracy(train, test)

    experiment.publish_stats(adult_learning_stats, experiment_root)

def make_data_points():
    train, test = adult.get_frames()

    ks = [100]
    max_sups = [0, 10, 50]
    losses = [information_loss.prec, information_loss.dm_star, information_loss.entropy]

    train = adult.make_list(train)

    tqdm = lambda x: x
    
    for k in tqdm(ks):
        for sup in tqdm(max_sups):
            for loss in tqdm(losses):
                [anon_train, gen], ola_stats = ola.run(train, gen_rules, k=k, max_sup=sup, info_loss=loss, logs=False)

                anon_train = adult.make_frame(anon_train)
                gen_test = adult.make_frame(data_transform.apply_gen(adult.make_list(test), gen, gen_rules))

                acc, adult_learning_stats = predict_adult.train_and_get_accuracy(anon_train, gen_test)

                stats = experiment.merge_stats(ola_stats, adult_learning_stats)
                experiment.publish_stats(stats, experiment_root)

    
def graph_loss_accuracy():
    prec = experiment.look_for_stats(experiment_root, params={'loss_metric': 'Prec'})
    prec = experiment.filter_many_stats(prec, results=['info_loss', 'adult_test_accuracy'])
    prec = sorted([tuple(point.values()) for point in prec])
    # Normalize Prec for comparison
    prec = [(loss/9, acc) for loss, acc in prec]

    dm_star = experiment.look_for_stats(experiment_root, params={'loss_metric': 'DM*'})
    dm_star = experiment.filter_many_stats(dm_star, results=['info_loss', 'adult_test_accuracy'])
    dm_star = sorted([tuple(point.values()) for point in dm_star])
    # Consider the difference with the root node for DM*
    train, _ = adult.get_frames()
    records = adult.make_list(train)
    initial_dm_star = information_loss.dm_star_formula(gen_rules, None, records, records)
    dm_star = [(loss-initial_dm_star, acc) for loss, acc in dm_star]

    entropy = experiment.look_for_stats(experiment_root, params={'loss_metric': 'Entropy'})
    entropy = experiment.filter_many_stats(entropy, results=['info_loss', 'adult_test_accuracy'])
    entropy = sorted([tuple(point.values()) for point in entropy])


    experiment.graph_points({
        'Prec': prec,
    }, 'information loss', 'accuracy', filename=os.path.join(experiment_root, 'Prec.pdf'))

    experiment.graph_points({
        'DM*': dm_star,
    }, 'information loss', 'accuracy', filename=os.path.join(experiment_root, 'DM*.pdf'))

    experiment.graph_points({
        'Entropy': entropy,
    }, 'information loss', 'accuracy', filename=os.path.join(experiment_root, 'Entropy.pdf'))

# train_full()
# make_data_points()
# graph_loss_accuracy()
