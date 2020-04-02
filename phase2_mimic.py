import time
import os
import pickle

import numpy as np

import inverse_ola
import quasi_identifiers
import information_loss
import m_concealing
from dataset import mimic
import experiment


experiment_root = os.path.expanduser('~/Documents/uni/dissertation/results/p2.3/')
age_sample_file = os.path.join(experiment_root, 'age_sample.pickle')

gen_rules = {
    0: quasi_identifiers.mimic_generalize_insurance_rule,        # insurance
    1: quasi_identifiers.suppress_rule,                          # language
    2: quasi_identifiers.suppress_rule,                          # religion
    3: quasi_identifiers.mimic_generalize_marital_status_rule,   # marital status
    4: quasi_identifiers.suppress_rule,                          # gender
    5: quasi_identifiers.mimic_generalize_dob_rule,              # dob
    6: quasi_identifiers.mimic_generalize_proc_rule,             # procedure 1
    7: quasi_identifiers.mimic_generalize_proc_rule,             # procedure 2
    8: quasi_identifiers.mimic_generalize_proc_rule,             # procedure 3
}

# weights = [1, 1, 1, 2, 2, 2]
weights = None

demographics = mimic.get_patient_demographics()

def procedures_by_seq_num(seq_nums):
    # Get only interesting columns
    proc = mimic.get_procedures()[['HADM_ID', 'ICD9_CODE', 'SEQ_NUM']]
    # Get only sequence numbers we want
    proc = proc.loc[proc['SEQ_NUM'].isin(seq_nums)]
    # Stack horizontally sequence numbers and remove nans
    proc = proc.pivot(index='HADM_ID', columns='SEQ_NUM', values='ICD9_CODE')
    
    return proc

seq_num = [1,2,3]
proc = procedures_by_seq_num(seq_num)
joint_table = demographics.merge(proc,left_on=['HADM_ID'],right_on=['HADM_ID'], how='inner')
joint_table = joint_table[['INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'GENDER', 'DOB', *seq_num]].dropna()

def describe_records():
    columns = joint_table.columns
    for i, c in enumerate(columns):
        print(i,c)

    print(len(joint_table))

def sample_age():
    sample = np.round(np.random.normal(1980, 33, len(joint_table)))

    with open(age_sample_file, 'wb') as handle:
        pickle.dump(sample, handle)

def make_data_points():
    with open(age_sample_file, 'rb') as handle:
        joint_table['DOB'] = pickle.load(handle)
    records = mimic.make_list(joint_table)

    max_sups = [0, 10, 50]
    losses = {information_loss.weighted_norm_prec: [0.2, 0.5, 0.8]}
    
    c = 0
    for metric in losses.keys():
        for value in losses[metric]:
            for sup in max_sups:
                [release, gen], ola_stats = inverse_ola.run(records, gen_rules, max_loss=value, max_sup=sup, info_loss=metric, logs=False, weights=weights)
                experiment.publish_stats(ola_stats, experiment_root)


# describe_records()
# sample_age()
# make_data_points()