import pandas as pd

from dataset import mimic


# BUILD PIVOT TABLE
def procedures_by_seq_num(seq_nums):
    # Get only interesting columns
    proc = mimic.get_procedures()[['HADM_ID', 'ICD9_CODE', 'SEQ_NUM']]
    # Get only sequence numbers we want
    proc = proc.loc[proc['SEQ_NUM'].isin(seq_nums)]
    # Stack horizontally sequence numbers and remove nans
    proc = proc.pivot(index='HADM_ID', columns='SEQ_NUM', values='ICD9_CODE').dropna()
    

    return proc



seq_num = [1,2,3]
proc = procedures_by_seq_num(seq_num)
num_admissions = len(proc)

# COUNT EQUIVALENCE CLASSES FOR ORDERED PROCEDURES
e_classess = proc.groupby(seq_num)
ec_sizes = e_classess.size()

num_class_one = sum([1 for s in ec_sizes if s == 1])
avg = ec_sizes.mean()

print('ORDERED SEQUENCES')
print('Num admissions', num_admissions)
print('Num size-one classes', num_class_one)
print('Percentage', num_class_one/num_admissions*100)
print('Average', avg)
print('\n\n')


# COUNT EQUIVALENCE CLASSES FOR UNORDERED PROCEDURES
unordered_procedures = {}
for p in proc.iterrows():
    key = tuple(sorted(p[1].tolist()))
    if key in unordered_procedures.keys():
        unordered_procedures[key] += 1
    else:
        unordered_procedures[key] = 1


num_class_one = sum([1 for s in unordered_procedures.values() if s == 1])
avg = sum(unordered_procedures.values()) / len(unordered_procedures)

print('UNORDERED SEQUENCES')
print('Num admissions', num_admissions)
print('Num size-one classes', num_class_one)
print('Percentage', num_class_one/num_admissions*100)
print('Average', avg)

