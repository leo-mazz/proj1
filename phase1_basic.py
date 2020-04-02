from dataset import mimic
import pandas as pd
import numpy as np

patients = mimic.get_patient_demographics()
print(len(patients))

avg_avg = 0
avg_num_class_one = 0
min_class_one = 30000
max_class_one = 0

trials = 1000
for i in range(trials):
    # Gaussian-distributed age
    patients['DOB'] = np.round(np.random.normal(1980, 33, len(patients)))
    # Uniform distributed age
    # patients['DOB'] = np.round(np.random.uniform(0, 99, len(patients)))

    class_sizes = patients.groupby(by=['DOB', 'GENDER', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS']).size()
    avg_avg += class_sizes.mean()

    num_class_one = sum([1 for s in class_sizes if s == 1])
    avg_num_class_one += num_class_one

    min_class_one = min(num_class_one, min_class_one)
    max_class_one = max(num_class_one, max_class_one)


avg_avg /= trials
avg_num_class_one /= trials

print('avg avg', avg_avg)
print('avg num class one', avg_num_class_one)
print('min num class one', min_class_one)
print('max num class one', max_class_one)



