import math

import numpy as np

import epsilon_safe
from dataset import adult

records = adult.retrieve()

epsilon = [0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.6, 1, 2]
betas = [0.01, 0.1, 0.25, 0.33, 0.44, 0.55, 0.66, 0.77]

points = {}

for i, beta in enumerate(betas):
    for j, eps in enumerate(epsilon):
        eps_prime, delta = epsilon_safe.parameters(25, beta, eps, len(records))
        if eps_prime is not None:
            if beta in points.keys():
                points[beta].append((eps, eps_prime))
            else:
                points[beta] = [(eps,eps_prime)]
            print(f'beta {beta} \t epsilon {np.round(eps, 5)} \t\t epsilon_prime {eps_prime} \t {delta<10e-4} ')
