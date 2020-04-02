import os
import time
import json

from matplotlib import pyplot
import numpy as np

def merge_stats(original, *updates):
    """ Merge the fields of two nested dictionaries representing statistics """
    for update in updates:
        for key, value in update.items(): 
            if key not in original:
                original[key] = value
            elif isinstance(value, dict):
                original[key] = merge_stats(value, original[key]) 
    return original

def publish_stats(stats, experiment_root, nonce=None):
    """ Write statistics dictionary to a file """
    timestamp = f"{time.strftime('%d.%m-%H.%M.%S')}"
    if nonce is not None:
        timestamp += f'.{nonce}.txt'
    else:
        timestamp += '.txt'
    filename = os.path.join(experiment_root, timestamp)

    os.makedirs(os.path.dirname(filename), exist_ok=True)   

    with open(filename, 'w+') as file:
        file.write(json.dumps(stats, indent=4))


def look_for_stats(experiment_root, **kwargs):
    """ Look, in a directory, for runs whose stats match some criteria.
    Currently can only nest two keys deep """
    stats = []
    for filename in os.listdir(experiment_root):
        if filename.endswith('.txt'):
            file = os.path.join(experiment_root, filename)
            with open(file) as run:
                select = True
                data = json.load(run)
                for key, dic in kwargs.items():
                    for prop, value in dic.items():
                        if key not in data.keys() or prop not in data[key].keys() or data[key][prop] != value:
                            select = False
                
                if select:
                    stats.append(data)

    return stats

def filter_stats(stats, **kwargs):
    """ Get only the values associated with the specified target, 2-levels deep in the nesting """
    filtered = {}
    for k, vs in kwargs.items():
        for v in vs:
            filtered[f'{k}.{v}'] = stats[k][v]
    
    return filtered

def filter_many_stats(stats, **kwargs):
    return [filter_stats(s, **kwargs) for s in stats]


def graph_points(bags_of_points, x_axis, y_axis, scale='linear', line=True, baseline=-1, filename=None):
    """ Produce a plot from a 2-dimensional data points """
    for i, (legend, points) in enumerate(bags_of_points.items()):
        xs, ys = zip(*points)
        if i == baseline:
            pyplot.xlim(min(xs)-0.01, max(xs)+0.01)
            linspace = np.linspace(min(xs)-0.01, max(xs)+0.01, 2)
            g = pyplot.plot(linspace, [ys[0]]*len(linspace), ':', label=legend, color="black", linewidth=2)
            g[-1].set_zorder(100)
        else:
            if line:
                pyplot.plot(xs, ys, '-o', label=legend)
            else:
                pyplot.scatter(xs, ys, label=legend)
        pyplot.yscale(scale)
        pyplot.xlabel(x_axis)
        pyplot.ylabel(y_axis)
        pyplot.legend()

    if filename is None:
        pyplot.show()
    else:
        pyplot.savefig(filename)

    pyplot.clf()



class Logger():
    def __init__(self, active=True, verbose=False):
        self.active = active
        self.verbose = verbose
        self.beginning = True

    def log_step(self, step):
        if self.active:
            if not self.beginning:
                print('Ok.\n--------------------\n')
            self.beginning = False
            print(step)

    def print(self, message):
        if self.active:
            print(message)


def step(name):
    def step_decorator(func):
        def func_wrapper(*args, **kwargs):
            start = time.time()
            result, stats = func(*args, **kwargs)
            end = time.time()
            elapsed = round(end - start, 4)

            print(f'DONE with {name} in {elapsed}s')

            return result, merge_stats(stats, {'runtime': {name: elapsed}})
        return func_wrapper
    return step_decorator


class AverageMeter():
    def __init__(self):
        self.total = 0
        self.count = 0

    def add(self, value):
        self.total += value
        self.count += 1

    def get(self):
        return self.total / self.count

class MinMeter():
    def __init__(self):
        self.beginning = True
        self.min = None
    
    def add(self, value):
        if self.beginning:
            self.beginning = False
            self.min = value
        else:
            self.min = min(self.min, value)

    def get(self):
        return self.min


class MaxMeter():
    def __init__(self):
        self.beginning = True
        self.max = None
    
    def add(self, value):
        if self.beginning:
            self.beginning = False
            self.max = value
        else:
            self.max = max(self.max, value)

    def get(self):
        return self.max