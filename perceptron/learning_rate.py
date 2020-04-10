import numpy as np
import utils

def constant(model, learning_rate):
    return learning_rate

memory = 0
cycle_clock = -1
#lower bound, upper bound, decay constant
def warm_restarts(model, lb, ub, decay_const, iterations):
    global memory
    global cycle_clock
    cycle_clock += 1

    ub *= np.exp(-1*decay_const*memory)
    lb *= np.exp(-1*decay_const*memory)
    lr = lb + (1/2)*(ub - lb)*(1 + np.cos((cycle_clock/iterations) * np.pi))

    if (cycle_clock/iterations == 1):
        memory = model.total_iterations
        cycle_clock = -1

    return lr

def step_decay(model, initial, decay_const, iterations):
    if (model.iterations % iterations == 0):
        if (model.epoch == 1 and model.iterations == 0):
            return initial
        else:
            return model.learning_rate*decay_const
    else:
        return model.learning_rate

def exp_decay(model, initial, decay_const):
    return initial*np.exp(-1*decay_const*model.total_iterations)

def func_dict():
    dict = {
        "const": constant,
        "warm restarts": warm_restarts,
        "step decay": step_decay,
        "exp decay": exp_decay,
    }
    return dict
