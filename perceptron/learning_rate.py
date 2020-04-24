import numpy as np
import utils

def constant(model, learning_rate):
    return learning_rate

memory = 0
cycle_clock = -1
#lower bound, upper bound, decay constant
def warm_restarts(model, lb, ub, dc, its):
    global memory
    global cycle_clock
    cycle_clock += 1

    ub *= np.exp(-1*dc*memory)
    lb *= np.exp(-1*dc*memory)
    lr = lb + (1/2)*(ub - lb)*(1 + np.cos((cycle_clock/its) * np.pi))

    if (cycle_clock/its == 1):
        memory = model.total_iterations
        cycle_clock = -1

    return lr

#initial rate, decay constant, iterations
def step_decay(model, init, dc, its):
    if (model.iterations % its == 0):
        if (model.epoch == 1 and model.iterations == 0):
            return init
        else:
            return model.learning_rate*dc
    else:
        return model.learning_rate

def exp_decay(model, init, dc):
    return init*np.exp(-1*dc*model.total_iterations)

def func_dict():
    dict = {
        "const": constant,
        "warm restarts": warm_restarts,
        "step decay": step_decay,
        "exp decay": exp_decay,
    }
    return dict
