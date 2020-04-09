import numpy as np
import utils

def constant(model, learning_rate, batch_size):
    return learning_rate

#lower bound, upper bound, decay constant, batch_size.
def warm_restarts(model, lb, ub, dc, bs):
    if (utils.batch_check(model, bs)):
        B = np.floor(np.floor(model.size/bs))

        ub *= dc**(model.epoch - 1)
        lb *= dc**(model.epoch - 1)

        adjusted_rate = lb + (1/2)*(ub - lb)*(1 + np.cos((model.iterations/B) * np.pi))

        return adjusted_rate
    else:
        return model.learning_rate

#takes the network object, how many iterations (completed batches) until decay, the initial learning rate, and the decay constant.
def step_decay(model, duration, initial, decay, batch_size):
    if (utils.batch_check(model, batch_size)):
        if (model.iterations % duration == 0):
            if (model.epoch == 1 and model.iterations == 0):
                return initial
            else:
                return model.learning_rate*decay
        else:
            return model.learning_rate

def func_dict():
    dict = {
        "const": constant,
        "warm restarts": warm_restarts,
        "step decay": step_decay,
    }
    return dict
