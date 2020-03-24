import numpy as np
import utils

def constant(learning_rate, batch_size):
    return learning_rate

#lower bound, upper bound, batch_size.
def warm_restarts(model, lb, ub, bs):
    if (utils.batch_check(model, bs)):
        B = np.floor(np.floor(model.size/bs))
        adjusted_rate = lb + (1/2)*(ub - lb)*(1 + np.cos((model.iterations/B) * np.pi))

        return adjusted_rate

def func_dict():
    dict = {
        "const": constant,
        "warm restarts": warm_restarts,
    }
    return dict
