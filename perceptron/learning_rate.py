import numpy as np

def warm_restarts(self, lower_bound, upper_bound, batch_size, cycles_per_epoch):
    if (self.batch_clock % batch_size == 0):
        B = np.floor(np.floor(self.size/batch_size)*(1/cycles_per_epoch))
        self.iterations += 1
        self.learning_rate = lower_bound + (1/2)*(upper_bound - lower_bound)*(1 + np.cos((self.iterations/B) * np.pi))

        if(self.iterations >= B):
            self.iterations = 0

def check_lr_fuction(self):
    #switcher {
    #    "warm_restarts":
    #}
    print(":)")
