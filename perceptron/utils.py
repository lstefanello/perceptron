#Many updates to the network are applied upon completion of a batch.
#Multiple functions need to know if a batch is finished.
def batch_check(model, batch_size):
    if (model.batch_clock % batch_size == 0):
        return True
    else:
        return False
