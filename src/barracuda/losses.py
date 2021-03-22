import numpy

def mse(y_true, y_pred):
    y_true = numpy.array(y_true)
    y_pred = numpy.array(y_pred)
    return numpy.mean(numpy.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    y_true = numpy.array(y_true)
    y_pred = numpy.array(y_pred)
    return 2*(y_pred-y_true)/y_true.size


class MSELoss(object):
    def __init__(self):
        self.normal = mse
        self.derivative = mse_prime