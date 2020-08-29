import numpy as np


class FunctionWithDerivative:
    def __init__(self, f, df, extra_args=None):
        self.f = f
        self.df = df


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return np.diag(np.exp(-x) / ((1 + np.exp(-x)) ** 2))


sigmoid_fd = FunctionWithDerivative(sigmoid, d_sigmoid)


def relu(x):
    return np.maximum(x, 0.0)


def d_relu(x):
    return np.diag(np.where(x > 0, 1.0, 0.0))


relu_fd = FunctionWithDerivative(relu, d_relu)


def tanh(x, scale=1.0):
    return scale * np.tanh(x)


def d_tanh(x, scale=1.0):
    return np.diag(scale / (np.cosh(x) ** 2))


tanh_fd = FunctionWithDerivative(tanh, d_tanh)


def square_diff(y0, y):
    return np.dot(y0 - y, y0 - y)


def d_square_diff(y0, y):
    return -2 * (y0 - y)


square_diff_fd = FunctionWithDerivative(square_diff, d_square_diff)


def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp)


def d_softmax(x):
    exp = np.exp(x)
    denum = np.sum(exp)
    diag = np.diag(exp / denum)

    off_diag = np.array(np.matrix([
        np.array([-(exp[i] * exp[j]) / denum ** 2
                  for j in range(len(x))
                  ]) for i in range(len(x))
    ]))
    return diag + off_diag


softmax_fd = FunctionWithDerivative(softmax, d_softmax)


def softmax_i(x, i):
    exp = np.exp(x)
    return exp[i] / np.sum(exp)


def d_softmax_i(x, i):
    exp = np.exp(x)
    return -(exp[i] / np.sum(exp)) ** 2


softmax_i_fd = FunctionWithDerivative(softmax_i, d_softmax_i)


def log(x, ind):
    return np.log(x)


def d_log(x):
    return 1 / x


log_fd = FunctionWithDerivative(log, d_log)


def log_policy(ind, x):
    return np.log(x[ind])


def d_log_policy(ind, x):
    return np.array([min(1 / x[i], 1000) if i == ind else 0 for i in range(len(x))])


log_policy_fd = FunctionWithDerivative(log_policy, d_log_policy)


def cross_entropy(p, p0):
    return -np.dot(p0, np.log(p))


def d_cross_entropy(p, p0):
    return -np.dot(p0, 1 / p)


cross_entropy_fd = FunctionWithDerivative(cross_entropy, d_cross_entropy)


def linear(x):
    return x


def d_linear(x):
    return 1


linear_fd = FunctionWithDerivative(linear, d_linear)


def huber(y0, y):
    sq = np.dot(y0 - y, y0 - y)
    if sq < 1:
        return sq / 2
    else:
        return np.sqrt(sq) - 0.5


def d_huber(y0, y):
    sq = np.dot(y0 - y, y0 - y)

    if sq < 1:
        return -(y0 - y)
    else:
        return -(y0 - y) / np.linalg.norm(y0 - y)


huber_fd = FunctionWithDerivative(huber, d_huber)

string_to_differentiable_function = {
    'sigmoid': sigmoid_fd,
    'relu': relu_fd,
    'tanh': tanh_fd,
    'square_diff': square_diff_fd,
    'softmax': softmax_fd,
    'huber': huber_fd,
    'log': log_fd,
    'log_policy': log_policy_fd,
    'cross_entropy': cross_entropy_fd,
    'linear': linear_fd,
    'none': None
}


def get_function(function_string):
    return string_to_differentiable_function[function_string].f


def get_derivative(function_string):
    return string_to_differentiable_function[function_string].df
