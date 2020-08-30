import numpy as np
from differentiable_functions import get_function, get_derivative


class DenseLayer:
    """A single layer of a dense neural net, stores the weights in addition to the gradients of the last evaluation"""

    def __init__(self, input_size, output_size, activation_string, initial_variance=0.2):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_string = activation_string
        self.initial_variance = initial_variance
        self.weights = np.random.normal(0, initial_variance, [self.output_size, self.input_size + 1])
        self.input = np.ones(self.input_size + 1)
        self.output = np.zeros(self.output_size)
        self.s = np.zeros(self.output_size)  # Outputs before activation is applied
        self.derivative_buffer = 0 * self.weights

    def __repr__(self):
        def add_buffer(input_string, max_length):
            buffer_length = max_length - len(input_string)
            if buffer_length > 0:
                return input_string + ' ' * buffer_length
            return input_string

        params_string = add_buffer('Dense Layer with parameters of shape {}, '.format(self.weights.shape), 50)
        neuron_string = add_buffer(' {} neuron(s),'.format(self.output_size), 15)
        activation_string = add_buffer(' {} activation'.format(self.activation_string), 20)
        return params_string + neuron_string + activation_string

    def activate(self, inputs):
        self.input[1:] = inputs
        self.s = np.dot(self.weights, self.input)
        activation_function = get_function(self.activation_string)
        self.output = activation_function(self.s)
        return self.output

    def mutate(self, rate):
        self.weights += np.random.normal(0, rate,
                                         [self.output_size, self.input_size + 1]
                                         )

    def reset(self, var=0.2):
        self.weights = np.random.normal(0, var, [self.output_size, self.input_size + 1])
        self.input = np.ones(self.input_size + 1)
        self.output = np.zeros(self.output_size)
        self.s = np.zeros(self.output_size)
        self.derivative_buffer = 0 * self.weights


class DenseNet:
    """A collection of layers of neural net, with functions to call the model, run back propagation and share weights"""

    def __init__(self,
                 input_size,
                 fitness_function_string,
                 recursive=False,
                 initial_variance=0.01,
                 learning_rate=0.0001
                 ):
        self.input_size = input_size
        self.fitness_function_string = fitness_function_string
        self.initial_variance = initial_variance
        self.layers = []
        self.frozen = []
        self.recursive = recursive
        self.learning_rate = learning_rate
        if self.recursive:
            self.input_with_state = []

    def __repr__(self):
        pass

    def add_layer(self, output_size, activation_string, initial_variance=0.2):
        """Adds a single new layer to the model"""
        self.frozen.append(False)
        if len(self.layers) == 0:
            self.layers.append(
                DenseLayer(self.input_size, output_size, activation_string,
                           initial_variance))
        else:
            if self.recursive:
                self.layers.append(
                    DenseLayer(self.layers[-1].output_size, output_size, activation_string,
                               initial_variance))
                self.layers[0] = DenseLayer(
                    self.input_size + self.layers[-1].input_size,
                    self.layers[0].output_size,
                    self.layers[0].activation_string,
                    self.layers[0].initial_variance)
                self.input_with_state = np.zeros(
                    self.input_size + self.layers[-1].input_size)
            else:
                self.layers.append(
                    DenseLayer(self.layers[-1].output_size, output_size, activation_string,
                               initial_variance))

    def activate(self, inputs):
        """Evaluates the inputs for the model"""
        if self.recursive:
            self.input_with_state[:self.input_size] = inputs
            output = self.input_with_state
            for layer in self.layers:
                output = layer.activate(output)
            self.input_with_state[self.input_size:] = self.layers[-1].input[1:]
            return output
        else:
            output = inputs
            for layer in self.layers:
                output = layer.activate(output)
            return output

    def mutate(self, rate):
        """Adds random noise to weights. Useful for evolution"""
        if type(rate) == float:
            for layer in self.layers:
                layer.mutate(rate)
        else:
            assert (len(self.layers) == len(rate))
            for i in range(len(self.layers)):
                self.layers[i].mutate(rate[i])

    def reset(self, rate):
        """Resets weights on all layers"""
        if type(rate) == float:
            for layer in self.layers:
                layer.reset(rate)
        else:
            assert (len(self.layers) == len(rate))
            for i in range(len(self.layers)):
                self.layers[i].reset(rate[i])

    def _fitness(self, y0):
        """Calculates fitness -- assumes that x has already been forward propagated"""
        assert self.fitness_function_string != 'none', 'No fitness function defined'
        fitness_function = get_function(self.fitness_function_string)
        y = self.layers[-1].output
        return fitness_function(y, y0)

    def _calculate_derivatives(self, y0):
        """Calculates derivatives in all layers,Assumes that x has already been passed through"""
        fitness_derivative = get_derivative(self.fitness_function_string)
        dldx = fitness_derivative(self.layers[-1].output, y0)
        for layer in self.layers[::-1]:
            activation_derivative = get_derivative(layer.activation_string)
            dlds = np.dot(activation_derivative(layer.s), dldx)
            layer.derivative = np.outer(dlds, layer.input)
            dldx = np.dot(dlds, layer.weights)[1:]

    def _update(self, learning_rate=None):
        """Updates the weights with a single SDG step. Assumes that the derivatives have already been calculated,
        and skips frozen layers"""
        if learning_rate is None:
            learning_rate = self.learning_rate
        for layer, frozen in zip(self.layers, self.frozen):
            if not frozen:
                layer.weights += learning_rate * layer.derivative

    def __repr__(self):
        output = 'Dense Net with size {} input and {} objective\n'.format(self.input_size, self.fitness_function_string)
        for layer, frozen in zip(self.layers, self.frozen):
            frozen_string = "(NOT learning)" if frozen else '(learning)'
            output += layer.__repr__() + frozen_string + '\n'
        return output

    def update(self, vx, vy, a=-0.001):
        fitness = 0
        for x, y in zip(vx, vy):
            self.activate(x)
            fitness += self._fitness(y)
            self._calculate_derivatives(y)
            self._update(a)
        return fitness


@staticmethod
def add_shared_layer(output_size, activation_string, models, frozen_list=None, input_size=None, initial_variance=0.2):
    """Adds layer to two or more models. The weight are shared between the models, and weight updates can be toggled
    with the frozen list array. Useful for making GANs or sharing weights """
    if frozen_list is None:
        frozen_list = [False for _ in models]
    assert len(models) == len(frozen_list), "Mismatched number of objects"
    if input_size is None:
        for model in models:
            if len(model.layers) != 0:
                input_size = len(model.layers[-1])
                break
        assert input_size is not None, "No valid input size found"
    for model in models:
        assert not model.recursive, "recursion not enabled for shared layers"
        if len(model.layers) != 0:
            assert input_size == len(
                model.layers[-1]), "Mismatch between input size and last layer output size in dense net"
    new_layer = DenseLayer(input_size, output_size, activation_string,
                           initial_variance)
    for model, frozen in zip(models, frozen_list):
        model.frozen.append(frozen)
        model.layer.append(new_layer)


def update_trajectory(self, trajectory, discount=1.0, a=-0.001):
    vx = trajectory['state']
    vy = trajectory['action']
    rewards = trajectory['reward']
    reward_derivative = []
    for reward in rewards:
        if len(reward_derivative) == 0:
            reward_derivative.append(reward)
        else:
            reward_derivative.append(discount * reward_derivative[-1] + reward)
    reward_derivative = reward_derivative[::-1]
    for x, y, z in zip(vx, vy, reward_derivative):
        self.activate(x)
        self._calculate_derivatives(y)
        self._update(a * z)


def simple_test(recursive):
    net = DenseNet(3, 'square_difference', recursive)
    net.add_layer(2, 'sigmoid')
    net.add_layer(2, 'sigmoid')
    print(net.activate(np.array([1, 1, 1])))
    print(net.activate(np.array([1, 1, 1])))
    print(net.activate(np.array([1, 1, 1])))
    net.mutate(0.01)
    print(net.activate(np.array([1, 1, 1])))
    print(net.activate(np.array([1, 1, 1])))
    print(net._fitness([0, 0]))
    net._calculate_derivatives([0, 0])


def run_stress(recursive):
    net = DenseNet(2, 128, 'sigmoid', 'square_diff', recursive)
    net.add_layer(128, 'sigmoid')
    net.add_layer(128, 'sigmoid')
    net.add_layer(128, 'sigmoid')
    net.add_layer(128, 'sigmoid')
    net.add_layer(2, 'sigmoid')
    input = np.array([1, 1, 1])
    for i in range(1000):
        net.activate(np.array(input))


def training_stress():
    net = DenseNet(1, 'square_diff', False)
    net.add_layer(128, 'sigmoid')
    net.add_layer(32, 'sigmoid')
    net.add_layer(1, 'linear')
    print(net)
    for i in range(1000000):
        x = np.random.uniform(-4, 4)

        input = np.array([x])
        net.activate(input)
        net._fitness(input)
        if i % 50000 == 0:
            print(net._fitness(input))

        net._calculate_derivatives(input)
        net._update(-0.0001)
        if i % 10000 == 0:
            print(net.activate(np.array([-1])), net.activate(0.1), net.activate(1))


def time_test(recursive):
    import cProfile
    cProfile.run("run_stress({})".format(recursive))


def train_time_test():
    import cProfile
    cProfile.run("training_stress()")


if __name__ == "__main__":
    training_stress()
    simple_test(True)
    """  simple_test(False)
    time_test(True)
    time_test(False)
    train_time_test()"""
