"""
This module contains the implementation of the Restricted Boltzmann Machine
(RBM) algorithm, a generative stochastic artificial neural network that can
learn a probability distribution over its set of inputs.
"""
import os
import argparse  # for argument parsing
import numpy as np  # for numerical operations
import matplotlib.pyplot as plt  # for plotting the energy graph
from layers import HiddenLayer, VisibleLayer
import networkx as nx  # for plotting the network nicely
from DataSet import DataSet
from formulas import get_probability


NUMBER_OF_HIDDEN_UNITS = 16
CONSOLE_LOGGING = True
DEFAULT_DATA_FILE = "iris.data"


class Synapses:
    """
    Synapses class represents the weights of the connections
    between the visible and hidden layers
    """

    def __init__(self, visible_units, hidden_units):
        # weights matrix of size visible_units x hidden_units
        # initialized with random small values
        self.weights = np.random.randn(visible_units, hidden_units) * 0.01

    def update_weights(self, new_weights):
        """
        setter for the weights
        """
        self.weights = new_weights  # TODO: use proper setter


class RBM:
    """
    RBM class represents the Restricted Boltzmann Machine model
    """

    def __init__(self, data=None):
        self.data = data  # DataSet object
        # VisibleLayer object including the input and output units
        self.visible_layer = VisibleLayer(data)
        # HiddenLayer object. The number of hidden units is a hyperparameter
        self.hidden_layer = HiddenLayer(NUMBER_OF_HIDDEN_UNITS)
        self.synapses = Synapses(
            len(self.visible_layer), len(self.hidden_layer))

    def setDataSet(self, data):
        self.data = data
        self.visible_layer = VisibleLayer(data)
        self.synapses = Synapses(
            len(self.visible_layer), len(self.hidden_layer))

    def energy(self, visible_units, hidden_units):
        """
        calculate the energy of the model using the formula:
        -Σ(vi * a) - Σ(hj * b) - Σ(Σ(vi * wji) * hj)

        args:
        - visible_units: the visible layer units
        - hidden_units: the hidden layer units
        """
        visible_bias = self.visible_layer.bias
        hidden_bias = self.hidden_layer.bias
        weights = self.synapses.weights
        return -np.dot(visible_bias, visible_units)\
            - np.dot(hidden_bias, hidden_units) - \
            np.dot(visible_units.T @ weights,
                   hidden_units)  # note the @ operator

    def init_graph(self, title="Energy over time"):
        """
        initialize the graph for the energy over time.
        This feature is used for debugging and visualization, set off by default
        due to the overhead of plotting the graph
        """
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Energy")
        plt.show()
        return ax


    def train(self, epochs=128, learning_rate=1):
        """
        train the model using the Contrastive Divergence algorithm.

        args:
        - epochs: number of iterations
            more iterations will take longer to train
            currently not very useful
        - learning_rate: the learning rate of the model. Currently not used.
        """
        for _ in range(epochs):
            # shuffle the data to sample randomly
            np.random.shuffle(self.data.instances)

            for instance in self.data.instances:
                self.visible_layer.free_locked_units()  # make sure all units are free
                self.visible_layer.set_input(instance)  # turn on input units
                self.visible_layer.set_output(instance)  # turn on output units
                self.visible_layer.lock_units()  # lock the input and output units

                # Calculate hidden probabilities and sample hidden states
                hidden_probs = get_probability(
                    v=self.visible_layer.units, w=self.synapses.weights, b=self.hidden_layer.bias)
                hidden_states = np.random.binomial(1, hidden_probs)

                # Positive phase value represent the outer product
                # of the visible and hidden states
                # used instead of the dot product to calculate the gradient
                pos_phase = np.outer(self.visible_layer.units, hidden_states)

                # Sample visible units from hidden states for negative phase
                visible_activation = np.dot(
                    hidden_states, self.synapses.weights.T) + self.visible_layer.bias

                # Calculate the probability of the visible units using the formula:
                # p(vi = 1|h) = 1 / (1 + exp(-Σ(hj * wji) - ai))
                visible_probs = 1 / (1 + np.exp(-visible_activation))
                visible_states = np.random.binomial(1, visible_probs)

                # Recalculate hidden states from sampled visible units
                hidden_probs_neg = get_probability(v=visible_states, w=self.synapses.weights, b=self.hidden_layer.bias)
                hidden_states_neg = np.random.binomial(1, hidden_probs_neg)

                # Negative phase value represent the outer product
                # of the visible and hidden states after sampling
                # again used instead of the dot product to calculate the gradient
                neg_phase = np.outer(visible_states, hidden_states_neg)

                # Update weights and biases
                self.synapses.weights += learning_rate * \
                    (pos_phase - neg_phase)
                self.visible_layer.bias += learning_rate * \
                    (self.visible_layer.units - visible_states)
                self.hidden_layer.bias += learning_rate * \
                    (hidden_states - hidden_states_neg)

    def isConverged(self, energy_values, threshold=0.01):
        """
        check if the model has converged
        """
        if len(energy_values) < 2:
            # not enough data to check
            return False
        return np.abs(energy_values[-1] - energy_values[-2]) < threshold

    def classify(self, instance, isTraining=True, iterations=1024,
                 temprature=32, console_output=False, plot=False) -> str:
        """
        use iterative approach to classify the instance

        args:
        - instance: the instance to classify
        - isTraining: whether the model is currently training or not
            used to determine whether to bing the output units (training) or not (testing)
        - iterations: number of maximum iterations to run
        - temprature: the initial temprature of the model
            higher temprature will make the model more random
            temprature is reduced over time until it reaches 1
        - console_output: whether to print the progress of the model
        - plot: whether to plot the energy function over time

        returns:
        - the label of the instance as a string, based on the model's decision
        """
        # commented out due to the overhead of plotting the graph
        if plot is True:
            ax = self.init_graph("Deciding : " + str(instance))
        energy_values = []
        energy_values.append(np.inf)
        energy_values.append(-np.inf)
        self.visible_layer.free_locked_units()
        self.visible_layer.set_input(instance)
        if isTraining is True:
            self.visible_layer.set_output(instance)
        self.visible_layer.lock_units()
        self.visible_layer.randomize_units()
        self.hidden_layer.randomize_units()
        if console_output:
            print('-'*60)
            print("starting classification initial visible layer")

        for i in range(iterations):
            if console_output:
                print('iteration:', i, end=' ')
                print(self.visible_layer, end=' ')
                print(self.hidden_layer)

            # vector e calculate the energy gap of the hidden layer
            e = np.zeros_like(self.hidden_layer.bias)
            for k in range(self.hidden_layer.bias.shape[0]):
                e[k] = self.hidden_layer.bias[k] + \
                    np.sum(self.synapses.weights[:, k]
                           * self.visible_layer.units)
            # calculate the probability of the hidden layer
            p = np.zeros_like(self.hidden_layer.bias)
            for k in range(self.hidden_layer.bias.shape[0]):
                p[k] = 1 / (1 + np.exp(-e[k]/temprature))

            # update the hidden layer units based on the probability
            self.hidden_layer.units = np.random.binomial(1, p)

            # now repeat the process on unlocked visible layer
            e = np.zeros_like(self.visible_layer.bias)
            for k in range(self.visible_layer.bias.shape[0]):
                e[k] = self.visible_layer.bias[k] + \
                    np.sum(self.synapses.weights[k, :]
                           * self.hidden_layer.units)
            p = np.zeros_like(self.visible_layer.bias)
            for k in range(self.visible_layer.bias.shape[0]):
                p[k] = 1 / (1 + np.exp(-e[k]/temprature))
            self.visible_layer.units = np.random.binomial(1, p)
            self.visible_layer.turn_on_locked_units()

            # reduce the temprature, max used to prevent float overflow
            temprature = max(0.0001, temprature * 0.9)

            # track the energy of the model to check for early convergence
            energy_values.append(self.energy(
                self.visible_layer.units, self.hidden_layer.units))
            if self.isConverged(energy_values):
                break

            if plot:
                ax.plot(energy_values, "r-")
                plt.draw()
                plt.pause(0.0001)

        # model energy has converged or reached maximum iterations
        if console_output:
            print(f'Final layer: {self.visible_layer} after {i} iterations')
        plt.ioff()

        return self.visible_layer.decide()

    def classify_all(self) -> float:
        """
        classify all instances and return the accuracy

        returns:
        - the accuracy of the model as a float
        """
        correct = 0
        for instance in self.data.instances:
            if self.classify(instance, isTraining=False) == instance.label:
                correct += 1
        return correct / len(self.data.instances)

    def classify_all_verbose(self) -> dict:
        """
        classify all instances and print the results

        returns:
        - a dictionary of the form:
            {
                "label1": {"correct": 0, "total": 0},
                "label2": {"correct": 0, "total": 0},
                ...
            }
        """
        correct = 0
        stats = {}
        for label in self.data.labels:
            stats[label] = {"correct": 0, "total": 0}
        for instance in self.data.instances:
            model_decision = self.classify(
                instance, isTraining=False, console_output=CONSOLE_LOGGING)
            print(f"Classifying {instance} as {model_decision}", end=" ")
            if model_decision == instance.label:
                correct += 1
                stats[instance.label]["correct"] += 1
                print("Correct")
            else:
                print("Incorrect")
            stats[instance.label]["total"] += 1
        if CONSOLE_LOGGING:
            print(f"Accuracy: {correct / len(self.data.instances) * 100:.2f}%")
        return stats

    def test(self) -> dict:
        """
        test the model and return the accuracy

        returns:
        - the accuracy of the model as a dictionary of the form:
            {
                "label1": {"correct": 0, "total": 0},
                "label2": {"correct": 0, "total": 0},
                ...
            }
        """
        return self.classify_all_verbose()

    def save_model(self, directory="models"):
        """
        save the model biases and weights to a directory
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.savetxt(os.path.join(directory, "weights.txt"),
                   self.synapses.weights)
        np.savetxt(os.path.join(directory, "visible_bias.txt"),
                   self.visible_layer.bias)
        np.savetxt(os.path.join(directory, "hidden_bias.txt"),
                   self.hidden_layer.bias)

    def load_weights(self, w_file):
        self.synapses.weights = np.loadtxt(w_file)

    def load_visible_bias(self, a_file):
        self.visible_layer.bias = np.loadtxt(a_file)

    def load_hidden_bias(self, b_file):
        self.hidden_layer.bias = np.loadtxt(b_file)

    def load_model(self, directory=None):
        if directory is None:
            # use current directory
            directory = os.getcwd()
        self.synapses.weights = np.loadtxt(
            os.path.join(directory, "weights.txt"))
        self.visible_layer.bias = np.loadtxt(
            os.path.join(directory, "visible_bias.txt"))
        self.hidden_layer.bias = np.loadtxt(
            os.path.join(directory, "hidden_bias.txt"))

    def plotWeights(self):
        """
        plot the weights of the model
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(self.synapses.weights, cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.xlabel("Hidden Units")
        plt.ylabel("Visible Units")
        plt.show()

    def plotBias(self):
        """
        plot the bias of the model
        """
        plt.figure(figsize=(8, 6))
        plt.plot(self.visible_layer.bias, label="Visible Layer Bias")
        plt.plot(self.hidden_layer.bias, label="Hidden Layer Bias")
        plt.xlabel("Unit")
        plt.ylabel("Bias")
        plt.legend()
        plt.show()

    def plotNetwork(self):
        """
        plot the network : visible layer, hidden layer and synapses
        """
        G = nx.DiGraph()
        # Add nodes for the visible layer
        for i in range(len(self.visible_layer)):
            G.add_node(f'v{i}', layer='visible')

        # Add nodes for the hidden layer
        for i in range(len(self.hidden_layer.units)):
            G.add_node(f'h{i}', layer='hidden')

        # Add edges for the synapses
        for i in range(len(self.visible_layer.units)):
            for j in range(len(self.hidden_layer.units)):
                G.add_edge(f'v{i}', f'h{j}',
                           weight=self.synapses.weights[i, j])

        pos = nx.multipartite_layout(G, subset_key='layer')

        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_color='skyblue',
                node_size=1200, font_size=10)
        plt.show()



# if user runs with -train flag, train the model
# if user runs with -test flag, test the model
# if user runs with -plot flag, plot the model
# if user runs with -save flag, save the model
# if user runs with -load flag, load the model

parser = argparse.ArgumentParser(description='Restricted Boltzmann Machine')
parser.add_argument('-train', action='store_true', help='Train the model')
parser.add_argument('-test', action='store_true', help='Test the model')
parser.add_argument('-plot', action='store_true', help='Plot the model')
parser.add_argument('-save', action='store_true', help='Save the model')
parser.add_argument('-load', action='store_true', help='Load the model')
args = parser.parse_args()

# load default dataset
try:
    data = DataSet.createDataSet(DEFAULT_DATA_FILE, isRealValues=True)
    rbm = RBM(data)
except Exception as e:
    print(str(e))

if args.load:
    print("Loading model")
    rbm.load_model()
    rbm.test()
    rbm.plotWeights()
    rbm.plotBias()
    rbm.plotNetwork()
    exit()



if args.train:
    rbm.train(epochs=32)

if args.test:
    rbm.test()

if args.plot:
    rbm.plotWeights()
    rbm.plotBias()
    rbm.plotNetwork()

if args.save:
    rbm.save_model()
