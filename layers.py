import numpy as np

DEFAULT_OUTPUT_UNITS = 3
DEFAULT_INPUT_UNITS = 12


class Layer:
    def __init__(self, num_of_units):
        self.num_of_units = num_of_units
        self.units = np.zeros(num_of_units, dtype=int)
        self.bias = np.zeros(num_of_units, dtype=float)

    def update_units(self, new_units):
        self.units = new_units

    def update_bias(self, new_bias):
        self.bias = new_bias

    def get_gradient(self, eta=0.1) -> np.array:
        """
        calculate the gradient of the model
        """
        return np.gradient(self.units, eta)

    def __len__(self):
        return len(self.units)

    def __getitem__(self, key):
        return self.units[key]

    def __setitem__(self, key, value):
        self.units[key] = value

    def __iter__(self):
        return iter(self.units)


class HiddenLayer(Layer):
    def __init__(self, num_of_hidden_units=10):
        super().__init__(num_of_hidden_units)

    def randomize_units(self):
        random_numbers = np.random.rand(len(self.units))
        for i in range(len(self.units)):
            self.units[i] = random_numbers[i] > 0.5

    def __str__(self):
        ret = "Hidden units: [" + \
            " ".join(["1" if u == 1 else "0" for u in self.units])
        ret += "]"
        return ret


class VisibleLayer(Layer):
    def __init__(self, data=None, descrete_values=1):
        if data is None:
            self.num_of_input_units = DEFAULT_INPUT_UNITS
            self.num_of_output_units = DEFAULT_OUTPUT_UNITS
        else:
            self.num_of_input_units = sum(
                len(data.attributes[attribute]) for attribute in data.attributes)
            self.num_of_output_units = len(np.unique(data.labels))
        super().__init__(self.num_of_input_units + self.num_of_output_units)
        self.output_unit_to_class = {}
        self.class_to_output_unit = {}
        if data is not None:
            self.init_labels(data.labels)
        else:
            self.init_labels(np.array([]))
        self.locked_units = np.zeros(len(self.units), dtype=int)

    def set_output(self, instance):
        output_unit_index = self.class_to_output_unit[instance.label]
        self.units[output_unit_index] = 1

    def set_input(self, instance: 'Instance'):
        self.units = np.zeros(len(self.units), dtype=int)
        offset = 0
        for attribute in instance.attributes:
            self.units[int(attribute) + offset - 1] = 1
            # Adjust based on attribute size
            offset += 3  # TODO: Change to descrete_values

    def lock_units(self):
        self.locked_units = self.units.copy()

    def randomize_units(self):
        random_numbers = np.random.rand(len(self.units))
        for i in range(len(self.units)):
            if self.locked_units[i] != 1:
                self.units[i] = random_numbers[i] > 0.5

    def init_labels(self, labels):
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            output_unit_index = self.num_of_input_units + i
            self.output_unit_to_class[output_unit_index] = label
            self.class_to_output_unit[label] = output_unit_index

    def decide(self):
        class_index = np.argmax(self.units[self.num_of_input_units:])
        return self.output_unit_to_class[self.num_of_input_units + class_index]

    def free_locked_units(self):
        self.locked_units = np.zeros(len(self.units), dtype=int)

    def turn_on_locked_units(self):
        for i in range(len(self.units)):
            if self.locked_units[i] == 1:
                self.units[i] = 1

    def __str__(self):
        ret = "Input units:  ["
        for i in range(len(self.units)):
            if self.units[i] == 1 and self.locked_units[i] == 1:
                ret += "L"
            elif self.units[i] == 1:
                ret += "1"
            elif self.locked_units[i] == 1 and self.units[i] == 0:
                ret += "BUG"
            else:
                ret += "0"
            if i == self.num_of_input_units - 1:
                ret += " ]\tOutput units: ["
            elif i < len(self.units) - 1:
                ret += " "
        ret += "]"
        return ret
