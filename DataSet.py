import numpy as np


class Instance:
    def __init__(self, label=None, attributes=None):
        self.label = label
        self.attributes = attributes if attributes is not None else []

    def __getitem__(self, index):
        return self.attributes[index]

    def __setitem__(self, index, value):
        self.attributes[index] = value

    def __str__(self):
        return str(self.attributes) + " " + self.label

    def __repr__(self):
        return str(self)


class DataSet:
    def __init__(self):
        self.labels = np.array([])
        self.attributes = {}  # Dictionary of attribute names and their possible values
        self.instances = np.array([])  # list of Instance objects
        self.DELIMITER = ","  # Used to split input strings
        self.descreteValues = 1

    def convertToDiscrete(self, descreteValues=3):
        """
        convert real values to 3 discrete values
        """
        self.descreteValues = descreteValues
        for i in range(len(self.attributes.keys())):  # for each attribute
            sorted_instances = sorted(self.instances, key=lambda x: x[i])
            threshold1 = sorted_instances[int(len(sorted_instances) / 3)][i]
            threshold2 = sorted_instances[int(
                len(sorted_instances) * 2 / 3)][i]
            for instance in self.instances:
                if instance.attributes[i] <= threshold1:
                    instance.attributes[i] = int(1)
                elif instance.attributes[i] <= threshold2:
                    instance.attributes[i] = int(2)
                else:
                    instance.attributes[i] = int(3)

        # convert all instances to int
        for instance in self.instances:
            instance.attributes = instance.attributes.astype(int)

    def addLabels(self, line):
        # Assuming labels are at the start of the file prefixed with %%
        self.labels = line.strip().split(self.DELIMITER)[1:]  # Skip the prefix

    def addAttribute(self, line):
        # Assuming attributes are defined with names (no need for value lists)
        attributeName = line.strip().split(self.DELIMITER)[
            1]  # Skip the prefix
        attributeValues = line.strip().split(self.DELIMITER)[
            2:]  # Skip the prefix and name
        self.attributes[attributeName] = attributeValues

    def addInstance(self, line):
        splitline = line.strip().split(self.DELIMITER)
        if len(splitline) != len(self.attributes) + 1:  # Assuming last value is the label
            raise ValueError(
                "Instance format incorrect, does not match attribute count")

        instance = Instance()
        # Assuming the last element is the label
        instance.label = splitline[-1]
        # Convert attribute values to float
        # All but the last element
        instance.attributes = np.array(splitline[0:-1], dtype=float)
        self.instances = np.append(self.instances, instance)

    def sameMetaValues(self, other):
        # Check if both datasets have the same labels and attributes
        if self.labels != other.labels or self.attributes != other.attributes:
            return False
        return True

    @staticmethod
    def createDataSet(file_path, isRealValues=True):
        dataset = DataSet()
        with open(file_path, 'r') as f:
            for line in f:
                prefix = line[:2]
                if prefix == "//":  # Comment line
                    continue
                elif prefix == "%%":  # Label line
                    dataset.addLabels(line)
                elif prefix == "##":  # Attribute name line
                    dataset.addAttribute(line)
                else:  # Instance data line
                    dataset.addInstance(line)
        if isRealValues:
            dataset.convertToDiscrete()

        return dataset
