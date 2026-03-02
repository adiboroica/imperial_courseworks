import torch
import torch.nn as nn
import pickle
import pandas as pd
import random
from sklearn import preprocessing, metrics
import numpy as np


class Model(nn.Module):
    def __init__(self, input_size, neurons, activations):
        super().__init__()

        self.input_size = input_size
        self.net = [nn.Linear(self.input_size, neurons[0])]
        for i in range(len(neurons)):
            if i != len(neurons) - 1:
                layers_to_add = []
                if activations[i] == 'relu':
                    layers_to_add.append(nn.ReLU())
                elif activations[i] == 'sigmoid':
                    layers_to_add.append(nn.Sigmoid())
                self.net.extend(layers_to_add)
                self.net.append(nn.Linear(neurons[i], neurons[i + 1]))
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = self.net(x)
        return x


class Regressor():

    def __init__(
        self,
        x,
        nb_epoch=893,
        learning_rate=0.099,
        batch_size=100,
        neurons=[31, 26, 27, 24, 29, 1],
        activations=["relu", "relu", "relu", "relu", "relu", "relu"]
    ):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        self.lb_dict = dict()
        self.xscaler = preprocessing.MinMaxScaler()
        self.yscaler = preprocessing.MinMaxScaler()

        # Preprocess the data.
        X, _ = self._preprocessor(x, training=True)

        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.model = Model(self.input_size, neurons, activations)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y=None, training=False):
        """ 
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        x = x.copy()
        features = x.columns

        # Remove values if there are more than half missing features.
        x = x[x.isnull().sum(axis=1) < len(features) / 2]
        # Handle missing values.
        for feature in features:
            if feature == "ocean_proximity":
                x[feature] = x[feature].fillna(x[feature].mode()[0])
            else:
                x[feature] = x[feature].fillna(x[feature].median())

        if training:
            # One hot encoding only needs to be done for ocean_proximity
            lb = preprocessing.LabelBinarizer()
            lb.fit(x["ocean_proximity"])
            self.lb_dict["ocean_proximity"] = lb
            x = x.join(pd.DataFrame(
                lb.fit_transform(x["ocean_proximity"]),
                columns=lb.classes_,
                index=x.index
            ))

        new_x = pd.DataFrame()
        for feature in features:
            if feature == 'ocean_proximity':
                new_x = new_x.join(pd.DataFrame(
                    self.lb_dict[feature].transform(x["ocean_proximity"]),
                    columns=self.lb_dict[feature].classes_,
                    index=x.index
                ))
            else:
                new_x = pd.concat([new_x, x[feature]], axis=1)

        # Normalize the x data.
        if training:
            new_x = self.xscaler.fit_transform(new_x)
        else:
            new_x = self.xscaler.transform(new_x)

        # Normalize the y data.
        if y is not None and training:
            new_y = self.yscaler.fit_transform(y)
        if y is not None:
            new_y = self.yscaler.transform(y)

        new_x = torch.tensor(new_x)
        new_y = torch.tensor(new_y) if isinstance(y, pd.DataFrame) else None

        # Return preprocessed x and y, return None for y if it was None.
        return new_x, new_y

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Preprocess the data.
        X, Y = self._preprocessor(x, y=y, training=True)  # Do not forget

        criterion = nn.MSELoss()
        optimiser = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate
        )

        for _ in range(self.nb_epoch):
            # Split the data into batches.
            perm = np.random.permutation(len(X))
            input_batches = torch.split(X[perm], self.batch_size)
            target_batches = torch.split(Y[perm], self.batch_size)

            # Run gradient descent on every batch.
            for i in range(len(input_batches)):
                output = self.model(input_batches[i].float())
                loss = criterion(output, target_batches[i].float())
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Preprocess the data.
        X, _ = self._preprocessor(x, training=False)  # Do not forget

        prediction = self.model(X.float())

        return self.yscaler.inverse_transform(prediction.detach().numpy())

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Preprocess the data.
        _, Y = self._preprocessor(x, y=y, training=False)  # Do not forget

        prediction = self.predict(x)

        accuracy = metrics.mean_squared_error(
            self.yscaler.inverse_transform(Y.detach().numpy()),
            prediction,
            squared=False
        )

        return accuracy

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model):
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch(x_train, y_train, x_val, y_val):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.

    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    rmse = np.inf

    hyperparameters = {
        "nb_epoch": None,
        "learning_rate": None,
        "neurons": None,
        "batch_size": None,
        "activations": None,
    }

    for _ in range(10):
        # Set a random number of epochs.
        nb_epoch = np.random.randint(700, 1000)
        # Set a random permutation of layers.
        neuron_layers = np.random.randint(15, 35, (np.random.randint(4, 10), ))
        neuron_layers = np.append(neuron_layers, 1)
        # Set a random batch size.
        batch_size = np.random.randint(100, 150)
        # Set a random learning rate.
        learning_rate = np.random.uniform(0.05, 0.2)
        # Set up the activations.
        # (this uses relu layers now, but can be extended to allow more activation functions)
        activation_layers = ["relu"]
        activations = random.choices(activation_layers, k=len(neuron_layers))
        # Set up the last layer.
        activations.append("relu")

        # Construct the regressor.
        regressor = Regressor(
            x_train,
            nb_epoch=nb_epoch,
            learning_rate=learning_rate,
            neurons=neuron_layers,
            batch_size=batch_size,
            activations=activations
        )
        print('start training...')
        # Train the model on the train dataset.
        regressor.fit(x_train, y_train)

        # Find the error of the model on the val dataset.
        score = regressor.score(x_val, y_val)
        print('rmse on validation: ', score)
        if score < rmse:
            # Save the hyperparameters and update the best score.
            rmse = score
            hyperparameters = {
                "nb_epoch": nb_epoch,
                "learning_rate": learning_rate,
                "neurons": neuron_layers,
                "batch_size": batch_size,
                "activations": activations,
            }

    # Return the chosen hyper parameters.
    return hyperparameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def regressor_on_dataset(dataset_path):

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv(dataset_path)

    # Split the data in train and val.
    train_data = data.sample(frac=0.9)
    x_train = train_data.loc[:, train_data.columns != output_label]
    y_train = train_data.loc[:, [output_label]]
    test_data = data.drop(train_data.index)
    x_test = test_data.loc[:, test_data.columns != output_label]
    y_test = test_data.loc[:, [output_label]]

    # Construct the regressor.
    regressor = Regressor(x_train, nb_epoch=1000)
    # Train the model on the train dataset.
    regressor.fit(x_train, y_train)

    # Find the error of the model on the test dataset.
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))

    # Save regressor.
    save_regressor(regressor)


def regressor_hyperparameter_search_on_dataset(dataset_path):

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv(dataset_path)

    # Split the data in train, val and test.
    train_data, val_data, test_data = np.split(
        data.sample(frac=1),
        [int(.8*len(data)), int(.9*len(data))]
    )
    x_train = train_data.loc[:, train_data.columns != output_label]
    y_train = train_data.loc[:, [output_label]]
    x_val = val_data.loc[:, val_data.columns != output_label]
    y_val = val_data.loc[:, [output_label]]
    x_test = test_data.loc[:, test_data.columns != output_label]
    y_test = test_data.loc[:, [output_label]]

    # Perform hyperparameter search.
    hyperparameters = RegressorHyperParameterSearch(
        x_train, y_train, x_val, y_val)

    # Construct the regressor.
    regressor = Regressor(
        x_train,
        nb_epoch=hyperparameters["nb_epoch"],
        learning_rate=hyperparameters["learning_rate"],
        batch_size=hyperparameters["batch_size"],
        neurons=hyperparameters["neurons"],
        activations=hyperparameters["activations"],
    )
    # Train the model on the train dataset.
    regressor.fit(x_train, y_train)

    # Find the error on the test dataset.
    error = regressor.score(x_test, y_test)
    print("\nRegressor error with param tuning: {}\n".format(error))

    # Find the error on the whole dataset.
    error = regressor.score(
        pd.concat([x_train, x_val, x_test]),
        pd.concat([y_train, y_val, y_test]),
    )
    print("\nRegressor error on whole dataset: {}\n".format(error))

    # Save regressor.
    save_regressor(regressor)


def test_regressor_on_dataset(dataset_path):

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv(dataset_path)

    # Split the data in input and output.
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    # Load the regressor.
    regressor = load_regressor()

    # Find the error of the model on the test dataset.
    error = regressor.score(x, y)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    test_regressor_on_dataset("housing.csv")
