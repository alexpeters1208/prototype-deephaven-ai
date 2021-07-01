# This example demonstrates a case where a user function creates partial tensors for each row.
# These partial tensors are aggregated into tensors before evaluating the model.  
# The aggregation should result in more efficient use of the AI machinery.  
# The model function is then evaluated for each row to create results for the row.

################################################################################################################################
# Everything here would be part of a DH library
################################################################################################################################

from deephaven import QueryScope
from deephaven import npy
import numpy as np
import jpy

class Input:
    def __init__(self, columns, gather):
        if type(columns) is list: 
            self.columns = columns
        else:
            self.columns = [columns]

        self.gather = gather

class Output:
    def __init__(self, column, scatter, col_type="java.lang.Object"):
        self.column = column
        self.scatter = scatter
        self.col_type = col_type


#TODO: clearly in production code there would need to be extensive testing of inputs and outputs (e.g. no null, correct size, ...)
#TODO: ths is a static example, real time requires more work
#TODO: this is not written in an efficient way.  it is written quickly to get something to look at

# this handles input so that user does not always have to enter every column they want to use
def _parse_input(inputs, table):
    # what are all possible cases
    new_inputs = inputs
    # input length zero - problem
    if len(inputs) == 0:
        raise ValueError('The input list cannot have length 0.')
    # first input list of features
    elif len(inputs) == 1:
        # if list of features is empty, replace with all columns and return
        if len(inputs[0].columns) == 0:
            new_inputs[0].columns = list(table.getMeta().getColumn("Name").getDirect())
            return new_inputs
        else:
            return new_inputs
    else:
        # now that we know input length at least 2, ensure target non-empty
        if len(inputs[0].columns) == 0:
            raise ValueError('Target input cannot be empty.')
        else:
            target = inputs[0].columns
            # look through every other input to find empty list
            for i in range(1,len(inputs)):
                if len(inputs[i].columns) == 0:
                    new_inputs[i].columns = list(table.dropColumns(target).getMeta().getColumn("Name").getDirect())
                else:
                    pass
            return new_inputs
        
        
def _gather_input(table, input):
    # converts selected columns to numpy and removes axes of length 1
    npy_table = np.squeeze(npy.numpy_slice(table.view(input.columns), 0, table.size()))
    return input.gather(*npy_table)

def _gather_input_original(table, input):
    #TODO: getDirect is probably terribly slow here, but it makes short code
    data = [ table.getColumn(col).getDirect() for col in input.columns ]
    return input.gather(*data)


def ai_eval(table=None, model_func=None, inputs=[], outputs=[]):
    print("SETUP")
    # append default inputs to inputs if needed
    inputs = _parse_input(inputs, table)

    print("GATHER")
    # note that the default is now row-wise, which makes sense to me. Add feature to allow user to select axis of compression
    gathered = [ _gather_input(table, input) for input in inputs ]

    # if there are no outputs given, we just want to return whatever model_func returns
    if outputs == None:
        print("COMPUTE NEW DATA")
        return model_func(*gathered)

    else:
        print("COMPUTE NEW DATA")
        output_values = model_func(*gathered)

        print(type(output_values))

        print("POPULATE OUTPUT TABLE")
        rst = table.by()
        n = table.size()

        for output in outputs:
            print(f"GENERATING OUTPUT: {output.column}")
            #TODO: maybe we can infer the type?
            data = jpy.array(output.col_type, n)

            #TODO: python looping is slow.  should avoid or numba it
            for i in range(n):
                data[i] = output.scatter(output_values, i)

            QueryScope.addParam("__temp", data)
            rst = rst.update(f"{output.column} = __temp")

        return rst.ungroup()

################################################################################################################################
# Everything here would be user created -- or maybe part of a DH library if it is common functionality
################################################################################################################################

import torch
import torchsummary
import torch.nn as nn
from torch.optim import SGD
from torch import optim
import optuna

from numpy import argmax
from numpy import vstack
from sklearn.metrics import accuracy_score

from deephaven.TableTools import readCsv

# set seed for reproducibility
torch.manual_seed(17306168389181004404)

# import data from sample data directory
iris = readCsv("/data/examples/iris/csv/iris.csv")

# since Class is categorical, we need to convert it to numeric
# TODO: tihs is not great, a function to do all of this for me would be nice
iris = iris.aj(iris.by("Class")\
    .update("idx = i"), "Class", "idx")\
    .dropColumns("Class")\
    .renameColumns("Class = idx")


# first, we need to define an optuna version of our MLP
def define_model(trial: optuna.trial.Trial) -> nn.Sequential:

    n_layers = trial.suggest_int("n_layers", 1, 5)
    layers = []

    in_features = 4
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_1{}".format(i), 3, 15)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())

        in_features = out_features
    layers.append(nn.Linear(in_features, 3))
    layers.append(nn.Softmax(dim=1))

    return nn.Sequential(*layers)


# define function to return accuracy
def train_and_validate(model, target, features, optimizer):

    # first, since we pass ai_eval one DH table, we must perform train/test split here
    split_permutation = torch.randperm(features.size()[0])
    num_train = round(features.size()[0] * split)
    train_ind = split_permutation[0 : num_train - 1]
    test_ind = split_permutation[num_train : features.size()[0] - 1]
    train_target, train_features = target[train_ind], features[train_ind]
    test_target, test_features = target[test_ind], features[test_ind]
    # first, we train the model using the code from train_model given above.
    # enumerate epochs, one loop represents one full pass through dataset
    for epoch in range(epochs):
        # create permutation for selecting mini batches
        permutation = torch.randperm(train_features.size()[0])
        # enumerate mini batches, one loop represents one batch for updating gradients and loss
        for i in range(0, train_features.size()[0], batch_size):
            # compute indices for this batch and split
            indices = permutation[i:i+batch_size]
            target_batch, features_batch = train_target[indices], train_features[indices]
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(features_batch)
            # calculate loss
            loss = criterion(yhat, target_batch)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
    # print out a model summary using the torchsummary package
    torchsummary.summary(model, (1,) + tuple(features.size()))

    # now that we've trained the model, we perform validation on our test set, again using the code above
    predictions, actuals = list(), list()
    # evaluate the model on the test set
    yhat = model(test_features)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    actual = test_target.numpy()
    # convert to class labels
    yhat = argmax(yhat, axis=1)
    # reshape for stacking
    actual = actual.reshape((len(actual), 1))
    yhat = yhat.reshape((len(yhat), 1))
    # store
    predictions.append(yhat)
    actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    print("Accuracy score: " + str(acc))

    return acc


# now, we must define an objective function that takes a model, data and returns a score
def objective(trial, target, features):

    # define model
    model = define_model(trial)
    # generate optimizers
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # call train and validate to compute accuracy
    accuracy = train_and_validate(model, target, features, optimizer)

    return accuracy


# now we use this function to call all the others
def tune_model(target, features):

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, target, features), n_trials=10)

    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return None


# define parameters that are not to be tuned
criterion = nn.CrossEntropyLoss()
epochs = 500
batch_size = 20
split = .75


# define gather functions
def to_tensor_long(*data):
    return torch.tensor(data).long()

def to_tensor_float(*data):
    return torch.tensor(data).float()

ai_eval(table = iris, model_func = tune_model,
    inputs = [Input("Class", to_tensor_long), Input([], to_tensor_float)], outputs = None)
