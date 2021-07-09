# This example demonstrates a case where a user function creates partial tensors for each row.
# These partial tensors are aggregated into tensors before evaluating the model.  
# The aggregation should result in more efficient use of the AI machinery.  
# The model function is then evaluated for each row to create results for the row.

################################################################################################################################
# Everything here would be part of a DH library
################################################################################################################################

from deephaven import QueryScope
from deephaven import npy
from deephaven import listen
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

#TODO: this should be implemented in Java for speed.  This efficiently iterates over the indices in multiple index sets.  Works for hist and real time.
class IndexSetIterator:
    def __init__(self, *indexes):
        self.indexes = indexes

    def __len__(self):
        rst = 0

        for index in self.indexes:
            rst += index.size()

        return rst

    def __iter__(self):
        for index in self.indexes:
            it = index.iterator()

            while it.hasNext():
                yield it.next()


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


# this will be called from AI eval and create the necessary output
def _create_output(table=None, model_func=None, gathered=[], outputs=[]):
    # if there are no outputs, we just want to call model_func and return nothing
    if outputs == None:
        print("COMPUTE NEW DATA")
        model_func(*gathered)
        return

    else:
        print("COMPUTE NEW DATA")
        output_values = model_func(*gathered)
        print(output_values)

        print("POPULATE OUTPUT TABLE")
        rst = table.by()

        return

        n = table.size()

        for output in outputs:
            print(f"GENERATING OUTPUT: {output.column}")
            #TODO: maybe we can infer the type
            data = jpy.array(output.col_type, n)

            #TODO: python looping is slow.  should avoid or numba it
            # this is the line that breaks, the bad logic is probably elsewhere
            for i in range(n):
                data[i] = output.scatter(output_values, i)

            QueryScope.addParam("__temp", data)
            rst = rst.update(f"{output.column} = __temp")
            
        return rst.ungroup()


# this is what the user calls to make DH data interact with standard PT/TF functions that they create
def ai_eval(table=None, model_func=None, live=False, inputs=[], outputs=[]):
    print("SETUP")
    inputs = _parse_input(inputs, table)
    col_sets = [ [ table.getColumnSource(col) for col in input.columns ] for input in inputs ]

    print("GATHER")
    # this is where we need to begin making the distinction between static and live data
    if live:
        # instantiate class to listen to updates and update output accordingly
        listener = ListenAndReturn(table, model_func, inputs, outputs, col_sets)
        handle = listen(table, listener, replay_initial=True)

    else:
        idx = IndexSetIterator(table.getIndex())
        gathered = [ input.gather(idx, col_set) for (input,col_set) in zip(inputs,col_sets) ]
        return _create_output(table, model_func, gathered, outputs)


class ListenAndReturn:
    def __init__(self, table, model_func, inputs, outputs, col_sets):
        self.table = table
        self.model_func = model_func
        self.inputs = inputs
        self.outputs = outputs
        self.col_sets = col_sets
        self.newTable = None

    def onUpdate(self, isReplay, update):
        self.idx = IndexSetIterator(update.added, update.modified)
        self.gathered = [ input.gather(self.idx, col_set) for (input,col_set) in zip(self.inputs, self.col_sets) ]
        self.newTable = _create_output(self.table, self.model_func, self.gathered, self.outputs)

################################################################################################################################
# Everything here would be user created -- or maybe part of a DH library if it is common functionality
################################################################################################################################

import torch
import torchsummary
import torch.nn as nn
from torch.optim import SGD

from numpy import argmax
from numpy import vstack
from sklearn.metrics import accuracy_score

import numpy as np

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

# create model, this does not change with how you interact with ai_eval, so we put it at the top

# model definition
class MLP(nn.Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = nn.Linear(n_inputs, 10)
        nn.init.kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        # second hidden layer
        self.hidden2 = nn.Linear(10, 8)
        nn.init.kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()
        # third hidden layer and output
        self.hidden3 = nn.Linear(8, 3)
        nn.init.xavier_uniform_(self.hidden3.weight)
        self.act3 = nn.Softmax(dim=1)
 
    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # output layer
        X = self.hidden3(X)
        X = self.act3(X)
        return X


# define model and set hyperparameters
model = MLP(4)
criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
epochs = 500
batch_size = 20
split = .75

def train_and_validate(target, features):
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

    predicted_classes = torch.argmax(model(features),1)

    return predicted_classes


def tensor_2d(idx, cols):
    rst = torch.empty(len(idx), len(cols), dtype=torch.float32)

    for (i,kk) in enumerate(idx):
        for (j,col) in enumerate(cols):
            rst[i,j] = col.get(kk)

    return rst

def tensor_1d(idx, col):
    rst = torch.empty(len(idx), dtype=torch.long)

    for (i,kk) in enumerate(idx):
        rst[i] = col[0].get(kk)

    return rst

def to_scalar(data, i):
    return int(data[i])


# supervised learning on all features, target first
predicted = ai_eval(table = iris, model_func = train_and_validate, live=False,
    inputs = [Input("Class", tensor_1d), Input([], tensor_2d)],
    outputs = [Output("Predicted", to_scalar, "int")])

################################################################################################################################
# Generating iris dataset to classify incoming observations
################################################################################################################################

from deephaven import jpy
import threading, time

# Step 1: Fetch the object
DynamicTableWriter = jpy.get_type("io.deephaven.db.v2.utils.DynamicTableWriter")

# Step 2: Create the object
tableWriter = DynamicTableWriter(["SepalLengthCM", "SepalWidthCM", "PetalLengthCM", "PetalWidthCM"],
    [jpy.get_type("double"), jpy.get_type("double"), jpy.get_type("double"), jpy.get_type("double")])

# set name of live table
live_iris = tableWriter.getTable()

# define function to create live table
def thread_func():
    for i in range(100):
        sepLen = np.absolute(np.around(np.random.normal(5.8433, 0.6857, 1)[0], 1))
        sepWid = np.absolute(np.around(np.random.normal(3.0573, 0.19, 1)[0], 1))
        petLen = np.absolute(np.around(np.random.normal(3.7580, 3.1163, 1)[0], 1))
        petWid = np.absolute(np.around(np.random.normal(1.1993, 0.5810, 1)[0], 1))
        # The logRow method adds a row to the table
        tableWriter.logRow(sepLen, sepWid, petLen, petWid)
        time.sleep(5)

# create thread to execute data creation function
thread = threading.Thread(target = thread_func)
thread.start()


################################################################################################################################
# Classify LIVE iris dataset using the exact same framework as for the static case!
################################################################################################################################

# function to make prediction on observations of the iris dataset
def make_predictions(features):
    # use model to predict probabilities and take the maximum probability
    predicted_classes = torch.argmax(model(features),1)
    return predicted_classes

# function to collect input as tensor
def tensor_2d(idx, cols):
    rst = torch.empty(len(idx), len(cols), dtype=torch.float32)

    for (i,kk) in enumerate(idx):
        for (j,col) in enumerate(cols):
            rst[i,j] = col.get(kk)

    return rst

# function to distribute output to scalar
def to_scalar(data, i):
    return int(data[i])


more_predictions = ai_eval(table = live_iris, model_func = make_predictions, live=True,
    inputs = [Input([], tensor_2d)], outputs = [Output("Predicted", to_scalar, "int")])
