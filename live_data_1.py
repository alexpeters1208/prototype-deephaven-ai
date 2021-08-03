from deephaven import learn
from deephaven.TableTools import readCsv
import jpy

import torch
import torchsummary
import torch.nn as nn
from torch.optim import SGD

from numpy import argmax
from numpy import vstack
from sklearn.metrics import accuracy_score

# set seed for reproducibility
torch.manual_seed(17306168389181004404)

############################################################################
# import data from sample data directory
iris = readCsv("/data/examples/iris/csv/iris.csv")

# since Class is categorical, we need to convert it to numeric
# TODO: tihs is not great, a function to do all of this for me would be nice
iris = iris.aj(iris.by("Class")\
    .update("idx = i"), "Class", "idx")\
    .dropColumns("Class")\
    .renameColumns("Class = idx")


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
        self.act3 = nn.Softmax()
 
    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X.float())
        X = self.act1(X.float())
        # second hidden layer
        X = self.hidden2(X.float())
        X = self.act2(X.float())
        # output layer
        X = self.hidden3(X.float())
        X = self.act3(X.float())
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
            loss = criterion(yhat, target_batch.long())
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


def pt_tensor(idx, cols):

    rst = torch.empty(idx.size(), len(cols), dtype=torch.double)
    iter = idx.iterator()
    i = 0

    while(iter.hasNext()):
        it = iter.next()
        j = 0
        for col in cols:
            rst[i,j] = col.get(it)
            j=j+1
        i=i+1

    return torch.squeeze(rst)
    #return rst

def to_scalar(data, k):
    return int(data[k])

predicted = learn.eval(table = iris, model_func = train_and_validate,
    inputs = [learn.Input(["Class"], pt_tensor), learn.Input(["SepalLengthCM","SepalWidthCM","PetalLengthCM","PetalWidthCM"], pt_tensor)],
    outputs = learn.Output(["Predicted"], to_scalar, "int"))

###########################################################################################################

from deephaven import jpy
import numpy as np
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
        time.sleep(3)

# create thread to execute data creation function
thread = threading.Thread(target = thread_func)
thread.start()

###########################################################################################################

def make_predictions(features):
    # use model to predict probabilities and take the maximum probability
    predicted_classes = torch.argmax(model(features), 1)
    return predicted_classes

def to_scalar(data, k):
    return int(data[k])

def pt_tensor(idx, cols):

    rst = torch.empty(idx.size(), len(cols), dtype=torch.double)
    iter = idx.iterator()
    i = 0

    while(iter.hasNext()):
        it = iter.next()
        j = 0
        for col in cols:
            rst[i,j] = col.get(it)
            j=j+1
        i=i+1

    return rst

live_predicted = learn.eval(table = live_iris, model_func = make_predictions,
    inputs = [learn.Input(["SepalLengthCM","SepalWidthCM","PetalLengthCM","PetalWidthCM"], pt_tensor)],
    outputs = learn.Output(["PredictedClass"], to_scalar, "int"), batch_size=10)
