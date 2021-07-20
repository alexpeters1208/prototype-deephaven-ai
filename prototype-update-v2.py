# COMPLETELY DOES NOT WORK

from deephaven import QueryScope
from deephaven import npy
from deephaven import listen
import numpy as np
import jpy

class Input:
    """
    The Input class provides an interface for converting Deephaven tables to objects that Python deep learning libraries
    are familiar with. Input objects are intended to be used as the input argument of an eval() function call.
    """
    def __init__(self, columns, gather):
        """
        :param columns: the list of column names from a Deephaven table that you want to use in modelling
        :param gather: the function that determines how data from a Deephaven table is collected
        """
        if type(columns) is list: 
            self.columns = columns
        else:
            self.columns = [columns]

        self.gather = gather

        
class Output:
    """
    The Output class provides an interface for converting Python objects (such as tensors or dataframes) into Deephaven
    tables. Output objects are intended to be used as the output argument of an eval() function call.
    """
    def __init__(self, column, scatter, col_type="java.lang.Object"):
        """
        :param column: the string name of the column you would like to create to store your output
        :param scatter: the function that determines how data from a Python object is stored in a Deephaven table
        :param col_type: optional string that defaults to 'java.lang.Object', determines the type of output column
        """
        self.column = column
        self.scatter = scatter
        self.col_type = col_type


#TODO: this should be java
class IndexSet:
    def __init__(self, max_size):
        self.max_size = max_size
        self.current = -1
        self.idx = np.empty(max_size, dtype=np.int64)

    def clear(self):
        self.current = -1
        self.idx = np.empty(self.max_size, dtype=np.int64)

    def add(self, kk):
        if self.current == self.idx.size:
            raise Exception("Adding more indices than can fit")

        self.current += 1
        self.idx[self.current] = kk

    def is_full(self):
        return len(self) >= self.idx.size

    def __len__(self):
        return self.current + 1

    def __getitem__(self, i):
        if i >= len(self):
            raise Exception("Index out of bounds")

        return self.idx[i]



#TODO: this should probably be java
class Future:
    def __init__(self, func, inputs, col_sets, batch_size):
        self.func = func
        self.inputs = inputs
        self.col_sets = col_sets
        self.index_set = IndexSet(batch_size)
        self.called = False
        self.result = None

    def clear(self):
        self.result = None

    def get(self):
        if not self.called:
            self.result = self.index_set.idx[:len(self.index_set)]
            #gathered = [ input.gather(self.index_set.idx[:len(self.index_set)], col_set) for (input,col_set) in zip(self.inputs, self.col_sets) ]
            #self.result = self.func(gathered)
            self.index_set.clear()
            self.called = True
            self.func = None
            self.index_set = None

        return self.result


#TODO: this should probably be java
class Computer:
    def __init__(self, func, inputs, col_sets, batch_size):
        self.func = func
        self.inputs = inputs # this is just an Input object, low memory cost!
        self.col_sets = col_sets # this is just a Python list of references to column sources, low memory cost!
        self.batch_size = batch_size
        self.current = None

    def clear(self):
        self.current = None

    def compute(self, kk):
        # only instantiate a new future if we do not have one, or the current one is full, meaning we've reached the batch size
        if self.current == None or self.current.index_set.is_full():
            self.current = Future(self.func, self.inputs, self.col_sets, self.batch_size)

        self.current.index_set.add(kk)
        return self.current


#TODO: this should be java
class Scatterer:
    def __init__(self, batch_size, scatter_func):
        self.batch_size = batch_size
        self.count = -1
        self.scatter_func = scatter_func

    def clear(self):
        self.count = -1

    def scatter(self, data):
        self.count += 1
        offset = self.count % self.batch_size
        return self.scatter_func(data, offset)


def _parse_input(inputs, table):
    """
    Converts the list of user inputs into a new list of inputs with the following rules:
    
        inputs = [Input([], gather)]
        will be transformed into a list containing a new Input object, with every column in the table as an element in
        Input.columns. This allows users to not have to type all column names to use all features.
        
        inputs = [Input(["target"], gather), Input([], gather)]
        will be transformed into a list containing two Input objects; the first will be unchanged and represent the
        target variable, the second will be transformed to an Input object containing all column names in the dataset
        except for the target. This allows users to not have to type all column names to use all features.
        
        If inputs is of length 2 or greater, we assume that the first Input object is the target variable and insist
        that it be non-empty.
        
    :param inputs: the list of Input objects that gets passed to eval()
    :param table: the Deephaven table that gets passed to eval()
    """
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


def eval(table, model_func, inputs, outputs, batch_size=1000):

    # first we transform inputs and gather into column sets
    inputs = _parse_input(inputs, table)
    col_sets = [ [ table.getColumnSource(col) for col in input.columns ] for input in inputs ]

    # instantiate objects used for input and output in .update
    computer = Computer(model_func, inputs, col_sets, batch_size)
    #scatterer = Scatterer(batch_size, scatter_func)

    return computer

    #TODO: python is having major problems.  It doesn't resolve these variables inside of a function, and when you try to add them, it complains they aren't java
    #TODO: may need to implement this function in Java as well to avoid some problems.  Right now, it doesn't run.
    #QueryScope.addParam("computer", computer)
    #QueryScope.addParam("scatterer_x", scatterer_x)

    def cleanup(future):
        computer.clear()
        future.clear()
        scatterer_x.clear()

    #return table.update("Future = computer.compute(kk)", "X = (double) scatterer_x(Future.get())", "Clean = cleanup(Future)") \
    #    .dropColumns("Future", "Clean")


################################################################################################################################
import torch
from deephaven.TableTools import readCsv

# import data from sample data directory
iris = readCsv("/data/examples/iris/csv/iris.csv")

# since Class is categorical, we need to convert it to numeric
# TODO: tihs is not great, a function to do all of this for me would be nice
iris = iris.aj(iris.by("Class")\
    .update("idx = i"), "Class", "idx")\
    .dropColumns("Class")\
    .renameColumns("Class = idx")


def identity(*args):
    return args

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
computer = eval(table = iris, model_func = identity,
    inputs = [Input("Class", tensor_1d), Input([], tensor_2d)], outputs = [Output("Predicted", to_scalar, "int")], batch_size=5)
