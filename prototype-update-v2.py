
# This example demonstrates prototype of the update model that Ryan, Alex, and Chip discussed.

################################################################################################################################
# Everything here would be part of a DH library
################################################################################################################################

import numpy as np
from deephaven import QueryScope

#TODO: this should be java
class IndexSet:
    def __init__(self, max_size):
        self.current = -1
        self.idx = np.zeros(max_size, dtype=np.int64)

    def clear(self):
        self.current = -2
        self.idx = None

    def add(self, kk):
        if self.current == self.idx.size:
            raise Exception("Adding more indices than can fit")

        self.current += 1
        self.idx[self.current] = kk

    def is_full(self):
        return self.size() >= self.idx.size

    def __len__(self):
        return self.current + 1

    def __getitem__(self, i):
        if i >= len(self):
            raise Exception("Index out of bounds")

        return self.idx[i]



#TODO: this should probably be java
class Future:
    def __init__(self, func, batch_size):
        self.func = func
        self.index_set = IndexSet(batch_size)
        self.called = False
        self.result = None

    def clear(self):
        self.result = None

    def get(self):
        if not self.called:
            # data should be gathered here and then passed to model instead of doing it all in one go. That means were
            # going to need to get the input objects here somehow, and I think that complicates things quite a bit.
            
            # self.func gets passed an index set, but I think it should get passed the gathered data.
            # otherwise, how does the interface not change?
            self.result = self.func(self.index_set)
            self.index_set.clear()
            self.called = True
            self.func = None
            self.index_set = None

        return self.result


#TODO: this should probably be java
class Computer:
    def __init__(self, batch_size, func):
        self.func = func
        self.batch_size = batch_size
        self.current = None

    def clear(self):
        self.current = None

    def compute(self, kk):
        if self.current == None or self.current.index_set.is_full():
            self.current = Future(self.func, self.batch_size)

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


def do_magic(table, model, scatter_func, batch_size):
    #TODO: horrible hack
    def gather_it(index_set):
        print("Calling gather_it")
        data = np.zeros([len(index_set), 3], dtype=np.float64)
        
        for i in range(len(index_set)):
            data[i,0] = table.getColumnSource("A", index_set[i])
            data[i,1] = table.getColumnSource("B", index_set[i])
            data[i,2] = table.getColumnSource("C", index_set[i])

        return data

    #TODO: horrible hack
    def eval_func(index_set):
        print("Calling eval_func")
        data = gather_it(index_set)
        return model(data)

    computer = Computer(batch_size, eval_func)
    scatterer_x = Scatterer(batch_size, scatter_func)

    #TODO: python is having major problems.  It doesn't resolve these variables inside of a function, and when you try to add them, it complains they aren't java
    #TODO: may need to implement this function in Java as well to avoid some problems.  Right now, it doesn't run.
    QueryScope.addParam("computer", computer)
    QueryScope.addParam("scatterer_x", scatterer_x)

    def cleanup(future):
        computer.clear()
        future.clear()
        scatterer_x.clear()

    return table.update("Future = computer.compute(kk)", "X = (double) scatterer_x.scatter(Future.get())", "Clean = cleanup(Future)") \
        .dropColumns("Future", "Clean")


################################################################################################################################
# Everything here would be part of user code
################################################################################################################################


def model(data):
    return np.sum(data, axis=1)

def scatter(data,i):
    return data[i]

from deephaven.TableTools import timeTable 
source = timeTable("00:00:01").update("A=i", "B=sqrt(i)", "C=i*i") 

batch_size = 10

result = do_magic(source, model, scatter, batch_size)
