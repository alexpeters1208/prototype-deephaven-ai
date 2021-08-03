from deephaven import learn
from deephaven.TableTools import readCsv
import torch
import jpy

############################################################################
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

def to_scalar(data, i):
    return int(data[i])


predicted = learn.eval(table = iris, model_func = identity,
    inputs = [learn.Input(["Class"], pt_tensor), learn.Input(["SepalLengthCM","SepalWidthCM","PetalLengthCM","PetalWidthCM"], pt_tensor)],
    outputs = [learn.Output(["Prob0","Prob1","Prob2"], to_scalar, "float"), learn.Output(["Predicted"], to_scalar, "int")])

###########################################################################################################

from deephaven import jpy
import numpy as np
import threading, time

# Step 1: Fetch the object
DynamicTableWriter = jpy.get_type("io.deephaven.db.v2.utils.DynamicTableWriter")

# Step 2: Create the object
tableWriter = DynamicTableWriter(["SepalLengthCM", "SepalWidthCM", "PetalLengthCM", "PetalWidthCM", "Class"],
    [jpy.get_type("double"), jpy.get_type("double"), jpy.get_type("double"), jpy.get_type("double"), jpy.get_type("int")])

# set name of live table
live_iris = tableWriter.getTable()

# define function to create live table
def thread_func():
    for i in range(100):
        sepLen = np.absolute(np.around(np.random.normal(5.8433, 0.6857, 1)[0], 1))
        sepWid = np.absolute(np.around(np.random.normal(3.0573, 0.19, 1)[0], 1))
        petLen = np.absolute(np.around(np.random.normal(3.7580, 3.1163, 1)[0], 1))
        petWid = np.absolute(np.around(np.random.normal(1.1993, 0.5810, 1)[0], 1))
        cls = int(np.random.randint(0, 3, 1)[0])

        # The logRow method adds a row to the table
        tableWriter.logRow(sepLen, sepWid, petLen, petWid, cls)
        time.sleep(5)

# create thread to execute data creation function
thread = threading.Thread(target = thread_func)
thread.start()

###########################################################################################################

live_predicted = learn.eval(table = live_iris, model_func = identity,
    inputs = [learn.Input(["Class"], pt_tensor), learn.Input(["SepalLengthCM","SepalWidthCM","PetalLengthCM","PetalWidthCM"], pt_tensor)],
    outputs = [learn.Output(["Predicted"], to_scalar, "int")], batch_size=10)
