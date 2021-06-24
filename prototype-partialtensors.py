
# This example demonstrates a case where a user function creates partial tensors for each row.
# These partial tensors are aggregated into tensors before evaluating the model.  
# The aggregation should result in more efficient use of the AI machinery.  
# The model function is then evaluated for each row to create results for the row.

################################################################################################################################
# Everything here would be part of a DH library
################################################################################################################################

from deephaven import QueryScope
import jpy

#TODO: clearly in production code there would need to be extensive testing of inputs and outputs (e.g. no null, correct size, ...)
#TODO: ths is a static example, real time requires more work
#TODO: this is not written in an efficient way.  it is written quickly to get something to look at

def ai_eval(table=None, model=None, inputs=[], outputs=[], gather=[], scatter=[]):
    print("SETUP")
    columns = [ table.getColumn(col) for col in inputs ]

    print("GATHER")
    #TODO: getDirect is probably terribly slow here, but it makes short code
    gathered = [ g(c.getDirect()) for (g,c) in zip(gather,columns)]

    print("COMPUTE NEW DATA")
    output_values = model(*gathered)

    print("POPULATE OUTPUT TABLE")
    rst = table.by()
    n = table.size()

    for (out_col, scatter_func, out_vals) in zip(outputs, scatter, output_values):
        #TODO: not sure what the right output column types are.  Maybe some sort of python object?
        #TODO: hard coded as int32 (https://jpy.readthedocs.io/en/latest/reference.html)
        data = jpy.array("java.lang.Object", n)
        # data = jpy.array("[I", n)

        #TODO: python looping is slow.  should avoid or numba it
        for i in range(n):
            data[i] = scatter_func(out_vals, i)

        QueryScope.addParam("__temp", data)
        rst = rst.update(f"{out_col} = __temp")

    return rst.ungroup()



################################################################################################################################
# Everything here would be user created -- or maybe part of a DH library if it is common functionality
################################################################################################################################

import random
import numpy as np
from deephaven.TableTools import emptyTable

class ZNugget:
    def __init__(self, payload):
        self.payload = payload

def make_z(x):
    return ZNugget([random.randint(4,11)+x for z in range(5)])
    # return np.random.rand(3,2) + x

def gather_x(data):
    return np.array(data)

def gather_y(data):
    return np.array(data)

def gather_z(data):
    return np.array([ d.payload for d in data ])

def scatter_a(data, i):
    return int(data[i])

def scatter_b(data, i):
    return data[i]

def scatter_c(data, i):
    return data[i]

def model_func(a,b,c):
    return 3*a, b+11, b + 32

t = emptyTable(10).update("X = i", "Y = sqrt(X)")
t2 = t.update("Z = make_z(X)")
t3 = ai_eval(table=t2, model=model_func, inputs=["X", "Y", "Z"], outputs=["A", "B", "C"], gather=[gather_x, gather_y, gather_z], scatter=[scatter_a, scatter_b, scatter_c])

#TODO: dropping weird column types to avoid some display bugs
meta2 = t2.getMeta()
t2 = t2.dropColumns("Z")
meta3 = t3.getMeta()
t3 = t3.dropColumns("Z", "B", "C")