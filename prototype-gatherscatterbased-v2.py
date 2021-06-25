
# This example demonstrates a case where a user function creates partial tensors for each row.
# These partial tensors are aggregated into tensors before evaluating the model.  
# The aggregation should result in more efficient use of the AI machinery.  
# The model function is then evaluated for each row to create results for the row.

################################################################################################################################
# Everything here would be part of a DH library
################################################################################################################################

from deephaven import QueryScope
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

def __gather_input(table, input):
    #TODO: getDirect is probably terribly slow here, but it makes short code
    data = [ table.getColumn(col).getDirect() for col in input.columns ]
    return input.gather(*data)


#TODO: clearly in production code there would need to be extensive testing of inputs and outputs (e.g. no null, correct size, ...)
#TODO: ths is a static example, real time requires more work
#TODO: this is not written in an efficient way.  it is written quickly to get something to look at

def ai_eval(table=None, model=None, inputs=[], outputs=[]):
    print("SETUP")
    # columns = [ table.getColumn(col) for col in inputs ]

    print("GATHER")
    gathered = [ __gather_input(table, input) for input in inputs ]

    print("COMPUTE NEW DATA")
    output_values = model(*gathered)

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

import random
import numpy as np
from math import sqrt
from deephaven.TableTools import emptyTable

class ZNugget:
    def __init__(self, payload):
        self.payload = payload

def make_z(x):
    return ZNugget([random.randint(4,11)+x for z in range(5)])

def gather_x(data):
    return np.array(data)

def gather_xy(x_data, y_data):
    return np.array(x_data) + np.array(y_data)

def gather_z(data):
    return np.array([ d.payload for d in data ])

def scatter_a(data, i):
    return int(data[0][i])

def scatter_b(data, i):
    return data[1][i]

def scatter_c(data, i):
    return sqrt(data[2][i] + data[1][i])

def model_func(a,b,c):
    return 3*a, b+11, b + 32

t = emptyTable(10).update("X = i", "Y = sqrt(X)")
t2 = t.update("Z = make_z(X)")
t3 = ai_eval(table=t2, model=model_func, inputs=[Input("X", gather_x), Input(["X", "Y"], gather_xy), Input("Z", gather_z)], outputs=[Output("A",scatter_a, col_type="int"), Output("B",scatter_b), Output("C",scatter_c)])

#TODO: dropping weird column types to avoid some display bugs
meta2 = t2.getMeta()
t2 = t2.dropColumns("Z")
meta3 = t3.getMeta()
t3 = t3.dropColumns("Z", "B", "C")