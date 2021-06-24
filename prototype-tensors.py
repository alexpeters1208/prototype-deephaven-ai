
# This example demonstrates a case where a user function creates tensors for each row.  
# The model function is then evaluated for each row to create results for the row.

################################################################################################################################
# Everything here would be part of a DH library
################################################################################################################################

from deephaven import QueryScope
import jpy

#TODO: clearly in production code there would need to be extensive testing of inputs and outputs (e.g. no null, correct size, ...)
#TODO: ths is a static example, real time requires more work
#TODO: this is not written in an efficient way.  it is written quickly to get something to look at

def ai_eval(table=None, model=None, inputs=[], outputs=[]):
    print("SETUP")
    n = table.size()
    #TODO: not sure what the right output column types are.  Maybe some sort of python object?
    outs = [ jpy.array("java.lang.Object", n) for o in outputs ]
    columns = [ table.getColumn(col) for col in inputs ]

    print("COMPUTE NEW DATA")
    # python looping is slow.  should avoid or numba it
    for i in range(n):
        input_values = [ col.get(i) for col in columns ]
        output_values = model(*input_values)

        for (j,ov) in enumerate(output_values):
            outs[j][i] = ov

    print("POPULATE OUTPUT TABLE")
    rst = table.by()

    for (col,data) in zip(outputs, outs):
        QueryScope.addParam("__temp", data)
        rst = rst.update(f"{col} = __temp")

    return rst.ungroup()



################################################################################################################################
# Everything here would be user created -- or maybe part of a DH library if it is common functionality
################################################################################################################################

import numpy as np
from deephaven.TableTools import emptyTable

t = emptyTable(10).update("X = i", "Y = sqrt(X)")

def make_tensor(x):
    return np.random.rand(3,2) + x

t2 = t.update("Z = make_tensor(X)")

def model_func(a,b,c):
    return 3*a, b+11, b + 32


t3 = ai_eval(table=t2, model=model_func, inputs=["X", "Y", "Z"], outputs=["A", "B", "C"])

#TODO: dropping weird column types to avoid some display bugs
meta2 = t2.getMeta()
t2 = t2.dropColumns("Z")
meta3 = t3.getMeta()
t3 = t3.dropColumns("Z", "B", "C")