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
    inputs = _parse_input(inputs, table)
    col_sets = [ [ table.getColumnSource(col) for col in input.columns ] for input in inputs ]

    print("GATHER")
    #TODO: for real time, the IndexSetIterator would be populated with the ADD and MODIFY indices
    idx = IndexSetIterator(table.getIndex())
    gathered = [ input.gather(idx, col_set) for (input,col_set) in zip(inputs,col_sets) ]

    # if there are no outputs, we just want to call model_func and return nothing
    if outputs == None:
        print("COMPUTE NEW DATA")
        model_func(*gathered)
        return

    else:
        print("COMPUTE NEW DATA")
        output_values = model_func(*gathered)

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


from deephaven import TableTools as ttools
from deephaven import listen

data = ttools.timeTable("00:00:10").update("X=i", "Y=new int[]{0,1,2}").tail(5).ungroup()
static_data = ttools.emptyTable(15).snapshot(data)

# here, f is known as the listener
def f(update):
    
    print(type(update.added))
    print(f"ADDED: ", update.added)
    print(f"MODIFIED: ", update.modified)
    print(f"DELETED: ", update.removed)
    print(f"REINDEXED: ", update.shifted)
    print(f"ModifiedColumnSet: ", update.modifiedColumnSet)

handle = listen(table, f)





new_table = ai_eval(table = data, model_func = do_computation,
                    inputs = [Input("X", gather)], outputs = [Output("New", scatter)])
