Here is a list of things that suck about this example, in order of appearance in the code. (!'s are suck level, 1-5, developer perspective)

##

(!)
1: The "default input" way of doing things (line 39) is not optimal. We use a string variable as a key argument and apparently that is not so good.
   For now, this works, but it would be nice to find a better way to do things.
   
   Potential solutions:
   
   I have no idea.
      
##
      
(!!!!!)
2: The forced transformations in ai_eval (lines 60 & 61) suck hard as they do not generalize at all. They are solving the problem of getting input data
   into the dimsions that PyTorch likes (rows x columns, flat target tensor) rather than the dimensions that are produced from gather (columns x rows).
   It is not obvious to me that modifying the relevant gather functions is enough, because there is a .getColumn in __gather_input__ that insists that
   data be parsed columnwise by the gather function.
   
   Potential solutions:
   
   1. R will often ask a user if they want to perform a given operation row-wise or column-wise, in the form of a 1 or 2 function argument.
      This is nice, but doesn't generalize to higher dimensions
         
   2. We can ask the user for a vector of dimensions that they want to impose on their data in the ai_eval function call.
      This leaves this issue entirely in the hands of the user, which gives flexibility but possibly reduces ease of use.
         
   3. We can try to figure out from the given model_func what shapes the model expects and transform given data as needed. This sunds like it
      would suck to implement, as I'm not sure if it's possible to get such information just from model_func.
         
   4. Ideally, we generalize the __gather_input__ function to be able to aggregate the data in whatever way the user wants to do, without needing
      to specify additional dimensionality arguments. Not sure how to make this happen.
 
##
         
(!!)
3: The output column is currently just a Java object. There is a "TODO: maybe we can infer the type?", does this mean figure out what kind of output the
   model generates and create the appropriate column type with jpy? I think so. This probably isn't too difficult, just have to figure it out.
    
   Potential solutions:
    
   1. Figure out how to get datatypes from returned tensors, independent of Tensorflow or PyTorch, etc. Then use this type to create column of correct type.
       
   2. Have users declare desired output type in the output class.
    
##

(!!!!)
4: Converting Deephaven columns from categorical to numeric (line 124) is a little difficult, and I think this difficulty would increase exponentially with
   the number of categorical variables.
   
   Potential solutions:
   
   1. Create a Deephaven function to perform this kind of categorical encoding on a table that we would provide to the user out of the box.
      
   2. Find a really easy way to do this kind of encoding with the query language in a way that does not get harder the more variables you have.
      I could not figure this out and wrote some garbage code to make it happen. There's gotta be a better way.
 
 ##

(!!!!!)
5: This way of declaring hyperparameters is apparently not illegal (closure), but will not allow for hyperparameter tuning. I view hyperparameter tuning
   as an essential tool in any deep learning framework, so this is an unnacceptable way of doing things.
   
   Potential solutions:
   
   1. The thing I will likely have to do is develop some kind of interface whereby the user determines hyperparameters, but in a way that allows for tuning
      from various outside libraries that we don't want to try to anticipate. I've been trying to think of what this might look like but I don't have any
      concrete ideas yet. 
