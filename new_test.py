
import BayesianOptim as bo
import numpy as np
import pandas as pa

input_info = []
input_info.append({"name":"var1", "type":"real", "scale":"lin", "bounds":[0,1]})
input_info.append({"name":"var2", "type":"categorical", "scale":"none", "bounds":["blue","green","red"]})

input_info = pa.DataFrame(input_info)
print(input_info)
bo.check_input_info(input_info)
dummy = bo.dummify_input_info(input_info)
print("\n")
print(dummy)

output_info = []
output_info.append({"name":"var2", "constraints":None, "scale":"lin"})
input_info = pa.DataFrame(output_info)