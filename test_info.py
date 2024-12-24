
import BayesianOptim as bo
import numpy as np
import pandas as pa


input_info = []
input_info.append({"name":"var1", "type":"real", "scale":"lin", "bounds":[0, 1]})
input_info.append({"name":"var2", "type":"categorical", "scale":"", "bounds":["red", "blue", "green"]})
input_info.append({"name":"var3", "type":"categorical", "scale":"", "bounds":["sphere", "cube"]})
input_info.append({"name":"var4", "type":"discrete", "scale":"lin", "bounds":[0, 1]})
input_info.append({"name":"var5", "type":"real", "scale":"lin", "bounds":[0, 1]})

input_info = pa.DataFrame(input_info)
dummy = bo.dummify_input_info(input_info)

data = pa.DataFrame(columns=dummy["name"])
data.loc[len(data)] = [0.41, 0.501, 1, 0.712, 0.214, 0.11]
print(data)
print(bo.categorify_and_discretize_data(data, input_info))