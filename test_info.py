
import BayesianOptim as bo
import numpy as np
import pandas as pa


input_info = []
input_info.append({"name":"var1", "type":"real", "scale":"lin", "bounds":[0, 100]})
input_info.append({"name":"var2", "type":"categorical", "scale":"none", "bounds":["red", "blue", "green"]})
input_info.append({"name":"var3", "type":"categorical", "scale":"none", "bounds":["sphere", "cube"]})
input_info.append({"name":"var4", "type":"discrete", "scale":"lin", "bounds":[0, 100]})
input_info.append({"name":"var5", "type":"real", "scale":"log", "bounds":[0.1, 100]})

input_info = pa.DataFrame(input_info)

print("\nInput info:\n")
print(input_info)
bo.check_input_info(input_info)
dummy_input_info = bo.dummify_input_info(input_info)
print("\nDummy input info:\n")
print(dummy_input_info)

dummy_inputs = pa.DataFrame(columns=dummy_input_info["name"])
dummy_inputs.loc[len(dummy_inputs)] = [0., 0, 0., 0., 0., 0.]
dummy_inputs.loc[len(dummy_inputs)] = [1., 1., 1., 1., 1., 1.]
dummy_inputs.loc[len(dummy_inputs)] = [0.41, 0.3, 0.501, 0.712, 0.214, 0.11]
print("\nDummy inputs:\n")
print(dummy_inputs)
print("\nDeprocessed inputs:\n")
deprocessed_inputs = bo.postprocess_inputs(dummy_inputs, input_info)
print(deprocessed_inputs)

print("\nOutput info:\n")
output_info = []
output_info.append({"name":"var1", "constraints":"objective", "scale":"lin"})
output_info.append({"name":"var2", "constraints":[["<", 4.]], "scale":"log"})
output_info = pa.DataFrame(output_info)
print(output_info)
bo.check_output_info(output_info)

print("\nOutputs:\n")
outputs = pa.DataFrame(columns=output_info["name"])
outputs.loc[len(outputs)] = [0.5, 5]
outputs.loc[len(outputs)] = [0.74, 4]
print(outputs)

print("\nProcessed outputs:\n")
normalized_outputs, normalized_out_info = bo.preprocess_outputs_and_info(outputs, output_info)
print(normalized_outputs)
print("\nProcessed output_info:\n")
print(normalized_out_info)

print("\nDeprocessed outputs:\n")
deprocessed_outputs = bo.postprocess_output(normalized_outputs, outputs, output_info)
print(deprocessed_outputs)

bo.check_data(dummy_inputs, normalized_outputs)