
import math
import BayesianOptim as bo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pa



input_info = []
input_info.append({"name":"x", "type":"real", "scale":"lin", "bounds":[0, 1]})
input_info = pa.DataFrame(input_info)

output_info = []
output_info.append({"name":"y", "constraints":"objective", "scale":"lin"})
output_info = pa.DataFrame(output_info)

def func(x):
    return 0.1*x**2 * math.sin(5 * math.pi * x)**6



if __name__ == '__main__':
    input_list = np.arange(0, 1.1, 0.1)
    output_list = np.array([func(x) for x in input_list])
    input_df = pa.DataFrame(input_list.reshape(-1, 1), columns=["x"])
    output_df = pa.DataFrame(output_list.reshape(-1, 1), columns=["y"])

    model = bo.BayesianOptimizer(title="model", input_info=input_info, output_info=output_info,
                                seed=100, threads=7, iso="iso", epsilon=1e-13)

    model.add_data(dummy_inputs=input_df, outputs=output_df)
    #raw_next_pts, next_pts = model.next_points(n=30, a=5)

    #model.pairPlot()
    #model.parallelPlot()