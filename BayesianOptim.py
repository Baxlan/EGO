
import math
import numpy as np
import scipy as sp
import multiprocessing
import warnings
import copy
import plotly.graph_objects as go
import seaborn
import pandas as pa
import matplotlib.pyplot as plt
import sys
from scipy.stats import beta



def error(text : str) -> None:
    print(text)
    sys.exit(-1)



# =================================================================
# =================================================================
# ===============  DATA HANDLING TOOLS  ===========================
# =================================================================
# =================================================================



def check_input_info(input_info : pa.DataFrame) -> None:
    if "name" not in input_info.columns:
        error("\"name\" column is missing in input_info")
    if "type" not in input_info.columns:
        error("\"type\" column is missing in input_info")
    if "scale" not in input_info.columns:
        error("\"scale\" column is missing in input_info")
    if "bounds" not in input_info.columns:
        error("\"bounds\" column is missing in input_info")

    for i in range(len(input_info)):
        name = input_info.loc[i, "name"]
        if type(name) is not str:
            error(str(i+1) + "th \"name\" in input_info must be a string")
        if "~" in name:
            error(str(i+1) + "th \"name\" in input_info contains a \"~\", which is forbidden")
        if input_info.loc[i, "type"] != "real" and input_info.loc[i, "type"] != "discrete" and input_info.loc[i, "type"] != "categorical":
            error(name + "'s \"type\" in input_info must be \"real\", \"discrete\" or \"categorical\"")
        if type(input_info.loc[i, "bounds"]) != list:
            error(name + "'s \"bounds\" in input_info must be a list")

        if input_info.loc[i, "type"] == "categorical":
            if input_info.loc[i, "scale"] != "none":
                error(name + "'s \"scale\" in input_info must be \"none\", because its type is \"categorical\"")
            if len(input_info.loc[i, "bounds"]) <= 1:
                error(name + "'s \"bounds\" length in input_info must be at least 2")
            for j in range(len(input_info.loc[i, "bounds"])):
                if type(input_info.loc[i, "bounds"][j]) != str:
                    error(str(j+1) +"th item in " + name + "'s \"bounds\" in input_info must be a string because its type is \"categorical\"")

        else: # if input_info.loc[i, "type"] != "categorical":
            if input_info.loc[i, "scale"] == "none":
                error(name + "'s \"scale\" in input_info cannot be \"none\", because its type is \"real\" or \"discrete\"")
            if input_info.loc[i, "scale"] != "lin" and input_info.loc[i, "scale"] != "log":
                error(name + "'s \"scale\" in input_info must either be \"real\" or \"discrete\"")
            if len(input_info.loc[i, "bounds"]) != 2:
                error(name + "'s \"bounds\" length in input_info must be exaclty 2")
            for j in range(2):
                if type(input_info.loc[i, "bounds"][j]) != float and type(input_info.loc[i, "bounds"][j]) != int:
                    error(str(j+1) +"th item in " + name + "'s \"bounds\" in input_info must be a float or an integer")
            if input_info.loc[i, "bounds"][0] >= input_info.loc[i, "bounds"][1]:
                error("Lower boundary of the " + name + " variable in input_info must be strictly inferior to its upper boundary")
            if input_info.loc[i, "scale"] == "log" and (input_info.loc[i, "bounds"][0] == 0 or input_info.loc[i, "bounds"][1] == 0):
                error("Boundaries of the " + name + " variable in input_info cannot be 0 because its scale is \"log\"")
            if input_info.loc[i, "scale"] == "log" and input_info.loc[i, "bounds"][0] < 0 and input_info.loc[i, "bounds"][1] > 0:
                error("Boundaries of the " + name + " variable in input_info must be of the same sign because its scale is \"log\"")



def check_constraints(constr1, constr2):
    # check if constr2 is always true when constr is true
    return False



def check_output_info(output_info : pa.DataFrame) -> None:
    if "name" not in output_info.columns:
        error("\"name\" column is missing in output_info")
    if "constraints" not in output_info.columns:
        error("\"constraints\" column is missing in output_info")
    if "scale" not in output_info.columns:
        error("\"scale\" column is missing in output_info")

    objective = False

    for i in range(len(output_info)):
        name = output_info.loc[i, "name"]
        if type(name) is not str:
            error(name + "'s \"name\" in output_info must be a string")
        if output_info.loc[i, "scale"] != "lin" and output_info.loc[i, "scale"] != "log":
            error(name + "'s \"scale\" in output_info must either be \"lin\" or \"log\"")

        if output_info.loc[i, "constraints"] == "objective":
            if objective:
                error("Only one objective can be defined in output_info")
            else:
                objective = True
        elif type(output_info.loc[i, "constraints"]) != list:
            error(name + "'s \"constraint\" in output_info must be a list (or \"objective\")")
        elif len(output_info.loc[i, "constraints"]) == 0:
            error(name + "'s \"constraint\" list in output_info cannot be empty")
        else:
            for j in range(len(output_info.loc[i, "constraints"])):
                if type(output_info.loc[i, "constraints"][j]) != list:
                    error(str(j+1) + "th " + name + "'s \"constraint\" in output_info must be a list")
                elif len(output_info.loc[i, "constraints"][j]) not in  [2, 4]:
                    error(str(j+1) + "th " + name + "'s \"constraint\" length in output_info must be 2 or 4")
                else:
                    for k in range(len(output_info.loc[i, "constraints"][j])):
                        if k % 2 == 1 and type(output_info.loc[i, "constraints"][j][k]) != float:
                            error(str(k+1) + "th item in the " + str(j+1) + "th " + name + "'s \"constraint\" in output_info must be a float")
                        elif k % 2 == 0 and output_info.loc[i, "constraints"][j][k] not in ["<", ">"]:
                            error(str(k+1) + "th item in the " + str(j+1) + "th " + name + "'s \"constraint\" in output_info must be either \"<\" or \">\"")
                    if len(output_info.loc[i, "constraints"][j]) == 4:
                        a = ["a", "a"]
                        if output_info.loc[i, "constraints"][j][0] == "<":
                            a[0] = output_info.loc[i, "constraints"][j][1]
                        else:
                            a[1] = output_info.loc[i, "constraints"][j][1]
                        a[a.index("a")] = output_info.loc[i, "constraints"][j][3]
                        if a[0] < a[1]:
                            error(str(j+1) + "th " + name + "'s \"constraint\" in output_info is never satisfied")
                        elif a[0] == a[1]:
                            error(str(j+1) + "th " + name + "'s \"constraint\" in output_info is always satisfied")
                for k in range(len(output_info.loc[i, "constraints"])):
                    if k == j:
                        continue
                    if check_constraints(output_info.loc[i, "constraints"][j], output_info.loc[i, "constraints"][k]):
                        error("When the " + str(j) + "th " + name + "'s \"constraint\" is satisfied, its " + str(k) + "th one is also ALWAYS satisfied. They are redundant")

    if not objective:
        error("No objective defined in output_info")



def dummify_input_info(input_info : pa.DataFrame) -> pa.DataFrame:
    new_input_info = []

    for i in range(len(input_info)):
        if input_info.loc[i, "type"] == "real" or input_info.loc[i, "type"] == "discrete":
            new_input_info.append(input_info.loc[i].to_dict())

        else: #if "type" == "categorical"
            for j in range(len(input_info.loc[i, "bounds"])-1):
                new_input_info.append({"name":input_info.loc[i, "name"] + "~" + str(j+1), "type":"real", "scale":"lin", "bounds":[0, 1]})

    return pa.DataFrame(new_input_info)



def preprocess_inputs(inputs : pa.DataFrame, input_info : pa.DataFrame, force : bool = False) -> pa.DataFrame:
    if not force and "categorical" in input_info.loc[:, "type"]:
        error("Inputs cannot be processed with categorical variables")

    # TEMPORARY : CATEGORICAL PREPROCESSING MUST BE IMPLEMENTED
    if "categorical" in input_info.loc[:, "type"]:
        error("Categorical preprocessing is not implemented yet")

    inp = copy.deepcopy(inputs)
    inp_info = copy.deepcopy(input_info)
    inp_info.index = inp_info["name"]

    for name in inp.columns:
        inf = inp_info.loc[name, "bounds"][0]
        sup = inp_info.loc[name, "bounds"][1]
        # linearize log scaled variables
        if inp_info.loc[name, "scale"] == "log":
            sign = 1
            if inf < 0:
                sign = -1
                inf, sup = -sup, -inf

            inp.loc[:, name] = np.log(sign*inp.loc[:, name])

            # normalize log variables
            inp.loc[:, name] -= np.log(inf)
            inp.loc[:, name] /= (np.log(sup) - np.log(inf))

        else:
            # normalize non log variables
            inp.loc[:, name] -= inf
            inp.loc[:, name] /= (sup-inf)

    return inp



def preprocess_outputs_and_info(outputs : pa.DataFrame, output_info : pa.DataFrame) -> (pa.DataFrame, pa.DataFrame):
    out = copy.deepcopy(outputs)
    out_info = copy.deepcopy(output_info)
    out_info.index = out_info["name"]

    # normalize outputs
    for name in out.columns:
        inf = np.min(out[name])
        sup = np.max(out[name])
        #linearize log scaled variables
        if out_info.loc[name, "scale"] == "log":
            sign = 1
            if inf < 0:
                sign = -1
                inf, sup = -sup, -inf

            out.loc[:, name] = np.log(sign*out.loc[:, name])

            # normalize log variables
            out.loc[:, name] -= np.log(inf)
            out.loc[:, name] /= (np.log(sup) - np.log(inf))

            # adapting output_info for log scaled constraints
            if out_info.loc[name, "constraints"] != "objective":
                for i in range(len(out_info.loc[name, "constraints"])):
                    for j in range(len(out_info.loc[name, "constraints"][i])):
                        if j % 2 == 1:
                            out_info.loc[name, "constraints"][i][j] = np.log(sign*out_info.loc[name, "constraints"][i][j])
                            out_info.loc[name, "constraints"][i][j] -= np.log(inf)
                            out_info.loc[name, "constraints"][i][j] /= (np.log(sup) - np.log(inf))
                        else:
                            if sign * (np.log(sup) - np.log(inf)) < 0:
                                if out_info.loc[name, "constraints"][i][j] == "<":
                                    out_info.loc[name, "constraints"][i][j] = ">"
                                else:
                                    out_info.loc[name, "constraints"][i][j] = "<"

        else:
            #normalize linear scaled variables
            out.loc[:, name] -= inf
            out.loc[:, name] /= (sup - inf)

            # adapting output_info for log scaled variables
            if out_info.loc[name, "constraints"] != "objective":
                for i in range(len(out_info.loc[name, "constraints"])):
                    for j in range(len(out_info.loc[name, "constraints"][i])):
                        if j % 2 == 1:
                            out_info.loc[name, "constraints"][i][j] -= inf
                            out_info.loc[name, "constraints"][i][j] /= (sup-inf)
                        else:
                            if sup-inf < 0:
                                if out_info.loc[name, "constraints"][i][j] == "<":
                                    out_info.loc[name, "constraints"][i][j] = ">"
                                else:
                                    out_info.loc[name, "constraints"][i][j] = "<"

    out_info.index = range(len(out_info.index))
    return out, out_info



def spherical_to_cartesian(spherical_coords : np.array) -> np.array:
    # spherical_coords are not really spherical coordinates, they are uniformly
    # distributed variables that must be converted to beta distributed ones

    N = len(spherical_coords)
    angles = [math.acos(abs((2*beta.ppf(spherical_coords[i], (N-i)/2, (N-i)/2)-1))) for i in range(len(spherical_coords))]

    N += 1 # cartesian coords
    cartesian_coords = np.zeros(N)
    sin_cumul = 1

    for i in range(0, N-1):
        cartesian_coords[i] = np.cos(angles[i]) * sin_cumul
        sin_cumul *= np.sin(angles[i])

    cartesian_coords[-1] = sin_cumul
    return cartesian_coords



def postprocess_inputs(dummy_inputs : pa.DataFrame, input_info : pa.DataFrame) -> pa.DataFrame:
    inp = copy.deepcopy(dummy_inputs)
    in_info = copy.deepcopy(input_info)
    in_info.index = in_info["name"]

    # de-normalize normalized data
    for name in inp.columns:
        if "~" in name:
            continue

        inf = in_info.loc[name, "bounds"][0]
        sup = in_info.loc[name, "bounds"][1]

        if in_info.loc[name, "scale"] == "log":
            sign = 1
            inf = in_info.loc[name, "bounds"][0]
            sup = in_info.loc[name, "bounds"][1]
            if inf < 0:
                sign = -1
                inf, sup = -sup, -inf

            # denormalize log scaled variables
            inp.loc[:, name] *= (np.log(sup) - np.log(inf))
            inp.loc[:, name] += np.log(inf)
            # exponentiate log scaled variables
            inp.loc[:, name] = sign*np.exp(inp.loc[:, name])

        # denormalize linear scaled variables
        else:
            inp.loc[:, name] *= (sup-inf)
            inp.loc[:, name] += inf


    postprocessed_inputs = pa.DataFrame(columns=in_info["name"])

    # categorify dummy data
    for i in range(inp.shape[0]):
        new_row = {}
        for name in in_info["name"]:

            if in_info.loc[name, "type"] == "categorical":
                categorical_parameters = []

                for k in range(1, len(in_info.loc[name, "bounds"])):
                    categorical_parameters.append(inp.loc[i, name+"~"+str(k)])
                coords = spherical_to_cartesian(categorical_parameters)

                dist = []
                for a in range(len(coords)):
                    category = np.zeros(len(coords))
                    category[a] = 1
                    dist.append(math.dist(coords, category))
                new_row[name] = in_info.loc[name, "bounds"][dist.index(min(dist))]

            elif in_info.loc[name, "type"] == "discrete":
                new_row[name] = round(inp.loc[i, name])
            else:
                new_row[name] = inp.loc[i, name]

        postprocessed_inputs.loc[postprocessed_inputs.shape[0]] = new_row
    return postprocessed_inputs



def postprocess_output(normalized_outputs : pa.DataFrame, outputs : pa.DataFrame, output_info : pa.DataFrame, normalized_sigma : pa.DataFrame = []) -> (pa.DataFrame, pa.DataFrame):
    out = copy.deepcopy(normalized_outputs)
    sig = copy.deepcopy(normalized_sigma)
    output_info.index = output_info["name"]

    for name in outputs.columns:
        inf = np.min(outputs[name])
        sup = np.max(outputs[name])

        if output_info.loc[name, "scale"] == "log":
            sign = 1
            if inf < 0:
                sign = -1
                inf, sup = -sup, -inf

            # denormalize log scaled variables
            out.loc[:, name] *= (np.log(sup)-np.log(inf))
            out.loc[:, name] += np.log(inf)
            # exponentiate log scaled variables
            out.loc[:, name] = sign*np.exp(out.loc[:, name])

            if len(sig) != 0:
                sig.loc[:, name] *= (np.log(sup)-np.log(inf))
                sig.loc[:, name] *= sig.loc[:, name]/np.log(sign*sig.loc[:, name]) # NOT SURE ABOUT THIS LINE

        # denormalize linear scaled variables
        else:
            out.loc[:, name] *= (sup-inf)
            out.loc[:, name] += inf

            if len(sig) != 0:
                sig.loc[:, name] *= (max-min)

    if len(sig) == 0:
        return out
    else:
        return out, sig



def check_data(inputs : pa.DataFrame, outputs : pa.DataFrame) -> None:
    for name in inputs.columns:
        zero = False
        one = False
        if not 1. in inputs[name] or not 0. in inputs[name] or not inputs[name].between(0., 1.).all():
            error("Inputs have not been preprocessed")

    for name in outputs.columns:
        zero = False
        one = False
        if not 1. in outputs[name] or not 0. in outputs[name] or not outputs[name].between(0., 1.).all():
            error("Outputs have not been preprocessed")



# =================================================================
# =================================================================
# ================== GAUSSIAN KERNEL ==============================
# =================================================================
# =================================================================



def check_metric(x : pa.DataFrame, metric : np.ndarray) -> np.ndarray:
    dim = x.shape[1]
    if isinstance(metric, (int, float, np.floating)):
        M = np.zeros((dim, dim))
        np.fill_diagonal(M, metric)
        return M
    elif (type(metric) == list or (type(metric) == np.ndarray and metric.ndim == 1)) and len(metric) == 1:
        M = np.zeros((dim, dim))
        np.fill_diagonal(M, metric[0])
        return M
    else:
        metric = np.array(metric)
        if metric.ndim == 1 and len(metric) == dim:
            return np.diag(metric)
        elif metric.ndim == 1 and len(metric) == dim*(dim+1)/2:
            return make_symmetric_matrix_from_list(metric)
        elif metric.ndim == 2 and len(metric) == dim and len(metric[0]) == dim and np.allclose(metric, metric.T, rtol=1e-9, atol=1e-12):
            return metric
        else:
            error("The \"metric\" parameter must either be a scalar, a 1D array of length N (problem dimensionality), or a 2D SYMMETRIC N*N array")



def get_triangular_matrix_rank_from_list(vals : list) -> int:
    # vals must be a list containing elements of the upper triangular matrix
    N = (-1+math.sqrt(1+8*len(vals)))/2
    n = int(N)
    if (n-N)%1 != 0:
        error("Number of elements doesn't match with a squarre triangular matrix")
    return n



def make_symmetric_matrix_from_list(vals : list) -> np.ndarray:
    n = get_triangular_matrix_rank_from_list(vals)
    m = np.zeros([n,n], dtype=np.double)
    xs,ys = np.triu_indices(n)
    m[xs,ys] = vals
    m[ys,xs] = vals
    return m



def make_diff_list(inputs : pa.DataFrame) -> np.array:
    x = inputs.to_numpy()
    diffs = []
    for i in range(len(x)):
        for j in range(len(x)-i):
            diffs.append(x[i] - x[j+i])

    return np.array(diffs)



def make_kernel(diffs, metric):
    K = [np.exp(-np.sum(diff.transpose() * metric * diff)) for diff in diffs]
    return make_symmetric_matrix_from_list(K)



def add_to_kernel(K, x, x_new, metric):
    diffs = []
    for point in x:
        diffs.append(x_new - point)

    diffs = [np.exp(-np.sum(diff.transpose() * metric * diff)) for diff in diffs]

    K = np.vstack([K, diffs])
    K = np.hstack([K, np.transpose([diffs+[1.0]])])
    return K



# =================================================================
# =================================================================
# =================== METRIC OPTIMIZER ============================
# =================================================================
# =================================================================



def log_marginal_likelihood(K, y):
    try:
        L = sp.linalg.cholesky(K, lower = True)
    except:
        return -math.inf
    S1 = sp.linalg.solve_triangular(L, y, lower = True)
    S2 = sp.linalg.solve_triangular(L.T, S1, lower=False)
    return -np.sum(np.log(np.diagonal(L))) - 0.5*np.array(y).dot(S2) - 0.5*len(y)*np.log(2*np.pi)



def delinearize_metric(x, metric, bounds):
    metric = check_metric(x, metric)
    for i in range(len(metric)):
        for j in range(len(metric[0])):
            # normalize to [0, 1]
            metric[i, j] = (metric[i, j] - bounds[0])/(bounds[1] - bounds[0])
            # exponentiate
            metric[i, j] = 10**bounds[0] * math.exp(metric[i, j] * math.log(10**bounds[1]/10**bounds[0]))
    return metric



def param_optimizer(M, *args):
    metric = delinearize_metric(args[3], M, args[2])
    K = make_kernel(args[0], metric=metric)
    return -log_marginal_likelihood(K, args[1])



def optimized_metric(diffs, x, y, isotropy, seed, initial, bounds, method):
    if type(initial) != list and type(initial) != np.ndarray:
        error("\"initial\" parameter must be a list or an array")

    n = len(x[0])

    if isotropy == "iso":
        b = [bounds]
        if len(initial) != len(b):
            error("Initial point of metric optimization must be of length 1. It is " + str(len(initial)))

    elif isotropy == "diag":
        b = [bounds for i in range(n)]
        if len(initial) != len(b):
            error("Initial point of metric optimization must be of length " + str(len(b)) + ". It is " + str(len(initial)))

    elif isotropy == "aniso":
        b = [bounds for i in range(int(n*(n+1)/2))]
        if len(initial) != len(b):
            error("Initial point of metric optimization must be of length " + str(len(b)) + ". It is " + str(len(initial)))

    else:
        error("\"isotropy\" parameter must be \"iso\", \"diag\" or \"aniso\"")



    if method=="stochastic":
        warnings.filterwarnings("ignore")

        response = sp.optimize.differential_evolution( \
            func=param_optimizer, bounds=b, x0=initial, \
            args=(diffs, y, bounds, x), seed=seed)

        warnings.filterwarnings("default")

    elif method=="gradient":
        warnings.filterwarnings("ignore")

        response = sp.optimize.minimize( \
            fun=param_optimizer, bounds=b, x0=initial, \
            args=(diffs, y, bounds, x), method="L-BFGS-B")

        warnings.filterwarnings("default")

    else:
        error("\"method\" parameter must be \"stochastic\" or \"gradient\"")

    M = response.x
    if isotropy == "aniso":
        M = make_symmetric_matrix_from_list(M)

    M = delinearize_metric(x, M, bounds)

    if isotropy != "aniso":
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if i != j:
                    M[i,j] = 0

    return M, -response.fun



def optimized_metric_tuple(args):
    return optimized_metric(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7])



def optimal_metric(diffs, x, y, bounds, iso, seed, threads):
    if bounds[0] >= bounds[1]:
        error("Lower bound must be strictly inferior to upper bound")

    isotropies = {"iso" : 1, "diag" : len(x[0]), "aniso" : int(len(x[0])*(len(x[0])+1)/2)}
    methods = ["gradient", "stochastic"]
    args = []

    n = 10 * isotropies[iso] * math.ceil(math.sqrt(isotropies[iso]))
    m = math.ceil(math.log(n)/math.log(2))
    pool = multiprocessing.Pool(threads)
    generator = sp.stats.qmc.Sobol(d=isotropies[iso], seed=seed)
    for method in methods:
        initial = generator.random_base2(m=m)[:n]
        initial = [init*(bounds[1]-bounds[0])+bounds[0] for init in initial]
        for init in initial:
            args.append((diffs, x, y, iso, seed, init, bounds, method))

    metrics_lmls = pool.map(optimized_metric_tuple, args)

    # keys are lml, values are metrics
    metrics = {}

    for i in range(len(metrics_lmls)-1):
        metrics[metrics_lmls[i][1]] = metrics_lmls[i][0]

    if -math.inf in metrics.keys():
        del metrics[-math.inf]

    metrics = dict(sorted(metrics.items(), reverse=True))

    for lml, metric in metrics.items():
        K = make_kernel(diffs, metric)
        if (K > 1e-4).all():
            continue
        else:
            return metric, lml

    error("No optimal metric found. Try to change bounds, seed or number of points")



# =================================================================
# =================================================================
# ================== PROBLEM MODELISATION =========================
# =================================================================
# =================================================================



def bound_combinations(bounds):
    if not bounds:
        return [[]]

    first_bound = bounds[0]
    rest_bounds = bounds[1:]

    sub_combinations = bound_combinations(rest_bounds)
    result = []

    for bound_value in range(first_bound[0], first_bound[1] + 1):
        for sub_combination in sub_combinations:
            result.append([bound_value] + sub_combination)

    return result



def first_points(input_info, n, seed):
    points = random_points(input_info, n, seed)
    bounds = bound_combinations([[0,1] for i in range(input_info.shape[0])])
    bounds = pa.DataFrame(bounds, columns=input_info["name"])
    return pa.concat([bounds, points], ignore_index=True)



def random_points(input_info, n, seed):
    m = math.ceil(math.log(n)/math.log(2))
    points_generator = sp.stats.qmc.Sobol(d=input_info.shape[0], seed=seed)
    points = points_generator.random_base2(m=m)[:n]
    return pa.DataFrame(points, columns=input_info["name"])



def predict(model, x, x_new):
        metric = check_metric(x, model[2])

        y_mean = []
        y_sigma = []
        for new in x_new:
            local_kernel = add_to_kernel(model[0], x, new, metric)
            K = local_kernel[:len(x), :len(x)]
            k = local_kernel[:len(x), -1]
            inv_K = np.linalg.inv(K)
            y_mean.append(np.dot(np.dot(k, inv_K), model[1]))
            y_sigma.append(local_kernel[-1,-1] - np.dot(np.dot(k, inv_K),k))

        return np.array(y_mean), np.array(y_sigma)



def next_points(models, x, input_info, constraints, n, seed, a, epsilon=1e-13, threads=1):
    pool = multiprocessing.Pool(threads)
    points = random_points(input_info, math.ceil(n/2), seed)
    args = [(models, x, point, a, epsilon, constraints) for point in points.to_numpy()]
    res  = pool.map(find_max_ei_gradient, args)

    points = random_points(input_info, math.floor(n/2), seed+1)
    args = [(models, x, point, a, epsilon, constraints, seed) for point in points.to_numpy()]
    res2 = pool.map(find_max_ei_stochastic, args)
    pool.close()

    res = res + res2
    results_dict = {}
    for i in range(len(res)):
        # keys are ei, values are points
        results_dict[res[i][0]] = res[i][1]

    # sorting in descending ei
    results_dict = dict(sorted(results_dict.items(), reverse=True))

    # convertring to dataframe then keeping only the n first values
    results = pa.DataFrame(results_dict.values(), columns=points.columns)
    ei = list(results_dict.keys())

    if n < results.shape[0]:
            results = results[:n]
            ei = ei[:n]
    elif n > results.shape[0]:
        results = results + random_points(input_info=input_info, n=n-results.shape[0], seed=seed)

    return ei, results



# =================================================================
# =================================================================
# ================== ACQUISITION FUNCTIONS ========================
# =================================================================
# =================================================================



def log1mexp(x):
    if x > -math.log(2):
        return math.log(-np.expm1(x))
    else:
        return np.log1p(-math.exp(x))



def h(z):
    return sp.stats.norm.pdf(z) + z*sp.stats.norm.cdf(z)



def log_h(z, epsilon):
    c1 = math.log(2*math.pi)/2

    if z > -1:
        return math.log(h(z))

    elif z > -1/math.sqrt(epsilon):
        c2 = math.log(math.pi/2)/2
        tmp = math.log(sp.special.erfcx(-z/math.sqrt(2)) * abs(z))

        return -(z**2)/2 - c1 + log1mexp(tmp+c2)

    else:
        return -(z**2)/2 - c1 - 2*math.log(abs(z))



def expected_improvement(y_mean, y_sigma, y_best, a, epsilon):
        ei = []
        for i in range(len(y_mean)):
            if y_sigma[i] <= 0:
                ei.append(0)

            else:
                sigma = a*y_sigma[i]
                z = (y_mean[i] - y_best) / sigma
                ei.append(math.exp(log_h(z, epsilon)+math.log(sigma)))

                if ei[i] < 0:
                    ei[i] = 0

        return ei



def find_max_ei_gradient(args):
    models = args[0]
    x = args[1]
    point = args[2]
    a = args[3]
    epsilon = args[4]
    constraints = args[5]

    response = sp.optimize.minimize( \
        fun=acquisition_function, x0=point, \
        args=(models, x, a, epsilon, constraints), method="L-BFGS-B", \
        bounds=[[0, 1] for i in range(len(x[0]))])

    return (-response.fun, response.x)



def find_max_ei_stochastic(args):
    models = args[0]
    x = args[1]
    point = args[2]
    a = args[3]
    epsilon = args[4]
    constraints = args[5]
    seed = args[6]

    response = sp.optimize.differential_evolution( \
        func=acquisition_function, x0=point, seed=seed, \
        args=(models, x, a, epsilon, constraints), \
        bounds=[[0, 1] for i in range(len(x[0]))])

    return (-response.fun, response.x)



def acquisition_function(X_new, *args):
    models = args[0]
    x = args[1]
    a = args[2]
    epsilon = args[3]
    constraints = args[4]

    if len(models) > 1:
        for i in range(len(models)-1):
            pred, sigma = predict(models[i], x, [X_new])
            if not are_contraint_satifcation_probable(pred, sigma, a, constraints[i]):
                return 0
                # RETOURNER QUELQUE CHOSE DE PLUS SMOOTH QUE 0

    pred, sigma = predict(models[0], x, [X_new])
    return -expected_improvement(pred, sigma, max(models[0][1]), a, epsilon)[0]



def are_contraint_satifcation_probable(value, sigma, a, constraint):
    for nested_constraints in constraint[1]:
        if are_nested_contraint_satifcation_probable(value, sigma, a, nested_constraints):
            return True

    return False



def are_nested_contraint_satifcation_probable(value, sigma, a, constraints):
    for i in range(len(constraints)):
        if i % 2 != 0:
            continue

        if constraints[i] == "<":
            if value - a*sigma >= constraints[i+1]:
                return False
        elif constraints[i] == ">":
            if value + a*sigma <= constraints[i+1]:
                return False
        else:
            error("Constraint condition must be either > or <")

    return True



# =================================================================
# =================================================================
# ============================ PLOT ===============================
# =================================================================
# =================================================================



def parallelPlot(inputs, outputs, input_info, output_info):
    all_labels = list(input_info["name"]) + list(output_info["name"])
    all_scales = list(input_info["scale"]) + list(output_info["scale"])
    # put data in the same order as the ones in the infos
    all_data = pa.concat([inputs[input_info["name"]], outputs[output_info["name"]]], axis=1).to_numpy()

    dimensions = []
    print(all_scales)

    # next block is for log-scaling log scaled variables because plotly doesn't support it trivialy
    for i in range(len(all_labels)):
        data = all_data[:,i]
        if all_scales[i] == "log":
            for j in range(len(data)):
                data[j] = math.log(data[j])
            n = 15
            ticks = [max(data)*i/n for i in range(n+1)]
            text = [format(math.exp(ticks[i]), "1.2E") for i in range(len(ticks))]
            d = dict(label=all_labels[i], values = data, tickvals=ticks, ticktext=text)
        else:
            d = dict(label=all_labels[i], values = data)

        dimensions.append(d)

    # EST CE QUE CA MARCHE AVEC DES CATEGORIES ?
    fig = go.Figure(data=go.Parcoords(dimensions=dimensions))
    fig.show()



def pairPlot(inputs, outputs, input_info, output_info):
    all_labels = list(input_info["name"]) + list(output_info["name"])
    all_scales = list(input_info["scale"]) + list(output_info["scale"])
    # put data in the same order as the ones in the infos
    all_data = pa.concat([inputs[input_info["name"]], outputs[output_info["name"]]], axis=1)

    log_labels = []
    for i in range(len(all_labels)):
        if all_scales[i] == "log":
            log_labels.append(all_labels[i])

    # COMMENT NE TRACER QUE LES LIGNES AYANT Y EN ORDONEE ?
    # FAUT-IL NE PAS TRACER CE QUI EST APRES LA DIAGONALE ?
    fig = seaborn.pairplot(all_data, x_vars=all_labels[:inputs.shape[1]], y_vars=all_labels[inputs.shape[1]:])
    for ax in fig.axes.flat:
        if ax.get_xlabel() in log_labels:
            ax.set(xscale="log")
        if ax.get_ylabel() in log_labels:
            ax.set(yscale="log")

    plt.show()



# =================================================================
# =================================================================
# ============= BAYESIAN OPTIMIZATION CLASS =======================
# =================================================================
# =================================================================



class BayesianOptimizer:
    def __init__(self,
                title : str,
                input_info : pa.DataFrame,
                output_info : pa.DataFrame,
                seed : int,
                threads : int,
                iso : str = "diag",
                epsilon : float = 1e-13) -> None:

        check_input_info(input_info)
        check_output_info(output_info)

        self.title = title
        self.input_info = input_info
        self.dummy_input_info = dummify_input_info(input_info)
        self.output_info = output_info
        self.original_seed = seed
        self.seed = seed
        self.threads = threads
        self.dummy_inputs = None
        self.outputs = None
        self.iso = iso
        self.epsilon = epsilon
        self.kernel = None
        self.metric = None



    def add_data(self, dummy_inputs : pa.DataFrame, outputs : pa.DataFrame) -> None:
        #outputs are real ones, not processed ones

        for name in self.input_info["name"]:
            if name not in dummy_inputs.columns:
                error("Column \"" + name + "\" not found in the added inputs")
        for name in dummy_inputs.columns:
            if name not in self.input_info["name"].values:
                error("Column \"" + name + "\" have been found in the added inputs but is not specified in the input_info")
        for name in self.output_info["name"]:
            if name not in outputs.columns:
                error("Column \"" + name + "\" not found in the added outputs")
        for name in outputs.columns:
            if name not in self.output_info["name"].values:
                error("Column \"" + name + "\" have been found in the added outputs but is not specified in the input_info")

        if dummy_inputs.shape[0] != outputs.shape[0]:
            error("The added inputs and outputs must have the same number of rows")

        if self.dummy_inputs == None:
            self.dummy_inputs = dummy_inputs
            self.outputs = outputs
        else:
            self.dummy_inputs = pa.concat([self.dummy_inputs, dummy_inputs], ignore_index=True)
            self.outputs = pa.concat([self.outputs, outputs], ignore_index=True)

        self.kernel = None
        self.metric = None




    def first_points(self, n):
        self.seed += 1
        dummy_points = first_points(self.dummy_input_info, n, self.seed)
        real_points = postprocess_inputs(dummy_inputs=dummy_points, input_info=self.input_info)
        return dummy_points, real_points



    def next_points(self, n, a, metric_bounds=[-12, 12]):
        inputs = self.dummy_inputs
        outputs, out_info = preprocess_outputs_and_info(outputs=self.outputs, output_info=self.output_info)
        self.seed += 1

        print("Calculating optimal metrics", flush=True)
        diffs = make_diff_list(inputs=inputs)
        print(diffs)

        metrics = []
        kernels = []
        for i in range(out_info.shape[0]):
            metric, lml = optimal_metric(diffs=diffs,
                                        x=inputs.to_numpy(),
                                        y=outputs.to_numpy()[:,i],
                                        bounds=metric_bounds,
                                        iso=self.iso,
                                        seed=self.seed,
                                        threads=self.threads)
            metrics.append(metric)
            kernels.append(make_kernel(diffs=diffs, metric=metric))

        print(metrics[0], flush=True)
        print("lml = " + str(lml), flush=True)
        np.set_printoptions(formatter={'float':"{0:0.3f}".format})
        print(kernels[0], flush=True)

        self.seed += 1
        m = 100 * len(self.dummy_input_info) * math.ceil(math.sqrt(self.dummy_input_info.shape[0]))
        print("Calculating next points", flush=True)

        models = []
        for i in range(len(metrics)):
            models.append((kernels[i], outputs.to_numpy()[:,i], metrics[i]))

        # GERER LA POSITION DE L'OBJECTIF DANS LES OUTPUTS ET DANS L'OUTPUT_INFO
        # GERER L'ORDRE DES CONTRAINTES
        ei, raw_next_pts = list(next_points(models=models,
                                            x=inputs.to_numpy(),
                                            input_info=self.dummy_input_info,
                                            constraints=None,
                                            n=m, seed=self.seed, a=a, epsilon=self.epsilon,
                                            threads=self.threads))

        self.kernel = kernels
        self.metric = metrics
        return raw_next_pts, postprocess_inputs(raw_next_pts, self.input_info)


    def plot1D(self):
        # plot constraints too
        pass


    def plot2D(self):
        pass


    def parallelPlot(self):
        parallelPlot(postprocess_inputs(self.dummy_inputs, self.input_info), self.outputs, self.input_info, self.output_info)


    def pairPlot(self):
        pairPlot(postprocess_inputs(self.dummy_inputs, self.input_info), self.outputs, self.input_info, self.output_info)


    def save(self):
        pass



    def load(self):
        pass
