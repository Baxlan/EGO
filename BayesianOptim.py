
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



# =================================================================
# =================================================================
# ===============  DATA HANDLING TOOLS  ===========================
# =================================================================
# =================================================================



def check_input_info(input_info : pa.DataFrame) -> None:
    if "name" not in input_info.columns:
        raise Exception("\"name\" column is missing in input_info")
    if "type" not in input_info.columns:
        raise Exception("\"type\" column is missing in input_info")
    if "scale" not in input_info.columns:
        raise Exception("\"scale\" column is missing in input_info")
    if "bounds" not in input_info.columns:
        raise Exception("\"bounds\" column is missing in input_info")

    for i in range(len(input_info)):
        if type(input_info.loc[i, "name"]) is not str:
            raise Exception(str(i+1) + "th \"name\" in input_info must be a string")
        if "~" in input_info.loc[i, "name"]:
            raise Exception(str(i+1) + "th \"name\" in input_info contains a \"~\", which is forbidden")
        if input_info.loc[i, "type"] != "real" and input_info.loc[i, "type"] != "discrete" and input_info.loc[i, "type"] != "categorical":
            raise Exception(str(i+1) + "th \"type\" in input_info must be \"real\", \"discrete\" or \"categorical\"")
        if type(input_info.loc[i, "bounds"]) != list:
            raise Exception(str(i+1) + "th bounds in input_info must be a list")

        if input_info.loc[i, "type"] == "categorical":
            if input_info.loc[i, "scale"] != "none":
                raise Exception(str(i+1) + "th \"scale\" in input_info must be \"none\", because its type is \"categorical\"")
            if len(input_info.loc[i, "bounds"]) <= 1:
                raise Exception(str(i+1) + "th \"bounds\" length must be at least 2")
            for j in range(len(input_info.loc[i, "bounds"])):
                if type(input_info.loc[i, "bounds"][j]) != str:
                    raise Exception(str(i+1) + "th \"bounds\"/"+ str(j+1) +"th item must be a string because its type is \"categorical\"")
                if "~" in input_info.loc[i, "bounds"][j]:
                    raise Exception(str(i+1) + "th \"bounds\" in input_info contains a \"~\", which is forbidden")

        if input_info.loc[i, "type"] != "categorical":
            if input_info.loc[i, "scale"] == "none":
                raise Exception(str(i+1) + "th \"scale\" in input_info cannot be \"none\", because its type is \"real\" or \"discrete\"")
            if len(input_info.loc[i, "bounds"]) != 2:
                raise Exception(str(i+1) + "th \"bounds\" length must be exaclty 2")
            for j in range(2):
                if type (input_info.loc[i, "bounds"][j]) != float and type(input_info.loc[i, "bounds"][j]) != int:
                    raise Exception(str(i+1) + "th \"bounds\"/"+ str(j+1) +"th item must be a floa or an integer")
            if input_info.loc[i, "bounds"][0] >= input_info.loc[i, "bounds"][1]:
                raise Exception("Lower boundary of the " + str(i+1) + "th variable must be strictly inferior to its upper boundary")
            if input_info.loc[i, "scale"] == "log" and (input_info.loc[i, "bounds"][0] == 0 or input_info.loc[i, "bounds"][1] == 0):
                raise Exception("Boundaries of the " + str(i+1) + "th variable cannot be 0 because its scale is \"log\"")
            if input_info.loc[i, "scale"] == "log" and input_info.loc[i, "bounds"][0] < 0 and input_info.loc[i, "bounds"][1] > 0:
                raise Exception("Boundaries of the " + str(i+1) + "th variable must be of the same sign because its scale is \"log\"")



def check_output_info(output_info : pa.DataFrame) -> None:
    print("Nothing done")




def dummify_input_info(input_info : pa.DataFrame) -> pa.DataFrame:
    new_input_info = []

    for i in range(len(input_info)):
        if input_info.loc[i, "type"] == "real" or input_info.loc[i, "type"] == "discrete":
            new_input_info.append(input_info.loc[i].to_dict())

        else: #if "type" == "categorical"
            for j in range(len(input_info.loc[i, "bounds"])-1):
                new_input_info.append({"name":input_info.loc[i, "name"] + "~" + str(j+1), "type":"real", "scale":"lin", "bounds":[0, 1]})

    return pa.DataFrame(new_input_info)



def preprocess_outputs_and_info(outputs : pa.DataFrame, output_info : pa.DataFrame) -> pa.DataFrame:
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

            if out_info.loc[name, "constraints"] != "none":
                out_info.loc[name, "constraints"] = np.log(sign*out_info.loc[name, "constraints"])
                out_info.loc[name, "constraints"] -= np.log(inf)
                out_info.loc[name, "constraints"] /= (np.log(sup) - np.log(inf))
                out_info.loc[name, "constraints"] = list(out_info.loc[name, "constraints"])


        #normalize linear scaled variables
        else:
            out.loc[:, name] -= inf
            out.loc[:, name] /= (sup - inf)
            if out_info.loc[name, "constraints"] != "none":
                out_info.loc[name, "constraints"] -= inf
                out_info.loc[name, "constraints"] /= (sup-inf)
                out_info.loc[name, "constraints"] = list(out_info.loc[name, "constraints"])

    return out, out_info



def spherical_to_cartesian(spherical_coords : list) -> list:
    """
    It is assumed that all the arguments (except the last one) are cosines of the polar angles.
    while the last argument is the normalized angle [0, 1] of the azimuth.
    """
    angles = [math.acos(spherical_coords[i]) for i in range(len(spherical_coords)-1)] + [spherical_coords[-1]*math.pi/2]

    N = len(angles) + 1
    cartesian_coords = np.zeros(N)
    cartesian_coords[0] = np.cos(angles[0])

    for i in range(1, N-1):
        cartesian_coords[i] = np.cos(angles[i]) * np.prod(np.sin(angles[:i]))

    cartesian_coords[N-1] = np.prod(np.sin(angles))
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
    for i in range(len(inp)):
        new_row = {}
        for name in in_info["name"]:

            if in_info.loc[name, "type"] == "categorical":
                cosines = []

                for k in range(1, len(in_info.loc[name, "bounds"])):
                    cosines.append(inp.loc[i, name+"~"+str(k)])
                coords = spherical_to_cartesian(cosines)

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

        postprocessed_inputs.loc[len(postprocessed_inputs)] = new_row
    return postprocessed_inputs



def postprocess_output(normalized_outputs : pa.DataFrame, outputs : pa.DataFrame, output_info : pa.DataFrame, normalized_sigma : pa.DataFrame = []):
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



def check_data(inputs : pa.DataFrame, outputs : pa.DataFrame):
    for name in inputs.columns:
        zero = False
        one = False
        if 1. in inputs[name] and 0. in inputs[name] and inputs[name].between(0., 1.).all():
            pass
        else:
            raise Exception("Inputs have not been preprocessed")

    for name in outputs.columns:
        zero = False
        one = False
        if 1. in outputs[name] and 0. in outputs[name] and outputs[name].between(0., 1.).all():
            pass
        else:
            raise Exception("Outputs have not been preprocessed")



# =================================================================
# =================================================================
# ================== GAUSSIAN KERNEL ==============================
# =================================================================
# =================================================================



def check_metric(x, metric):
    if isinstance(metric, (int, float, np.floating)):
        M = np.zeros((len(x[0]), len(x[0])))
        np.fill_diagonal(M, metric)
        return M
    elif (type(metric) == list or (type(metric) == np.ndarray and metric.ndim == 1)) and len(metric) == 1:
        M = np.zeros((len(x[0]), len(x[0])))
        np.fill_diagonal(M, metric[0])
        return M
    else:
        metric = np.array(metric)
        if metric.ndim == 1 and len(metric) == len(x[0]):
            return np.diag(metric)
        elif metric.ndim == 1 and len(metric) == len(x[0])*(len(x[0])+1)/2:
            return make_symmetric_matrix_from_list(metric)
        elif metric.ndim == 2 and len(metric) == len(x[0]) and len(metric[0]) == len(x[0]) and np.allclose(metric, metric.T, rtol=1e-9, atol=1e-12):
            return metric
        else:
            raise Exception("The \"metric\" parameter must either be a scalar, a 1D array of length N (problem dimensionality), or a 2D SYMMETRIC N*N array")



# vals must be a list containing elements of the upper triangular matrix
def get_triangular_matrix_rank_from_list(vals):
    N = (-1+math.sqrt(1+8*len(vals)))/2
    n = int(N)
    if (n-N)%1 != 0:
        raise Exception("Number of elements doesn't match with a squarre triangular matrix")
    return n



def make_symmetric_matrix_from_list(vals):
    n = get_triangular_matrix_rank_from_list(vals)
    m = np.zeros([n,n], dtype=np.double)
    xs,ys = np.triu_indices(n)
    m[xs,ys] = vals
    m[ys,xs] = vals
    return m



def make_diff_list(x, input_info, uncertainties=False):
    diffs = []
    for i in range(len(x)):
        for j in range(len(x)-i):
            diffs.append(x[i] - x[j+i])

            if uncertainties:
                for k in len(x[i]):
                    diffs[len(diffs)-1][k] += input_info[k] * math.sqrt(x[i][k]**2 + x[j+i][k]**2)


    return np.array(diffs)



def make_kernel(diffs, noise, metric):
    K = [np.exp(-np.sum(diff.transpose() * metric * diff)) for diff in diffs]
    K = make_symmetric_matrix_from_list(K)
    return K + noise * np.eye(len(K))



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
    metric = delinearize_metric(args[4], M, args[3])
    K = make_kernel(args[0], args[2], metric=metric)
    return -log_marginal_likelihood(K, args[1])



def optimized_metric(diffs, x, y, noise, isotropy, seed, initial, bounds, method):
    if type(initial) != list and type(initial) != np.ndarray:
        raise Exception("\"initial\" parameter must be a list or an array")

    n = len(x[0])

    if isotropy == "iso":
        b = [bounds]
        if len(initial) != len(b):
            raise Exception("Initial point of metric optimization must be of length 1. It is " + str(len(initial)))

    elif isotropy == "diag":
        b = [bounds for i in range(n)]
        if len(initial) != len(b):
            raise Exception("Initial point of metric optimization must be of length " + str(len(b)) + ". It is " + str(len(initial)))

    elif isotropy == "aniso":
        b = [bounds for i in range(int(n*(n+1)/2))]
        if len(initial) != len(b):
            raise Exception("Initial point of metric optimization must be of length " + str(len(b)) + ". It is " + str(len(initial)))

    else:
        raise Exception("\"isotropy\" parameter must be \"iso\", \"diag\" or \"aniso\"")



    if method=="stochastic":
        warnings.filterwarnings("ignore")

        response = sp.optimize.differential_evolution( \
            func=param_optimizer, bounds=b, x0=initial, \
            args=(diffs, y, noise, bounds, x), seed=seed)

        warnings.filterwarnings("default")

    elif method=="gradient":
        warnings.filterwarnings("ignore")

        response = sp.optimize.minimize( \
            fun=param_optimizer, bounds=b, x0=initial, \
            args=(diffs, y, noise, bounds, x), method="L-BFGS-B")

        warnings.filterwarnings("default")

    else:
        raise Exception("\"method\" parameter must be \"stochastic\" or \"gradient\"")

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
    return optimized_metric(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8])



def optimal_metric(diffs, x, y, noise, bounds, iso, seed, threads):
    if bounds[0] >= bounds[1]:
        raise Exception("Lower bound must be strictly inferior to upper bound")

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
            args.append((diffs, x, y, noise, iso, seed, init, bounds, method))

    metrics_lmls = pool.map(optimized_metric_tuple, args)

    # keys are lml, values are metrics
    metrics = {}

    for i in range(len(metrics_lmls)-1):
        metrics[metrics_lmls[i][1]] = metrics_lmls[i][0]

    if -math.inf in metrics.keys():
        del metrics[-math.inf]

    metrics = dict(sorted(metrics.items(), reverse=True))

    for lml, metric in metrics.items():
        K = make_kernel(diffs, noise, metric)
        if (K > 1e-4).all():
            continue
        else:
            return metric, lml

    raise Exception("No optimal metric found. Try to change bounds, seed or number of points")



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
    bounds = bound_combinations([[0,1] for i in range(len(input_info))])
    return np.vstack([bounds, points])



def random_points(input_info, n, seed):
    check_data_info([np.ones(len(input_info))], input_info)
    m = math.ceil(math.log(n)/math.log(2))
    points_generator = sp.stats.qmc.Sobol(d=len(input_info), seed=seed)
    points = points_generator.random_base2(m=m)[:n]
    return points



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
    results = {}
    pool = multiprocessing.Pool(threads)
    points = random_points(input_info, math.ceil(n/2), seed)
    args = [(models, x, point, a, epsilon, constraints) for point in points]
    res  = pool.map(find_max_ei_gradient, args)

    points = random_points(input_info, math.floor(n/2), seed+1)
    args = [(models, x, point, a, epsilon, constraints, seed) for point in points]
    res2 = pool.map(find_max_ei_stochastic, args)
    pool.close()

    res = res + res2

    for i in range(len(res)):
        results[res[i][0]] = res[i][1]

    # keys are ei, values are points
    return dict(sorted(results.items(), reverse=True))



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
    response = sp.optimize.minimize( \
        fun=acquisition_function, x0=args[2], \
        args=(args[0], args[1], args[3], args[4], args[5]), method="L-BFGS-B", \
        bounds=[[0, 1] for i in range(len(args[1][0]))])

    return (-response.fun, response.x)



def find_max_ei_stochastic(args):
    response = sp.optimize.differential_evolution( \
        func=acquisition_function, x0=args[2], seed=args[6], \
        args=(args[0], args[1], args[3], args[4], args[5]), \
        bounds=[[0, 1] for i in range(len(args[1][0]))])

    return (-response.fun, response.x)



def acquisition_function(X_new, *args):
    if len(args[0]) > 1:
        for i in range(len(args[0])-1):
            pred, sigma = predict(args[0][i], args[1], [X_new])
            if not are_contraint_satifcation_probable(pred, sigma, args[2], args[4][i]):
                return 0

    pred, sigma = predict(args[0][0], args[1], [X_new])
    return -expected_improvement(pred, sigma, max(args[0][0][1]), args[2], args[3])[0]



def are_contraint_satifcation_probable(value, sigma, a, constraint):
    for nested_constraints in constraint[1]:
        if are_nested_contraint_satifcation_probable(value, sigma, a, nested_constraints):
            return True

    return False



def are_nested_contraint_satifcation_probable(value, sigma, a, constraints):
    if len(constraints) % 2 != 0:
        raise Exception("Nested constraint must have a pair length")

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
            raise Exception("Constraint condition must be either > or <")

    return True



# =================================================================
# =================================================================
# ============================ PLOT ===============================
# =================================================================
# =================================================================



def parallelPlot(x, y, input_info, output_info):
    all_data = np.hstack([x, y])
    all_labels = np.hstack([[input_info[i][0] for i in range(len(input_info))], [output_info[i][0] for i in range(len(output_info))]])
    all_scales = np.hstack([[input_info[i][2] for i in range(len(input_info))], [output_info[i][1] for i in range(len(output_info))]])
    dimensions = []

    # next block is for log-scaling log scaled variables because plotly doesn't support it trivialy
    for i in range(len(all_labels)):
        data = all_data[:,i]
        if all_scales[i] == "log":
            for j in range(len(data)):
                data[j] = math.log(data[j])
            ticks = [max(data)*i/10 for i in range(11)]
            text = [format(math.exp(ticks[i]), "1.2E") for i in range(len(ticks))]
            d = dict(label=all_labels[i], values = data, tickvals=ticks, ticktext=text)
        else:
            d = dict(label=all_labels[i], values = data)

        dimensions.append(d)

    fig = go.Figure(data=go.Parcoords(dimensions=dimensions))
    fig.show()



def pairPlot(x, y, input_info, output_info):
    all_data = pa.DataFrame(np.hstack([x, y]))
    all_labels = np.hstack([[input_info[i][0] for i in range(len(input_info))], [output_info[i][0] for i in range(len(output_info))]])
    all_data.columns = all_labels
    all_scales = np.hstack([[input_info[i][2] for i in range(len(input_info))], [output_info[i][1] for i in range(len(output_info))]])

    log_labels = []
    for i in range(len(all_labels)):
        if all_scales[i] == "log":
            log_labels.append(all_labels[i])

    fig = seaborn.pairplot(all_data, y_vars=[output_info[i][0] for i in range(len(output_info))], x_vars=[input_info[i][0] for i in range(len(input_info))])
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
    def __init__(self, title, noise, input_info, constraints, seed, threads, iso="diag", epsilon=1e-13):
        if np.array(noise).size != len(constraints)+1:
            raise Exception("Noise length must have same length than constraint one + 1 (ie: " + str(len(constraints)+1) + "). " + str(len(noise)) + " provided")
        self.title = title
        self.noise = noise
        self.input_info = input_info
        self.dummy_data_info = dummify_data_info(input_info)
        self.constraints = constraints
        self.seed = seed
        self.threads = threads
        self.x = None
        self.y = None
        self.iso = iso
        self.epsilon = epsilon
        self.kernel = None
        self.metric = None



    def add_data(self, x, y):
        x = np.array(x)
        y = np.array(y)
        if  x.ndim != 2:
            raise Exception("X data must be 2-dimensional")
        if  y.ndim != 2:
            raise Exception("Y data must be 2-dimensional")
        if y.shape[1] != len(self.constraints)+1:
            raise Exception("Y data must containt as much data as constraints + 1 (ie: " + str(len(self.constraints)+1) + "). " + str(y.shape[1]) + " provided")
        if self.x == None:
            self.x = x
            self.y = y
        else:
            self.x = np.vstack([self.x, x])
            self.y = np.vstack([self.y, y])




    def first_points(self, n):
        self.seed += 1
        return first_points(self.input_info, n, self.seed)



    def next_points(self, n, a, metric_bounds=[-12, 12]):
        x = preprocess_inputs(self.x, self.input_info)
        y = preprocess_outputs(self.y)

        print(np.hstack([x, y]), flush=True)

        self.seed += 1
        print("Calculating optimal metrics", flush=True)
        diffs = make_diff_list(x, self.input_info, uncertainties=False)

        metrics = []
        kernels = []
        for i in range(len(self.constraints)+1):
            metric, lml = optimal_metric(diffs, x, y[:,i], self.noise[i], metric_bounds, self.iso, self.seed, self.threads)
            metrics.append(metric)
            kernels.append(make_kernel(diffs, self.noise[i], metric))

        print(metrics[0], flush=True)
        print("lml = " + str(lml), flush=True)
        np.set_printoptions(formatter={'float':"{0:0.3f}".format})
        print(kernels[0], flush=True)

        self.seed += 1
        m = 100 * len(self.dummy_data_info) * math.ceil(math.sqrt(len(self.dummy_data_info)))
        print("Calculating next points", flush=True)

        models = []
        for i in range(len(metrics)):
            models.append((kernels[i], y[:,i], metrics[i]))


        next_pts = np.array(list(next_points(models, x, self.dummy_data_info, m, self.seed, a, self.epsilon, self.threads).values()))[:n]
        if n > next_pts.shape[0]-1:
            n =  next_pts.shape[0]-1

        self.kernel = kernels[0]
        self.metric = metrics[0]
        return postprocess_inputs(next_pts, self.dummy_data_info)
    
    
    def save(self):
        pass
    
    def load(self):
        pass
    
    