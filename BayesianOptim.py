
import math
import numpy as np
import scipy as sp
import multiprocessing
import warnings
import copy



# =================================================================
# =================================================================
# ===============  DATA HANDLING TOOLS  ===========================
# =================================================================
# =================================================================



def check_data_info(x, data_info):
    if type(data_info) is not list:
        raise Exception("\"data_info\" parameter must be a list")
    if len(data_info) is not len(x[0]):
        raise Exception("\"data_info\" parameter length must be equal to the number of variables in the dataset")
    for i in range(len(data_info)):
        if type(data_info[i]) is not tuple and type(data_info[i]) is not list:
            raise Exception("\"data_info\" elements must be tuples or lists." + str(i) + "th element is of type " + type(data_info[i]))
        if type(data_info[i][0]) != str:
            raise Exception("First data info must be of type str (name of the variable). Those of the " + str(i) + "th element is of type " + type(data_info[i][0]))
        if data_info[i][1] != "real"  and  data_info[i][1] != "discrete":
            raise Exception("Second data info must be \"real\" or \"discrete\". If it is \"categorical\", the dummify_data_info() function must be used. The value of the " + str(i) + "th element is " + str(data_info[i][1]))
        if data_info[i][2] != "lin" and  data_info[i][2] != "log":
            raise Exception("Third data info must be \"lin\" or \"log\". Those of the " + str(i) + "th element is " + str(data_info[i][2]))
        if type(data_info[i][3]) is not list and type(data_info[i][3]) is not tuple:
            raise Exception("Fourth data info must be of type list or tuple (boundaries of the variable). Those of the " + str(i) + "th element is of type " + type(data_info[i][3]))
        if len(data_info[i][3]) != 2:
            raise Exception("Fourth data info length must be 2 (lower and upper boundaries of the variable). Those of the " + str(i) + "th element is " + str(len(data_info[i][3])))
        if data_info[i][3][0] >= data_info[i][3][1]:
            raise Exception("Lower boundary of the " + str(i) + "th variable must be strictly inferior to its upper boundary")
        if data_info[i][2] == "log" and data_info[i][3][0] == 0:
            raise Exception("Lower boundary of the " + str(i) + "th variable cannot be 0 because its scale is \"log\"")
        if data_info[i][2] == "log" and data_info[i][3][0] < 0 and data_info[i][3][1] > 0:
            raise Exception("Boundaries of the " + str(i) + "th variable must be of the same sign because its scale is \"log\"")



def dummify_data_info(data_info):
    new_data_info = []
    for i in range(len(data_info)):
        if data_info[i][1] == "real" or data_info[i][1] == "discrete":
            new_data_info.append(data_info[i])

        elif data_info[i][1] == "categorical":
            for j in range(len(data_info[3])):
                if type(data_info[3][j]) != str:
                    raise Exception("Categories of categorical variables must be str")

                new_data_info.append([data_info[0] + "~" + data_info[3][j], "real", "lin", [0, 1]])
        else:
            raise Exception("Second data_info must be \"real\", \"discrete\" or \"categorical\". The value of the " + str(i) + "th element is " + str(data_info[i][1]))
    return new_data_info



def categorify_and_discretize_data(x, data_info):
    out = []

    for data in x:
        out.append([])
        skipper = 0
        for i in range(len(data_info)):
            if data_info[i][1] == "categorical":
                category = ""
                val = -1
                for j in range(len(data_info[i][3])):
                    if data[i+skipper] > val:
                        category = data_info[i][3][j]
                        val = data[i+skipper]
                    skipper += 1
                skipper -= 1
                out[len(out)-1].append(category)

            elif data_info[i][1] == "discrete":
                out[len(out)-1].append(round(data[i+skipper]))

            else:
                out[len(out)-1].append(data[i+skipper])
    return out



def preprocess_inputs(x, data_info):
    check_data_info(x, data_info)
    inp = copy.deepcopy(x)

    for i in range(len(data_info)):

        # linearize log scaled variables
        if data_info[i][2] == "log":
            sign = 1
            inf = data_info[i][3][0]
            sup = data_info[i][3][1]
            if data_info[i][3][0] < 0:
                sign = -1
                inf = -data_info[i][3][1]
                sup = -data_info[i][3][0]
            inp[:, i] = np.log(sign*inp[:, i])

            # normalize log variables
            inp[:, i] -= np.log(inf)
            inp[:, i] /= np.log(sup)

        # normalize non log variables
        else:
            inp[:, i] -= data_info[i][3][0]
            inp[:, i] /= (data_info[i][3][1] - data_info[i][3][0])

    return inp



def preprocess_outputs(y):
    out = np.array(copy.deepcopy(y))

    # normalize outputs
    for i in range(out.shape[1]):
        min = np.min(out[:, i])
        max = np.max(out[:, i])
        out[:, i] -= min
        out[:, i] /= (max - min)


    return out



def postprocess_inputs(scaled_x, data_info):
    check_data_info(scaled_x, data_info)
    inp = copy.deepcopy(scaled_x)

    for i in range(len(data_info)):
        if data_info[i][2] == "log":
            sign = 1
            inf = data_info[i][3][0]
            sup = data_info[i][3][1]
            if data_info[i][3][0] < 0:
                sign = -1
                inf = -data_info[i][3][1]
                sup = -data_info[i][3][0]

            # denormalize log variables
            inp[:, i] *= np.log(sup)
            inp[:, i] += np.log(inf)
            # exponentiate log scaled variables
            inp[:, i] = sign*np.exp(inp[:, i])

        # denormalize non log variables
        else:
            inp[:, i] *= (data_info[i][3][1]-data_info[i][3][0])
            inp[:, i] += data_info[i][3][0]

    return inp



def postprocess_output(scaled_y, y, scaled_sigma = []):
    out = copy.deepcopy(scaled_y)

    # denormalize outputs
    min = np.min(y)
    max = np.max(y)
    out *= (max-min)
    out += min

    if len(scaled_sigma) == 0:
        return out
    else:
        sigma = copy.deepcopy(scaled_sigma)
        sigma *= (max-min)
        return out, sigma



def check_data(x, y):
    for i in range(x.shape[1]):
        zero = False
        one = False
        for j in range(x.shape[0]):
            if x[i, j] == 0:
                zero = True
            elif x[i, j] == 1:
                one = True
            elif x[i, j] < 0 or x[i, j] > 1:
                raise Exception("Data have not been preprocessed")
        if not zero or not one:
            raise Exception("Data have not been preprocessed")

    for i in range(y.shape[1]):
        zero = False
        one = False
        for j in range(y.shape[0]):
            if y[i, j] == 0:
                zero = True
            elif y[i, j] == 1:
                one = True
            elif y[i, j] < 0 or x[i, j] > 1:
                raise Exception("Data have not been preprocessed")
        if not zero or not one:
            raise Exception("Data have not been preprocessed")



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



def make_diff_list(x):
    diffs = []
    for i in range(len(x)):
        for j in range(len(x)-i):
            diffs.append(x[i] - x[j+i])

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



def first_points(data_info, n, seed):
    points = random_points(data_info, n, seed)
    bounds = bound_combinations([[0,1] for i in range(len(data_info))])
    return np.vstack([bounds, points])



def random_points(data_info, n, seed):
    check_data_info([np.ones(len(data_info))], data_info)
    m = math.ceil(math.log(n)/math.log(2))
    points_generator = sp.stats.qmc.Sobol(d=len(data_info), seed=seed)
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



def next_points(models, x, data_info, constraints, n, seed, a, epsilon=1e-13, threads=1):
    results = {}
    pool = multiprocessing.Pool(threads)
    points = random_points(data_info, math.ceil(n/2), seed)
    args = [(models, x, point, a, epsilon, constraints) for point in points]
    res  = pool.map(find_max_ei_gradient, args)

    points = random_points(data_info, math.floor(n/2), seed+1)
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
# ============= BAYESIAN OPTIMIZATION CLASS =======================
# =================================================================
# =================================================================



class BayesianOptimizer:
    def __init__(self, title, noise, data_info, constraints, seed, threads, iso="diag", epsilon=1e-13):
        if np.array(noise).size != len(constraints)+1:
            raise Exception("Noise length must have same length than constraint one + 1 (ie: " + str(len(constraints)+1) + "). " + str(len(noise)) + " provided")
        self.title = title
        self.noise = noise
        self.data_info = data_info
        self.dummy_data_info = dummify_data_info(data_info)
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
        return first_points(self.data_info, n, self.seed)



    def next_points(self, n, a, metric_bounds=[-12, 12]):
        x = preprocess_inputs(self.x, self.data_info)
        y = preprocess_outputs(self.y)

        print(np.hstack([x, y]), flush=True)

        self.seed += 1
        print("Calculating optimal metrics", flush=True)
        diffs = make_diff_list(x)

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