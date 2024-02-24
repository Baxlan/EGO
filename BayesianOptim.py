
import math
import numpy as np
import scipy as sp
import multiprocessing
import collections



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
        if data_info[i][1] != "real" and  data_info[i][1] != "discrete":
            raise Exception("Second data info must be \"real\" or \"discrete\". Those of the " + str(i) + "th element is " + str(data_info[i][1]))
        if data_info[i][2] != "lin" and  data_info[i][2] != "log":
            raise Exception("Third data info must be \"lin\" or \"log\". Those of the " + str(i) + "th element is " + str(data_info[i][2]))
        if type(data_info[i][3]) is not list and type(data_info[i][3]) is not tuple:
            raise Exception("Fourth data info must be of type list or tuple (boundaries of the variable). Those of the " + str(i) + "th element is of type " + type(data_info[i][3]))
        if len(data_info[i][3]) != 2:
            raise Exception("Fourth data info length must be 2 (lower and upper boundaries of the variable). Those of the " + str(i) + "th element is " + str(len(data_info[i][3])))
        if data_info[i][3][0] >= data_info[i][3][1]:
            raise Exception("Lower boundary of the " + str(i) + "th variable must be strictly inferior to its upper boundary")
        if data_info[i][2] == "log" and data_info[i][3][0] <= 0:
            raise Exception("Lower boundary of the " + str(i) + "th variable must be strictly positive because its scale is \"log\"")



# convert original data_info, deleting categorical variables and adding dummy ones to replace them
def convert_data_info(x, data_info):
    check_data_info(x, data_info)



# convert categorical variables to dummy ones
def convert_inputs(inputs, original_data_info, converted_data_info):
    pass



# convert dummy variables to categorical ones
def convert_outputs(out, original_data_info, converted_data_info):
    pass



def preprocess_data(x):
    pass
    # linearize log scaled variables

    # dummyfy categorical variables

    return x



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
    diffs=[]
    for i in range(len(x)):
        for j in range(len(x)-i):
            diffs.append(x[i] - x[j+i])

    return np.array(diffs)



def make_kernel(x, noise, metric):
    diffs = make_diff_list(x)
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
    metric = delinearize_metric(args[0], M, args[3])
    K = make_kernel(args[0], args[2], metric=metric)
    return -log_marginal_likelihood(K, args[1])



def optimized_metric(x, y, noise, isotropy, seed=32, initial=-2, bounds=[-9, 9], method="stochastic"):
    x = np.array(x)
    n = len(x[0])

    if isotropy == "iso":
        elem = [initial]
        b = [bounds]
    elif isotropy == "diag":
        elem = np.ones(n)*initial
        b = [bounds for i in range(n)]
    elif isotropy == "aniso":
        elem = np.ones(int(n*(n+1)/2))*initial
        b = [bounds for i in range(int(n*(n+1)/2))]
    else:
        raise Exception("\"isotropy\" parameter must be \"iso\", \"diag\" or \"aniso\"")

    if method=="stochastic":
        response = sp.optimize.differential_evolution( \
            func=param_optimizer, bounds=b, x0=elem, \
            args=(x, y, noise, bounds), seed=seed)
    elif method=="gradient":
        response = sp.optimize.minimize( \
            fun=param_optimizer, bounds=b, x0=elem, \
            args=(x, y, noise, bounds), method="L-BFGS-B")
    else:
        raise Exception("\"method\" parameter must be \"stochastic\" or \"gradient\"")

    M = response.x
    if isotropy == "aniso":
        M = make_symmetric_matrix_from_list(M)

    return delinearize_metric(x, M, bounds)



def max_metric(x):
    diffs = make_diff_list(x)
    m = np.max(np.abs(diffs), axis=0)
    return np.diag(1/np.square(m))



def mean_metric(x):
    diffs = make_diff_list(x)
    m = np.mean(np.abs(diffs), axis=0)
    return np.diag(1/np.square(m))



def optimal_metric():
    pass



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



def scale_point(point, data_info):
    for d in range(len(point)):
        if data_info[d][0] == "lin":
            point[d] *= (data_info[d][3] - data_info[d][2])
            point[d] += data_info[d][2]
        elif data_info[d][0] == "log":
            point[d] = data_info[d][2] * math.exp(point[d] * math.log(data_info[d][3]/data_info[d][2]))

    return point



def first_points(data_info, n, seed):
    points = random_points(data_info, n, seed)

    bounds = bound_combinations([[0,1] for i in range(len(data_info))])
    bounds = [scale_point(bound, data_info) for bound in bounds]

    return np.vstack([bounds, points])



def random_points(data_info, n, seed):
    m = math.ceil(math.log(n)/math.log(2))
    points_generator = sp.stats.qmc.Sobol(d=len(data_info), seed=seed)
    points = points_generator.random_base2(m=m)[:n]
    return [scale_point(point, data_info) for point in points]



def predict(kernel, x, y, x_new, metric):
        metric = check_metric(x, metric)

        y_mean = []
        y_sigma = []
        for new in x_new:
            local_kernel = add_to_kernel(kernel, x, new, metric)
            K = local_kernel[:len(x), :len(x)]
            k = local_kernel[:len(x), -1]
            inv_K = np.linalg.inv(K)
            y_mean.append(np.dot(np.dot(k, inv_K), y))
            y_sigma.append(local_kernel[-1,-1] - np.dot(np.dot(k, inv_K),k))

        return np.array(y_mean), np.array(y_sigma)



def next_points(kernel, x, y, data_info, n, seed, metric, threads, ei_a=1, ei_log=True, ei_epsilon=1e-13):
    check_data_info(x, data_info)
    points = random_points(data_info, 2*n, seed)

    results = {}
    pool = multiprocessing.Pool(threads)
    args = [(kernel, x, y, point, metric, data_info, seed, (ei_a, ei_log, ei_epsilon)) for point in points]
    res  = pool.map(find_max_ei_gradient, args)
    res2 = pool.map(find_max_ei_stochastic, args)
    pool.close()

    res = np.vstack([res, res2])

    for i in range(len(res)):
        results[res[i][0]] = res[i][1]

    # key is ei, value is point
    return collections.OrderedDict(sorted(results.items())[:n])



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



def expected_improvement(y_mean, y_sigma, y_best, a, log, epsilon):
        ei = []
        for i in range(len(y_mean)):
            if y_sigma[i] <= 0:
                ei.append(0)

            else:
                sigma = a*y_sigma[i]
                z = (y_mean[i] - y_best) / sigma

                if not log:
                    ei.append(sigma*h(z))
                else:
                    ei.append(math.exp(log_h(z, epsilon)+math.log(sigma)))

                if ei[i] < 0:
                    ei[i] = 0

        return ei



def find_max_ei_gradient(args):
    response = sp.optimize.minimize( \
        fun=acquisition_function, x0=args[3], \
        args=(args[0], args[1], args[2], args[4], args[7]), method="L-BFGS-B", \
        bounds=[[args[5][i][1], args[5][i][2]] for i in range(len(args[0][0]))])
    return (-response.fun, response.x)



def find_max_ei_stochastic(args):
    response = sp.optimize.differential_evolution( \
        func=acquisition_function, x0=args[3], seed=args[6], \
        args=(args[0], args[1], args[2], args[4], args[7]), method="L-BFGS-B", \
        bounds=[[args[5][i][1], args[5][i][2]] for i in range(len(args[0][0]))])
    return (-response.fun, response.x)



def acquisition_function(X_new, *args):
    pred, sigma = predict(args[0], args[1], args[2], [X_new], args[3])
    return -expected_improvement(pred, sigma, max(args[2]), args[4][0], args[4][1], args[4][2])[0]



# =================================================================
# =================================================================
# ============= BAYESIAN OPTIMIZATION CLASS =======================
# =================================================================
# =================================================================