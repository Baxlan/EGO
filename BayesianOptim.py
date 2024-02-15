
import math
import numpy as np
import scipy as sp



# =================================================================
# =================================================================
# ===============  UTILITARIES  ===================================
# =================================================================
# =================================================================



def check_scales_bounds(X, scales_bounds):
    if type(scales_bounds) is not list:
        raise Exception("\"scales_bounds\" parameter must be a list")
    if len(scales_bounds) is not len(X[0]):
        raise Exception("\"scales_bounds\" parameter length must be equal to the problem dimentionality")
    for i in range(len(scales_bounds)):
        if type(scales_bounds[i]) is not tuple:
            raise Exception("\"scales_bounds\" elements must be tuples." + str(i) + "th element is " + type(scales_bounds[i]))
        if scales_bounds[i][0] != "lin" and  scales_bounds[i][0] != "log":
            raise Exception("\"scales_bounds\" tuple's first element must be \"lin\" or \"log\"")
        if type(scales_bounds[i][1]) is not int and type(scales_bounds[i][1]) is not float:
            raise Exception("\"scales_bounds\" tuple's second element must be an int or a float (lower boundary of parameter)")
        if type(scales_bounds[i][2]) is not int and type(scales_bounds[i][1]) is not float:
            raise Exception("\"scales_bounds\" tuple's third element must be an int or a float (higher boundary of parameter)")
        if scales_bounds[i][1] >= scales_bounds[i][2]:
            raise Exception("\"scales_bounds\" tuple's second element must be strictly inferior to the third element")
        if scales_bounds[i][0] == "log" and scales_bounds[i][1] <= 0:
            raise Exception("\"scales_bounds\" tuple's second element must be strictly positive when scale is \"log\"")



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



# =================================================================
# =================================================================
# ================== DATA PREPROCESSING ===========================
# =================================================================
# =================================================================



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



def add_to_kernel(K, x, x_new):
    diffs = []
    for point in x:
        diffs.append(x_new - point)

    diffs = [np.exp(-np.sum(diff.transpose() * metric * diff)) for diff in diffs]

    K = np.vstack([K, diffs])
    K = np.hstack([K, diffs.T+[0]])
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



def delinearize_metric(metric, bounds):
    return metric



def param_optimizer(M, *args):
    metric = delinearize_metric(M, args[3])
    K = make_kernel(args[0], args[2], metric=metric)
    return -log_marginal_likelihood(K, args[1])



def optimized_metric(x, y, noise, isotropy, seed=32, initial=1e-2, bounds=[1e-9, 1e1], method="stochastic"):
    x = np.array(x)
    n = len(x[0])

    # linearize bounds
    lin_bounds = bounds

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
            func=param_optimizer, bounds=lin_bounds, x0=elem, \
            args=(x, y, noise, bounds), seed=seed)
    elif method=="gradient":
        response = sp.optimize.minimize( \
            fun=param_optimizer, bounds=lin_bounds, x0=elem, \
            args=(x, y, noise, bounds), method="L-BFGS-B")
    else:
        raise Exception("\"method\" parameter must be \"stochastic\" or \"gradient\"")

    M = response.x
    if isotropy == "aniso":
        M = make_symmetric_matrix_from_list(M)

    return delinearize_metric(M, bounds)



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
# ================== BAYESIAN OPTIMIZATION ========================
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



def scale_point(point, scales_bounds):
    for d in range(len(point)):
        if scales_bounds[d][0] == "lin":
            point[d] *= (scales_bounds[d][2] - scales_bounds[d][1])
            point[d] += scales_bounds[d][1]
        elif scales_bounds[d][0] == "log":
            point[d] = scales_bounds[d][1] * math.exp(point[d] * math.log(scales_bounds[d][2]/scales_bounds[d][1]))

    return point



def first_points(scales_bounds, m=5, seed=0):
    points_generator = scipy.stats.qmc.Sobol(d=len(scales_bounds), seed=seed)
    points = points_generator.random_base2(m=m)
    bounds = bound_combinations([[0,1] for i in range(len(scales_bounds))])

    points = [scale_point(point, scales_bounds) for point in points]
    bounds = [scale_point(bound, scales_bounds) for bound in bounds]

    return np.vstack([bounds, points])



def predict(x, y, x_new, noise, metric):
        metric = check_metric(x, metric)
        kernel = make_kernel(x, noise, metric)

        y_mean = []
        y_sigma = []
        for new in x_new:
            local_kernel = add_to_kernel(kernel, new)
            K = local_kernel[:len(x), :len(x)]
            k = local_kernel[:len(x), -1]
            inv_K = np.linalg.inv(K)
            y_mean.append(np.dot(np.dot(k, inv_K), y))
            y_sigma.append(local_kernel[-1,-1] - np.dot(np.dot(k, inv_K),k))

        return np.array(y_mean), np.array(y_sigma)



def expected_improvement(y_mean, y_sigma, y_best):
        ei = []
        for i in range(len(y_mean)):
            if y_sigma[i] == 0:
                ei.append(0)

            else:
                z = (y_mean[i] - y_best) / y_sigma[i]
                ei.append((y_mean[i] - y_best) * scipy.stats.norm.cdf(z) + y_sigma[i] * scipy.stats.norm.pdf(z))
                if ei[i] < 0:
                    ei[i] = 0

        return ei



def next_points(X, y, scales_bounds, m, m_rand, seed, metric, threads=1):
    check_scales_bounds(X, scales_bounds)

    seed += 1
    points_generator = scipy.stats.qmc.Sobol(d=len(X[0]), seed=seed)
    points = points_generator.random_base2(m=m)
    seed += 1
    points_generator = scipy.stats.qmc.Sobol(d=len(X[0]), seed=seed)
    explore_points = points_generator.random_base2(m=m_rand)

    points = [scale_point(point, scales_bounds) for point in points]

    results = {}
    pool = multiprocessing.Pool(threads)
    args = [(X, y, point, metric, scales_bounds) for point in points]
    res = pool.map(POOL_get_optimized_ei_and_point, args)
    pool.close()

    for i in range(len(res)):
        results[res[i][0]] = res[i][1]

    return collections.OrderedDict(sorted(results.items())), explore_points



def POOL_get_optimized_ei_and_point(args):
    response = scipy.optimize.minimize( \
        fun=_acquisition_function, x0=args[2], \
        args=(args[0], args[1], args[3]), method="L-BFGS-B", \
        bounds=[[args[4][i][1], args[4][i][2]] for i in range(len(args[0][0]))])
    return (-response.fun, response.x)



def _acquisition_function(X_new, *args):
    pred, sigma = predict(args[0], args[1], [X_new], args[2])
    return -expected_improvement(pred, sigma, max(args[1]))[0]