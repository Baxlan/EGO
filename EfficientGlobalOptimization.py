
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import collections
import time
import multiprocessing

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



def check_metric(X, metric):
    if isinstance(metric, (int, float, np.floating)):
        metric = np.diag(np.full(len(X[0]), metric))
    elif type(metric) == list and len(metric) == 1:
        metric = np.diag(np.full(len(X[0]), metric[0]))
    else:
        metric = np.array(metric)
        if metric.ndim == 1 and len(metric) == len(X[0]):
            metric = np.diag(metric)
        elif metric.ndim == 2 and len(metric) == len(X[0]) and len(metric[0]) == len(X[0]) and (metric == metric.T).all():
            pass
        else:
            raise Exception("The \"metric\" parameter must either be a scalar, a 1D array of length N (problem dimensionality), or a 2D symmetric N*N array")



# =================================================================
# =================================================================
# ================== EGO CLASS  ===================================
# =================================================================
# =================================================================



def RBF_kernel(X, metric = 1, noise = 0):
    K = np.zeros([len(X), len(X)])
    for i in range(len(X)):
        for j in range(len(X)):
            diff = X[i] - X[j]
            K[i, j] = np.exp(-0.5 * sum(diff.transpose() * metric * diff))

    return K + noise * np.eye(len(K))



def RBF_kernel_tuple(args):
    return RBF_kernel(args[0], args[1], args[2])



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



class EGO:
    def __init__(self, noise, seed=0, m=7, m_rand=4):
        self.noise = noise
        self.seed = seed # used as seed for Sobol distribution
        self.m = m # power of 2 of sobol points to explore the parameter space
        self.m_rand = m_rand



    def predict(self, X, y, X_new, metric = 1):
        check_metric(X, metric)

        y_mean = []
        y_sigma = []
        for x_new in X_new:
            kernel = RBF_kernel(np.vstack([X, x_new]), metric, self.noise)
            K = kernel[:len(X), :len(X)]
            k = kernel[:len(X), -1]
            inv_K = np.linalg.inv(K)
            y_mean.append(np.dot(np.dot(k, inv_K), y))
            y_sigma.append(kernel[-1,-1] - np.dot(np.dot(k, inv_K),k))

        return np.array(y_mean), np.array(y_sigma)



    def expected_improvement(self, y_mean, y_sigma, y_best):
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



    def next_points(self, X, y, scales_bounds, metric = 1, threads=1):
        check_scales_bounds(X, scales_bounds)
        check_metric(X, metric)

        self.seed += 1
        points_generator = scipy.stats.qmc.Sobol(d=len(X[0]), seed=self.seed)
        points = points_generator.random_base2(m=self.m)
        self.seed += 1
        points_generator = scipy.stats.qmc.Sobol(d=len(X[0]), seed=self.seed)
        explore_points = points_generator.random_base2(m=self.m_rand)

        points = [scale_point(point, scales_bounds) for point in points]

        results = {}
        pool = multiprocessing.Pool(threads)
        args = [(X, y, point, metric, scales_bounds) for point in points]
        res = pool.map(self.POOL_get_optimized_ei_and_point, args)
        pool.close()

        for i in range(len(res)):
            results[res[i][0]] = res[i][1]

        return collections.OrderedDict(sorted(results.items())), explore_points



    def POOL_get_optimized_ei_and_point(self, args):
        response = scipy.optimize.minimize( \
            fun=self._acquisition_function, x0=args[2], \
            args=(args[0], args[1], args[3]), method="L-BFGS-B", \
            bounds=[[args[4][i][1], args[4][i][2]] for i in range(len(args[0][0]))])
        return (-response.fun, response.x)



    def _acquisition_function(self, X_new, *args):
        pred, sigma = self.predict(args[0], args[1], [X_new], args[2])
        return -self.expected_improvement(pred, sigma, max(args[1]))[0]



# =================================================================
# =================================================================
# ================== METRIC OPTIMIZER  ============================
# =================================================================
# =================================================================



def delete_inf(list1, list2):
    # removes infs from list 1 and delete the same indexes from list 2
    i = 0
    while i < len(list1):
        if list1[i] == math.inf:
            list1 = list1[:i]+list1[i+1:]
            list2 = list(list2)[:i]+list(list2)[i+1:]
            i = 0
        else:
            i += 1
    return list1, list2



def log_marginal_likelihood(K, y):
    try:
        L = scipy.linalg.cholesky(K)
    except:
        return math.inf

    S1 = scipy.linalg.solve_triangular(L, y, lower = True)
    S2 = scipy.linalg.solve_triangular(L.T, S1, lower=False)
    return np.sum(np.log(np.diagonal(L))) + 0.5*np.array(y).dot(S2) + 0.5*len(y)*np.log(2*np.pi)


def log_marginal_likelihood_tuple(arg):
    return log_marginal_likelihood(arg[0], arg[1])

class metric_optimizer:
    def __init__(self, noise, isotropy = "iso", seed=0, bounds=[1e-4, 1e7], it=4, m=7, m_pts_per_it=5, threads=1):
        # 2^m points are tested at each iteration
        # 2^m_pts_per_it are kept as results for the next iteration
        # (if type is "iso", there is only 1 iteration whatever the value of "it")
        self.noise = noise
        self.isotropy = isotropy
        self.seed = seed
        self.bounds = bounds
        self.it = it
        self.m = m
        self.m_pts_per_it = m_pts_per_it
        self.threads = threads



    def optimal_metric(self, X, y, plot=False):
        X = np.array(X)
        if self.bounds[0] <= 0:
            raise Exception("Lower bound must be strictly positive")

        if self.bounds[1] <= self.bounds[0]:
            raise Exception("higher bound must be strictly greater than lower bound")

        if self.isotropy == "iso" or len(X[0]) == 1:
            print("calculating isotropic metric", flush=True)
            start_t = time.time()

            metric = np.logspace(math.log(self.bounds[0], 10), math.log(self.bounds[1], 10), 2**self.m)
            pool = multiprocessing.Pool(self.threads)
            args = [(X, metr, self.noise) for metr in metric]
            K = pool.map(RBF_kernel_tuple, args)
            args = [(k, y) for k in K]
            lml = pool.map(log_marginal_likelihood_tuple, args)
            pool.close()

            end_t = time.time()
            print(str(round(end_t - start_t)) + " sec", flush=True)

            if plot:
                plt.scatter(metric, lml)
                plt.xscale("log")
                plt.yscale("log")
                plt.title("Log marginal likelihood function of the isotropic metric")
                plt.xlabel("Isotropic metric")
                plt.ylabel("Log marginal likelihood")
                plt.grid()
                plt.show()

            return metric[lml.index(min(lml))]

        elif self.isotropy == "diag":
            bounds = [("log", self.bounds[0], self.bounds[1]) for i in range(len(X[0]))]
            points = first_points(bounds, m=5, seed=self.seed)
            self.seed += 1
            pool = multiprocessing.Pool(self.threads)
            args = [(X, metr, self.noise) for metr in points]
            K = pool.map(RBF_kernel_tuple, args)
            args = [(k, y) for k in K]
            lml = pool.map(log_marginal_likelihood_tuple, args)
            pool.close()
            lml, points = delete_inf(lml, points)

            lowest_lml = [min(lml)]
            lowest_metric = [points[lml.index(min(lml))]]

            model = EGO(noise=0, seed=self.seed, m=self.m, m_rand=self.m_pts_per_it-1)
            self.seed += 1
            metric_maker = metric_optimizer(noise=0, isotropy="iso", bounds=[1e-3, 1e9], m=9)

            for i in range(1, self.it+1):
                metr_iso = metric_maker.optimal_metric(points, np.array(lml))
                print("Inferring LML in diagonal metric space : iteration " + str(i), flush=True)
                start_t = time.time()

                next_pts, random = model.next_points(points, np.array(lml), bounds, metr_iso, threads=self.threads)
                print(next_pts.keys())
                next_pts = np.vstack([list(next_pts.values())[:2**(self.m_pts_per_it-1)], random])

                pool = multiprocessing.Pool(self.threads)
                args = [(X, metr, self.noise) for metr in next_pts]
                K = pool.map(RBF_kernel_tuple, args)
                args = [(k, y) for k in K]
                lml = lml + pool.map(log_marginal_likelihood_tuple, args)
                pool.close()

                points = np.vstack([points, next_pts])
                lml, points = delete_inf(lml, points)

                lowest_lml.append(min(lml))
                lowest_metric.append(points[lml.index(min(lml))])

                end_t = time.time()

                print("Lowest LML : " + "{:.2E}".format(min(lowest_lml)) + ", " + str(round(end_t - start_t)) + " sec", flush=True)

            if plot:
                plt.scatter(range(1, len(lowest_lml)+1), lowest_lml)
                plt.title("Lowest log marginal likelihood at each iteration")
                plt.xlabel("Iteration")
                plt.ylabel("Lowest log marginal likelihood")
                plt.grid()
                plt.show()

            return lowest_metric[lowest_lml.index(min(lowest_lml))]

        else:
            raise Exception("\"type\" parameter must be \"iso\" or \"diag\"")



if __name__ == '__main__':
    pass