# Data Preparation #
import sys
import GPy
import numpy as np
import numpy # fix up
import pandas as pd
from scipy import optimize
import multiprocessing as mp
#import gc
import random
from scipy.stats import norm
from itertools import chain
from itertools import zip_longest
import matplotlib.pyplot as plt
import csv
import PairwiseCV as pairwiseCV
#from concurrent.futures import ProcessPoolExecutor, as_completed


def sparse_to_dense(data):
    """Convert sparse gptoolbox matrices to dense format.

    Inputs:

    mat_x -- X matrix

    mat_m -- M matrix

    vec_y -- y vector

    """

    assert len(data) == 3

    mat_x = np.array(data[0])
    vec_y = np.array(data[2])
    mat_m = np.zeros((len(vec_y), mat_x.shape[0]))

    idx_ii, idx_ij = np.unravel_index(data[1][0], dims=mat_m.shape, order='C')  # pylint: disable=unbalanced-tuple-unpacking)

    idx_ji, idx_jj = np.unravel_index(data[1][1], dims=mat_m.shape, order='C')  # pylint: disable=unbalanced-tuple-unpacking)

    mat_m[idx_ii, idx_ij] = -1

    mat_m[idx_ji, idx_jj] = 1

    return mat_x, mat_m, vec_y

# Initialise new
# speech ### change write and read files for reduceNoise ###
def initialise_new(context):

    df = pd.read_json('06_11_19_gt5_dump.json', orient='records')

    df = df[df.environment.str.contains(context)]  # most interesting so far are home, noisyVenue
    print(df)

    mat_y = df.y

    #compars = settingsInd.apply(pd.DataFrame)  # reshuffle to a logical df
    #print(mat_m.iloc[1])
    #print(len(settingsInd))

    toDrop = []
    for rec in range(len(mat_y)):
        print(mat_y.iloc[rec])
        if len(mat_y.iloc[rec]) > 40:
            #toDrop.append(rec)
            print('yes')
        nonzero_c = np.count_nonzero(np.array(mat_y.iloc[rec]) == 0.5)
        #if(nonzero)
        #print(nonzero_c)
        if nonzero_c / len(mat_y.iloc[rec]) == 1:  # check its not just all 0s
            toDrop.append(rec)
    print(toDrop)
    df.drop(df.index[toDrop], inplace=True)  # may be uneccessary
    #df.drop(df.index[toDrop], inplace=True)

    # for i in range(0, len(compars)): maybe another way of deleting duplciates
    #    if np.all(Ypooled[Xpooled[:,  6]==i] == Ypooled[Xpooled[:, 6]==i+1])
    #        np.delete

    return(df) # need df?

def pool_data_new(N, df):
    #data = np.array(compars.iloc[0:N])
    Ypooled = []
    Xpooled = []
    bestPooled = []

    recLoc = []   # keep track of where the record starts and finishes
    recLoc.append(0)  # begin
    for i in range(0, N):  # For each record
        bestPooled.append(df.iloc[i].learned_best_setting)
        X = []  # unique settings
        Xind = []  # comparisons
        Xtrain = np.empty([0, 7])  # could define relative function or absolute (a-b rather than a,b)

        #print(Xind)
        X, mat_m, vec_y = sparse_to_dense((np.array(np.array(df.X)[i]), df.iloc[i].M_, df.iloc[i].y))
        t_d = []
        for j in range(len(mat_m)):
            if np.count_nonzero(mat_m[j]) > 0:  # if no bug
                Xind.append([int(np.flatnonzero(mat_m[j] == -1)), int(np.flatnonzero(mat_m[j] == 1))])
            else:
                print('error on', i)
                t_d.append(j)
        if t_d:
            vec_y = np.delete(vec_y, t_d, 0)

        for k in range(0, len(Xind)):  # for each comparison
            print(Xind[k][1])
            Xtrain = np.vstack((Xtrain, (np.hstack((X[Xind[k][0]], X[Xind[k][1]], i)))))  # [A, B, session]

        recLoc.append(len(Xind) + recLoc[-1])
        #vec_y = (vec_y * 2.0) - 1.0
        #vec_y = vec_y / np.max(np.abs(vec_y))
        Ypooled.append(np.array(vec_y))

        # print(Xtrain)
        Xpooled.append(Xtrain)

    Ypooled = np.concatenate(Ypooled).ravel()  # flatten
    Ypooled = (Ypooled * 2.0) - 1.0  # Set range of Y to -1,1
    # Ypooled = (((Ypooled - np.min(Ypooled)) * 2) / (np.max(Ypooled) - np.min(Ypooled))) - 1
    Ypooled = np.reshape(Ypooled, (-1, 1))
    Xpooled = np.concatenate(Xpooled)
    bestPooled = np.array(bestPooled)

    return (bestPooled, Xpooled, Ypooled, recLoc)

# Sort the data for PairwiseCV #
def sort_data(Xpooled, Ypooled):
    # Get all unique settings and the indexes where they appear
    X_not_u = np.reshape(Xpooled[:, 0:6], ((len(Xpooled) * 2, 3)))
    X, Xind = np.unique(X_not_u, axis=0, return_inverse=True)
    AB = np.reshape(Xind, (len(Xpooled), 2))
    # print(AB)
    # print(np.expand_dims(Xpooled[:,6],axis=0))
    Y = np.concatenate((AB.astype(int), np.expand_dims(Xpooled[:, 6].astype(int), axis=0).T), axis=1)
    print(Y)
    # Y = np.concatenate((Y, Xpooled[:,6].T), axis=1)
    return (X, Y)

    # Store in a list
    #


# Models #

# Batch GP
class BGP(object):
    def __init__(self, Xtrain, Ytrain):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain

    def train(self, hypers):
        k0 = GPy.kern.RBF(input_dim=1, active_dims=[6],
                          lengthscale=0.0000000001)  # a kernel for the users/session (Delta, implement)
        k1 = GPy.kern.PjkRbf(input_dim=6, active_dims=[0, 1, 2, 3, 4, 5], variance=hypers[0],
                             lengthscale=np.array(hypers[1:4]).ravel(), ARD=True)
        self.kf = GPy.kern.Prod(kernels=[k0, k1])
        m = GPy.models.GPRegression(self.Xtrain, self.Ytrain, kernel=self.kf, noise_var=hypers[4])
        return m

    def model_wrapper(self, coefs):
        m = self.train(coefs)
        lml = m._log_marginal_likelihood
        return -1.0 * lml

    def optimise(self): # could abstract this further
        bounds = [(0.01, 1), (0.01, 30), (0.01, 30), (0.01, 30), (0.01, 1)]
        results = []
        for i in range(0, 5):
            self.hyperparams = [numpy.random.uniform(0, 1, 1), numpy.random.randint(1, 30, size=1), numpy.random.randint(1, 30, size=1), numpy.random.randint(1, 30, size=1), numpy.random.uniform(0, 1, 1)]
            print(self.hyperparams)

            nlm0 = self.model_wrapper(self.hyperparams)  # get initial (negative) log marginal likelihoods

            ### Minimize negative log marginal likelihood
            res = optimize.minimize(self.model_wrapper, self.hyperparams, method='L-BFGS-B',
                                        bounds=bounds, tol=1e-6, options={'ftol': 1e-6, 'gtol': 1e-5, 'disp': False})
            ## Disp the final values
            print("Initial (positive) log marginal likelihood: %f" % (-nlm0))
            print("Final (positive) log marginal likelihood: %f" % (-res['fun']))
            for i in range(0, len(self.hyperparams)):
                print("Found hypers: %f" % (res['x'][i]))
            #self.hyperparams = res['x'][:]

            results.append((-res['fun'], list(res['x'][:])))
            #gc.collect()

            #print(results)

        results_dict = dict(results)

        results = results_dict[max(results_dict)]

        return results, max(results_dict)

# Polynomial Mean GP (Need the 'PairwiseLinear' mapping in your GPy, file name 'pairwise_linear.py' in repo)
class PMGP(object):
    def __init__(self, order, Xtrain, Ytrain, hypers):
        self.p_order = order
        self.var = hypers[4]
        self.mf = GPy.mappings.PairwiseLinear(input_dim=7, output_dim=1, order=self.p_order)
        self.A_dims = self.mf.A_dims
        # self.coefs = numpy.random.uniform(-0.05, 0.05, self.A_dims)
        k0 = GPy.kern.RBF(input_dim=1, active_dims=[6],
                          lengthscale=0.0000000001)  # a kernel for the users/session (Delta, implement)
        k1 = GPy.kern.PjkRbf(input_dim=6, active_dims=[0, 1, 2, 3, 4, 5], variance=hypers[0],
                             lengthscale=hypers[1:4], ARD=True)

        self.kf = GPy.kern.Prod(kernels=[k0, k1])
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain

    def train(self, coefs):
        A = coefs.reshape(-1, 1)
        self.mf.A = A
        m = GPy.models.GPRegression(self.Xtrain, self.Ytrain, kernel=self.kf, mean_function=self.mf,
                                    noise_var=self.var)
        # can add priors on hypers here
        return m

    def model_wrapper(self, coefs):
        m = self.train(coefs)
        lml = m._log_marginal_likelihood
        return -1.0 * lml

    def optimise(self): # could abstract this further
        # tune the coefficients
        bounds = [(-0.1, 0.1), ] * ((3 ** self.p_order))  # worth testing w.r.t these bounds perhaps
        results = []
        for i in range(0, 1):  # the change in results between successive runs is very small
            self.coefs = numpy.random.uniform(-0.1, 0.1, (3 ** self.p_order))

            nlm0 = self.model_wrapper(self.coefs)  # get initial (negative) log marginal likelihood

            ### Minimize negative log marginal likelihood
            res = optimize.minimize(self.model_wrapper, self.coefs, method='L-BFGS-B',
                                    bounds=bounds, tol=1e-6, options={'ftol': 1e-6, 'gtol': 1e-5, 'disp': False})

            ## Disp the final values

            print("Initial (positive)? log marginal likelihood: %f" % (-nlm0))
            print("Final (positive)? log marginal likelihood: %f" % (-res['fun']))
            for i in range(0, len(self.coefs)):
                print("Found coef: %f" % (res['x'][i]))

            results.append((-res['fun'], list(res['x'][:])))
            #gc.collect()

        results_dict = dict(results)

        results = results_dict[max(results_dict)]

        return results, max(results_dict)

# GP/Pooled :qGP
class GP(object):
    def __init__(self, Xtrain, Ytrain, ard_b=True, priors=True, prin=False):
        self.Xtrain = Xtrain[:, 0:6]
        self.Ytrain = Ytrain
        self.ard_b = ard_b
        self.prin = True
        self.kf = GPy.kern.PjkRbf(input_dim=6, active_dims=[0, 1, 2, 3, 4, 5], variance=0., lengthscale=1., ARD=self.ard_b)
        self.likelihood = GPy.likelihoods.Gaussian(variance=1)
        if priors:
            self.likelihood.variance.set_prior(GPy.priors.Gamma(1.5, 5.))
            self.kf.variance.set_prior(GPy.priors.Gamma(1.3, 3.))
            self.kf.lengthscale.set_prior(GPy.priors.Gamma(1.1, 0.05))

    def train(self, hypers):
        self.kf.variance = hypers[0]
        if self.ard_b:
            self.kf.lengthscale = np.asarray(hypers[1:4]).ravel()
            self.likelihood.variance = hypers[4]
        else:
            self.kf.lengthscale = np.asarray(hypers[1]).ravel()
            self.likelihood.variance = hypers[2]
        m = GPy.core.GP(self.Xtrain, self.Ytrain, kernel=self.kf, likelihood=self.likelihood)
        return m

    def model_wrapper(self, hypers):
        m = self.train(hypers)
        lml = m._log_marginal_likelihood + m.log_prior()
        return -1.0 * lml

    def optimise(self):  # could abstract this further
        # tune the coefficients
        if self.ard_b:
            bounds = [(0.01, 1), (0.01, 100), (0.01, 100), (0.01, 100), (0.01, 1)]  # perhaps hyper-priors here
        else:
            bounds = [(0.01, 1), (0.01, 100), (0.01, 1)]
        results = []
        for i in range(0, 1):
            if self.ard_b:
                self.hypers = [numpy.random.uniform(0.01, 1, 1), numpy.random.uniform(0.01, 40, size=1), numpy.random.uniform(0.01, 40, size=1),
                               numpy.random.uniform(0.01, 40, size=1), numpy.random.uniform(0.01, 1, 1)]
            else:
                self.hypers = [numpy.random.uniform(0.01, 1, 1), numpy.random.uniform(0.01, 40, size=1),
                               numpy.random.uniform(0.01, 1, size=1)]

            # print(self.hyperparams)
            nlm0 = self.model_wrapper(self.hypers)  # get initial (negative) log marginal likelihood

            ### Minimize negative log marginal likelihood
            res = optimize.minimize(self.model_wrapper, self.hypers, method='L-BFGS-B',
                                    bounds=bounds, tol=1e-6, options={'ftol': 1e-6, 'gtol': 1e-5, 'disp': False})
            ## Disp the final values

            if self.prin:
                print("Initial (positive) log marginal likelihood: %f" % (-nlm0))
                print("Final (positive) log marginal likelihood: %f" % (-res['fun']))
                for i in range(0, len(self.hypers)):
                   print("Found hypers: %f" % (res['x'][i]))


            results.append((-res['fun'], list(res['x'][:])))

            print(results)

        results_dict = dict(results)

        results = results_dict[max(results_dict)]

        return results, max(results_dict)

# Hierarchical GP (m(x) ~ GP)
class HGP(object):
    def __init__(self, Xtrain, Ytrain, hypers):  # init H_hypers
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.var = hypers[4]  # likelihood, (loses accuracy around boundaries)
        self.kern_lower = GPy.kern.PjkRbf(input_dim=6, active_dims=[0, 1, 2, 3, 4, 5], variance=hypers[0], lengthscale=hypers[1:4], ARD=True)
        self.kern_upper = GPy.kern.PjkRbf(input_dim=6, active_dims=[0, 1, 2, 3, 4, 5], variance=hypers[0], lengthscale=hypers[1:4], ARD=True)

    def train(self, H_hypers):
        self.kern_upper.variance = H_hypers[0]
        self.kern_upper.lengthscale = np.asarray(H_hypers[1:4]).ravel()

        self.kern_lower.variance = H_hypers[4]
        self.kern_lower.lengthscale = np.asarray(H_hypers[5:8]).ravel()

        k_hierarchy = GPy.kern.Hierarchical(kernels=[self.kern_upper, self.kern_lower])
        m = GPy.models.GPRegression(self.Xtrain, self.Ytrain, noise_var=self.var, kernel=k_hierarchy)
        return m

    def model_wrapper(self, coefs):
        m = self.train(coefs)
        lml = m._log_marginal_likelihood
        return -1.0 * lml

    def optimise(self): # could abstract this further
        bounds = [(0.01, 1), (0.01, 30), (0.01, 30), (0.01, 30), (0.01, 1), (0.01, 30), (0.01, 30), (0.01, 30)]  # keep bounds reasonable given the problem
        results = []
        for i in range(0, 1):
            self.hyperparams = [numpy.random.uniform(0, 1, 1), numpy.random.randint(1, 30, size=1), numpy.random.randint(1, 30, size=1), numpy.random.randint(1, 30, size=1),
                                numpy.random.uniform(0, 1, 1), numpy.random.randint(1, 30, size=1), numpy.random.randint(1, 30, size=1), numpy.random.randint(1, 30, size=1)]
            print(self.hyperparams)

            nlm0 = self.model_wrapper(self.hyperparams)  # get initial (negative) log marginal likelihood

            ### Minimize negative log marginal likelihood
            res = optimize.minimize(self.model_wrapper, self.hyperparams, method='L-BFGS-B',
                                        bounds=bounds, tol=1e-6, options={'ftol': 1e-6, 'gtol': 1e-5, 'disp': False})
            ## Disp the final values
            print("Initial (positive) log marginal likelihood: %f" % (-nlm0))
            print("Final (positive) log marginal likelihood: %f" % (-res['fun']))
            for j in range(0, len(self.hyperparams)):
                print("Found hypers: %f" % (res['x'][j]))
            #self.hyperparams = res['x'][:]

            results.append((-res['fun'], list(res['x'][:])))

            #gc.collect()

            # print(results)

        # def visualise:

        results_dict = dict(results)

        results = results_dict[max(results_dict)]

        return results, max(results_dict)



def opt_csv_write(context, best_hypers, best_p_hypers, best_coefs, best_H_hypers):

    with open('opt_results_{}_ARD.csv'.format(context), mode='w', encoding="ISO-8859-1", newline='') as res:
        wr = csv.writer(res)
        wr.writerow(("best_hypers", "best_p_hypers", "best_coefs", "best_H_hypers"))

        wr.writerow((best_hypers, best_p_hypers, best_coefs, best_H_hypers))
        res.close()

def csv_write(context, results): # can refactor this
    np.set_printoptions(threshold=sys.maxsize)
    #print(results)
    gp_noARD_ideal_pred = chain_res(results[:,0])
    gp_ideal_pred = chain_res(results[:,1])
    gp_test_pred = chain_res(results[:,2])
    gp_noARD_test_pred = chain_res(results[:,3])
    gp_train_pred = chain_res(results[:,4])
    indiv_true_train = chain_res(results[:,5])
    pgp_test_pred = chain_res(results[:,6])
    pgp_train_pred = chain_res(results[:,7])
    pmgp_test_pred = chain_res(results[:,8])
    pmgp_train_pred = chain_res(results[:,9])
    hgp_test_pred = chain_res(results[:,10])
    hgp_train_pred = chain_res(results[:,11])
    bgp_test_pred = chain_res(results[:,12])
    bgp_train_pred = chain_res(results[:,13])
    E_train = chain_res(results[:,14])
    true_test = chain_res(results[:,15])
    y_train_t = chain_res(results[:,16])
    bgp_ideal_pred = chain_res(results[:,17])
    pgp_ideal_pred = chain_res(results[:,18])
    pmgp_ideal_pred = chain_res(results[:,19])
    hgp_ideal_pred = chain_res(results[:,20])
    gp_loglik = chain_res(results[:,21])
    bgp_loglik = chain_res(results[:,22])
    hgp_loglik = chain_res(results[:,23])
    pmgp_loglik = chain_res(results[:,24])
    gp_noARD_loglik = chain_res(results[:,25])
    pgp_loglik = chain_res(results[:,26])
    gp_hyps = chain_res(results[:,27])
    gp_noARD_hypers = chain_res(results[:,28])
    ideal_gp_hyps_l = chain_res(results[:,29])
    rows = [gp_noARD_ideal_pred, gp_ideal_pred, gp_test_pred, gp_noARD_test_pred, gp_train_pred,
            indiv_true_train,
            pgp_test_pred,
            pgp_train_pred,
            pmgp_test_pred, pmgp_train_pred, hgp_test_pred, hgp_train_pred, bgp_test_pred,
            bgp_train_pred,
            E_train, true_test, y_train_t, bgp_ideal_pred, pgp_ideal_pred, pmgp_ideal_pred,
            hgp_ideal_pred,
            gp_loglik, bgp_loglik, hgp_loglik, pmgp_loglik, gp_noARD_loglik, pgp_loglik, gp_hyps,
            gp_noARD_hypers,
            ideal_gp_hyps_l]
    export = zip_longest(*rows, fillvalue='')
    with open('training_{}_ARD.csv'.format(context), mode='w', encoding="ISO-8859-1", newline='') as res:
        wr = csv.writer(res)
        wr.writerow(("gp_noARD_ideal_pred", "gp_ideal_pred", "gp_test_pred", "gp_noARD_test_pred", "gp_train_pred",
                     "indiv_true_train",
                     "pgp_test_pred",
                     "pgp_train_pred",
                     "pmgp_test_pred", "pmgp_train_pred", "hgp_test_pred", "hgp_train_pred", "bgp_test_pred",
                     "bgp_train_pred",
                     "E_train", "true_test", "y_train_t", "bgp_ideal_pred", "pgp_ideal_pred", "pmgp_ideal_pred",
                     "hgp_ideal_pred",
                     "gp_loglik", "bgp_loglik", "hgp_loglik", "pmgp_loglik", "gp_noARD_loglik", "pgp_loglik", "gp_hyps",
                     "gp_noARD_hypers",
                     "ideal_gp_hyps_l"))
        wr.writerows(export)
        res.close()

# Testing methods #
# Cold start

def cold_start(fold_N):
    rec = fold_N
    # incremental Y, associations included
    X_indiv = Xpooled[recLoc[rec]:recLoc[rec + 1], :]  # select C testing x
    Y_indiv = Ypooled[recLoc[rec]:recLoc[rec + 1], :]  # select C testing t

    X_train = np.delete(Xpooled, np.arange(recLoc[rec], recLoc[rec + 1], 1), 0)  # select N-C training x
    Y_train = np.delete(Ypooled, np.arange(recLoc[rec], recLoc[rec + 1], 1), 0)  # select N-C training y

    order = 2
    pmgp = PMGP(order, X_train, Y_train, best_hypers)
    best_coefs, pmgp_marg = pmgp.optimise()
    best_coefs = np.array(best_coefs)
    pmgp_coefs = np.expand_dims(best_coefs, axis=0)

    true_test = []
    gp_test_pred = []  # allocating memory for cold start
    bgp_test_pred = []  # allocating memory for cold start
    pmgp_test_pred = []
    gp_noARD_test_pred = []  # allocating memory for cold start
    hgp_test_pred = []  # allocating memory for cold start
    E_pred = []
    session_ind = []

    gp_loglik = []  # allocating memory for cold start
    bgp_loglik = []  # allocating memory for cold start
    pmgp_loglik = []
    gp_noARD_loglik = []  # allocating memory for cold start
    hgp_loglik = []  # allocating memory for cold start

    gp_hypers = []
    gp_noARD_hypers = []

    for i in range(1, len(Y_indiv)):
        # delete all data from pooled by the individual comparisons
        X_train = np.delete(Xpooled, np.arange(recLoc[rec] + i, recLoc[rec + 1], 1), 0)  # select N-C training x
        Y_train = np.delete(Ypooled, np.arange(recLoc[rec] + i, recLoc[rec + 1], 1), 0)  # select N-C training y

        # print(X_train, Y_train)
        # at first delete whole rec-1, then keep adding one
        # print(recLoc[rec]+i, recLoc[rec+1])

        pmgp = PMGP(order, X_train, Y_train, best_hypers)
        pmgp_m = pmgp.train(best_coefs)

        hgp = HGP(X_train, Y_train, best_hypers)
        hgp_m = hgp.train(best_H_hypers)

        X_train_indiv = X_indiv[:i]
        # print(X_train_indiv)
        Y_train_indiv = Y_indiv[:i]

        bgp = BGP(X_train_indiv, Y_train_indiv)
        bgp_m = bgp.train(best_hypers)

        X_test = np.expand_dims(X_indiv[i], axis=0)
        Y_test = np.expand_dims(Y_indiv[i], axis=0)

        # print(X_test, Y_test)

        gp = GP(X_train_indiv, Y_train_indiv, ard_b=True)
        best_gp_hyps, disc = gp.optimise()
        gp_m = gp.train(best_gp_hyps)
        gp_hypers.append(best_gp_hyps)

        gp_noARD = GP(X_train_indiv, Y_train_indiv, ard_b=False)
        best_gp_noARD_hyps, disc = gp_noARD.optimise()
        gp_noARD_m = gp_noARD.train(best_gp_noARD_hyps)
        gp_noARD_hypers.append(best_gp_noARD_hyps)

        # To do: append predictions
        # gp_test_pred.append(mu)
        # mu, var = gp_m.predict(X_test)
        # gp_loglik.append(norm.pdf(Y_test, mu, var))

        true_test.append(Y_test)

        mu, var = gp_m.predict(X_test)
        for i in range(0, len(Y_test)):
            gp_loglik.append(np.log(norm.pdf(Y_test[i], mu[i], np.sqrt(var[i]))))
        gp_test_pred.append(mu.flatten())

        mu, var = bgp_m.predict(X_test)
        for i in range(0, len(Y_test)):
            bgp_loglik.append(np.log(norm.pdf(Y_test[i], mu[i], np.sqrt(var[i]))))
        bgp_test_pred.append(mu.flatten())

        mu, var = hgp_m.predict(X_test)
        for i in range(0, len(Y_test)):
            hgp_loglik.append(np.log(norm.pdf(Y_test[i], mu[i], np.sqrt(var[i]))))
        hgp_test_pred.append(mu.flatten())

        mu, var = gp_noARD_m.predict(X_test)
        for i in range(0, len(Y_test)):
            gp_noARD_loglik.append(np.log(norm.pdf(Y_test[i], mu[i], np.sqrt(var[i]))))
        gp_noARD_test_pred.append(mu.flatten())

        mu, var = pmgp_m.predict(X_test)
        for i in range(0, len(Y_test)):
            pmgp_loglik.append(np.log(norm.pdf(Y_test[i], mu[i], np.sqrt(var[i]))))
        pmgp_test_pred.append(mu.flatten())

        # pred E(Y_f)
        E_pred.append(np.mean(Y_train_indiv))

        session_ind.append(fold_N)

    return (gp_test_pred, gp_noARD_test_pred, pmgp_test_pred, hgp_test_pred, bgp_test_pred, true_test, session_ind, gp_hypers,
    gp_noARD_hypers, E_pred, gp_loglik, bgp_loglik, pmgp_loglik, gp_noARD_loglik, hgp_loglik)

def LOO_Pair_CV(fold_N):  # if parallel, remove the for loop
    bgp = BGP(Xpooled, Ypooled)
    bgp_ideal_m = bgp.train(best_hypers)
    pmgp = PMGP(order, Xpooled, Ypooled, best_hypers)
    pmgp_ideal_m = pmgp.train(best_coefs)
    pgp = GP(Xpooled, Ypooled)
    pgp_ideal_m = pgp.train(best_p_hypers)
    hgp = HGP(Xpooled, Ypooled, best_hypers)
    hgp_ideal_m = hgp.train(best_H_hypers)

    bgp_ideal_pred = []
    pgp_ideal_pred = []
    pmgp_ideal_pred = []
    hgp_ideal_pred = []
    gp_ideal_pred = []
    gp_noARD_ideal_pred = []
    ideal_gp_hyps_l = []  # empty list for gp hyper analysis
    gp_hyps = []  # found gp hyps
    gp_noARD_hypers = []
    gp_noARD_test_pred = []

    # train/test
    gp_test_pred = []
    gp_train_pred = []
    bgp_test_pred = []
    bgp_train_pred = []
    pgp_test_pred = []
    pgp_train_pred = []
    pmgp_test_pred = []
    pmgp_train_pred = []
    hgp_test_pred = []
    hgp_train_pred = []


    true_test = []  # true y
    E_train = []  # true x
    indiv_true_train = []  # for indiv GP
    y_train_t = []  # seperate true y for training errors (big list)

    pairwiseCV.pCV.CV5(X, Y, fold_N, 'xindex_loo', Y)

    c_ind = []
    for count, y in enumerate(pairwiseCV.pCV.orgYLeaveOutidx):  # for each comparison to leave out
        # print(np.all(results.orgY == np.any(results.orgYLeaveOutidx), 1))
        ind = np.where(np.all(pairwiseCV.pCV.orgY == y, 1))[0]  # indices in the pooled data where to leave out
        for c, indi in enumerate(ind):  # to account for the same comparison existing in the same session...
            c_ind.append(int(indi))

    X_train = np.delete(Xpooled, c_ind, 0)  # select N-C training x
    Y_train = np.delete(Ypooled, c_ind, 0)  # select N-C training y

    bgp = BGP(X_train, Y_train)  # initialise the models
    bgp_m = bgp.train(best_hypers)

    pmgp = PMGP(order, X_train, Y_train, best_hypers)
    pmgp_m = pmgp.train(best_coefs)

    pgp = GP(X_train, Y_train)
    pgp_m = pgp.train(best_p_hypers)

    hgp = HGP(X_train, Y_train, best_hypers)
    hgp_m = hgp.train(best_H_hypers)

    X_fold = Xpooled[c_ind, :]  # select C testing x
    Y_fold = Ypooled[c_ind]  # select C testing t

    print(fold_N)
    #print('test quant', len(X_fold))

    if fold_N < 8:  # collect training errors for other 'pooled' models
        y_train_t.append(Y_train.flatten())

        mu_t, var_t = pgp_m.predict(X_train)
        pgp_train_pred.append(mu_t.flatten())

        mu_t, var_t = hgp_m.predict(X_train)
        hgp_train_pred.append(mu_t.flatten())

        mu_t, var_t = pmgp_m.predict(X_train)
        pmgp_train_pred.append(mu_t.flatten())

        mu_t, var_t = bgp_m.predict(X_train)
        bgp_train_pred.append(mu_t.flatten())

    # print(X_fold[j, :], Y_fold[j])
    # true_test.append(Y_fold)

    # print('j', j))
    Y_test = np.array(Y_fold)
    # print(Y_test)
    true_test.append(Y_test.flatten())  # true test needs to be the same shape as test_pred
    X_test = np.array(X_fold)
    # print(X_test)

    bgp_loglik = np.empty(len(Y_test))
    hgp_loglik = np.empty(len(Y_test))
    pmgp_loglik = np.empty(len(Y_test))
    pgp_loglik = np.empty(len(Y_test))

    gp_loglik = []
    gp_noARD_loglik = []


    mu, var = bgp_m.predict(X_test)
    # could set really small values to 0 to save memory i.e ~if (mu < eps): mu = 0
    for i in range(0, len(Y_test)):
        bgp_loglik[i] = np.log(norm.pdf(Y_test[i], mu[i], np.sqrt(var[i])))
    bgp_test_pred.append(mu.flatten())
    # print('bgp', var)
    mu, var = bgp_ideal_m.predict(X_test)  # baseline
    bgp_ideal_pred.append(mu.flatten())

    mu, var = pgp_m.predict(X_test)
    for i in range(0, len(Y_test)):
        pgp_loglik[i] = np.log(norm.pdf(Y_test[i], mu[i], np.sqrt(var[i])))
    pgp_test_pred.append(mu.flatten())
    # print('pgp', var)
    mu, var = pgp_ideal_m.predict(X_test)  # baseline
    pgp_ideal_pred.append(mu.flatten())

    mu, var = pmgp_m.predict(X_test)
    for i in range(0, len(Y_test)):
        pmgp_loglik[i] = np.log(norm.pdf(Y_test[i], mu[i], np.sqrt(var[i])))
    pmgp_test_pred.append(mu.flatten())
    # print('pmgp', var)
    mu, var = pmgp_ideal_m.predict(X_test)  # baseline
    pmgp_ideal_pred.append(mu.flatten())

    mu, var = hgp_m.predict(X_test)
    for i in range(0, len(Y_test)):
        hgp_loglik[i] = np.log(norm.pdf(Y_test[i], mu[i], np.sqrt(var[i])))
    hgp_test_pred.append(mu.flatten())
    # print('hgp', var)
    mu, var = hgp_ideal_m.predict(X_test)  # baseline
    hgp_ideal_pred.append(mu.flatten())


    for j, u in enumerate(np.unique(X_fold[:, 6])):  # accounting for settings in multiple sessions for the IGP

        X_indiv = Xpooled[np.where(Xpooled[:, 6] == u)]
        Y_indiv = Ypooled[np.where(Xpooled[:, 6] == u)]
        X_train_indiv = X_train[np.where(X_train[:, 6] == u)]  # get the relevant session for training
        Y_train_indiv = Y_train[np.where(X_train[:, 6] == u)]
        # print(X_fold)
        X_test_indiv = X_fold[np.where(X_fold[:, 6] == u)]
        Y_test_indiv = Y_fold[np.where(X_fold[:, 6] == u)]

        # print(Y_test_indiv)

        gp_ideal = GP(X_indiv, Y_indiv)
        ideal_gp_hyps, disc = gp_ideal.optimise()
        gp_ideal_m = gp_ideal.train(ideal_gp_hyps)
        mu, var = gp_ideal_m.predict(X_test_indiv)
        gp_ideal_pred.append(mu.flatten())

        gp_noARD_ideal = GP(X_indiv, Y_indiv, ard_b=False)
        ideal_gp_noARD_hyps, disc = gp_ideal.optimise()
        gp_noARD_ideal_m = gp_noARD_ideal.train(ideal_gp_noARD_hyps)
        mu, var = gp_noARD_ideal_m.predict(X_test_indiv)
        gp_noARD_ideal_pred.append(mu.flatten())

        if len(X_train_indiv) != 0:  # if there is training data

            gp = GP(X_train_indiv, Y_train_indiv, ard_b=True)
            best_gp_hyps, disc = gp.optimise()
            gp_m = gp.train(best_gp_hyps)

            gp_noARD = GP(X_train_indiv, Y_train_indiv, ard_b=False)
            best_gp_noARD_hyps, disc = gp_noARD.optimise()
            gp_noARD_m = gp_noARD.train(best_gp_noARD_hyps)

            E_t_a = []
            for e in range(len(Y_test_indiv)):
                E_t_a.append(np.mean(Y_train_indiv))  # more than one prediction, needs another pred for E(y_train)
            gp_hyps.append(best_gp_hyps)
            gp_noARD_hypers.append(best_gp_noARD_hyps)
            ideal_gp_hyps_l.append(ideal_gp_hyps)
            E_t_a = np.array(E_t_a)
            E_train.append(E_t_a.flatten())

            mu, var = gp_m.predict(X_test_indiv)
            for i in range(0, len(Y_test_indiv)):
                gp_loglik.append(np.log(norm.pdf(Y_test_indiv[i], mu[i], np.sqrt(var[i]))))
            gp_test_pred.append(mu.flatten())

            mu, var = gp_noARD_m.predict(X_test_indiv)
            for i in range(0, len(Y_test_indiv)):
                gp_noARD_loglik.append(np.log(norm.pdf(Y_test_indiv[i], mu[i], np.sqrt(var[i]))))
            gp_noARD_test_pred.append(mu.flatten())


            if fold_N < 700:  # collect training errors for indiv GP
                mu_t, var_t = gp_m.predict(X_train_indiv)
                gp_train_pred.append(mu_t.flatten())
                indiv_true_train.append(Y_train_indiv.flatten())

        else:
            print('else')
            E_train.append(np.zeros(len(Y_test_indiv)))
            gp_test_pred.append(np.zeros(len(Y_test_indiv)))
            gp_loglik.append(np.zeros(len(Y_test_indiv)))
            gp_noARD_loglik.append(np.zeros(len(Y_test_indiv)))


    # assert (len(gp_test_pred) == len(bgp_test_pred) == len(pgp_test_pred) == len(pmgp_test_pred) == len(
    # hgp_test_pred) == len(true_test))

    return(gp_noARD_ideal_pred, gp_ideal_pred, gp_test_pred, gp_noARD_test_pred, gp_train_pred, indiv_true_train,
                pgp_test_pred,
                pgp_train_pred,
                pmgp_test_pred, pmgp_train_pred, hgp_test_pred, hgp_train_pred, bgp_test_pred,
                bgp_train_pred,
                E_train, true_test, y_train_t, bgp_ideal_pred, pgp_ideal_pred, pmgp_ideal_pred,
                hgp_ideal_pred,
                gp_loglik, bgp_loglik, hgp_loglik, pmgp_loglik, gp_noARD_loglik, pgp_loglik, gp_hyps,
                gp_noARD_hypers,
                ideal_gp_hyps_l)

def chain_res(res):
    l = list(chain(res))
    l = list(chain(*l))
    #l = list(chain(*l))
    return np.array(l)

def log_result(result):
    results.append(result)


# Do the predictive performance tests #

# Main code #
# Data preparation #
context = 'largeHall'
df = initialise_new(context)


if len(df) > 100:
    num_recs = 100
else:
    num_recs = len(df)

bestPooled, Xpooled, Ypooled, recLoc = pool_data_new(num_recs, df)
X, Y = sort_data(Xpooled, Ypooled)

# Optimise
bgp = BGP(Xpooled, Ypooled)
best_hypers = bgp.optimise()

pgp = GP(Xpooled, Ypooled, prin=True)
best_p_hypers = pgp.optimise()

order = 2
pmgp = PMGP(order, Xpooled, Ypooled, best_hypers)
best_coefs = pmgp.optimise()

hgp = HGP(Xpooled, Ypooled, best_hypers)
best_H_hypers = hgp.optimise()

#opt_csv_write(context, (best_hypers, bgp_marg), (best_p_hypers, pgp_marg), (best_coefs, pmgp_marg),
#              (best_H_hypers, hgp_marg))


if __name__ == "__main__":
    results = []
    pool = mp.Pool(3)
    for i in range(8):
        pool.apply_async(LOO_Pair_CV, args=(i,), callback=log_result)
    pool.close()
    pool.join()
    results = np.array(results)
    csv_write(context, results)
