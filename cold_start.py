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
import csv

# Data Preparation #

# Initialise data from the 'context'
def initialise_old(context):
    df = pd.read_json('28_03_19_gt12_dump.json', orient='records')

    df = df[df.Name.str.contains(context)]  # most interesting (or significant) so far are Tv/TV, Traffic, Speech, Outdoor
    df = df[df.BaseProgram.str.contains("Universal")]  # change the marker based on this

    # drop all the duplicates, requires string version of th df
    df["z_tuple"] = df.Comparisons.apply(lambda x: str(x))
    df = df.drop_duplicates(subset="z_tuple", keep="first")
    df.drop("z_tuple", axis=1, inplace=True)  # drop all the du

    settingsInd = df.Comparisons

    compars = settingsInd.apply(pd.DataFrame)  # reshuffle to a logical df
    #print(compars.iloc[1].Settings)


    toDrop = []
    for rec in range(len(compars)):
        if len(compars.iloc[rec].Settings) > 30 or len(compars.iloc[rec].Settings) < 7:  # or < 3, >40
            toDrop.append(rec)
        nonzero_c = np.count_nonzero(np.array(compars.iloc[rec].Value == 0.500), axis=0)
        # print(nonzero_c)
        if nonzero_c / len(compars.iloc[rec].Value) == 1:  # check its not just all 0s
            toDrop.append(rec)
        if np.all(np.array(compars.iloc[rec].Settings) == np.array(compars.iloc[rec-1].Settings)):
            #print(np.array(compars.iloc[rec].Value))
            toDrop.append(rec)

    compars.drop(compars.index[toDrop], inplace=True)  # may be uneccessary
    df.drop(df.index[toDrop], inplace=True)

    # just incase, though we already did this
    compars.drop_duplicates()

    # for i in range(0, len(compars)): maybe another way of deleting duplciates
    #    if np.all(Ypooled[Xpooled[:,  6]==i] == Ypooled[Xpooled[:, 6]==i+1])
    #        np.delete

    print(compars.shape)

    return(compars, df) # need df?

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
def initialise_new(context):

    df = pd.read_json('/export/home/2423923k/testing/06_11_19_gt5_dump.json', orient='records')

    df = df[df.environment.str.contains(context)]
    print(df)

    mat_y = df.y

    #compars = settingsInd.apply(pd.DataFrame)  # reshuffle to a logical df
    #print(mat_m.iloc[1])
    #print(len(settingsInd))

    toDrop = []
    for rec in range(len(mat_y)):
        #print(mat_y.iloc[rec])
        if len(mat_y.iloc[rec]) > 40:
            #toDrop.append(rec)
            print('yes')
        nonzero_c = np.count_nonzero(np.array(mat_y.iloc[rec]) == 0.5)
        #if(nonzero)
        #print(nonzero_c)
        if nonzero_c / len(mat_y.iloc[rec]) == 1:  # check its not just all 0s
            toDrop.append(rec)
    print('todrop', len(toDrop))
    df.drop(df.index[toDrop], inplace=True)

    # for i in range(0, len(compars)): maybe another way of deleting duplicates
    #    if np.all(Ypooled[Xpooled[:,  6]==i] == Ypooled[Xpooled[:, 6]==i+1])
    #        np.delete

    return(df)

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

        #Xind[Xind == [-1]] = 1  # pretty sure Widex said this was a bug
        for k in range(0, len(Xind)):  # for each comparison
            print(Xind[k][1])
            Xtrain = np.vstack((Xtrain, (np.hstack((X[Xind[k][0]], X[Xind[k][1]], i)))))  # [A, B, session]

        recLoc.append(len(Xind) + recLoc[-1])
        Ypooled.append(np.array(vec_y))

        # print(Xtrain)
        Xpooled.append(Xtrain)

    Ypooled = np.concatenate(Ypooled).ravel()  # flatten
    Ypooled = (Ypooled * 2.0) - 1.0  # Set range of Y to -1,1
    #Ypooled = (((Ypooled - np.min(Ypooled)) * 2) / (np.max(Ypooled) - np.min(Ypooled))) - 1
    Ypooled = np.reshape(Ypooled, (-1, 1))
    Xpooled = np.concatenate(Xpooled)
    bestPooled = np.array(bestPooled)

    return (bestPooled, Xpooled, Ypooled, recLoc)

# Pool the data
def pool_data(N, compars, df):
    data = np.array(compars.iloc[0:N])
    Ypooled = []
    Xpooled = []
    bestPooled = []

    recLoc = []  # keep track of where the record starts and finishes
    recLoc.append(0)  # begin
    for i in range(0, N):  # For each record
        Ypooled.append(np.array(data[i].Value))
        bestPooled.append(df.iloc[i].LearnedBestSetting)
        X = []  # unique settings
        Xind = []  # comparisons
        Xtrain = np.empty([0, 7])  # could define relative function or absolute (a-b rather than a,b)
        X = np.array(np.array(df.Data)[i])
        # print(X)
        Xind = np.array(data[i].Settings.tolist())
        Xind[Xind == [-1]] = 1  # pretty sure Widex said this was a bug
        # print(Xind)
        for j in range(0, len(Xind)):  # for each comparison
            Xtrain = np.vstack((Xtrain, (np.hstack((X[Xind[j, 0]], X[Xind[j, 1]], i)))))  # [A, B, session]

        recLoc.append(len(Xind) + recLoc[-1])

        # print(Xtrain)
        Xpooled.append(Xtrain)

    Ypooled = np.concatenate(Ypooled).ravel()  # flatten
    Ypooled = (Ypooled * 2.0) - 1.0  # Set range of Y to -1,1
    Ypooled = np.reshape(Ypooled, (-1, 1))
    Xpooled = np.concatenate(Xpooled)
    bestPooled = np.array(bestPooled)

    return (bestPooled, Xpooled, Ypooled, recLoc)


# Models #

# Batch GP
class BGP(object):
    #__slots__ = ('Xtrain', 'Ytrain', 'kf', 'hyperparams')
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
    #__slots__ = ('p_order', 'Xtrain', 'Ytrain', 'hypers', 'mf', 'A_dims', 'kf', 'var', 'coefs')
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

# GP/Pooled GP
class GP(object):
    #__slots__ = ('Xtrain', 'Ytrain', 'ard_b', 'kf', 'likelihood', 'hypers')
    def __init__(self, Xtrain, Ytrain, ard_b=True, priors=True):
        self.Xtrain = Xtrain[:, 0:6]
        self.Ytrain = Ytrain
        self.ard_b = ard_b
        self.kf = GPy.kern.PjkRbf(input_dim=6, active_dims=[0, 1, 2, 3, 4, 5], variance=0., lengthscale=1., ARD=self.ard_b)	
        self.likelihood = GPy.likelihoods.Gaussian(variance=1)
        if priors:
                self.likelihood.variance.set_prior(GPy.priors.Gamma(1.5, 5.))
                #self.kf.variance.set_prior(GPy.priors.Gaussian(2, 1.5))
                self.kf.lengthscale.set_prior(GPy.priors.Gamma(1.1, 0.05))

    def train(self, hypers):
        self.kf.variance = 0.1
        if self.ard_b:
            self.kf.lengthscale = np.asarray(hypers[0:3]).ravel()
            self.likelihood.variance = hypers[3]
        else:
            self.kf.lengthscale = np.asarray(hypers[0]).ravel()
            self.likelihood.variance = hypers[1]
        m = GPy.core.GP(self.Xtrain, self.Ytrain, kernel=self.kf, likelihood=self.likelihood)
        return m

    def model_wrapper(self, hypers):
        m = self.train(hypers)
        lml = m._log_marginal_likelihood + m.log_prior()
        return -1.0 * lml

    def optimise(self): # could abstract this further
        # tune the coefficients
        if self.ard_b:
            bounds = [(0.01, 100), (0.01, 100), (0.01, 100), (0.01, 1)]  # perhaps hyper-priors here
        else:
            bounds = [(0.01, 100), (0.01, 1)]
        results = []
        for i in range(0, 5):
            if self.ard_b:
                self.hypers = [numpy.random.uniform(0.01, 40, size=1), numpy.random.uniform(0.01, 40, size=1),
                numpy.random.uniform(0.01, 40, size=1), numpy.random.uniform(0.01, 1, 1)]
            else:
                self.hypers = [numpy.random.uniform(0.01, 40, size=1),
                               numpy.random.uniform(0.01, 1, size=1)]

            #print(self.hyperparams)
            nlm0 = self.model_wrapper(self.hypers)  # get initial (negative) log marginal likelihood

            ### Minimize negative log marginal likelihood
            res = optimize.minimize(self.model_wrapper, self.hypers, method='L-BFGS-B',
                                    bounds=bounds, tol=1e-6, options={'ftol': 1e-6, 'gtol': 1e-5, 'disp': False})
            ## Disp the final values

            '''
            print("Initial (positive) log marginal likelihood: %f" % (-nlm0))
            print("Final (positive) log marginal likelihood: %f" % (-res['fun']))
            for i in range(0, len(self.hypers)):
                print("Found hypers: %f" % (res['x'][i]))
            '''

            results.append((-res['fun'], list(res['x'][:])))

            # print(results)

        results_dict = dict(results)

        results = results_dict[max(results_dict)]

        return results, max(results_dict)

# Hierarchical GP (m(x) ~ GP)
class HGP(object):
    #__slots__ = ('Xtrain', 'Ytrain', 'var', 'kern_lower', 'kern_upper', 'hyperparams')
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
        for i in range(0, 5):
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


def csv_write(context, results):
    #print(results)
    gp_test_pred = chain_res(results[:, 0])
    gp_noARD_test_pred = chain_res(results[:, 1])
    pmgp_test_pred = chain_res(results[:, 2])
    hgp_test_pred = chain_res(results[:, 3])
    bgp_test_pred = chain_res(results[:, 4])
    true_test = chain_res(results[:, 5])
    session_ind = chain_res(results[:, 6])
    gp_hypers = chain_res(results[:, 7])
    gp_noARD_hypers = chain_res(results[:, 8])
    E_pred = chain_res(results[:, 9])
    gp_loglik = chain_res(results[:, 10])
    bgp_loglik = chain_res(results[:, 11])
    pmgp_loglik = chain_res(results[:, 12])
    gp_noARD_loglik = chain_res(results[:, 13])
    hgp_loglik = chain_res(results[:, 14])

    rows = [gp_test_pred, gp_noARD_test_pred, pmgp_test_pred, hgp_test_pred, bgp_test_pred, true_test, E_pred, session_ind,
            gp_loglik, bgp_loglik, pmgp_loglik, gp_noARD_loglik, hgp_loglik, gp_hypers, gp_noARD_hypers]
    export = zip_longest(*rows, fillvalue='')
    with open('results_{}_coldstart_fixedfvar.csv'.format(context), mode='w', encoding="ISO-8859-1", newline='') as res:
        wr = csv.writer(res)
        wr.writerow(("gp_test_pred", "gp_noARD_test_pred", "pmgp_test_pred", "hgp_test_pred", "bgp_test_pred", "true_test",
                     "E_pred", "session_ind", "gp_loglik", "bgp_loglik", "pmgp_loglik",
                     "gp_noARD_loglik", "hgp_loglik", "gp_hypers", "gp_noARD_hypers"))
        wr.writerows(export)
        res.close()


def opt_csv_write(context, best_hypers, best_p_hypers, best_coefs, best_H_hypers):
    with open('opt_results_{}_ARD.csv'.format(context), mode='w', encoding="ISO-8859-1", newline='') as res:
        wr = csv.writer(res)
        wr.writerow(("best_hypers", "best_p_hypers", "best_coefs", "best_H_hypers"))
        wr.writerow((best_hypers, best_p_hypers, best_coefs, best_H_hypers))
        res.close()

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
    #pmgp_coefs = np.expand_dims(best_coefs, axis=0)

    true_test = []
    gp_test_pred = []
    bgp_test_pred = []
    pmgp_test_pred = []
    gp_noARD_test_pred = []
    hgp_test_pred = []
    E_pred = []
    session_ind = []

    gp_loglik = []
    bgp_loglik = []
    pmgp_loglik = []	
    gp_noARD_loglik = []
    hgp_loglik = []

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
        Y_train_indiv = Y_indiv[:i]

        bgp = BGP(X_train_indiv, Y_train_indiv)
        bgp_m = bgp.train(best_hypers)

        X_test = np.expand_dims(X_indiv[i], axis=0)
        Y_test = np.expand_dims(Y_indiv[i], axis=0)

        # assert(len(X_test)==len(Y_test))

        gp = GP(X_train_indiv, Y_train_indiv, ard_b=True)
        best_gp_hyps, disc = gp.optimise()
        gp_m = gp.train(best_gp_hyps)
        gp_hypers.append(best_gp_hyps)

        gp_noARD = GP(X_train_indiv, Y_train_indiv, ard_b=False)
        best_gp_noARD_hyps, disc = gp_noARD.optimise()
        gp_noARD_m = gp_noARD.train(best_gp_noARD_hyps)
        gp_noARD_hypers.append(best_gp_noARD_hyps)

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


    return(gp_test_pred, gp_noARD_test_pred, pmgp_test_pred, hgp_test_pred, bgp_test_pred, true_test, session_ind, gp_hypers,
            gp_noARD_hypers, E_pred, gp_loglik, bgp_loglik, pmgp_loglik, gp_noARD_loglik, hgp_loglik)


def chain_res(res):
    l = list(chain(res))
    l = list(chain(*l))
    # l = list(chain(*l))
    return np.array(l)

def log_result(result):
    #print('result', result)
    cs_results.append(result)

# Do the predictive performance tests #

# Main code #
# Data preparation #

context = sys.argv[1]
df = initialise_new(context)

if len(df) > 100:
    num_recs = 100
else:
    num_recs = len(df)

bestPooled, Xpooled, Ypooled, recLoc = pool_data_new(num_recs, df)
X, Y = sort_data(Xpooled, Ypooled)

# Optimise or use previously found hyper-parameters
bgp = BGP(Xpooled, Ypooled)
best_hypers, bgp_marg = ([0.090982, 13.196800, 2.607379, 19.331999, 0.110907], -1590.9622204169473)

# print(best_p_hypers, pgp_marg)

hgp = HGP(Xpooled, Ypooled, best_hypers)
best_H_hypers, hgp_marg = ([0.029186, 27.917921, 23.278268, 29.920158, 0.084336, 13.127482, 2.359934, 20.985823], -1580.87403899717)

order = 2
pmgp = PMGP(order, Xpooled, Ypooled, best_hypers)
#best_coefs, pmgp_marg = ([7.14506211e-03, -1.05143820e-02, -4.69262312e-03,  9.84034127e-06,
#        1.50880253e-04,  1.76142554e-04,  1.37012923e-04,  9.91587883e-05,
#       -3.45803453e-05], -1579.545266774319)

#opt_csv_write(context, (best_hypers, bgp_marg), (best_p_hypers, pgp_marg), (best_coefs, pmgp_marg),
#              (best_H_hypers, hgp_marg))


cs_results = []

# Execute parallel tests
# Parallel opt, on windows we need the line below,
#if __name__ == '__main__':

pool = mp.Pool(31)
for i in range(num_recs):
    pool.apply_async(cold_start, args=(i, ), callback=log_result)
pool.close()
pool.join()
cs_results = np.array(cs_results)
csv_write(context, cs_results)