# library & dataset
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy import stats
import sys

#test_df = sns.load_dataset('iris')
#print(test_df)

def opt_load_n_summmary(context):
    df = pd.read_csv(r'/Users/jasperkirton/Documents/Glasgow/Widex/uofg-widex/sandbox/jasper/results_new_data/{}.csv'.format(context))
    ideal_opt_df = df['ideal_gp_hyps_l'].dropna()

    ideal_opt_df = ideal_opt_df.apply(lambda x:
                               np.fromstring(
                                   x.replace('\n', '')
                                       .replace('[', '')
                                       .replace(']', '')
                                       .replace('  ', ' '), sep=' '))

    ideal_gp_opt = np.stack(ideal_opt_df, axis=0)

    gp_test_params = df['gp_hyps_2'].dropna()

    test_opt_df = gp_test_params.apply(lambda x:
                               np.fromstring(
                                   x.replace('\n', '')
                                       .replace('[', '')
                                       .replace(']', '')
                                       .replace('  ', ' '), sep=' '))

    test_gp_opt = np.stack(test_opt_df, axis=0)

    return(ideal_gp_opt, test_gp_opt)

def load_n_summary(context):
    df = pd.read_csv(r'results_new_data/{}.csv'.format(context))


    # if results are jumbled

    df_l = pd.read_csv(r'results_new_data/{}.csv'.format(context), header=None)
    names = ["gp_noARD_ideal_pred", "gp_ideal_pred", "gp_test_pred", "gp_noARD_test_pred", "gp_train_pred", "indiv_true_train",
             "pgp_test_pred", "pgp_train_pred",
             "pmgp_test_pred", "pmgp_train_pred", "hgp_test_pred", "hgp_train_pred", "bgp_test_pred",
             "bgp_train_pred",
             "E_train", "true_test", "y_train_t", "bgp_ideal_pred", "pgp_ideal_pred", "pmgp_ideal_pred",
             "hgp_ideal_pred",
             "gp_loglik", "bgp_loglik", "hgp_loglik", "pmgp_loglik", "gp_noARD_loglik", "pgp_loglik", "gp_hyps",
             "gp_noARD_hypers", "ideal_gp_hyps_l", "true_test_2", "gp_loglik_2", "gp_test_pred_2", "gp_ideal_pred_2",
             "gp_train_pred_2", "gp_hyps_2", "indiv_true_train_2", "gp_noARD_test_pred_2", "gp_noARD_hyps_2",  "gp_noARD_loglik_2"] #"gp_noARD_ideal_pred_2"]

    df = pd.DataFrame()


    for i in range(0, 40): #40
        test = df_l.iloc[:, i].dropna().astype(str).apply(lambda x: np.fromstring(
                                                        x.replace('\n', '')
                                                        .replace('[', '')
                                                        .replace('"', '')
                                                        .replace(']', '')
                                                        .replace('  ', ' '), sep=' '))
        test = test.to_numpy()
        test = pd.Series(np.concatenate(test).ravel())
        df[names[i]] = test


    gp_test_class = [1 if x > 0 else 0 if x < 0 else random.randint(0, 1) for x in df['gp_test_pred'].dropna()]
    gp_no_ard_test_class = [1 if x > 0 else 0 if x < 0 else random.randint(0, 1) for x in df['gp_noARD_test_pred'].dropna()]
    gp_no_ard2_test_class = [1 if x > 0 else 0 if x < 0 else random.randint(0, 1) for x in df['gp_noARD_test_pred_2'].dropna()]
    gp_test_class_2 = [1 if x > 0 else 0 if x < 0 else random.randint(0, 1) for x in df['gp_test_pred_2'].dropna()]
    bgp_test_class = [1 if x > 0 else 0 if x < 0 else random.randint(0, 1) for x in df['bgp_test_pred'].dropna()]
    pgp_test_class = [1 if x > 0 else 0 if x < 0 else random.randint(0, 1) for x in df['pgp_test_pred'].dropna()]
    pmgp_test_class = [1 if x > 0 else 0 if x < 0 else random.randint(0, 1) for x in df['pmgp_test_pred'].dropna()]
    hgp_test_class = [1 if x > 0 else 0 if x < 0 else random.randint(0, 1) for x in df['hgp_test_pred'].dropna()]
    y_true_test_class = [1 if x > 0 else 0 if x < 0 else random.randint(0, 1) for x in df['true_test'].dropna()]
    y_true_test_class_2 = [1 if x > 0 else 0 if x < 0 else random.randint(0, 1) for x in df['true_test_2'].dropna()]

    '''
    bins = np.linspace(-20, 20, 200)
    plt.hist(df['gp_noARD_loglik_2'], bins=bins)
    print(df['gp_noARD_loglik_2'])
    plt.hist(df['gp_noARD_loglik'], bins=bins)
    print(df['gp_noARD_loglik'])
    plt.show()
    plt.hist(df['hgp_loglik'], bins=bins)
    print(df['hgp_loglik'])
    plt.show()
    '''

    assert(len(df['gp_noARD_loglik_2']) == len(df['gp_noARD_loglik']) == len(df['gp_loglik_2']) == len(df['gp_loglik'])
           == len(df['hgp_loglik']) == len(df['bgp_loglik']) == len(df['pmgp_loglik']))

    print(len(gp_no_ard_test_class))
    print(len(df['gp_noARD_test_pred']))
    bgp_ideal_e = df['bgp_ideal_pred'] - df['true_test']
    pgp_ideal_e = df['pgp_ideal_pred'] - df['true_test']
    pmgp_ideal_e = df['pmgp_ideal_pred'] - df['true_test']
    hgp_ideal_e = df['hgp_ideal_pred'] - df['true_test']
    gp_ideal_e = df['gp_ideal_pred'] - df['true_test']
    gp_no_ard_ideal_e = df['gp_noARD_ideal_pred'] - df['true_test']
    gp_ideal_e_2 = df['gp_ideal_pred_2'] - df['true_test_2']

    bgp_train_e = (df['bgp_train_pred'] - df['y_train_t'])
    pgp_train_e = (df['pgp_train_pred'] - df['y_train_t'])
    pmgp_train_e = (df['pmgp_train_pred'] - df['y_train_t'])
    hgp_train_e = (df['hgp_train_pred'] - df['y_train_t'])
    gp_train_e = (df['gp_train_pred'] - df['indiv_true_train'])
    gp_train_e_2 = (df['gp_train_pred_2'] - df['indiv_true_train_2'])


    bgp_e = (df['bgp_test_pred'] - df['true_test'])
    pgp_e = (df['pgp_test_pred'] - df['true_test'])
    pmgp_e = (df['pmgp_test_pred'] - df['true_test'])
    hgp_e = (df['hgp_test_pred'] - df['true_test'])
    gp_e = (df['gp_test_pred'] - df['true_test'])
    gp_noard_e = (df['gp_noARD_test_pred'] - df['true_test'])
    gp_noard2_e = (df['gp_noARD_test_pred_2'] - df['true_test_2'])
    gp_e_2 = (df['gp_test_pred_2'] - df['true_test_2'])


    assert(len(gp_noard_e) == len(gp_e))

    gp_no_ard_ae = (abs(df['gp_noARD_test_pred'] - df['true_test']))
    gp_no_ard2_ae = (abs(df['gp_noARD_test_pred_2'] - df['true_test_2']))
    gp_ae = (abs(df['gp_test_pred'] - df['true_test']))
    gp_ae_2 = (abs(df['gp_test_pred_2'] - df['true_test_2']))
    bgp_ae = (abs(df['bgp_test_pred'] - df['true_test']))
    pgp_ae = (abs(df['pgp_test_pred'] - df['true_test']))
    pmgp_ae = (abs(df['pmgp_test_pred'] - df['true_test']))
    hgp_ae = (abs(df['hgp_test_pred'] - df['true_test']))

    #df.loc[df['gp_loglik'] < 3E-300, 'gp_loglik'] = 3E-200
    #[df['gp_loglik'] = 3E-200 if df['gp_loglik'] < 3.0E-300]
    gp_pred_loglik = np.sum(df['gp_loglik'])  # underflow errors?
    gp_pred_loglik_2 = np.sum(df['gp_loglik_2'])
    gp_noARD_loglik = np.sum(df['gp_noARD_loglik'])
    gp_noARD_loglik_2 = np.sum(df['gp_noARD_loglik_2'])
    bgp_pred_loglik = np.sum(df['bgp_loglik'])
    hgp_pred_loglik = np.sum(df['hgp_loglik'])
    pmgp_pred_loglik = np.sum(df['pmgp_loglik'])
    pgp_pred_loglik = np.sum(df['pgp_loglik'])

    # Predict E(Y_f)
    ae_from_E = (abs(df['E_train'] - df['true_test']))

    # Predict 0
    ae_from_0 = (abs(0 - df['true_test']))
    y_len = len(df['true_test'].dropna())


    pred_0_class = np.random.randint(2, size=y_len)
    print('0 mae = {}, + - {}'.format(np.around(np.mean(ae_from_0), decimals=3), np.around(np.std(ae_from_0), decimals=3)))
    print('0 rmse = ', np.around(np.sqrt(np.mean(np.square(ae_from_0))), decimals=3))
    print(confusion_matrix(y_true_test_class, pred_0_class))
    tn, fp, fn, tp = confusion_matrix(y_true_test_class, pred_0_class).ravel()
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    a = (tp+tn)/(tp+fp+fn+tn)
    print('random class precision, recall, F1 = {}, {}, {}'.format(p, r, 2*((p*r)/(p+r))))
    print('random class accuracy = {}'.format(a))

    pred_E_class = [1 if x > 0 else 0 if x < 0 else random.randint(0, 1) for x in df['E_train'].dropna()]
    print('E mae = {}, + - {}'.format(np.around(np.mean(ae_from_E), decimals=3), np.around(np.std(ae_from_E), decimals=3)))
    print('E rmse =', np.around(np.sqrt(np.mean(np.square(ae_from_E))), decimals=3))
    print(confusion_matrix(y_true_test_class, pred_E_class))
    tn, fp, fn, tp = confusion_matrix(y_true_test_class, pred_E_class).ravel()
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    print('E class precision, recall, F1 = {}, {}, {}'.format(p, r, 2*((p*r)/(p+r))))
    print('E class accuracy = ', np.sum(np.equal(y_true_test_class, np.random.randint(1, size=y_len))) / y_len)  # always 0


    print('gp_noARD mae = {}, + - {}'.format(np.around(np.mean(gp_no_ard_ae), decimals=3), np.around(np.std(gp_no_ard_ae), decimals=3)))
    print('gp_noARD rmse =', np.around(np.sqrt(np.mean(np.square(gp_no_ard_ae))), decimals=3))
    print('gp_noARD loglik =', np.around(gp_noARD_loglik, decimals=3))
    print(confusion_matrix(y_true_test_class, gp_no_ard_test_class))
    tn, fp, fn, tp = confusion_matrix(y_true_test_class, gp_no_ard_test_class).ravel()
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    a = (tp+tn)/(tp+fp+fn+tn)
    print('gp_noARD class precision, recall, F1 = {}, {}, {}'.format(p, r, 2*((p*r)/(p+r))))
    print('gp_noARD class accuracy = {}'.format(a))


    print('gp_noARD2 mae = {}, + - {}'.format(np.around(np.mean(gp_no_ard2_ae), decimals=3), np.around(np.std(gp_no_ard2_ae), decimals=3)))
    print('gp_noARD2 rmse =', np.around(np.sqrt(np.mean(np.square(gp_no_ard2_ae))), decimals=3))
    print('gp_noARD2 loglik =', np.around(gp_noARD_loglik_2, decimals=3))
    print(confusion_matrix(y_true_test_class_2, gp_no_ard2_test_class))
    tn, fp, fn, tp = confusion_matrix(y_true_test_class_2, gp_no_ard2_test_class).ravel()
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    a = (tp + tn) / (tp + fp + fn + tn)
    print('gp_noARD2 class precision, recall, F1 = {}, {}, {}'.format(p, r, 2 * ((p * r) / (p + r))))
    print('gp_noARD2 class accuracy = {}'.format(a))


    print('gp mae = {}, + - {}'.format(np.around(np.mean(gp_ae), decimals=3), np.around(np.std(gp_ae), decimals=3)))
    print('gp rmse =', np.around(np.sqrt(np.mean(np.square(gp_ae))), decimals=3))
    print('gp loglik =', np.around(gp_pred_loglik, decimals=3))
    print(confusion_matrix(y_true_test_class, gp_test_class))
    tn, fp, fn, tp = confusion_matrix(y_true_test_class, gp_test_class).ravel()
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    a = (tp+tn)/(tp+fp+fn+tn)
    print('gp class precision, recall, F1 = {}, {}, {}'.format(p, r, 2*((p*r)/(p+r))))
    print('gp class accuracy = {}'.format(a))


    print('gp_2 mae = {}, + - {}'.format(np.around(np.mean(gp_ae_2), decimals=3), np.around(np.std(gp_ae_2), decimals=3)))
    print('gp_2 rmse =', np.around(np.sqrt(np.mean(np.square(gp_ae_2))), decimals=3))
    print('gp_2 loglik =', np.around(gp_pred_loglik_2, decimals=3))
    print(confusion_matrix(y_true_test_class_2, gp_test_class_2))
    tn, fp, fn, tp = confusion_matrix(y_true_test_class_2, gp_test_class_2).ravel()
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    a = (tp+tn)/(tp+fp+fn+tn)
    print('gp_2 class precision, recall, F1 = {}, {}, {}'.format(p, r, 2*((p*r)/(p+r))))
    print('gp_2 class accuracy = {}'.format(a))


    print('bgp mae = {}, + - {}'.format(np.around(np.mean(bgp_ae), decimals=3), np.around(np.std(bgp_ae), decimals=3)))
    print('bgp rmse =', np.around(np.sqrt(np.mean(np.square(bgp_ae))), decimals=3))
    print('bgp loglik =', np.around(bgp_pred_loglik, decimals=3))

    print(confusion_matrix(y_true_test_class, bgp_test_class))
    tn, fp, fn, tp = confusion_matrix(y_true_test_class, bgp_test_class).ravel()
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    a = (tp+tn)/(tp+fp+fn+tn)
    print('bgp class precision, recall, F1 = {}, {}, {}'.format(p, r, 2*((p*r)/(p+r))))
    print('bgp class accuracy = {}'.format(a))


    print('pgp mae = {}, + - {}'.format(np.around(np.mean(pgp_ae), decimals=3), np.around(np.std(pgp_ae), decimals=3)))
    print('pgp rmse =', np.around(np.sqrt(np.mean(np.square(pgp_ae))), decimals=3))
    print('pgp loglik =', np.around(pgp_pred_loglik, decimals=3))
    print(confusion_matrix(y_true_test_class, pgp_test_class))
    tn, fp, fn, tp = confusion_matrix(y_true_test_class, pgp_test_class).ravel()
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    a = (tp+tn)/(tp+fp+fn+tn)
    print('pgp class precision, recall, F1 = {}, {}, {}'.format(p, r, 2*((p*r)/(p+r))))
    print('pgp class accuracy = {}'.format(a))


    print('pmgp mae = {}, + - {}'.format(np.around(np.mean(pmgp_ae), decimals=3), np.around(np.std(pmgp_ae), decimals=3)))
    print('pmgp rmse =', np.around(np.sqrt(np.mean(np.square(pmgp_ae))), decimals=3))
    print('pmgp loglik =', np.around(pmgp_pred_loglik, decimals=3))
    print(confusion_matrix(y_true_test_class, pmgp_test_class))
    tn, fp, fn, tp = confusion_matrix(y_true_test_class, pmgp_test_class).ravel()
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    a = (tp+tn)/(tp+fp+fn+tn)
    print('pmgp class precision, recall, F1 = {}, {}, {}'.format(p, r, 2*((p*r)/(p+r))))
    print('pmgp class accuracy = {}'.format(a))
    #print('pmgp class F1 score = ', y_true_test_class, pmgp_test_class) / y_len)


    print('hgp mae = {}, + - {}'.format(np.around(np.mean(hgp_ae), decimals=3), np.around(np.std(hgp_ae), decimals=3)))
    print('hgp rmse =', np.around(np.sqrt(np.mean(np.square(hgp_ae))), decimals=3))
    print('hgp loglik =', np.around(hgp_pred_loglik, decimals=3))
    print(confusion_matrix(y_true_test_class, hgp_test_class))
    tn, fp, fn, tp = confusion_matrix(y_true_test_class, hgp_test_class).ravel()
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    a = (tp+tn)/(tp+fp+fn+tn)
    print('hgp class precision, recall, F1 = {}, {}, {}'.format(p, r, 2*((p*r)/(p+r))))
    print('hgp class accuracy = {}'.format(a))
    #print('hgp class F1 score = ', np.sum(np.equal(y_true_test_class, hgp_test_class)) / y_len)


    ideal_df = pd.concat([pgp_ideal_e, bgp_ideal_e, pmgp_ideal_e, hgp_ideal_e, gp_ideal_e], axis=1)
    ideal_df.columns = ['pgp', 'bgp', 'pmgp', 'hgp', 'Iso gp']

    train_df = pd.concat([pgp_train_e, bgp_train_e, pmgp_train_e, hgp_train_e, gp_train_e], axis=1) # train_e_2?
    train_df.columns = ['pgp', 'bgp', 'pmgp', 'hgp', 'gp']

    test_df = pd.concat([pgp_e, bgp_e, pmgp_e, hgp_e, gp_noard_e], axis=1)  # gp_e
    test_df.columns = ['PGP', 'BGP', 'PMGP', 'HGP', 'Iso GP']  # 'GP']

    df['gp_difference'] = test_df['HGP'] - test_df['BGP']
    print(stats.shapiro(df['gp_difference']))
    #df['bp_difference'].hist(bins=20)
    print(len(df['gp_difference'][df['gp_difference'] == 0]))
    print(stats.wilcoxon(df['gp_difference']))

    #df2 = pd.DataFrame(ae, columns = ['bgp', 'pgp', 'pmgp', 'hgp'])
    return(ideal_df, train_df, test_df, ae_from_E, ae_from_0, df)


def scatter(df): # To do: sns facet grid
    import seaborn as sns
    sns.set(style="whitegrid")

    # Scatter #

    fig, ax = plt.subplots(2, 5, figsize=(5, 10))

    for i in range(0, 2):
        for j in range(0, 5):
            ax[i, j].plot([-1, 1], [-1, 1], color='grey', lw=0.3, label='ideal fit')

    # flat_ae = np.array(ae).flatten().ravel()
    # print(pgp_train_pred, y_true_train)
    assert(len(df['gp_train_pred']) == len(df['y_train_t']))

    # Training scatters
    ax[0, 0].set_title('PGP Training')
    ax[0, 0].scatter(df['pgp_train_pred'], df['y_train_t'], s=0.1)
    ax[0, 0].set_ylabel('True Y')

    ax[0, 1].set_title('PMGP Training')
    ax[0, 1].scatter(df['pmgp_train_pred'], df['y_train_t'], s=0.1)

    ax[0, 2].set_title('BGP Training')
    ax[0, 2].scatter(df['bgp_train_pred'], df['y_train_t'], s=0.1)

    ax[0, 3].set_title('HGP Training')
    ax[0, 3].scatter(df['hgp_train_pred'], df['y_train_t'], s=0.1)

    ax[0, 4].set_title('GP Training')
    ax[0, 4].scatter(df['gp_train_pred'], df['indiv_true_train'], s=0.1)
    #plt.legend(loc="upper right")

    ax[1, 0].set_xlabel('Predicted Y')

    # Testing scatters
    ax[1, 0].set_title('PGP Testing')
    ax[1, 0].scatter(df['gp_test_pred'], df['true_test'], s=0.1)

    ax[1, 1].set_title('PMGP Testing')
    ax[1, 1].scatter(df['pmgp_test_pred'], df['true_test'], s=0.1)

    ax[1, 2].set_title('BGP Testing')
    ax[1, 2].scatter(df['bgp_test_pred'], df['true_test'], s=0.1)

    ax[1, 3].set_title('HGP Testing')
    ax[1, 3].scatter(df['hgp_test_pred'], df['true_test'], s=0.1)

    ax[1, 4].set_title('GP Testing')
    ax[1, 4].scatter(df['gp_noARD_test_pred_2'], df['true_test_2'], s=0.1)

    for i in range(0, 5):
        plt.setp(ax[0, i].set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0]))
        plt.setp(ax[1, i].set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0]))
        plt.setp(ax[0, i].get_xticklabels(), visible=False)
    for i in range(1, 5):
        plt.setp(ax[0, i].get_yticklabels(), visible=False)
        plt.setp(ax[1, i].get_yticklabels(), visible=False)
        #axes.set_xticklabels([i + 100 for i in x])


    #plt.tight_layout()
    plt.setp(ax, xlim=[-1, 1])

    plt.show()

def opt_histogram(ideal, test):
    import seaborn as sns
    fig, ax = plt.subplots(5, 2, figsize=(10, 10))

    sns.set(style="whitegrid")

    bins = np.linspace(0, 1, 40)
    # Train
    ax[0, 0].set_title('f_var')
    ax[0, 0].hist(ideal[:, 0], bins)

    ax[1, 0].set_title('n_var')
    ax[1, 0].hist(ideal[:, 2], bins)  # [:, 4]

    bins = np.linspace(0, 100, 50)

    ax[2, 0].set_title('l_s Bass')
    ax[2, 0].hist(ideal[:, 1], bins)

    ax[3, 0].set_title('l_s Mids')
    ax[3, 0].hist(ideal[:, 2], bins)

    ax[4, 0].set_title('l_s Highs')
    ax[4, 0].hist(ideal[:, 3], bins)

    ax[4, 0].set_xlabel('Ideal Individual Hyper-Parameters')
    plt.setp(ax[2:5, 0], ylim=[0, 500])

    bins = np.linspace(0, 1, 40)

    # Test
    ax[0, 1].set_title('f_var')
    ax[0, 1].hist(test[:, 0], bins)

    ax[1, 1].set_title('n_var')
    ax[1, 1].hist(test[:, 2], bins) # [:, 4]

    bins = np.linspace(0, 100, 50)

    ax[2, 1].set_title('l_s Bass')
    ax[2, 1].hist(test[:, 1], bins)

    ax[3, 1].set_title('l_s Mids')
    ax[3, 1].hist(test[:, 2], bins)

    ax[4, 1].set_title('l_s Highs')
    ax[4, 1].hist(test[:, 3], bins)

    ax[4, 1].set_xlabel('LOSO-CV Individual Hyper-Parameters')
    #plt.setp(ax[2:5, 1], ylim=[0, 15000])
    plt.setp(ax[0:, 0:], ylim=[0, 500])
    plt.tight_layout()
    plt.show() # sta #sta ''


# start here
def histogram(dfs):
    import seaborn as sns
    fig, ax = plt.subplots(5, 2, figsize=(10, 10))

    sns.set(style="whitegrid")

    #plt.tick_params(axis='both', which='major', labelsize=6)

    bins = np.linspace(-1, 1, 50)
    # Train
    ax[0, 0].set_title('PGP Training Error')
    ax[0, 0].hist(dfs[1]['pgp'], bins)

    ax[1, 0].set_title('PMGP Training')
    ax[1, 0].hist(dfs[1]['pmgp'], bins)

    ax[2, 0].set_title('BGP Training')
    ax[2, 0].hist(dfs[1]['bgp'], bins)

    ax[3, 0].set_title('HGP Training')
    ax[3, 0].hist(dfs[1]['hgp'], bins)

    ax[4, 0].set_title('Iso GP Training')
    ax[4, 0].hist(dfs[1]['gp'], bins)

    #ax[4, 0].set_xlabel('Training Error')
    for i in range(0, 5):
        plt.setp(ax[i, 1].get_yticklabels(), visible=False)
    plt.setp(ax[0:, 0], ylim=[0, 600])
    plt.setp(ax[0:, 0:], xlim=[-1, 1])

    # Test
    ax[0, 1].set_title('PGP Testing Error')
    ax[0, 1].hist(dfs[2]['PGP'], bins)

    ax[1, 1].set_title('PMGP Testing')
    ax[1, 1].hist(dfs[2]['PMGP'], bins)

    ax[2, 1].set_title('BGP Testing')
    ax[2, 1].hist(dfs[2]['BGP'], bins)

    ax[3, 1].set_title('HGP Testing')
    ax[3, 1].hist(dfs[2]['HGP'], bins)

    ax[4, 1].set_title('Iso GP Testing')
    ax[4, 1].hist(dfs[2]['Iso GP'], bins)

    #ax[4, 1].set_xlabel('Testing Error')
    plt.setp(ax[0:, 1], ylim=[0, 600])
    plt.setp(ax[0:, 1], xlim=[-1, 1])

    for i in range(0, 4):
        plt.setp(ax[i, 0].get_xticklabels(), visible=False)
        plt.setp(ax[i, 1].get_xticklabels(), visible=False)

    plt.tight_layout()
    plt.show()

    #g = sns.FacetGrid(dfs[2], col='size', col_wrap=3)
    #g = (g.map(plt.hist), "tip").set_titles("{col_name} error")

''' positive classification analysis precision/recall histograms
def histogram(dfs):
    import seaborn as sns
    fig, ax = plt.subplots(5, 2, figsize=(10, 10))

    sns.set(style="whitegrid")

    bins = np.linspace(-1, 1, 22)
    # Train
    ax[0, 0].set_title('PGP')
    ax[0, 0].hist(dfs[1]['pgp'], bins)

    ax[1, 0].set_title('PMGP')
    ax[1, 0].hist(dfs[1]['pmgp'], bins)

    ax[2, 0].set_title('BGP')
    ax[2, 0].hist(dfs[1]['bgp'], bins)

    ax[3, 0].set_title('HGP')
    ax[3, 0].hist(dfs[1]['hgp'], bins)

    ax[4, 0].set_title('GP')
    ax[4, 0].hist(dfs[1]['gp'], bins)

    ax[4, 0].set_xlabel('training error')

    # Test
    ax[0, 1].set_title('PGP')
    ax[0, 1].hist(dfs[2]['pgp'], bins)

    ax[1, 1].set_title('PMGP')
    ax[1, 1].hist(dfs[2]['pmgp'], bins)

    ax[2, 1].set_title('BGP')
    ax[2, 1].hist(dfs[2]['bgp'], bins)

    ax[3, 1].set_title('HGP')
    ax[3, 1].hist(dfs[2]['hgp'], bins)

    ax[4, 1].set_title('GP')
    ax[4, 1].hist(dfs[2]['gp'], bins)

    ax[4, 1].set_xlabel('testing error')
    plt.setp(ax[0:, 1], ylim=[0, 3000])  # change this adaptively

    plt.tight_layout()
    plt.setp(ax, xlim=[-1, 1])
    plt.show() # sta #sta ''


    #g = sns.FacetGrid(dfs[2], col='size', col_wrap=3)
    #g = (g.map(plt.hist), "tip").set_titles("{col_name} error")
'''

def viol_plot(dfs):

    sns.set(font_scale=1.4)
    sns.set_style("whitegrid")

    plt.plot([-0.5, 4.5], [np.mean(dfs[3]), np.mean(dfs[3])], color="lightgreen", lw=3,  label="MAE from E")  # for ideal
    plt.plot([-0.5, 4.5], [np.mean(dfs[4]), np.mean(dfs[4])], color="green", lw=3, label="MAE from 0")  # for ideal
    plt.legend(['MAE from E(Y)', 'MAE from 0'])

    #plt.plot([-1, 5], [-np.mean(dfs[3]), -np.mean(dfs[3])], color="lightgreen", lw=3,  label="ME from E")  # for ideal
    #plt.plot([-1, 5], [-np.mean(dfs[4]), -np.mean(dfs[4])], color="green", lw=3, label="ME from 0")  # for ideal

    ideal_df = dfs[0]  # .apply(np.abs)
    ideal_MAE = ideal_df.quantile(0.25, axis=0)
    ideal_MAE_2 = ideal_df.quantile(0.75, axis=0)
    print(ideal_MAE)

    # Annotation
    plt.plot([-0.35, 0.35], [ideal_MAE[0], ideal_MAE[0]], color="skyblue", lw=2.75, label="'ideal' model quartiles")
    plt.plot([-0.35, 0.35], [ideal_MAE_2[0], ideal_MAE_2[0]], color="skyblue", lw=2.75)
    for i in range(1, 5):
        plt.plot([-0.35 + i, 0.35 + i], [ideal_MAE[i], ideal_MAE[i]], color="skyblue", lw=2.75,
                 label="_not in legend")  # for ideal
        plt.plot([-0.35 + i, 0.35 + i], [ideal_MAE_2[i], ideal_MAE_2[i]], color="skyblue", lw=2.75,
                 label="_not in legend")  # for ideal
        # plt.plot([-0.4 + i, 0.4 + i], [-ideal_MAE[i], -ideal_MAE[i]], color="skyblue", lw=3,
        #         label="_not in legend")  # for ideal

    ax = sns.violinplot(data=dfs[2], cut=0, bw=.15, inner='quartiles', gridsize=1000)

    plt.setp(ax, ylim=[-2, 2])
    plt.setp(ax.patch, zorder=100)
    #plt.rcParams.update({'font.size': 55})
    ax.set_ylabel('Generalisation Error')

    #ax = sns.boxplot(data=dfs[2].apply(np.abs))

    # Add jitter with the swarmplot function.
    #ax = sns.swarmplot(data=dfs[2].apply(np.abs), color='lightgrey', size = 3)

    ax.legend(loc = "upper left")
    plt.tight_layout()
    plt.show()


context = 'results_restaurant_ARD'
ideal_hyps, test_hyps = opt_load_n_summmary(context)
opt_histogram(ideal_hyps, test_hyps)
dfs = load_n_summary(context)
viol_plot(dfs)
histogram(dfs)
scatter(dfs[5])
