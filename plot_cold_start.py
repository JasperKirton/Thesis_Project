import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix

def opt_load_n_summmary(context):
    df = pd.read_csv(r'cold_start_results/results_{}.csv'.format(context))
    ideal_opt_df = df['gp_hypers'].dropna()

    ideal_opt_df = ideal_opt_df.apply(lambda x:
                               np.fromstring(
                                   x.replace('\n', '')
                                       .replace('[', '')
                                       .replace(']', '')
                                       .replace('  ', ' '), sep=' '))

    ideal_gp_opt = np.stack(ideal_opt_df, axis=0)

    gp_test_params = df['gp_noARD_hypers'].dropna()

    test_opt_df = gp_test_params.apply(lambda x:
                               np.fromstring(
                                   x.replace('\n', '')
                                       .replace('[', '')
                                       .replace(']', '')
                                       .replace('  ', ' '), sep=' '))

    test_gp_opt = np.stack(test_opt_df, axis=0)

    return(ideal_gp_opt, test_gp_opt)

def load_n_summary(context):
    df_l = pd.read_csv(r'cold_start_results/results_{}.csv'.format(context))


    names = ["gp_test_pred", "gp_noARD_test_pred", "pmgp_test_pred", "hgp_test_pred", "bgp_test_pred", "true_test",
                     "E_pred", "session_ind",  "gp_loglik", "bgp_loglik", "pmgp_loglik",
                     "gp_noARD_loglik", "hgp_loglik", "gp_hypers", "gp_noARD_hypers"]  # when we have PMGP



    df = pd.DataFrame()

    for i in range(0, 15):
        test = df_l.iloc[:, i].dropna().astype(str).apply(lambda x: np.fromstring(
                                                        x.replace('\n', '')
                                                        .replace('[', '')
                                                        .replace('"', '')
                                                        .replace(']', '')
                                                        .replace('  ', ' '), sep=' '))
        test = test.to_numpy()
        test = pd.Series(np.concatenate(test).ravel())
        df[names[i]] = test

    gp_test_class = pd.Series([1 if x > 0 else 0 if x < 0 else random.randint(0, 1) for x in df['gp_test_pred'].dropna()])
    bgp_test_class =  pd.Series([1 if x > 0 else 0 if x < 0 else random.randint(0, 1) for x in df['bgp_test_pred'].dropna()])
    pmgp_test_class =  pd.Series([1 if x > 0 else 0 if x < 0 else random.randint(0, 1) for x in df['pmgp_test_pred'].dropna()])
    hgp_test_class =  pd.Series([1 if x > 0 else 0 if x < 0 else random.randint(0, 1) for x in df['hgp_test_pred'].dropna()])
    y_true_test_class =  pd.Series([1 if x > 0 else 0 if x < 0 else random.randint(0, 1) for x in df['true_test'].dropna()])

    bgp_e = (df['bgp_test_pred'] - df['true_test'])
    pmgp_e = (df['pmgp_test_pred'] - df['true_test'])
    hgp_e = (df['hgp_test_pred'] - df['true_test'])
    gp_e = (df['gp_test_pred'] - df['true_test'])
    gp_noARD_e = (df['gp_noARD_test_pred'] - df['true_test'])

    gp_ae = (abs(df['gp_test_pred'] - df['true_test']))
    bgp_ae = (abs(df['bgp_test_pred'] - df['true_test']))
    gp_noARD_ae = (abs(df['gp_noARD_test_pred'] - df['true_test']))
    pmgp_ae = (abs(df['pmgp_test_pred'] - df['true_test']))
    hgp_ae = (abs(df['hgp_test_pred'] - df['true_test']))

    '''
    gp_pred_loglik = np.sum(df['gp_loglik'])  # underflow errors
    bgp_pred_loglik = np.sum(df['bgp_loglik'])
    hgp_pred_loglik = np.sum(df['hgp_loglik'])
    pmgp_pred_loglik = np.sum(df['pmgp_loglik'])

    print('gp_pred_loglik', gp_pred_loglik)
    print('bgp_pred_loglik', bgp_pred_loglik)
    print('hgp_pred_loglik', hgp_pred_loglik)
    print('pmgp_pred_loglik', pmgp_pred_loglik)
    '''

    # Predict E(Y_f)
    #ae_from_E = (abs(df['E_pred'] - df['true_test']), df['session_ind'])

    # Predict 0
    ae_from_0 = (abs(0 - df['true_test']))
    y_len = len(df['true_test'].dropna())

    '''
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

    '''

    test_df = pd.concat([gp_noARD_ae, bgp_ae, hgp_ae, pmgp_ae, gp_ae, df['session_ind']], axis=1)
    test_df.columns = ['Iso GP', 'BGP', 'HGP', 'PMGP', 'ARD GP', 's_id']

    stest_df = pd.concat([gp_noARD_e, bgp_e, hgp_e, pmgp_e, gp_e, df['session_ind']], axis=1)
    stest_df.columns = ['Iso GP', 'BGP', 'HGP', 'PMGP', 'ARD GP', 's_id']

    '''
    #stest_df['GP'].hist(bins=20)
    #stest_df['BGP'].hist(bins=20)
    df['gp_difference'] = stest_df['GP'] - stest_df['BGP']
    print(stats.shapiro(df['gp_difference']))
    #df['bp_difference'].hist(bins=20)
    #print(df['gp_difference'][df['gp_difference'] == 0])
    print(stats.wilcoxon(df['gp_difference']))
    '''

    class_df = pd.concat([gp_test_class, bgp_test_class, hgp_test_class, pmgp_test_class, y_true_test_class, df['session_ind']], axis=1)
    class_df.columns = ['GP', 'BGP', 'HGP', 'PMGP', 'y_true_class', 's_id']

    #df2 = pd.DataFrame(ae, columns = ['bgp', 'pgp', 'pmgp', 'hgp'])
    return(test_df, class_df)#, ae_from_E, ae_from_0, df)


def plot_cold_start(results):
    s_id = results['s_id']
    gp_noARD_ae = results['Iso GP']
    bgp_ae = results['BGP']
    pmgp_ae = results['PMGP']
    hgp_ae = results['HGP']
    gp_ae = results['ARD GP']

    df = pd.melt(results, id_vars=['s_id'], value_vars=['Iso GP', 'BGP', 'HGP', 'PMGP', 'ARD GP'])


    # ae_from_E = [5]
    # ae_from_0 = [6]

    gp_ae_all = []  # prior to mean
    bgp_ae_all = []
    hgp_ae_all = []
    pmgp_ae_all = []
    gp_noARD_ae_all = []

    _, idx = np.unique(s_id, return_index=True)
    sessions = s_id[np.sort(idx)]

    counter = 0
    comps = []
    for i, ses in enumerate(sessions):
        for j in range(len(s_id)):
            if s_id[j] == ses:
                counter += 1
                comps.append(counter)
            else:
                counter = 0
    comps_new = []
    for i in range(5):  # for each model
        comps_new.append(comps)

    comps_new = [item for sublist in comps_new for item in sublist]
    df.insert(0, "Comparisons", comps_new, True)


    for i in range(100):

        ind = list(s_id[s_id == i].index)
        gp_noARD_mae = gp_noARD_ae[ind].to_numpy()
        bgp_mae = bgp_ae[ind].to_numpy()
        pmgp_mae = pmgp_ae[ind].to_numpy()
        hgp_mae = hgp_ae[ind].to_numpy()
        gp_mae = gp_ae[ind].to_numpy()

        for j in range(0, 23 - len(gp_mae)):
            gp_mae = np.append(gp_mae, np.nan)
            bgp_mae = np.append(bgp_mae, np.nan)
            hgp_mae = np.append(hgp_mae, np.nan)
            pmgp_mae = np.append(pmgp_mae, np.nan)
            gp_noARD_mae = np.append(gp_noARD_mae, np.nan)

        gp_ae_all.append(gp_mae)
        bgp_ae_all.append(bgp_mae)
        hgp_ae_all.append(hgp_mae)
        pmgp_ae_all.append(pmgp_mae)
        gp_noARD_ae_all.append(gp_noARD_mae)

    # gp_mae = np.nanmean(np.array(gp_ae_all), axis=0)  # save the MAEs?
    # bgp_mae = np.nanmean(np.array(bgp_ae_all), axis=0)
    # hgp_mae = np.nanmean(np.array(hgp_ae_all), axis=0)
    # pmgp_mae = np.nanmean(np.array(pmgp_ae_all), axis=0)
    # pgp_mae = np.nanmean(np.array(pgp_ae_all), axis=0)

    df.rename({'value': 'Absolute Error', 'variable': 'Model'}, axis=1, inplace=True)
    plt_x = np.arange(0, 23, 1)
    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    plt.get_cmap('Dark2')
    plt.rcParams["axes.labelsize"] = 15
    sns.lineplot(data=df, x="Comparisons", y="Absolute Error", hue="Model", err_style='bars')  # err_band=...
    #sns.boxplot(data=df, x="Comp", y="value", hue="variable")
    plt.legend(loc='lower left')
    plt.show()


    fig, ax = plt.subplots(2, 3, figsize=(10, 10))

    # plot every groups, but discreet
    for j in range(0, len(gp_ae_all)):
        ax[0, 0].plot(plt_x, gp_ae_all[j], marker='', color='grey', linewidth=0.6, alpha=0.3)
        ax[0, 0].set_title('ARD GP')
        ax[0, 0].set_xlabel('')
        ax[0, 1].plot(plt_x, bgp_ae_all[j], marker='', color='grey', linewidth=0.6, alpha=0.3)
        ax[0, 1].set_title('BGP')
        ax[0, 1].set_xlabel('')
        ax[0, 2].plot(plt_x, gp_noARD_ae_all[j], marker='', color='grey', linewidth=0.6, alpha=0.3)
        ax[0, 2].set_title('Iso GP')
        ax[0, 2].set_xlabel('')

        ax[1, 0].plot(plt_x, hgp_ae_all[j], marker='', color='grey', linewidth=0.6, alpha=0.3)
        ax[1, 0].set_title('HGP')
        ax[1, 0].set_xlabel('Comparisons')
        ax[1, 1].plot(plt_x, bgp_ae_all[j], marker='', color='grey', linewidth=0.6, alpha=0.3)
        ax[1, 1].set_title('BGP')
        ax[1, 1].set_xlabel('')
        ax[1, 2].plot(plt_x, pmgp_ae_all[j], marker='', color='grey', linewidth=0.6, alpha=0.3)
        ax[1, 2].set_title('PMGP')
        ax[1, 1].set_xlabel('')

        # to do
        # plt.tight_layout()
        # plt.setp(ax, xlim=[-1, 1])


        plt.setp(ax, xlim=[0, 23], ylim=[0, 2])

    ax[0, 0].set_ylabel('Absolute Error')
    ax[1, 0].set_ylabel('Absolute Error')

    #plt.rcParams.update({'ytick.size': 8})

    for i in range(1, 3):
        plt.setp(ax[0, i].get_yticklabels(), visible=False)
        plt.setp(ax[1, i].get_yticklabels(), visible=False)

    for i in range(0, 3):
        ax[1, i].set_xlabel('Comparisons')
        plt.setp(ax[0, i].get_xticklabels(), visible=False)

    plt.rcParams.update({'font.size': 18})
    plt.show()
    # Plot the lineplot
    # plt.plot(plt_x, df[column], marker='', color=palette(num), linewidth=2.4, alpha=0.9, label=column)

    # Same limits for everybody

    # Not ticks everywhere
    # if num in range(7):
    # plt.tick_params(labelbottom='off')
    # if num not in [1, 4, 7]:
    # plt.tick_params(labelleft='off')

    # Add title
    # plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num))

    gp_mae = np.nanmean(np.array(gp_ae_all), axis=0)  # save the MAEs?
    bgp_mae = np.nanmean(np.array(bgp_ae_all), axis=0)
    hgp_mae = np.nanmean(np.array(hgp_ae_all), axis=0)
    pmgp_mae = np.nanmean(np.array(pmgp_ae_all), axis=0)
    # pgp_mae = np.nanmean(np.array(pgp_ae_all), axis=0)


def opt_histogram(ideal, test):
    import seaborn as sns
    fig, ax = plt.subplots(5, 1, figsize=(10, 10))

    sns.set(style="whitegrid")

    bins = np.linspace(0, 1, 50)
    # Train
    ax[0].set_title('f_var')
    ax[0].hist(ideal[:, 0], bins)

    ax[1].set_title('n_var')
    ax[1].hist(ideal[:, 4], bins)

    bins = np.linspace(0, 30, 50)

    ax[2].set_title('l_s Bass')
    ax[2].hist(ideal[:, 1], bins)

    ax[3].set_title('l_s Mids')
    ax[3].hist(ideal[:, 2], bins)

    ax[4].set_title('l_s Highs')
    ax[4].hist(ideal[:, 3], bins)

    ax[4].set_xlabel('GP Hyper-Parameters')
    #plt.setp(ax[2:5, 0], ylim=[0, 15000])

    '''
    bins = np.linspace(0, 1, 40)
    # Test
    ax[0, 1].set_title('f_var')
    ax[0, 1].hist(test[:, 0], bins)

    ax[1, 1].set_title('n_var')
    ax[1, 1].hist(test[:, 4], bins)

    bins = np.linspace(0, 100, 50)

    ax[2, 1].set_title('l_s Bass')
    ax[2, 1].hist(test[:, 1], bins)

    ax[3, 1].set_title('l_s Mids')
    ax[3, 1].hist(test[:, 2], bins)

    ax[4, 1].set_title('l_s Highs')
    ax[4, 1].hist(test[:, 3], bins)

    ax[4, 1].set_xlabel('GP_noARD Individual Hyper-Parameters')
    '''
    #plt.setp(ax[2:5, 1], ylim=[0, 15000])
    #plt.setp(ax[0:, 0:], ylim=[0, 10000])
    plt.tight_layout()
    plt.show() # sta #sta ''



context = 'classroom_coldstart'
test_df, class_df = load_n_summary(context)
#gp_hyp_df, gp_noARD_hyp_df = opt_load_n_summmary(context)
#opt_histogram(gp_hyp_df, gp_noARD_hyp_df)

plot_cold_start(test_df)
plot_cold_start(opt_dfs)

