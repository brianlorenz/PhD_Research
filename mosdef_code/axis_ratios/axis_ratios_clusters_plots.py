#

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from read_data import mosdef_df
from mosdef_obj_data_funcs import read_sed, read_mock_sed, get_mosdef_obj, read_composite_sed, setup_get_AV, get_AV, setup_get_ssfr, merge_ar_ssfr
from filter_response import lines, overview, get_index, get_filter_response
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.integrate as integrate
from query_funcs import get_zobjs
import initialize_mosdef_dirs as imd
import cluster_data_funcs as cdf
from axis_ratio_funcs import read_axis_ratio, read_interp_axis_ratio
from emission_measurements import read_emission_df, get_emission_measurements
from astropy.table import Table


save_dir = imd.cluster_dir + '/axis_ratio_cluster_plots/'


def plot_balmer_dec_AV_const_sfr_mass(n_groups, save_name=''):
    """Plots the balmer dec for each cluster

    Parameters:
    n_groups (int): NUmber of axis ratio groups
    save_name (list): Addition to the folder where the fils are saved
    """
    fit_dfs = [ascii.read(imd.cluster_dir + f'/emission_fitting/axis_ratio_clusters{save_name}/{axis_group}_emission_fits.csv').to_pandas() for axis_group in range(n_groups)]
    ar_dfs = [ascii.read(imd.cluster_dir + f'/composite_spectra/axis_stack{save_name}/{axis_group}_df.csv').to_pandas() for axis_group in range(n_groups)]

    fields, av_dfs = setup_get_AV()

    medians = [np.median(ar_df['use_ratio']) for ar_df in ar_dfs]
    # stds = [np.std(ar_df['use_ratio']) for ar_df in ar_dfs]
    stds = [np.abs(np.median(ar_df['use_ratio']) -
                   np.percentile(ar_df['use_ratio'], [16, 84])) for ar_df in ar_dfs]
    stds_low = [std[0] for std in stds]
    stds_high = [std[1] for std in stds]

    for ar_df in ar_dfs:
        mosdef_objs = [get_mosdef_obj(ar_df.iloc[i]['field'], ar_df.iloc[i][
                                      'v4id']) for i in range(len(ar_df))]
        Avs = [get_AV(fields, av_dfs, mosdef_obj)
               for mosdef_obj in mosdef_objs]
        ar_df['Av'] = Avs

    median_Av = [np.median(ar_df['Av']) for ar_df in ar_dfs]
    stds_Av = [np.std(ar_df['Av']) for ar_df in ar_dfs]

    balmer_decs = []
    balmer_errs_high = []
    balmer_errs_low = []
    colors = []
    for i in range(len(fit_dfs)):
        fit_df = fit_dfs[i]
        balmer_decs.append(fit_df['balmer_dec'].iloc[0])
        balmer_errs_low.append(fit_df['err_balmer_dec_low'].iloc[0])
        balmer_errs_high.append(fit_df['err_balmer_dec_high'].iloc[0])
        colors.append(ar_dfs[i]['plot_color'].iloc[0])
    # balmer_errs = np.transpose(np.array(balmer_errs))

    mass_medians = [np.median(ar_df['LMASS']) for ar_df in ar_dfs]
    ssfr_medians = [np.median(ar_df['L_SSFR']) for ar_df in ar_dfs]
    values_df = pd.DataFrame(zip(medians, stds_low, stds_high, balmer_decs, balmer_errs_low, balmer_errs_high, mass_medians, ssfr_medians, colors, median_Av, stds_Av), columns=[
        'ars', 'scat_ars_low', 'scat_ars_high', 'balmers', 'err_balmers_low', 'err_balmers_high', 'masses', 'ssfrs', 'plot_color', 'Av', 'scat_AV', ])

    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    # Figure for just the galaixes in that cluster
    fig, ax = plt.subplots(figsize=(8, 7))

    label1 = 1
    label2 = 1
    label3 = 1
    label4 = 1
    for i in range(len(values_df)):
        if values_df['masses'].iloc[i] < 9.95 and values_df['ssfrs'].iloc[i] > -8.575:
            color = 'royalblue'
            if label1 == 1:
                label = 'Low Mass, High sSFR'
                label1 = 0
            else:
                label = None
        elif values_df['masses'].iloc[i] < 9.95 and values_df['ssfrs'].iloc[i] < -8.575:
            color = 'black'
            if label2 == 1:
                label = 'Low Mass, Low sSFR'
                label2 = 0
            else:
                label = None
        elif values_df['masses'].iloc[i] > 9.95 and values_df['ssfrs'].iloc[i] > -8.725:
            color = 'firebrick'
            if label3 == 1:
                label = 'High Mass, High sSFR'
                label3 = 0
            else:
                label = None
        else:
            color = 'sandybrown'
            if label4 == 1:
                label = 'Low Mass, Low sSFR'
                label4 = 0
            else:
                label = None
        ax.errorbar(values_df['ars'].iloc[i], values_df['balmers'].iloc[i], xerr=[[values_df['scat_ars_low'].iloc[i]], [values_df['scat_ars_high'].iloc[i]]], yerr=[[values_df['err_balmers_low'].iloc[i]], [values_df['err_balmers_high'].iloc[i]]],
                    color=color, label=label, ls='None', marker='o')  # , ms=len(fit_dfs[i].iloc[i]) / 4)
    ax.legend(fontsize=axisfont)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 8)
    ax.set_xlabel('Axis Ratio', fontsize=axisfont)
    ax.set_ylabel('Balmer Decrement', fontsize=axisfont)
    ax.tick_params(labelsize=ticksize, size=ticks)
    fig.savefig(save_dir + save_name[1:] + f'/AR_Clusters_Balmer_Decs{save_name}.pdf')

    plt.close()

    # NOW DOING AVs

    # Figure for just the galaixes in that cluster
    fig, ax = plt.subplots(figsize=(8, 7))

    label1 = 1
    label2 = 1
    label3 = 1
    for i in range(len(values_df)):
        if values_df['masses'].iloc[i] < 9.95 and values_df['ssfrs'].iloc[i] > -8.575:
            color = 'royalblue'
            if label1 == 1:
                label = 'Low Mass, High sSFR'
                label1 = 0
            else:
                label = None
        elif values_df['masses'].iloc[i] < 9.95 and values_df['ssfrs'].iloc[i] < -8.575:
            color = 'black'
            if label2 == 1:
                label = 'Low Mass, Low sSFR'
                label2 = 0
            else:
                label = None
        elif values_df['masses'].iloc[i] > 9.95 and values_df['ssfrs'].iloc[i] > -8.725:
            color = 'firebrick'
            if label3 == 1:
                label = 'High Mass, High sSFR'
                label3 = 0
            else:
                label = None
        else:
            color = 'sandybrown'
            if label4 == 1:
                label = 'Low Mass, Low sSFR'
                label4 = 0
            else:
                label = None

        ax.plot(ar_dfs[i]['use_ratio'], ar_dfs[i]['Av'], color=color,
                marker='.', markersize=6, zorder=1, ls='None')
        ax.errorbar(values_df['ars'].iloc[i], values_df['Av'].iloc[i], xerr=[[values_df['scat_ars_low'].iloc[i]], [values_df['scat_ars_high'].iloc[i]]], yerr=values_df['scat_AV'].iloc[i],
                    color=color, ls='None', marker='o', label=label)

    ax.legend(fontsize=axisfont)

    ax.set_xlim(-0.05, 1.05)
    # ax.set_ylim(8, 12)
    ax.set_xlabel('Axis Ratio', fontsize=axisfont)
    ax.set_ylabel('FAST A_V', fontsize=axisfont)
    ax.tick_params(labelsize=ticksize, size=ticks)
    fig.savefig(save_dir + save_name[1:] + f'/AR_Clusters_FAST_AV{save_name}.pdf')
    plt.close()


def plot_axis_ratio_clusters_balmer_dec(n_groups, save_name=''):
    """Plots the balmer dec for each cluster

    Parameters:
    n_groups (int): NUmber of axis ratio groups
    save_name (list): Addition to the folder where the fils are saved
    """
    fit_dfs = [ascii.read(imd.cluster_dir + f'/emission_fitting/axis_ratio_clusters{save_name}/{axis_group}_emission_fits.csv').to_pandas() for axis_group in range(n_groups)]
    ar_dfs = [ascii.read(imd.cluster_dir + f'/composite_spectra/axis_stack{save_name}/{axis_group}_df.csv').to_pandas() for axis_group in range(n_groups)]

    medians = [np.median(ar_df['use_ratio']) for ar_df in ar_dfs]
    # stds = [np.std(ar_df['use_ratio']) for ar_df in ar_dfs]
    stds = [np.abs(np.median(ar_df['use_ratio']) -
                   np.percentile(ar_df['use_ratio'], [16, 84])) for ar_df in ar_dfs]
    stds_low = [std[0] for std in stds]
    stds_high = [std[1] for std in stds]

    balmer_decs = []
    balmer_errs_high = []
    balmer_errs_low = []
    for i in range(len(fit_dfs)):
        fit_df = fit_dfs[i]
        balmer_decs.append(fit_df['balmer_dec'].iloc[0])
        balmer_errs_low.append(fit_df['err_balmer_dec_low'].iloc[0])
        balmer_errs_high.append(fit_df['err_balmer_dec_high'].iloc[0])
    # balmer_errs = np.transpose(np.array(balmer_errs))

    if '_mass' in save_name:
        mass_medians = [np.median(ar_df['LMASS']) for ar_df in ar_dfs]
        values_df = pd.DataFrame(zip(medians, stds_low, stds_high, balmer_decs, balmer_errs_low, balmer_errs_high, mass_medians), columns=[
                                 'ars', 'scat_ars_low', 'scat_ars_high', 'balmers', 'err_balmers_low', 'err_balmers_high', 'masses'])
    if '_ssfr' in save_name:
        ssfr_medians = [np.median(ar_df['LSSFR']) for ar_df in ar_dfs]
        values_df = pd.DataFrame(zip(medians, stds_low, stds_high, balmer_decs, balmer_errs_low, balmer_errs_high, ssfr_medians), columns=[
                                 'ars', 'scat_ars_low', 'scat_ars_high', 'balmers', 'err_balmers_low', 'err_balmers_high', 'ssfrs'])

    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    # Figure for just the galaixes in that cluster
    fig, ax = plt.subplots(figsize=(8, 7))

    if '_ssfr' in save_name:
        high_ssfr = values_df['ssfrs'] > -8.9
        ax.errorbar(values_df[high_ssfr]['ars'], values_df[high_ssfr]['balmers'], xerr=(values_df[high_ssfr]['scat_ars_low'], values_df[high_ssfr]['scat_ars_high']), yerr=(values_df[high_ssfr]['err_balmers_low'], values_df[high_ssfr]['err_balmers_high']),
                    color='orange', ls='None', marker='o', label='>-8.9 LSSFR', ms=len(values_df[high_ssfr]['ars']) / 2)
        ax.errorbar(values_df[~high_ssfr]['ars'], values_df[~high_ssfr]['balmers'], xerr=(values_df[~high_ssfr]['scat_ars_low'], values_df[~high_ssfr]['scat_ars_high']), yerr=(values_df[~high_ssfr]['err_balmers_low'], values_df[~high_ssfr]['err_balmers_high']),
                    color='black', ls='None', marker='o', label='<-8.9 LSSFR', ms=len(values_df[~high_ssfr]['ars']) / 2)
        ax.legend(fontsize=axisfont)
    elif '_mass' in save_name:
        high_mass = values_df['masses'] > 10
        ax.errorbar(values_df[high_mass]['ars'], values_df[high_mass]['balmers'], xerr=(values_df[high_mass]['scat_ars_low'], values_df[high_mass]['scat_ars_high']), yerr=(values_df[high_mass]['err_balmers_low'], values_df[high_mass]['err_balmers_high']),
                    color='orange', ls='None', marker='o', label='>10 LMASS', ms=len(values_df[high_mass]['ars']) / 2)
        ax.errorbar(values_df[~high_mass]['ars'], values_df[~high_mass]['balmers'], xerr=(values_df[~high_mass]['scat_ars_low'], values_df[~high_mass]['scat_ars_high']), yerr=(values_df[~high_mass]['err_balmers_low'], values_df[~high_mass]['err_balmers_high']),
                    color='black', ls='None', marker='o', label='<10 LMASS', ms=len(values_df[~high_mass]['ars']) / 2)
        ax.legend(fontsize=axisfont)
    else:
        ax.errorbar(medians, balmer_decs, xerr=np.transpose(stds), yerr=[balmer_errs_low, balmer_errs_high],
                    color='black', ls='None', marker='o', ms=len(medians) / 2)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 8)
    ax.set_xlabel('Axis Ratio', fontsize=axisfont)
    ax.set_ylabel('Balmer Decrement', fontsize=axisfont)
    ax.tick_params(labelsize=ticksize, size=ticks)
    fig.savefig(save_dir + save_name[1:] + f'/AR_Clusters_Balmer_Decs{save_name}.pdf')
    plt.close()


def plot_axis_ratio_clusters_mass(n_groups, save_name=''):
    """Plots the mass distribution for each cluster

    Parameters:
    n_groups (int): NUmber of axis ratio groups
    """
    ar_dfs = [ascii.read(imd.cluster_dir + f'/composite_spectra/axis_stack{save_name}/{axis_group}_df.csv').to_pandas() for axis_group in range(n_groups)]

    # Append each galaxy's mass to its ar info
    for ar_df in ar_dfs:
        l_masses = [get_mosdef_obj(ar_df.iloc[i]['field'], ar_df.iloc[i]['v4id'])[
            'LMASS'] for i in range(len(ar_df))]
        ar_df['LMASS'] = l_masses

    medians = [np.median(ar_df['use_ratio']) for ar_df in ar_dfs]
    stds = [np.std(ar_df['use_ratio']) for ar_df in ar_dfs]
    median_mass = [np.median(ar_df['LMASS']) for ar_df in ar_dfs]
    stds_mass = [np.std(ar_df['LMASS']) for ar_df in ar_dfs]
    if 'ssfr' in save_name:
        median_ssfr = [np.median(ar_df['LSSFR']) for ar_df in ar_dfs]

    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    # Figure for just the galaixes in that cluster
    fig, ax = plt.subplots(figsize=(8, 7))

    if 'mass' in save_name:
        for ar_df in ar_dfs:
            if np.median(ar_df['LMASS']) > 10:
                color = 'orange'
            else:
                color = 'gray'
            [ax.plot(ar_df.iloc[i]['use_ratio'], ar_df.iloc[i]['LMASS'], color=color,
                     marker='.', markersize=6, zorder=1) for i in range(len(ar_df))]
    if 'ssfr' in save_name:
        for ar_df in ar_dfs:
            if np.median(ar_df['LSSFR']) > -8.9:
                color = 'orange'
            else:
                color = 'gray'
            [ax.plot(ar_df.iloc[i]['use_ratio'], ar_df.iloc[i]['LMASS'], color=color,
                     marker='.', markersize=6, zorder=1) for i in range(len(ar_df))]

    if 'ssfr' in save_name:
        values_df = pd.DataFrame(zip(medians, stds, median_mass, stds_mass, median_ssfr), columns=[
                                 'medians', 'stds', 'median_mass', 'stds_mass', 'median_ssfr'])
        high_ssfr = values_df['median_ssfr'] > -8.9
        ax.errorbar(values_df[high_ssfr]['medians'], values_df[high_ssfr]['median_mass'], xerr=values_df[high_ssfr]['stds'], yerr=values_df[high_ssfr]['stds_mass'],
                    color='orangered', ls='None', marker='o', zorder=2)
        ax.errorbar(values_df[~high_ssfr]['medians'], values_df[~high_ssfr]['median_mass'], xerr=values_df[~high_ssfr]['stds'], yerr=values_df[~high_ssfr]['stds_mass'],
                    color='black', ls='None', marker='o', zorder=2)
    else:
        for ar_df in ar_dfs:
            color = 'gray'
            [ax.plot(ar_df.iloc[i]['use_ratio'], ar_df.iloc[i]['LMASS'], color=color,
                     marker='.', markersize=6, zorder=1) for i in range(len(ar_df))]
        ax.errorbar(medians, median_mass, xerr=stds, yerr=stds_mass,
                    color='black', ls='None', marker='o', zorder=2)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(8, 12)
    ax.set_xlabel('Axis Ratio', fontsize=axisfont)
    ax.set_ylabel('log(Stellar Mass)', fontsize=axisfont)
    ax.tick_params(labelsize=ticksize, size=ticks)
    fig.savefig(save_dir + save_name[1:] + f'/AR_Clusters_LMASS{save_name}.pdf')
    plt.close()


def plot_axis_ratio_clusters_ssfr(n_groups, save_name=''):
    """Plots the mass distribution for each cluster

    Parameters:
    n_groups (int): NUmber of axis ratio groups
    """
    ar_dfs = [ascii.read(imd.cluster_dir + f'/composite_spectra/axis_stack{save_name}/{axis_group}_df.csv').to_pandas() for axis_group in range(n_groups)]

    # ssfr_mosdef_merge_no_dups = setup_get_ssfr()
    ssfr_ar_dfs = []
    # Append each galaxy's mass to its ar info
    for ar_df in ar_dfs:
        print(f'Before: {len(ar_df)}')
        # ar_df = merge_ar_ssfr(ar_df, ssfr_mosdef_merge_no_dups)
        ar_df = ar_df[ar_df['SSFR'] > 0]
        ar_df['LSSFR'] = np.log10(ar_df['SSFR'])
        print(f'After: {len(ar_df)}')
        print('\n')
        ssfr_ar_dfs.append(ar_df)

    medians = [np.median(ar_df['use_ratio']) for ar_df in ssfr_ar_dfs]
    stds = [np.std(ar_df['use_ratio']) for ar_df in ssfr_ar_dfs]
    median_ssfr = [np.median(ar_df['LSSFR']) for ar_df in ssfr_ar_dfs]
    stds_ssfr = [np.std(ar_df['LSSFR']) for ar_df in ssfr_ar_dfs]

    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    # Figure for just the galaixes in that cluster
    fig, ax = plt.subplots(figsize=(8, 7))

    for ar_df in ssfr_ar_dfs:
        # if np.median(ar_df['SSFR']) > 10:
        #     color = 'orange'
        # else:
        #     color = 'gray'
        color = 'gray'
        [ax.plot(ar_df.iloc[i]['use_ratio'], ar_df.iloc[i]['LSSFR'], color=color,
                 marker='.', markersize=6, zorder=1) for i in range(len(ar_df))]

    ax.errorbar(medians, median_ssfr, xerr=stds, yerr=stds_ssfr,
                color='black', ls='None', marker='o', zorder=2)

    ax.set_xlim(-0.05, 1.05)
    # ax.set_ylim(8, 12)
    ax.set_xlabel('Axis Ratio', fontsize=axisfont)
    ax.set_ylabel('log(sSFR)', fontsize=axisfont)
    ax.tick_params(labelsize=ticksize, size=ticks)
    fig.savefig(save_dir + save_name[1:] + f'/AR_Clusters_LSSFR{save_name}.pdf')
    plt.close()


def plot_axis_ratio_clusters_AV(n_groups, save_name=''):
    """Plots the Av distribution for each cluster

    Parameters:
    n_groups (int): NUmber of axis ratio groups
    """
    ar_dfs = [ascii.read(imd.cluster_dir + f'/composite_spectra/axis_stack{save_name}/{axis_group}_df.csv').to_pandas() for axis_group in range(n_groups)]

    fields, av_dfs = setup_get_AV()

    # Append each galaxy's mass to its ar info
    for ar_df in ar_dfs:
        mosdef_objs = [get_mosdef_obj(ar_df.iloc[i]['field'], ar_df.iloc[i][
                                      'v4id']) for i in range(len(ar_df))]
        Avs = [get_AV(fields, av_dfs, mosdef_obj)
               for mosdef_obj in mosdef_objs]
        ar_df['Av'] = Avs

    medians = [np.median(ar_df['use_ratio']) for ar_df in ar_dfs]
    stds = [np.std(ar_df['use_ratio']) for ar_df in ar_dfs]
    median_Av = [np.median(ar_df['Av']) for ar_df in ar_dfs]
    stds_Av = [np.std(ar_df['Av']) for ar_df in ar_dfs]

    if '_mass' in save_name:
        mass_medians = [np.median(ar_df['LMASS']) for ar_df in ar_dfs]
        values_df = pd.DataFrame(zip(medians, stds, median_Av, stds_Av, mass_medians), columns=[
                                 'ars', 'scat_ars', 'Av', 'scat_AV', 'masses'])
    if '_ssfr' in save_name:
        ssfr_medians = [np.median(ar_df['LSSFR']) for ar_df in ar_dfs]
        values_df = pd.DataFrame(zip(medians, stds, median_Av, stds_Av, ssfr_medians), columns=[
                                 'ars', 'scat_ars', 'Av', 'scat_AV', 'ssfrs'])

    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    # Figure for just the galaixes in that cluster
    fig, ax = plt.subplots(figsize=(8, 7))

    if '_mass' in save_name:
        for ar_df in ar_dfs:
            for i in range(len(ar_df)):
                if ar_df.iloc[i]['LMASS'] > 10:
                    color = 'skyblue'
                else:
                    color = 'grey'
                ax.plot(ar_df.iloc[i]['use_ratio'], ar_df.iloc[i]['Av'], color=color,
                        marker='.', markersize=6, zorder=1)
    if '_ssfr' in save_name:
        for ar_df in ar_dfs:
            for i in range(len(ar_df)):
                if ar_df.iloc[i]['LSSFR'] > -8.9:
                    color = 'skyblue'
                else:
                    color = 'grey'
                ax.plot(ar_df.iloc[i]['use_ratio'], ar_df.iloc[i]['Av'], color=color,
                        marker='.', markersize=6, zorder=1)

    if '_mass' in save_name:
        high_mass = values_df['masses'] > 10
        ax.errorbar(values_df[high_mass]['ars'], values_df[high_mass]['Av'], xerr=values_df[high_mass]['scat_ars'], yerr=values_df[high_mass]['scat_AV'],
                    color='mediumblue', ls='None', marker='o', label='>10 LMASS')
        ax.errorbar(values_df[~high_mass]['ars'], values_df[~high_mass]['Av'], xerr=values_df[~high_mass]['scat_ars'], yerr=values_df[high_mass]['scat_AV'],
                    color='black', ls='None', marker='o', label='<10 LMASS')
        ax.legend(fontsize=axisfont)
    elif '_ssfr' in save_name:
        high_mass = values_df['ssfrs'] > -8.9
        ax.errorbar(values_df[high_mass]['ars'], values_df[high_mass]['Av'], xerr=values_df[high_mass]['scat_ars'], yerr=values_df[high_mass]['scat_AV'],
                    color='mediumblue', ls='None', marker='o', label='>-8.9 LSSFR')
        ax.errorbar(values_df[~high_mass]['ars'], values_df[~high_mass]['Av'], xerr=values_df[~high_mass]['scat_ars'], yerr=values_df[high_mass]['scat_AV'],
                    color='black', ls='None', marker='o', label='<-8.9 LSSFR')
        ax.legend(fontsize=axisfont)
    else:
        for ar_df in ar_dfs:
            for i in range(len(ar_df)):
                color = 'grey'
                ax.plot(ar_df.iloc[i]['use_ratio'], ar_df.iloc[i]['Av'], color=color,
                        marker='.', markersize=6, zorder=1)
        ax.errorbar(medians, median_Av, xerr=stds, yerr=stds_Av,
                    color='black', ls='None', marker='o', zorder=2)

    ax.set_xlim(-0.05, 1.05)
    # ax.set_ylim(8, 12)
    ax.set_xlabel('Axis Ratio', fontsize=axisfont)
    ax.set_ylabel('FAST A_V', fontsize=axisfont)
    ax.tick_params(labelsize=ticksize, size=ticks)
    fig.savefig(save_dir + save_name[1:] + f'/AR_Clusters_FAST_AV{save_name}.pdf')
    plt.close()


def plot_re_axis_ratio():
    """Plot of axis ratio vs half-light radius

    Parameters:
    """

    ar_df = read_interp_axis_ratio()
    filter_bad_ratios = ar_df['err_use_ratio'] > 0.1

    axis_ratios = ar_df['use_ratio']
    err_axis_ratios = ar_df['err_use_ratio']

    ar_df_big = ascii.read(
        imd.mosdef_dir + '/axis_ratio_data/Merged_catalogs/mosdef_F125W_galfit_v4.0.csv')
    half_light = ar_df_big['re']
    err_half_light = ar_df_big['dre']

    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    # Figure for just the galaixes in that cluster
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.errorbar(axis_ratios, half_light, xerr=err_axis_ratios, yerr=err_half_light,
                color='black', ls='None', marker='o', markersize=4)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 2)
    ax.set_xlabel('Axis Ratio', fontsize=axisfont)
    ax.set_ylabel('Half-Light Radius (unit)', fontsize=axisfont)
    ax.tick_params(labelsize=ticksize, size=ticks)
    fig.savefig(save_dir + f'AxisRatio_HalfLight.pdf')
    plt.close()


def plot_mass_sfr_ar(n_bins):
    """Plots sfr vs mass, colored by ar

    Parameters:
    n_bins (int): Number of bins (2 or 3)
    """
    n_groups = 20
    save_name = '_ssfr'
    ar_dfs = [ascii.read(imd.cluster_dir + f'/composite_spectra/axis_stack{save_name}/{axis_group}_df.csv').to_pandas() for axis_group in range(n_groups)]

    c = 0
    for ar_df in ar_dfs:
        ar_df = ar_df[ar_df['SFR_CORR'] > 0]
        if c == 0:
            full_df = ar_df
        else:
            full_df = pd.concat([full_df, ar_df], ignore_index=True)
        c = 1

    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    # Figure for just the galaixes in that cluster
    fig, axarr = plt.subplots(1, n_bins, figsize=(n_bins * 7, 7))
    ax1 = axarr[0]
    ax2 = axarr[1]

    full_df['L_SFR'] = np.log10(full_df['SFR_CORR'])

    cutoffs = ['<0.5', '>=0.5']
    cut = full_df['use_ratio'] >= 0.5
    cut_low = full_df['use_ratio'] < 0.5
    if n_bins == 3:
        cut_low = full_df['use_ratio'] < 0.4
        cut = np.logical_and(full_df['use_ratio']
                             >= 0.4, full_df['use_ratio'] <= 0.6)
        cut_high = full_df['use_ratio'] > 0.6
        cutoffs = ['<0.4', '0.4-0.6', '>0.6']

    for ax in axarr:
        ax.scatter(full_df['LMASS'], full_df[
            'L_SFR'], s=4, color='grey')

    ax1.scatter(full_df[cut_low]['LMASS'], full_df[cut_low][
        'L_SFR'], label=f'Axis Ratio {cutoffs[0]}')  # , c=full_df[cut]['use_ratio'])
    ax2.scatter(full_df[cut]['LMASS'], full_df[cut][
        'L_SFR'], label=f'Axis Ratio {cutoffs[1]}')
    if n_bins > 2:
        ax3 = axarr[2]
        ax3.scatter(full_df[cut_high]['LMASS'], full_df[cut_high][
            'L_SFR'], label=f'Axis Ratio {cutoffs[2]}')

    for ax in axarr:
        #ax.set_ylim(-10, 500)
        ax.set_xlim(8, 12)
        ax.set_xlabel('LMASS', fontsize=axisfont)
        ax.set_ylabel('log(SFR)', fontsize=axisfont)
        ax.legend(fontsize=axisfont)
        ax.tick_params(labelsize=ticksize, size=ticks)
    fig.savefig(save_dir + f'SFR_MASS_ar_{n_bins}bins.pdf')
    plt.close()
