

from sys import builtin_module_names
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import initialize_mosdef_dirs as imd
import cluster_data_funcs as cdf
from mosdef_obj_data_funcs import get_mosdef_obj

def add_balmer_dec():
    '''Adds measurement of balmer decrement to the mosdef_all_cats.csv file generated in axis ratio funcs
    
    '''
    all_cats_df = ascii.read(imd.mosdef_dir + '/axis_ratio_data/Merged_catalogs/mosdef_all_cats.csv').to_pandas()
    linemeas_df = ascii.read(imd.loc_linemeas).to_pandas()
    
    
    fields = []
    v4ids = []

    agn_flags = []
    masses = []
    err_l_masses = []
    err_h_masses = []
    sfrs = []
    err_sfrs = []
    res = []
    err_res = []
    
    hb_values = []
    hb_errs = []
    ha_values = []
    ha_errs = []
    

    # Add the sfrs from the sfr_latest catalog
    dat = Table.read(imd.loc_sfrs_latest, format='fits')
    sfrs_df = dat.to_pandas()
    sfrs_df['FIELD_STR'] = [sfrs_df.iloc[i]['FIELD'].decode("utf-8").rstrip() for i in range(len(sfrs_df))]


    #Loop through the catalog and find the ha and hb value for each galaxy
    for i in range(len(all_cats_df)):
        v4id = all_cats_df.iloc[i]['v4id']
        field = all_cats_df.iloc[i]['field']
        v4ids.append(v4id)
        fields.append(field)

        obj = get_mosdef_obj(field, v4id)
        cat_id = obj['ID']

        agn_flags.append(obj['AGNFLAG'])
        masses.append(obj['LMASS'])
        err_l_masses.append(obj['L68_LMASS'])
        err_h_masses.append(obj['U68_LMASS'])
        res.append(obj['RE'])
        err_res.append(obj['DRE'])

        linemeas_slice = np.logical_and(linemeas_df['ID']==cat_id, linemeas_df['FIELD_STR']==field)
        sfrs_slice = np.logical_and(sfrs_df['ID']==cat_id, sfrs_df['FIELD_STR']==field)
        
        hb_values.append(linemeas_df[linemeas_slice].iloc[0]['HB4863_FLUX'])
        hb_errs.append(linemeas_df[linemeas_slice].iloc[0]['HB4863_FLUX_ERR'])
        ha_values.append(linemeas_df[linemeas_slice].iloc[0]['HA6565_FLUX'])
        ha_errs.append(linemeas_df[linemeas_slice].iloc[0]['HA6565_FLUX_ERR'])


        # USE SFR2 or SFR_CORR?
        sfrs.append(sfrs_df[sfrs_slice].iloc[0]['SFR2'])
        err_sfrs.append(sfrs_df[sfrs_slice].iloc[0]['SFRERR2'])
    

    





    to_merge_df = pd.DataFrame(zip(fields, v4ids, agn_flags, masses, err_l_masses, err_h_masses, sfrs, err_sfrs, res, err_res, hb_values, hb_errs, ha_values, ha_errs), columns=['field', 'v4id', 'agn_flag', 'log_mass', 'err_log_mass_d', 'err_log_mass_u', 'sfr', 'err_sfr', 'half_light', 'err_half_light', 'hb_flux', 'err_hb_flux', 'ha_flux', 'err_ha_flux'])
    merged_all_cats = all_cats_df.merge(to_merge_df, left_on=['v4id', 'field'], right_on=['v4id', 'field'])
    merged_all_cats.to_csv(imd.loc_axis_ratio_cat, index=False)


def plot_balmer_dec():

    axis_df = ascii.read(imd.loc_axis_ratio_cat).to_pandas()

    axis_ratios = []
    balmer_decs = []

    skipped = 0

    for i in range(len(axis_df)):
        row = axis_df.iloc[i]

        if row['ha_flux']>0 and row['hb_flux']>0:
            axis_ratios.append(row['use_ratio'])
            balmer_decs.append(row['ha_flux']/row['hb_flux'])
        else:
            skipped += 1
            continue
    
    print(f'Skipped {skipped}/{len(axis_df)} galxies')
    plt.scatter(axis_ratios, balmer_decs)
    plt.show()


def plot_balmer_dec():

    axis_df = ascii.read(imd.loc_axis_ratio_cat).to_pandas()

    axis_ratios = []
    balmer_decs = []

    skipped = 0

    for i in range(len(axis_df)):
        row = axis_df.iloc[i]

        if row['ha_flux']>0 and row['hb_flux']>0:
            axis_ratios.append(row['use_ratio'])
            balmer_decs.append(row['ha_flux']/row['hb_flux'])
        else:
            skipped += 1
            continue
    
    print(f'Skipped {skipped}/{len(axis_df)} galxies')
    fig, ax = plt.subplots(figsize=(8,8))
    
    ax.scatter(axis_ratios, balmer_decs, color='black')
    ax.set_xlabel('Axis Ratio', fontsize=14)
    ax.set_ylabel('Balmer Decrement (MOSDEF)', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.plot((-1, 2), (2.8, 2.8), ls='--', color='red')
    ax.set_ylim(0, 10)
    ax.set_xlim(-0.05, 1.05)
    fig.savefig(imd.axis_figure_dir + '/ar_vs_balmer.pdf')



def plot_balmer_vs_mass():

    axis_df = ascii.read(imd.loc_axis_ratio_cat).to_pandas()

    axis_ratios = []
    balmer_decs = []
    balmer_errs = []
    masses = []
    res = []
    sfrs = []


    skipped = 0

    for i in range(len(axis_df)):
        row = axis_df.iloc[i]

        if row['ha_flux']>0 and row['hb_flux']>0:
            balmer_dec = row['ha_flux']/row['hb_flux']
            if balmer_dec > 2.5 and balmer_dec < 10:
                axis_ratios.append(row['use_ratio'])
                balmer_decs.append(row['ha_flux']/row['hb_flux'])
                balmer_errs.append((row['ha_flux']/row['hb_flux'])*(row['err_ha_flux']/row['ha_flux'] + row['err_hb_flux']/row['hb_flux']))
                masses.append(row['log_mass'])
                sfrs.append(row['sfr'])
                res.append(row['half_light'])
        else:
            skipped += 1
            continue

    plot_df = pd.DataFrame(zip(axis_ratios, balmer_decs, balmer_errs, masses, res, sfrs), columns=['axis_ratio', 'balmer_dec', 'err_balmer_dec', 'log_mass', 'half_light', 'sfr'])
    plot_df['log_ssfr'] = np.log10((plot_df['sfr']) / (10**plot_df['log_mass']))
    

    def plot_mass_ssfr(mass_width, ssfr_width, starting_points):
        """Plots ssfr vs mass and shows the regions that we are slicing by 
        
        Parameters:
        mass_width (float): size of the bins in mass
        ssfr_width (float): size of the bins in ssfr
        starting_points (list of tuples): [(mass, ssfr)] coordinates of where to start
        """
        # Full sample mass/sfr
        fig, ax = plt.subplots(figsize=(8,8))

        # Make sure it is greater than zero
        filter_sfr = plot_df['sfr'] > 0
        ax.plot(plot_df[filter_sfr]['log_mass'], np.log10(plot_df[filter_sfr]['sfr']), color = 'black', ls='None', marker='o')
        ax.set_xlabel('log(Stellar Mass)', fontsize=14)
        ax.set_ylabel('log(sfr)', fontsize=14)
        ax.tick_params(labelsize=12)
        fig.savefig(imd.axis_output_dir + '/mass_sfr_sample.pdf')
        plt.close('all')

        fig, ax = plt.subplots(figsize=(8,8))
        ax.plot(plot_df['log_mass'], plot_df['log_ssfr'], color = 'black', ls='None', marker='o')

        # Splitting the groups and showing in ssfr/mass space
        bin_idxs = []
        colors = ['red', 'blue', 'orange', 'mediumseagreen']
        for i in range(len(starting_points)):
            mass_start = starting_points[i][0]
            ssfr_start = starting_points[i][1]
            mass_idx = np.logical_and(plot_df['log_mass']>mass_start, plot_df['log_mass']<=mass_start+mass_width)
            ssfr_idx = np.logical_and(plot_df['log_ssfr']>ssfr_start, plot_df['log_ssfr']<=ssfr_start+ssfr_width)
            bin_idxs.append(np.logical_and(mass_idx, ssfr_idx))
            ax.plot(plot_df[bin_idxs[i]]['log_mass'], plot_df[bin_idxs[i]]['log_ssfr'], color = colors[i], ls='None', marker='o')

        ax.set_xlabel('log(Stellar Mass)', fontsize=14)
        ax.set_ylabel('log(ssfr)', fontsize=14)
        ax.tick_params(labelsize=12)
        ax.set_ylim(-12, -7)
        ax.set_xlim(8.5, 11.5)
        fig.savefig(imd.axis_output_dir + '/mass_ssfr_split.pdf')
        plt.close('all')


        # Plotting the split groups
        fig, axarr = plt.subplots(1, 4, figsize=(16,8))
    
        count = 0
        for ax in axarr:
            ax.errorbar(plot_df[bin_idxs[count]]['axis_ratio'], plot_df[bin_idxs[count]]['balmer_dec'], yerr=plot_df[bin_idxs[count]]['err_balmer_dec'], color=colors[count], marker='o', ls='None')
            ax.set_xlabel('Axis Ratio', fontsize=14)
            ax.set_ylabel('Balmer Decrement (MOSDEF)', fontsize=14)
            ax.tick_params(labelsize=12)
            ax.set_ylim(0, 10)
            ax.set_xlim(-0.05, 1.01)
            count = count + 1
        fig.savefig(imd.axis_output_dir + '/mass_ssfr_balmer_dec.pdf')



    mass_width = 0.7
    ssfr_width = 0.5
    starting_points = [(9.3, -9.1), (9.3, -8.6), (10.0, -9.2), (10.0, -8.7)]

    plot_mass_ssfr(mass_width, ssfr_width, starting_points)

    


    def plot_cut(cut_var, cut_vals, savename, key='False'):
        """Slices the data and plots the axis ratio
        cut_var (str): Variable name to cut on
        cut_vals (list): Breakpoints of where to make the cut, include the outer boundaries e.g. [0, 0.4, 1] will cut 0 to 0.4 and 0.4 to 1
        savename (str): Name to save under
        key (str): Sets what to plot
        

        """
        n_plots = len(cut_vals)-1
        fig, axarr = plt.subplots(1, n_plots, figsize=(4+4*n_plots,8))
    
        count = 0
        for ax in axarr:
            cut = (cut_vals[count], cut_vals[count+1])

            filt = np.logical_and(plot_df[cut_var]>cut[0], plot_df[cut_var]<=cut[1])

            if key=='mass':
                ax.scatter(plot_df[filt]['log_mass'], plot_df[filt]['balmer_dec'], color='black')
                ax.set_xlabel('log(Stellar Mass)', fontsize=14)
                ax.set_ylabel('Balmer Decrement (MOSDEF)', fontsize=14)
                ax.tick_params(labelsize=12)
                ax.set_ylim(0, 10)
                ax.set_xlim(8, 11.5)
            
            if key=='size':
                ax.scatter(plot_df[filt]['half_light'], plot_df[filt]['balmer_dec'], color='black')
                ax.set_xlabel('Re', fontsize=14)
                ax.set_ylabel('Balmer Decrement (MOSDEF)', fontsize=14)
                ax.tick_params(labelsize=12)
                ax.set_ylim(0, 10)
                ax.set_xlim(0, 1.5)

            if cut_var=='log_mass':
                ax.scatter(plot_df[filt]['axis_ratio'], plot_df[filt]['balmer_dec'], color='black')
                ax.set_xlabel('Axis Ratio', fontsize=14)
                ax.set_ylabel('Balmer Decrement (MOSDEF)', fontsize=14)
                ax.tick_params(labelsize=12)
                ax.set_ylim(0, 10)
                ax.set_xlim(-0.05, 1.05)
            ax.set_title(f'{cut[0]} < {cut_var} < {cut[1]}')
            count += 1
    

        fig.savefig(imd.axis_figure_dir + f'/{savename}.pdf')

    plot_cut('axis_ratio', [0, 0.4, 0.7, 1], 'balmer_vs_mass', key='mass')
    plot_cut('axis_ratio', [0, 0.4, 0.7, 1], 'balmer_vs_re', key='size')
    plot_cut('log_mass', [0, 9.5, 10, 10.5, 13], 'balmer_vs_ar_mass_cut')
    
# add_balmer_dec()
plot_balmer_vs_mass()