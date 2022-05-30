# Plot a uvj diagram color coded by axis ratio
import numpy as np
import pandas as pd
from astropy.io import ascii
import matplotlib.pyplot as plt
import matplotlib as mpl
import initialize_mosdef_dirs as imd
import matplotlib.patheffects as path_effects
from ellipses_for_plotting import get_ellipse_shapes
from matplotlib.patches import Ellipse



def plot_median_uvj(n_bins, save_name):
    """UVJ diagram with medians from each group
    
    """
    fig, ax = plt.subplots(figsize=(8,8))

    summary_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/summary.csv').to_pandas()

    for axis_group in range(n_bins):
        ar_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_group_dfs/{axis_group}_df.csv').to_pandas()
        U_V_point = np.median(ar_df['U_V'])
        V_J_point = np.median(ar_df['V_J'])

        row = summary_df.iloc[axis_group]
        shape = row['shape']

        xlims = (-0.5, 2.0)
        ylims = (0, 2.5)
        ax.set_ylim(ylims)
        ax.set_xlim(xlims)
        cmap = mpl.cm.gist_heat_r 
        norm = mpl.colors.Normalize(vmin=0, vmax=2.5) 

        rgba = cmap(norm(row['log_use_sfr_median']))

        ellipse_width, ellipse_height = get_ellipse_shapes(xlims[1]-xlims[0], np.abs(ylims[1]-ylims[0]), shape, scale_factor=0.025)
        
        if row['log_mass_median'] > 10:
            size = 1.5
        else:
            size = 1
        ax.add_artist(Ellipse((V_J_point, U_V_point), size*ellipse_width, size*ellipse_height, facecolor=rgba, zorder=2))

    # UVJ diagram lines
    ax.plot((-100, 0.69), (1.3, 1.3), color='black')
    ax.plot((1.5, 1.5), (2.01, 100), color='black')
    xline = np.arange(0.69, 1.5, 0.001)
    yline = xline * 0.88 + 0.69
    ax.plot(xline, yline, color='black')

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('SFR median', fontsize=14)
    ax.set_xlabel('V-J', fontsize=14) 
    ax.set_ylabel('U-V', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_aspect(ellipse_width/ellipse_height)


    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/median_uvjs.pdf')



def plot_uvj_ar():
    '''Plot the uvj diagram colored by axis ratio
    
    '''
    # Read in the axis ratio data
    ar_df = ascii.read(imd.loc_axis_ratio_cat).to_pandas()

    # Read in the uvj colors
    uvj_df = ascii.read(imd.loc_eazy_uvj_cat).to_pandas()

    ar_uvj_df = ar_df.merge(uvj_df, left_on=['field', 'v4id'], right_on=['FIELD', 'V4ID'], how='left')

    # Remove galaxies that have poor UVJ measurements
    ar_uvj_df = ar_uvj_df[ar_uvj_df['UVJ_FLAG']==0] 
    
    # Keeps only the first instance where there are multiple masks
    ar_uvj_df = ar_uvj_df.drop_duplicates(subset=['field', 'v4id'])


    # Figure 1 - UVJ WIth everything
    fig, ax = plt.subplots(figsize=(9,8))

    cmap = mpl.cm.viridis 
    norm = mpl.colors.Normalize(vmin=0, vmax=1) 

    for i in range(len(ar_uvj_df)):
        row = ar_uvj_df.iloc[i]
        rgba = cmap(norm(row['use_ratio']))
        ax.plot(row['V_J'], row['U_V'], marker = 'o', ls='None', color=rgba)

    # UVJ diagram lines
    ax.plot((-100, 0.69), (1.3, 1.3), color='black')
    ax.plot((1.5, 1.5), (2.01, 100), color='black')
    xline = np.arange(0.69, 1.5, 0.001)
    yline = xline * 0.88 + 0.69
    ax.plot(xline, yline, color='black')

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('Axis Ratio', fontsize=14)
    ax.set_xlabel('V-J', fontsize=14) 
    ax.set_ylabel('U-V', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_xlim(-0.5, 2)
    ax.set_ylim(0, 2.5)
    text2 = ax.text(2.5, 0, 'Edge-on', fontsize=14, color=cmap(norm(0)))
    text1 = ax.text(2.5, 2.5, 'Face-on', fontsize=14, color=cmap(norm(1)))
    text1.set_path_effects([path_effects.Stroke(linewidth=1.7, foreground='black'),
                       path_effects.Normal()])
    text2.set_path_effects([path_effects.Stroke(linewidth=1.7, foreground='black'),
                       path_effects.Normal()])
    fig.savefig(imd.axis_output_dir + f'/uvj_ar.pdf')

    # Figure 2 - three-panel UVJ in mass bins, similar to https://arxiv.org/pdf/2109.14721.pdf
    fig, axarr = plt.subplots(1, 3, figsize=(35,8))


    cmap = mpl.cm.viridis 
    norm = mpl.colors.Normalize(vmin=0, vmax=1) 

    
    mass_95_100_filt = np.logical_and(ar_uvj_df['log_mass']>=9.5, ar_uvj_df['log_mass']<10)
    mass_100_105_filt = np.logical_and(ar_uvj_df['log_mass']>=10, ar_uvj_df['log_mass']<10.5)
    mass_105_110_filt = np.logical_and(ar_uvj_df['log_mass']>=10.5, ar_uvj_df['log_mass']<11)

    mass_filts = [mass_95_100_filt, mass_100_105_filt, mass_105_110_filt]
    titles = ['9.5 <= log(Stellar Mass) < 10', '10 <= log(Stellar Mass) < 10.5', '10.5 <= log(Stellar Mass) < 11']

    count = 0
    for ax in axarr:
        mass_filt = mass_filts[count]
        for i in range(len(ar_uvj_df[mass_filt])):
            row = ar_uvj_df[mass_filt].iloc[i]
            if row['V_J']>-20 and row['U_V']>-20:
                rgba = cmap(norm(row['use_ratio']))
                ax.plot(row['V_J'], row['U_V'], marker = 'o', ls='None', color=rgba)

        # UVJ diagram lines
        ax.plot((-100, 0.69), (1.3, 1.3), color='black')
        ax.plot((1.5, 1.5), (2.01, 100), color='black')
        xline = np.arange(0.69, 1.5, 0.001)
        yline = xline * 0.88 + 0.69
        ax.plot(xline, yline, color='black')
        
        ax.set_xlabel('V-J', fontsize=14) 
        ax.set_ylabel('U-V', fontsize=14)
        ax.tick_params(labelsize=12)
        ax.set_xlim(-0.5, 2)
        ax.set_ylim(0, 2.5)
        ax.set_title(titles[count])
        count += 1
    
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axarr)
    cbar.set_label('Axis Ratio', fontsize=14)
    text2 = ax.text(2.5, 0, 'Edge-on', fontsize=14, color=cmap(norm(0)))
    text1 = ax.text(2.5, 2.5, 'Face-on', fontsize=14, color=cmap(norm(1)))
    text1.set_path_effects([path_effects.Stroke(linewidth=1.7, foreground='black'),
                    path_effects.Normal()])
    text2.set_path_effects([path_effects.Stroke(linewidth=1.7, foreground='black'),
                    path_effects.Normal()])
    fig.savefig(imd.axis_output_dir + f'/uvj_ar_by_mass.pdf')





    # Figure 3 - Single UVJ with points
    def uvj_balmer_plot(ar_uvj_df, plot_num):
        """Makes the uvj balmer plot, using a funciton so I can make two nearly-identical plots
        
        ar_uvj_df (pd.DataFrame): Dataframe to plot from
        plot_num (int): Number of the plot, just used to differentiate first and second times through
        """
        fig, ax = plt.subplots(1, 1, figsize=(8,8))

        cmap = mpl.cm.viridis 
        norm = mpl.colors.Normalize(vmin=2, vmax=7) 

        ar_uvj_df['balmer_dec'] = (ar_uvj_df['ha_flux'] / ar_uvj_df['hb_flux'])

        ar_uvj_df, filter_names = filter_uvj_df(ar_uvj_df)

        num_lower_limit = 0
        if plot_num == 1:
            for i in range(len(ar_uvj_df)):
                row = ar_uvj_df.iloc[i]
                # if the hbeta line is not 3 sigma detected, make the circle open
                rgba = cmap(norm(row['balmer_dec']))
                # Open circle for hb flux S/N less than 2
                if (row['hb_flux']/row['err_hb_flux']) < 2:
                    num_lower_limit += 1
                    ax.plot(row['V_J'], row['U_V'], marker = 'o', ls='None', color=rgba, mfc='none')
                else:
                    ax.plot(row['V_J'], row['U_V'], marker = 'o', ls='None', color=rgba)

        if plot_num == 2:
            # Name to change if not on mean anymore
            filter_names.append('Using mean balmer dec')

            # Set the bin grid
            stepsize_x = 0.15
            stepsize_y = 0.15
            x_vals = np.arange(-0.5, 2, stepsize_x)
            y_vals = np.arange(0, 2, stepsize_y)
            for x in range(len(x_vals)):
                for y in range(len(y_vals)):
                    df_filt_x = np.logical_and(ar_uvj_df['V_J'] > x_vals[x], ar_uvj_df['V_J'] <= x_vals[x]+stepsize_x)
                    df_filt_y = np.logical_and(ar_uvj_df['U_V'] > y_vals[y], ar_uvj_df['U_V'] <= y_vals[y]+stepsize_y)
                    df_filt = np.logical_and(df_filt_x, df_filt_y)
                    n_gals = len(ar_uvj_df[df_filt])
                    # Skip if there are no galaxies in this bin
                    if n_gals  == 0:
                        continue
                    
                    ### UPDATE NAME ABOVE IF SWITCHING OFF OF MEAN DECREMENT
                    mean_dec = np.mean(ar_uvj_df[df_filt]['balmer_dec'])
                    rgba = cmap(norm(mean_dec))
                    ax.plot(x_vals[x]+(stepsize_x/2), y_vals[y]+(stepsize_y/2), marker = 's', ls='None', color=rgba, markersize=3+(n_gals/3))



        # UVJ diagram lines
        ax.plot((-100, 0.69), (1.3, 1.3), color='black')
        ax.plot((1.5, 1.5), (2.01, 100), color='black')
        xline = np.arange(0.69, 1.5, 0.001)
        yline = xline * 0.88 + 0.69
        ax.plot(xline, yline, color='black')
        
        ax.set_xlabel('V-J', fontsize=14) 
        ax.set_ylabel('U-V', fontsize=14)
        ax.tick_params(labelsize=12)
        ax.set_xlim(-0.5, 2)
        ax.set_ylim(0, 2.5)

        # Filter names
        filter_names.append(f'UVJ_FLAG = 0')
        filter_names.append(f'Open Circle for Hb S/N < 2 ({num_lower_limit})')
        for k in range(len(filter_names)):
            ax.text(-0.45, 2.27-(0.13*k), filter_names[k], fontsize=12, color='red')
        ax.text(-0.45, 2.4, f'{len(ar_uvj_df)} galaxies', fontsize=12, color='red')
        
        
        # Get the ids of the galaxies that are in weird spots on UVJ
        top_corner = np.logical_and(ar_uvj_df['U_V']>1.1, ar_uvj_df['V_J']>1.1)
        top_odd_gals_idx = np.logical_and(ar_uvj_df['balmer_dec']<3, top_corner)
        top_odd_gals_df = ar_uvj_df[top_odd_gals_idx]
        top_odd_gals_df.to_csv(imd.axis_output_dir + '/uvj_top_odd_gals.csv', index=False)

        bot_corner = np.logical_and(ar_uvj_df['U_V']<0.65, ar_uvj_df['V_J']<0.6)
        bot_odd_gals_idx = np.logical_and(ar_uvj_df['balmer_dec']>6.5, bot_corner)
        bot_odd_gals_df = ar_uvj_df[bot_odd_gals_idx]
        bot_odd_gals_df.to_csv(imd.axis_output_dir + '/uvj_bot_odd_gals.csv', index=False)




        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.set_label('Balmer Decrement', fontsize=14)
        if plot_num == 1:
            fig.savefig(imd.axis_output_dir + f'/uvj_balmer.pdf')
        if plot_num == 2:
            fig.savefig(imd.axis_output_dir + f'/uvj_balmerboxes.pdf')

    uvj_balmer_plot(ar_uvj_df, 1)
    uvj_balmer_plot(ar_uvj_df, 2)


def filter_uvj_df(df):
    """Given a dataframe, filter out all points that we don't want included in the UVJ diagram
    
    Parameters:
    df (pd.DataFrame): Dataframe with balmer decrement values already computed

    Returns:
    df_filt (pd.DataFrame): The filtered dataframe
    filter_names (str): Names of the filters to show on the plot
    """
    ### IF THE FILTER IS TRUE, THE OBJECT IS KEPT IN
    filts = []
    ### IF YOU CHANGE OR ADD A FILTER, UPDATE THE NAME
    filter_names = []


    # Make sure both halpha and hbeta are detected to 3 sigma
    filts.append((df['ha_flux']/df['err_ha_flux']) > 3)
    filter_names.append('Ha S/N > 3')

    # Remove any that don't have coverage in the lines
    filts.append((df['ha_flux']) > -99)
    filts.append((df['hb_flux']) > -99)
    filter_names.append('Ha and Hb flux > -999')

    # Make sure the agn flag is zero
    filts.append(df['agn_flag'] == 0)
    filter_names.append('Removed AGN')

    final_filt = np.ones(len(df))
    for i in range(len(filts)):
        final_filt = np.logical_and(final_filt, filts[i])

    df_filt = df[final_filt]
    return df_filt, filter_names




# plot_uvj_ar()
plot_median_uvj(8, 'both_sfms_4bin_median_2axis_boot100')