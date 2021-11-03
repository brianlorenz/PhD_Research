from astropy.io import ascii
import initialize_mosdef_dirs as imd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np



def plot_av_beta(n_groups, save_name='halpha_norm', use='False'):
    '''Plot AV vs Beta for the galaxies to see how they compare
    
    Parameters:
    n_groups (int): Number of groups
    save_name (str): FOlder to pull and save data from
    use (str): Set to A1600 to convert beta to A1600 and plot that
    '''
    

    fig, ax = plt.subplots(figsize=(9,8))

    cmap = mpl.cm.viridis 
    norm = mpl.colors.Normalize(vmin=-10, vmax=-8) 

    # loop through all the groups, so that we apply the same filters
    for axis_group in range(n_groups):
        ar_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_group_dfs/{axis_group}_df.csv').to_pandas()
        ar_df = ar_df[ar_df['beta']>-10]

        ar_df['log_ssfr'] = np.log10(ar_df['sfr']/(10**ar_df['log_mass']))

        if use == 'A1600':
            ar_df['A1600'] = 1.99*ar_df['beta']+4.43
            y_var = 'A1600'
            y_label = 'A1600'
        else:
            y_var = 'beta'
            y_label = 'Beta'
        
        for i in range(len(ar_df)):
            row = ar_df.iloc[i]
            rgba = cmap(norm(row['log_ssfr']))
            ax.scatter(row['AV'], row[y_var], color=rgba)
    ax.set_xlabel('FAST AV', fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.tick_params(labelsize=12)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('log(ssfr)', fontsize=14)

    fig.savefig(imd.axis_output_dir + f'/av_vs_{y_label}.pdf')

plot_av_beta(18)
plot_av_beta(18, use='A1600')