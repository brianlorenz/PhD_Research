from astropy.io import ascii
import initialize_mosdef_dirs as imd
import matplotlib.pyplot as plt


def plot_av_beta(n_groups, save_name='halpha_norm'):
    '''Plot AV vs Beta for the galaxies to see how they compare'''
    

    fig, ax = plt.subplots(figsize=(8,8))

    # loop through all the groups, so that we apply the same filters
    for axis_group in range(n_groups):
        ar_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_group_dfs/{axis_group}_df.csv')
        ar_df = ar_df[ar_df['beta']>-10]


        ax.scatter(ar_df['AV'], ar_df['beta'], color='black')
    ax.set_xlabel('FAST AV', fontsize=14)
    ax.set_ylabel('Beta', fontsize=14)
    ax.tick_params(labelsize=12)

    fig.savefig(imd.axis_output_dir + '/av_vs_beta.pdf')

plot_av_beta(18)