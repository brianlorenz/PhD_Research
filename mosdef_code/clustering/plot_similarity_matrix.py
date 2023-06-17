from astropy.io import ascii
import initialize_mosdef_dirs as imd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

def plot_sim_matrix(n_clusters, ssfr_order=False):
    
    sim_matrix = ascii.read(imd.cluster_dir+'/similarity_matrix.csv').to_pandas()
    zobjs_clustered = ascii.read(imd.cluster_dir+'/zobjs_clustered.csv').to_pandas()
    

    
    # ORder by ssfr
    if ssfr_order == True:
        cluster_summary_df = ascii.read(imd.loc_cluster_summary_df).to_pandas()
        ssfr_sorted = cluster_summary_df.sort_values(['median_log_ssfr']).index[::-1] #most to least star-forming
        ssfr_sorted = ssfr_sorted.tolist()
        #Add a new categorical varialbe to the groups
        zobjs_clustered['cluster_num_ssfr_sort'] = pd.Categorical(zobjs_clustered['cluster_num'], categories=ssfr_sorted, ordered=True)
        new_order = zobjs_clustered.sort_values(by=['cluster_num_ssfr_sort'])['original_zobjs_index']
        new_order_df = zobjs_clustered.sort_values(by=['cluster_num_ssfr_sort']).reset_index()
        add_str = '_ssfr_sort'
        cluster_order = ssfr_sorted
    else:
        new_order = zobjs_clustered.sort_values(by=['cluster_num'])['original_zobjs_index']
        new_order_df = zobjs_clustered.sort_values(by=['cluster_num']).reset_index()
        add_str = ''
        cluster_order = np.arange(n_clusters)
    
    col_names = [f'col{new_order.iloc[i]+1}' for i in range(len(new_order))]

    sim_matrix = sim_matrix.reindex(np.array(new_order))
    sim_matrix = sim_matrix[col_names]

    cluster_divisions = []
    for cluster_num in cluster_order:
        starting_index = new_order_df[new_order_df['cluster_num']==cluster_num].index[0]
        cluster_divisions.append((cluster_num, starting_index))


    fig, ax = plt.subplots(figsize=(8,8))

    plt.set_cmap('Reds')
    ax.matshow(sim_matrix)
    
    for i in range(n_clusters):
        ax.text(870, cluster_divisions[i][1], f'Group{cluster_divisions[i][0]}')
        ax.axhline(cluster_divisions[i][1], color='blue')
        ax.axvline(cluster_divisions[i][1], color='blue')


    fig.savefig(imd.cluster_dir+f'/similarity_matrix_vis{add_str}.pdf', bbox_inches='tight')
    

# plot_sim_matrix(19, ssfr_order=True)
# plot_sim_matrix(19)