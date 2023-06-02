from astropy.io import ascii
import initialize_mosdef_dirs as imd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

def plot_sim_matrix(n_clusters):
    
    sim_matrix = ascii.read(imd.cluster_dir+'/similarity_matrix.csv').to_pandas()
    zobjs_clustered = ascii.read(imd.cluster_dir+'/zobjs_clustered.csv').to_pandas()
    
    
    new_order = zobjs_clustered.sort_values(by=['cluster_num'])['original_zobjs_index']
    new_order_df = zobjs_clustered.sort_values(by=['cluster_num']).reset_index()
    col_names = [f'col{new_order.iloc[i]+1}' for i in range(len(new_order))]

    sim_matrix = sim_matrix.reindex(np.array(new_order))
    sim_matrix = sim_matrix[col_names]

    cluster_divisions = []
    for cluster_num in range(n_clusters):
        starting_index = new_order_df[new_order_df['cluster_num']==cluster_num].index[0]
        cluster_divisions.append((cluster_num, starting_index))


    fig, ax = plt.subplots(figsize=(8,8))

    plt.set_cmap('Reds')
    ax.matshow(sim_matrix)
    
    for i in range(n_clusters):
        ax.text(870, cluster_divisions[i][1], f'Group{cluster_divisions[i][0]}')
        ax.axhline(cluster_divisions[i][1], color='blue')
        ax.axvline(cluster_divisions[i][1], color='blue')


    fig.savefig(imd.cluster_dir+'/similarity_matrix_vis.pdf', bbox_inches='tight')
    

plot_sim_matrix(23)