'''Runs all of the plotting codes for clusters'''

from bpt_clusters import plot_bpt_cluster, plot_all_bpt_clusters
from emission_measurements import read_emission_df, get_emission_measurements


def generate_cluster_plots(groupID, emission_df):
    '''
    Generates all plots for the cluster with the given groupID
    
    Parameters:
    groupID (int): the groupID of the cluster
    emission_df (pd.DataFrame): Use read_emission_df

    Returns:
    '''
    plot_bpt_cluster(emission_df, groupID)


def generate_all_cluster_plots(n_clusters):
    '''
    Generates all plots for all the clusters
    
    Parameters:
    n_clusters (int): number of clusters

    Returns:
    '''
    emission_df = read_emission_df()

    for groupID in range(n_clusters):
        generate_cluster_plots(groupID, emission_df)
