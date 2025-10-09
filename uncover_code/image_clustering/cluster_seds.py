from data_paths import pixel_sed_save_loc, read_saved_pixels, read_sed, check_and_make_dir, get_cluster_save_path
import numpy as np
import matplotlib.pyplot as plt
import time
from cluster_algorithms.k_means import kmeans
from cluster_algorithms.similarity_spectral_clustering import spectral_cluster_cross_cor
from cluster_algorithms.gaussian_mixture import gaussian_mixture_model
from cluster_algorithms.dbscan import dbscan_clustering

def define_cluster_methods():
    cluster_dict = {
        'test': cluster_method_test,
        'kmeans': kmeans,
        'spectral_cross_cor': spectral_cluster_cross_cor,
        'gaussian_mixture': gaussian_mixture_model,
        'dbscan': dbscan_clustering
    }
    return cluster_dict


def cluster_pixels(id_dr3_list, cluster_method='test', norm_method=''):
    save_path = get_cluster_save_path(cluster_method, norm_method=norm_method)
    check_and_make_dir(save_path)
    for id_dr3 in id_dr3_list:
        save_location = save_path + f'{id_dr3}_clustered.npz'
        

        print(f'Reading cutouts for id_dr3 = {id_dr3}')
        pixel_data = read_saved_pixels(id_dr3)
        pixel_seds = pixel_data['pixel_seds']
        image_cutouts = pixel_data['image_cutouts'] # shape of (n_images, cutout_y_size, cutout_x_size)
        bad_image_idxs = pixel_data['bad_image_idxs']
        masked_indicies = pixel_data['masked_indicies']
        sed_data = read_sed(id_dr3)
        sed = sed_data['sed']
        err_sed = sed_data['err_sed']


        # Remove the images flagged as bad if there are any
        if len(bad_image_idxs) > 0:
            pixel_seds = np.delete(pixel_seds, bad_image_idxs, axis=0)
            image_cutouts = np.delete(image_cutouts, bad_image_idxs, axis=0)
            sed = np.delete(sed, bad_image_idxs, axis=0)
            err_sed = np.delete(err_sed, bad_image_idxs, axis=0)
        

        # Now need to cluster on the pixel_seds
        cluster_dict = define_cluster_methods()
        cluster_values = cluster_dict[cluster_method](pixel_seds, sed, err_sed, norm_method=norm_method)

        # Reconstruct the image but with clusteres given their categorical varaibles
        clustered_image = setup_cluster_image(cluster_values, masked_indicies, image_cutouts[0].shape)

        # Save the output
        np.savez(save_location, cluster_values=cluster_values, clustered_image=clustered_image)
        


def cluster_method_test(pixel_seds, *args):
    cluster_values = np.ones(pixel_seds.shape[1])
    cluster_values[:round(len(cluster_values)/2)] = 2
    return cluster_values

def setup_cluster_image(cluster_values, masked_indicies, image_shape):
    cluster_image = np.zeros(image_shape)
    row_idx, col_idx = masked_indicies
    cluster_image[row_idx, col_idx] = cluster_values
    return cluster_image
    





if __name__ == '__main__':
    cluster_pixels([46339, 44283, 30804], cluster_method='spectral_cross_cor', norm_method='')