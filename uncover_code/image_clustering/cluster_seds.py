from data_paths import pixel_sed_save_loc
import numpy as np
import matplotlib.pyplot as plt
import time

def cluster_pixels(id_dr3_list):
    for id_dr3 in id_dr3_list:
        print(f'Reading cutouts for id_dr3 = {id_dr3}')
        pixel_data = np.load(pixel_sed_save_loc + f'{id_dr3}_pixels.npz')
        pixel_seds = pixel_data['pixel_seds'] # shape of (n_images, pixel_ids)
        masked_indicies = pixel_data['masked_indicies'] # shape of (2, pixel_ids)
        image_cutouts = pixel_data['image_cutouts'] # shape of (n_images, cutout_y_size, cutout_x_size)
        wht_image_cutouts = pixel_data['wht_image_cutouts'] # shape of (n_images, cutout_y_size, cutout_x_size)
        boolean_segmap = pixel_data['boolean_segmap'] # shape of (cutout_y_size, cutout_x_size)
        filter_names = pixel_data['filter_names'] # shape of (n_images,)

        breakpoint()
        # Now need to cluster on the pixel_seds
        # Boolean segmap leaves holes, doesn't seem to be a great decider of what's in the galaxy

if __name__ == '__main__':
    cluster_pixels([44283, 30804])