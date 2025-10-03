from data_paths import pixel_sed_save_loc, read_saved_pixels
import numpy as np
import matplotlib.pyplot as plt
import time

def cluster_pixels(id_dr3_list):
    for id_dr3 in id_dr3_list:
        print(f'Reading cutouts for id_dr3 = {id_dr3}')
        pixel_data = read_saved_pixels(id_dr3)
        pixel_seds = pixel_data['pixel_seds']
        image_cutouts = pixel_data['image_cutouts'] # shape of (n_images, cutout_y_size, cutout_x_size)
        bad_image_idxs = pixel_data['bad_image_idxs']

        # Remove the images flagged as bad
        pixel_seds = np.delete(pixel_seds, bad_image_idxs, axis=0)
        image_cutouts = np.delete(image_cutouts, bad_image_idxs, axis=0)

        breakpoint()
        # Now need to cluster on the pixel_seds



if __name__ == '__main__':
    cluster_pixels([46339, 44283, 30804])