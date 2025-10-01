from read_images import prepare_images
from vizualize_outputs import plot_cutout_summary


def main(id_dr3_list, snr_thresh=3):
    prepare_images(id_dr3_list, snr_thresh=snr_thresh)
    plot_cutout_summary(id_dr3_list)

if __name__ == '__main__':
    main([46339, 44283, 30804], snr_thresh=3)