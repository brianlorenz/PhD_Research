from read_images import prepare_images
from vizualize_outputs import plot_cutout_overview, plot_cluster_summary
from cluster_seds import cluster_pixels


def main(id_dr3_list, snr_thresh=3, cluster_method='test'):
    # prepare_images(id_dr3_list, snr_thresh=snr_thresh)
    # plot_cutout_overview(id_dr3_list)
    cluster_pixels(id_dr3_list, cluster_method=cluster_method)
    plot_cluster_summary(id_dr3_list, cluster_method=cluster_method)


if __name__ == '__main__':
    main([46339, 44283, 30804], snr_thresh=3, cluster_method='kmeans')