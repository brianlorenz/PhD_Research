import random
import matplotlib.pyplot as plt
import numpy as np
from simple_make_dustmap import scale_linemap_to_other, compute_similarity_linemaps, plot_and_correlate_highsnr_pix
from scipy.stats import pearsonr
import pandas as pd



def test_cors():
    offset_amts = np.arange(0,19,3)
    flux_multipliers = np.arange(1, 4, 1)

    r_vals = []
    p_vals = []
    cc_vals = []
    sim_vals = []
    offset_amt_used = []
    flux_multiplier_used = []

    for offset_amt in offset_amts:
        for flux_multiplier in flux_multipliers:
            r_info, sim_info = generate_blobs(offset_amt, flux_multiplier)
            offset_amt_used.append(offset_amt)
            flux_multiplier_used.append(flux_multiplier)
            r_vals.append(r_info[0])
            p_vals.append(r_info[1])
            cc_vals.append(sim_info[0])
            sim_vals.append(sim_info[1])

    test_df = pd.DataFrame(zip(offset_amt_used, flux_multiplier_used, r_vals, p_vals, cc_vals, sim_vals), columns=['offset_amt', 'flux_multiplier', 'r_value', 'p_value', 'cross_cor_value', 'sim_value'])

    fig, axarr=plt.subplots(1, 3, figsize=(18, 6))
    ax_r = axarr[0]
    ax_cc = axarr[1]
    ax_sim = axarr[2]

    colors = ['black', 'orange', 'red']
    for i in range(len(test_df)):
        # breakpoint()
        ax_r.plot(test_df['offset_amt'].iloc[i], test_df['r_value'].iloc[i], marker='o', color=colors[test_df['flux_multiplier'].iloc[i]-1], ls='None')
        ax_cc.plot(test_df['offset_amt'].iloc[i], 1-test_df['cross_cor_value'].iloc[i], marker='o', color=colors[test_df['flux_multiplier'].iloc[i]-1], ls='None')
        ax_sim.plot(test_df['offset_amt'].iloc[i], test_df['sim_value'].iloc[i], marker='o', color=colors[test_df['flux_multiplier'].iloc[i]-1], ls='None')

    for ax in axarr:
        ax.set_xlabel('Offset amount (pix)', fontsize=14)
        ax.tick_params(labelsize=14)
    ax_r.set_ylabel('r value', fontsize=14)
    ax_cc.set_ylabel('cross-cor value', fontsize=14)
    ax_sim.set_ylabel('similarity value', fontsize=14)
    fig.savefig(f'/Users/brianlorenz/uncover/Figures/correlation_test/summary.pdf')

def generate_blobs(offset_amt, flux_multiplier):
    noise_level = 1e-9
    noise_arr = np.random.rand(100, 100)*noise_level
    noise_arr2 = np.random.rand(100, 100)*noise_level
    segmap = np.full((100, 100), False)



    gauss_arr_center = np.zeros([100, 100])
    gauss_arr_offset = np.zeros([100, 100])
    for i in range(100):
        for j in range(100):
            gauss_arr_center[i][j] = 1e-7 * gaus2d(i, j) * flux_multiplier
            gauss_arr_offset[i][j] = 1e-7 * gaus2d(i, j, mx=50-offset_amt, my=50-offset_amt)
            if i > 20 and i < 70:
                if j > 20 and j < 70:
                    segmap[i][j] = True

    ha_map = noise_arr + gauss_arr_center 
    cont_map = noise_arr2 + gauss_arr_offset 
    
    save_map(ha_map, cont_map, offset_amt, flux_multiplier)
    # r_value_info = plot_and_correlate_highsnr_pix(ha_map, cont_map, segmap, offset_amt, flux_multiplier)
    r_value, p_value = pearsonr(ha_map[segmap], cont_map[segmap])
    r_value_info = [r_value, p_value]

    ha_linemap_scaled_to_cont, map_scale_factor, cc_similarity = scale_linemap_to_other(ha_map, cont_map, segmap)
    sim_index = compute_similarity_linemaps(ha_linemap_scaled_to_cont, cont_map, segmap)
    sim_index_values = [cc_similarity, sim_index]

    return r_value_info, sim_index_values

def save_map(ha_map, cont_map, offset_amt, flux_multiplier):
    save_name = f'/Users/brianlorenz/uncover/Figures/correlation_test/maps/{offset_amt}_flux{flux_multiplier}.pdf'
    combined_map = ha_map + cont_map
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(combined_map)
    fig.savefig(save_name)
    plt.close('all')

def plot_and_correlate_highsnr_pix(map1, map2, segmap, offset_amt, flux_multiplier):
    map1_pixels = map1[segmap]
    map2_pixels = map2[segmap]
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(map2_pixels, map1_pixels, marker='o', ls='None', color='black')
    ax.set_xscale('log')
    ax.set_yscale('log')
    if len(map1_pixels) < 2 or len(map2_pixels) < 2:
        r_value = -99
        p_value = -99
    else:
        r_value, p_value = pearsonr(map1_pixels, map2_pixels)
    n_pixels = len(map1_pixels)
    ax.set_title(f'r_value = {r_value:0.3f}, p_value = {p_value:0.2e}')
    fig.savefig(f'/Users/brianlorenz/uncover/Figures/correlation_test/cors/{offset_amt}_flux{flux_multiplier}_correlation.pdf', bbox_inches='tight')
    plt.close('all')
    # file = open("/Users/brianlorenz/uncover/Data/generated_tables/r_values.txt", "a")
    # file.write(f"{id_dr3} {r_value} {p_value} {snr_thresh_map}\n")
    # file.close()
    r_value_info = [r_value, p_value]
    return r_value_info


# define normalized 2D gaussian
def gaus2d(x=0, y=0, mx=50, my=50, sx=6, sy=6):
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))




test_cors()