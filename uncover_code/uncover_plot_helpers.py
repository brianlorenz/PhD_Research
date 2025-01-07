import matplotlib as mpl

def get_color_info(target_var, cmap, color_var='None'):
    if color_var == 'sed_av':
        norm = mpl.colors.Normalize(vmin=0, vmax=3) 
        rgba = cmap(norm(lineratio_data_row['sed_av']))  
        cbar_label = 'Photometry AV'
    if color_var == 'ha_snr':
        norm = mpl.colors.LogNorm(vmin=2, vmax=100) 
        rgba = cmap(norm(ha_snr))
        cbar_label = 'H$\\alpha$ SNR'
    if color_var == 'pab_snr':
        norm = mpl.colors.LogNorm(vmin=2, vmax=50) 
        rgba = cmap(norm(pab_snr))
        cbar_label = 'Pa$\\beta$ SNR'
    if color_var != 'None':
        color_str = f'_{color_var}'
    else:
        color_str = ''