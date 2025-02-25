import os
from uncover_read_data import read_supercat, read_SPS_cat_all
import shutil

source_dir = '/Users/brianlorenz/uncover/Figures/PHOT_sample/interesting'
target_dir = '/Users/brianlorenz/uncover/Figures/PHOT_sample/interesting_zsort'

sps_df = read_SPS_cat_all()
files = os.listdir(source_dir)
id_dr3s = [int(file.split('_')[0]) for file in files]
redshifts = [sps_df[sps_df['id']==file_id].iloc[0]['z_50'] for file_id in id_dr3s]

for i in range(len(files)):
    file = files[i]
    redshift = redshifts[i]
    new_name = f'z{redshift:0.2f}_' + file
    shutil.copyfile(source_dir + '/' + file, target_dir + '/' +new_name)

