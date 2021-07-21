import initialize_mosdef_dirs as imd

run_name = 'first_savio'

print(f'rsync -r brianlorenz@dtn.brc.berkeley.edu:/global/scratch/brianlorenz/prospector_h5s/{run_name}_h5s  {imd.prospector_h5_dir}')
print(f'rsync -r brianlorenz@dtn.brc.berkeley.edu:/global/scratch/brianlorenz/prospector_csvs/{run_name}_csvs  {imd.prospector_fit_csvs_dir}')
print(f'rsync -r brianlorenz@dtn.brc.berkeley.edu:/global/scratch/brianlorenz/prospector_plots/{run_name}_plots  {imd.prospector_plot_dir}')