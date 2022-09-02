import initialize_mosdef_dirs as imd

run_name = 'two_groups_2nd_test'

print(f'rsync -r brianlorenz@dtn.brc.berkeley.edu:/global/scratch/users/brianlorenz/prospector_h5s/{run_name}_h5s  {imd.prospector_h5_dir}')
print(f'rsync -r brianlorenz@dtn.brc.berkeley.edu:/global/scratch/users/brianlorenz/prospector_csvs/{run_name}_csvs  {imd.prospector_fit_csvs_dir}')
print(f'rsync -r brianlorenz@dtn.brc.berkeley.edu:/global/scratch/users/brianlorenz/prospector_plots/{run_name}_plots  {imd.prospector_plot_dir}')