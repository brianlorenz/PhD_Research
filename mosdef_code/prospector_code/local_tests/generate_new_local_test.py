from re import sub
from shutil import copyfile
import initialize_mosdef_dirs as imd
from astropy.io import ascii

# On home computer
path_to_param_file = '/Users/brianlorenz/code/mosdef_code/prospector_code/prospector_composite_params_nonpar_localtest.py'
path_to_dynesty_file = '/Users/brianlorenz/code/mosdef_code/prospector_code/prospector_dynesty.py'
path_to_process_file = '/Users/brianlorenz/code/mosdef_code/prospector_code/process_completed_run_2group.py'
path_to_cosmo_file = '/Users/brianlorenz/code/mosdef_code/cosmology_calcs.py'
path_to_convert_file = '/Users/brianlorenz/code/mosdef_code/prospector_code/convert_flux_to_maggies.py'
path_to_prospector_scripts = '/Users/brianlorenz/prospector_scripts/scripts'

# Local output
desired_output_location = imd.prospector_h5_dir


def generate_all(groupID1, groupID2, run_name):
    '''Creates the taskfile and prospector scripts

    Parameters:
    n_clusters (int): number of composite groups
    run_name (str): Name of the run on prospector, will save to a folder with this name

    '''
    generate_scripts_2groups(groupID1, groupID2)
    generate_prospector_taskfile_2groups(groupID1, groupID2, run_name)
    generate_processing_taskfile_2groups(groupID1, groupID2, run_name)
    copy_files()
    print(f'Ready to run with run_name {run_name}. Run on this computer to make the directory')
    print(f'mkdir /Users/brianlorenz/mosdef/Clustering/prospector_outputs/prospector_h5s/{run_name}_h5s/')
    print(f'mkdir /Users/brianlorenz/mosdef/Clustering/prospector_outputs/prospector_csvs/{run_name}_csvs/')
    print(f'mkdir /Users/brianlorenz/mosdef/Clustering/prospector_outputs/prospector_plots/{run_name}_plots/')

def generate_scripts_2groups(groupID1, groupID2):
    '''Changes the groupID and generates the appropriate script with only 2 groups to compare fittings

    Parameters:
    groupID1 (int): ID of the first group
    groupID2 (int): ID of the second group

    '''
    file_in = path_to_param_file
    for i in range(2):
        file_out = f'/Users/brianlorenz/code/mosdef_code/prospector_code/local_tests/prospector_composite_params_group{groupID1}_trial{i}.py'

        with open(file_in, 'r') as f_in:
            with open(file_out, 'w') as f_out:
                for line in f_in:
                    if "'groupID': -1," in line:
                        line_change = sub('-1', f'{groupID1}', line)
                    else:
                        line_change = line

                    f_out.write(line.replace(line, line_change))

    print('Pausing to allow changes')
    print('Modify files 0 to 1, then press c to continue')
    breakpoint()
    for j in range(2):
        file_in = f'/Users/brianlorenz/code/mosdef_code/prospector_code/local_tests/prospector_composite_params_group{groupID1}_trial{j}.py'
        file_out = f'/Users/brianlorenz/code/mosdef_code/prospector_code/local_tests/prospector_composite_params_group{groupID2}_trial{j}.py'

        with open(file_in, 'r') as f_in:
            with open(file_out, 'w') as f_out:
                for line in f_in:
                    if f"'groupID': {groupID1}," in line:
                        line_change = sub(f'{groupID1}', f'{groupID2}', line)
                    else:
                        line_change = line

                    f_out.write(line.replace(line, line_change))


def generate_prospector_taskfile_2groups(groupID1, groupID2, run_name):
    '''Generates a taskfile that will run prospector on all of the composite sed groups

    Parameters:
    n_clusters (int): number of composite groups
    run_name (str): Name of the run on prospector, will save to a folder with this name


    '''
    file = open(f'/Users/brianlorenz/code/mosdef_code/prospector_code/local_tests/taskfile_run_prospector_2group', 'w')
    for i in range(20):
        current_groupID = get_current_group(i, groupID1, groupID2)
        j = i%10
        file.write(f'ipython -c "%run prospector_dynesty.py --param_file=\'prospector_composite_params_group{current_groupID}_trial{j}.py\' --outfile=\'{desired_output_location}/{run_name}_h5s/group{current_groupID}_trial{i}\'"\n')


def generate_processing_taskfile_2groups(groupID1, groupID2, run_name):
    '''Generates a taskfile that will process the images of a completed run

    Parameters:
    n_clusters (int): number of composite groups

    '''
    file = open(f'/Users/brianlorenz/code/mosdef_code/prospector_code/local_tests/taskfile_process_run_2group', 'w')
    for i in range(20):
        current_groupID = get_current_group(i, groupID1, groupID2)
        j = i%10
        file.write(f'ipython -c "run process_completed_run_2group.py {groupID1} {groupID2} {i} \'{run_name}\'"\n')


def copy_files():
    '''Copies necessary scripts to savio
    '''
    copyfile(path_to_dynesty_file, f'/Users/brianlorenz/code/mosdef_code/prospector_code/local_tests/prospector_dynesty.py')
    copyfile(path_to_process_file, f'/Users/brianlorenz/code/mosdef_code/prospector_code/local_tests/process_completed_run.py')
    copyfile(path_to_convert_file, f'/Users/brianlorenz/code/mosdef_code/prospector_code/local_tests/convert_flux_to_maggies.py')
    copyfile(path_to_cosmo_file, f'/Users/brianlorenz/code/mosdef_code/prospector_code/local_tests/cosmology_calcs.py')


def get_current_group(i, groupID1, groupID2):
    if i<10:
        current_groupID = groupID1
    else:
        current_groupID = groupID2
    return current_groupID


generate_all(4, 16, 'local_nonpar_sfh_fixedbins')
# generate_prospector_taskfile_2groups(1, 2, 'test_2groups')
