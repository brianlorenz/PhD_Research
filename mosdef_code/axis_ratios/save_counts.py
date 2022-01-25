import initialize_mosdef_dirs as imd


def save_count(df, name, long_name):
    """Prints the final count and saves the dataframe into the counts folder

    Parameters:
    df (pd.DataFrame): Dataframe to count
    name (str): csv name to save it for
    long_name (str): What to print in the terminal, more descriptive name
    
    """

    print(f'Count of {long_name}: {len(df)}')
    df.to_csv(imd.gal_counts_dir + f'/{name}.csv', index=False)