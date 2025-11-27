import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import glob
import os


# Return a DataFrame row with genome bins as columns and their corresponding log2 ratio values    
def process_folder(folder_path):

    extension = ".cna.seg"
    cna_col = ".logR"
    #extension = ".correctedDepth.txt"

    chrom_lengths_hg38 = {
    '1': 248956422, '2': 242193529, '3': 198295559, '4': 190214555,
    '5': 181538259, '6': 170805979, '7': 159345973, '8': 145138636,
    '9': 138394717, '10': 133797422, '11': 135086622, '12': 133275309,
    '13': 114364328, '14': 107043718, '15': 101991189, '16': 90338345,
    '17': 83257441, '18': 80373285, '19': 58617616, '20': 64444167,
    '21': 46709983, '22': 50818468, 'X': 156040895, 'Y': 57227415
    }

    bin_size = 1000000  # 1 Mb bins
    chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y']
    master_bins = []
    loc = 0

    print("Creating master mask...")
    for chrom in chromosomes:
        chrom_length = (chrom_lengths_hg38[chrom] // bin_size) * bin_size # Take only full bins
        for start in range(1, chrom_length+1, bin_size):
            end = start + bin_size - 1
            loc += bin_size
            master_bins.append((chrom, start, end, loc))
        
    df_master = pd.DataFrame(master_bins, columns=['chr', 'start', 'end', 'loc'])
    print(f"Master mask created with {len(df_master)} bins.")

    path_to_files = folder_path + "/*" + extension
    file_list = glob.glob(path_to_files)
    print(f"Found {len(file_list)} '{extension}' files in folder '{folder_path}'.")

    df_list = []
    global_min = float('inf')
    global_max = float('-inf')

    for file_path in file_list:
        try:
            patient_id = os.path.basename(file_path).replace(extension, '')

            patient_df = pd.read_csv(file_path, sep='\t', dtype={'chr': str}, na_values=['NA', 'inf'])

            col_name = patient_id + cna_col if extension == ".cna.seg" else 'log2_TNratio_corrected'

            meaningful_df = patient_df[['chr', 'start', 'end', col_name]]

            # Merge with master bins to ensure all bins are represented
            merged_df = pd.merge(df_master, meaningful_df, on=['chr', 'start', 'end'], how='left')

            # Fill NaN values with median log2 ratio
            median_log2R = patient_df[col_name].median()
            merged_df[col_name] = merged_df[col_name].fillna(median_log2R)

            min = merged_df[col_name].min()
            max = merged_df[col_name].max() 
            if min < global_min: global_min = min
            if max > global_max: global_max = max
            
            final_df = merged_df[['loc', col_name]].rename(columns = {col_name : cna_col.split(".")[1]}) # DF with absolute location of start and log2 ratio values 
            final_df.name = patient_id.split("_merged_")[0]

            df_list.append(final_df)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    if df_list:
        print(f"Processed {len(df_list)} people successfully")
        return df_list, global_min, global_max
    else:
        print("No valid patient data processed.")


def sample_single_dataframe(df, bins, sample_size, num_samples):
    
    num_mb = df.shape[0] 

    # Determine maximum start index for sampling
    max_start_index = num_mb - sample_size

    start_indices = np.random.randint(0, max_start_index + 1, size=num_samples)

    # Create indexes matrix 
    idx_matrix = start_indices[:, None] + np.arange(sample_size)

    # Extract logR values in a matrix num_samples x sample_size
    samples_matrix = df['logR'].values[idx_matrix]

    def get_rel_freq(row):
        counts, _ = np.histogram(row, bins=bins)
        total_counts = np.sum(counts)
        return counts / total_counts # NOTE: check this normalization
    
    # Compute relative frequency for each sample
    freq_matrix = np.apply_along_axis(get_rel_freq, 1, samples_matrix)

    # Create Dataframe 
    bin_labels = [f"bin_{i}" for i in range(len(bins) -1)]
    freq_df = pd.DataFrame(freq_matrix, columns=bin_labels)
    freq_df.name = df.name

    return freq_matrix, freq_df

def sample_dataframe_list(df_list, global_min, global_max, sample_size=30, num_samples=1000, num_bars=20):
     
    # Create histogram bins
    num_bars = 20
    bins = np.linspace(global_min, global_max, num_bars + 1)

    n_jobs = len(df_list) if len(df_list) < os.cpu_count() else os.cpu_count()
    results_list = Parallel(n_jobs)(
        delayed(sample_single_dataframe)(df, bins, sample_size, num_samples) for df in df_list
    )

    return results_list
