import pandas as pd
import numpy as np
import glob
import os


# Return a DataFrame row with genome bins as columns and their corresponding log2 ratio values    
def process_folder(folder_path):
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

    print("Creating master mask...")
    for chrom in chromosomes:
        chrom_length = (chrom_lengths_hg38[chrom] // bin_size) * bin_size # Take only full bins
        for start in range(1, chrom_length+1, bin_size):
            end = start + bin_size - 1
            bin_name = f"{chrom}:{start}-{end}"
            master_bins.append((chrom, start, end,bin_name))
        
    df_master = pd.DataFrame(master_bins, columns=['chr', 'start', 'end', 'bin_name'])
    print(f"Master mask created with {len(df_master)} bins.")

    path_to_files = folder_path + "/*.cna.seg"
    file_list = glob.glob(path_to_files)
    print(f"Found {len(file_list)} .cna.seg files in folder '{folder_path}'.")

    all_log2R_series = []

    for file_path in file_list:
        try:
            patient_id = os.path.basename(file_path).replace('.cna.seg', '')

            patient_df = pd.read_csv(file_path, sep='\t', dtype={'chr': str}, na_values=['NA', 'inf'])

            col_name = patient_id + '.logR'

            meaningful_df = patient_df[['chr', 'start', 'end', col_name]]

            # Merge with master bins to ensure all bins are represented
            merged_df = pd.merge(df_master, meaningful_df, on=['chr', 'start', 'end'], how='left')

            # Fill NaN values with median log2 ratio
            median_log2R = patient_df[col_name].median()
            merged_df[col_name] = merged_df[col_name].fillna(median_log2R)
            
            patient_series = merged_df.set_index('bin_name')[col_name]
            patient_series.name = os.path.basename(file_path).replace('_merged_al38_filofilter_srt.cna.seg', '')

            all_log2R_series.append(patient_series)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    if all_log2R_series:
        all_patients_df = pd.concat(all_log2R_series, axis=1).T #Transpose to have patients as rows
        all_patients_df.index.name = 'Patient_ID'
        all_patients_df.columns.name =  None
        all_patients_df.to_csv(folder_path + "/dataframe.csv", index=True)
        print(f"Processed {len(all_log2R_series)} patients successfully and saved to '{folder_path}/dataframe.csv'")
        print(all_patients_df.head())
        return all_patients_df
    else:
        print("No valid patient data processed.")