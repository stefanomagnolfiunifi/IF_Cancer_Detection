import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import glob
import os


# Return a list of DataFrames with genome bins as columns and their corresponding log2 ratio values    
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
    file_list = sorted(glob.glob(path_to_files))
    print(f"Found {len(file_list)} '{extension}' files in folder '{folder_path}'.")

    df_list = []

    for file_path in file_list:
        try:
            patient_id = os.path.basename(file_path).replace(extension, '')

            df = pd.read_csv(file_path, sep='\t', dtype={'chr': str}, na_values=['NA', 'inf'])

            col_name = patient_id + cna_col if extension == ".cna.seg" else 'log2_TNratio_corrected'

            meaningful_df = df[['chr', 'start', 'end', col_name]]

            # Merge with master bins to ensure all bins are represented
            merged_df = pd.merge(df_master, meaningful_df, on=['chr', 'start', 'end'], how='left')
            
            '''
            # Fill NaN values with median log2 ratio
            median_log2R = df[col_name].median()
            merged_df[col_name] = merged_df[col_name].fillna(median_log2R)
            '''
            
            final_df = merged_df[['loc', col_name]].rename(columns = {col_name : cna_col.split(".")[1]}) # DF with absolute location of start and log2 ratio values 
            final_df.attrs['name'] = patient_id.split("_merged_")[0]

            df_list.append(final_df)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    if df_list:
        print(f"Processed {len(df_list)} people successfully")
        return df_list
    else:
        print("No valid patient data processed.")

def plot_sample_densities(samples_matrix, patient_id, x_limits=(-1, 1)):
    n_rows, n_cols = 6,4
    n_plots = n_rows * n_cols
    lines_per_plot = 10
    total_samples_needed = n_plots * lines_per_plot
    
    # Color palette
    colors = sns.color_palette("tab10", lines_per_plot) 

    # Subset choice
    if len(samples_matrix) >= total_samples_needed:
        indices = np.random.choice(len(samples_matrix), total_samples_needed, replace=False)
        subset = samples_matrix[indices]
    else:
        subset = samples_matrix[:total_samples_needed]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 30), sharex=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        start = i * lines_per_plot
        end = start + lines_per_plot
        # Take the batch of samples for this subplot
        batch = subset[start:end]
        
        for row, color in zip(batch, colors):

            if np.std(row) < 1e-6:
                # With all identical values, plot a vertical line at the mean
                ax.axvline(x=np.mean(row), color="black", alpha=0.5, linestyle='--', linewidth=1.5)
            else:
                try:
                    sns.kdeplot(
                        data=row, 
                        ax=ax, 
                        fill=False, 
                        clip=x_limits,
                        color=color, 
                        alpha=0.6,        
                        linewidth=1.2,
                        warn_singular=False
                    )
                except Exception:
                    pass
        
        ax.tick_params(labelbottom=True)
        ax.set_xlim(x_limits)
        ax.grid(True, linestyle=':', alpha=0.3)
        ax.set_ylabel('')
        
        ax.set_title(f"Batch {i+1}", fontsize=10)
    
    fig.suptitle(f"Sample densities of {patient_id}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    filename = f"cna_density_plots/sample_densities/{patient_id}.png"
    plt.savefig(filename, dpi=150)
    plt.close(fig)

def plot_samples_with_gmm(samples_matrix, patient_id, x_limits=(-1, 1)):
    n_rows, n_cols = 6,4
    n_plots = n_rows * n_cols

    # Subset choice
    if len(samples_matrix) >= n_plots:
        indices = np.random.choice(len(samples_matrix), n_plots, replace=False)
        subset = samples_matrix[indices]
    else:
        subset = samples_matrix[:n_plots]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 30), sharex=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes):    
        row = subset[i]
        gmm = GaussianMixture(n_components=2, covariance_type='full', n_init=5, max_iter=200, random_state=42)
        gmm.fit(row.reshape(-1, 1))

        means = gmm.means_.flatten()
        covariances = gmm.covariances_.flatten()
        weights = gmm.weights_.flatten()

        idx_sorted = np.argsort(means)
        means_sorted = means[idx_sorted]
        covariances_sorted = covariances[idx_sorted]
        weights_sorted = weights[idx_sorted]
        
        try:
            sns.kdeplot(
                data=row, 
                ax=ax, 
                fill=False, 
                clip=x_limits,
                color="red",         
                linewidth=1.2,
                warn_singular=False,
                label='KDE'
            )
            
            x_plot = np.linspace(x_limits[0], x_limits[1], 1000)

            g1 = norm.pdf(x_plot, loc=means_sorted[0], scale=np.sqrt(covariances_sorted[0])) * weights_sorted[0]
            g2 = norm.pdf(x_plot, loc=means_sorted[1], scale=np.sqrt(covariances_sorted[1])) * weights_sorted[1]

            ax.plot(x_plot, g1, 'b:', linewidth=0.5, label=f'm: {means_sorted[0]:.2f}, w: {weights_sorted[0]:.2f}')
            ax.fill_between(x_plot, g1, color='blue', alpha=0.3)
            ax.plot(x_plot, g2, 'g:', linewidth=0.5, label=f'm: {means_sorted[1]:.2f}, w: {weights_sorted[1]:.2f}')
            ax.fill_between(x_plot, g2, color='green', alpha=0.3)

            '''
            g_total = g1 + g2
            ax.plot(x_plot, g_total, 'k--', linewidth=1.2, label='GMM Total')
            ax.fill_between(x_plot, g_total, color='gray', alpha=0.3)
            '''
        except Exception as e:
            print(f"Error plotting {patient_id}: {e}")
            
        
        ax.legend()
        ax.tick_params(labelbottom=True)
        ax.set_xlim(x_limits)
        ax.grid(True, linestyle=':', alpha=0.3)
        ax.set_ylabel('')
        
        ax.set_title(f"Sample {i+1}", fontsize=10)
    
    fig.suptitle(f"Sample densities of {patient_id}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    filename = f"cna_density_plots/distinct_gmm/{patient_id}.png"
    plt.savefig(filename, dpi=150)
    plt.close(fig)


def sample_single_dataframe(df, sample_size, num_samples):
    
    values = df['logR'].values

    # Mask of Nan values (1 if NaN, 0 otherwise)
    is_nan = np.isnan(values).astype(int)

    # Valid convolution to find number of NaNs in each window of size sample_size
    nan_counts_in_window = np.convolve(is_nan, np.ones(sample_size, dtype=int), mode='valid')

    # Take only the start indices where there are no NaNs in the window
    valid_start_indices = np.where(nan_counts_in_window <= (sample_size / 10))[0]

    if len(valid_start_indices) == 0:
        print(f"No valid samples available for dataframe '{df.attrs['name']}'. Returning empty results.")
        return None
    if len(valid_start_indices) < num_samples:
        print(f"Only {len(valid_start_indices)} valid samples available, requested {num_samples}. Reducing number of samples.")
        num_samples = len(valid_start_indices)

    start_indices = np.random.choice(valid_start_indices, size=num_samples, replace=False)
    # Create indexes matrix 
    idx_matrix = start_indices[:, None] + np.arange(sample_size)

    # Extract logR values in a matrix with dim: num_samples x sample_size; the few NaNs are filled with median log2R
    median_log2R = df['logR'].median()
    samples_matrix = df['logR'].fillna(median_log2R).values[idx_matrix]

    def get_gmm_params(row):

        row_reshaped = row.reshape(-1, 1)

        try:
            gmm = GaussianMixture(n_components=2, covariance_type='full', n_init=1, max_iter=100, random_state=42)
            gmm.fit(row_reshaped)

            means = gmm.means_.flatten()
            covariances = gmm.covariances_.flatten()
            weights = gmm.weights_.flatten()
            
            # Sort to have always the same order of gaussians
            idx_sorted = np.argsort(means)
            means_sorted = means[idx_sorted]
            covariances_sorted = covariances[idx_sorted]
            weights_sorted = weights[idx_sorted]

            return np.concatenate([means_sorted, covariances_sorted])
        except Exception:
            return np.array([np.nan, np.nan, np.nan, np.nan])
    
    # Compute gmm for each sample
    features_matrix = np.apply_along_axis(get_gmm_params, 1, samples_matrix)

    # Create Dataframe 
    features_df = pd.DataFrame(features_matrix, columns=['mean1', 'mean2', 'var1', 'var2'])
    features_df.attrs['name'] = df.attrs['name']

    
    plot_samples_with_gmm(samples_matrix, df.attrs['name'], x_limits=(-1,1))
    return features_matrix, features_df

def sample_dataframe_list(df_list, sample_size=30, num_samples=1000):
     
    n_jobs = len(df_list) if len(df_list) < os.cpu_count() else os.cpu_count()
    results_list = Parallel(n_jobs)(
        delayed(sample_single_dataframe)(df, sample_size, num_samples) for df in df_list
    )
    return results_list
