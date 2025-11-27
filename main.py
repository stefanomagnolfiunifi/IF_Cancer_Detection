import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import glob
import os
#from BamReader import BamReader
import CnaSegReader as csr
from sklearn.ensemble import IsolationForest

def plot_distributions(df: pd.DataFrame, file_path, left_quantile: float = 0.01, right_quantile: float = 0.99):

    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns

    # Calculate number of rows and columns for subplots
    n_cols = 3
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 12))

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    for i, col_name in enumerate(numeric_cols):
        ax = axes[i] # current axis
        
        # Calculate quantiles
        lower = df[col_name].quantile(left_quantile)
        upper = df[col_name].quantile(right_quantile)

        weights = np.ones(len(df)) / len(df)
        
        # Create histogram
        df[col_name].hist(
            ax=ax,
            range=(lower, upper),
            weights = weights
        )
        ax.set_title(f"'{col_name}' distribution ({left_quantile*100:.1f}%-{right_quantile*100:.1f}%)")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(file_path)


def create_and_save_df(bam_folder):


    # Initialize BamReader for training data
    bam_reader = BamReader(bam_folder)

    # Create persons Data Frame
    bam_reader.process_bam_folder()
    joined_persons_df = pd.concat(bam_reader.persons_dfs, ignore_index=True) # Concatenate all person DataFrames
    file_path = bam_folder + "/extracted_reads.csv"
    joined_persons_df.to_csv(file_path, index=False)
    return bam_reader.persons_dfs, joined_persons_df

def plot_cna_distributions():
    
    healthy_folder = "CHROMOSOME_INSTABILITY_hg38/HEALTHY"  
    tumor_folder = "CHROMOSOME_INSTABILITY_hg38"  
    output_folder = "cna_density_plots/chromosomes_with_total"
    col_value_name = "logR"

    
    healthy_files = sorted(glob.glob(os.path.join(healthy_folder, "*.c6.csv")))
    tumor_files = sorted(glob.glob(os.path.join(tumor_folder, "*.c6.csv")))

    selected_files = healthy_files + tumor_files
    
    print(f"Found files: {len(healthy_files)} healty, {len(tumor_files)} tumor.")
    
    if not selected_files:
        print("No file found.")
        return

    chroms = [str(i) for i in range(1, 23)] + ['X', 'Y']

    # chromosomes cicle
    for chrom in chroms:
        print(f"\n--- Plotting Chromosome {chrom} ---")
        

        cleaned_data_list = [] 
        global_min = float('inf')
        global_max = float('-inf')
        
        # Read and clean data from each file
        for file_idx, file_path in enumerate(selected_files):
            try:
                df = pd.read_csv(file_path)
                
                # Take the name of the column
                clean_name = file_path.replace(".cna.seg.c6.csv", '').split("/")[-1]        
                full_col_name = clean_name + "." + col_value_name 

                subset = df[df['chr'] == chrom]
                
                valid_series = None
                reason = "N/A"

                if not subset.empty and full_col_name in subset.columns:
                    data = subset[full_col_name].dropna()
                    
                    if not data.empty:
                        
                        q01 = data.quantile(0.01)
                        q99 = data.quantile(0.99)
                        
                        # Exclude outliers
                        data_clean = data[(data >= q01) & (data <= q99)]
                        
                        if not data_clean.empty:
                            valid_series = data_clean

                            # Update specific chromosome min/max values
                            curr_min = data_clean.min()
                            curr_max = data_clean.max()
                            if curr_min < global_min: global_min = curr_min
                            if curr_max > global_max: global_max = curr_max
                        else:
                            reason = "All Outliers"
                    else:
                        reason = "Empty Data"
                else:
                    reason = "No Chr Data"

                # Add to cleaned data list
                cleaned_data_list.append({
                    'data': valid_series,
                    'name': clean_name,
                    'group': 'Healthy' if file_idx < 8 else 'Tumor',
                    'reason': reason
                })
                
            except Exception as e:
                print(f"Error occurred while reading {file_path}: {e}")
                cleaned_data_list.append({'data': None, 'name': "Error", 'reason': str(e)})

        combined_healthy = pd.concat([person['data'] for person in cleaned_data_list if person['group'] == 'Healthy' and person['data'] is not None])
        combined_tumor = pd.concat([person['data'] for person in cleaned_data_list if person['group'] == 'Tumor' and person['data'] is not None])
        cleaned_data_list = cleaned_data_list[:22]
        
        cleaned_data_list.append({
            'data': combined_healthy,
            'name': 'Combined_Healthy',
            'group': 'Healthy',
            'reason': 'N/A'
        })
        cleaned_data_list.append({
            'data': combined_tumor,
            'name': 'Combined_Tumor',
            'group': 'Tumor',
            'reason': 'N/A'
        })

        # If no valid data found, skip plotting
        if global_min == float('inf'):
            print(f"No data for Chr {chrom}. Skip.")
            continue

        # Create subplots
        fig, axes = plt.subplots(6, 4, figsize=(20, 24))
        axes = axes.flatten()
        
        fig.suptitle(f"Chromosome {chrom} - {col_value_name} Distribution (Filtered 1-99%)", fontsize=20)
        
        # One plot for each person
        for i, person in enumerate(cleaned_data_list):
            if i >= len(axes): break 
            
            ax = axes[i]
            
            if person['data'] is not None:
                # Blue color for Healthy, Orange for Tumor
                colore = 'tab:blue' if person['group'] == 'Healthy' else 'tab:orange'
                label_grp = person['group']
                
                sns.kdeplot(
                    data=person['data'],
                    fill=True,
                    ax=ax,
                    color=colore,
                    alpha=0.6,
                    linewidth=1.5
                )
                
                # Global x-axis limits
                ax.set_xlim(global_min, global_max)
                
                ax.set_title(f"{person['name']} ({label_grp})", fontsize=10, fontweight='bold')
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.grid(True, linestyle=':', alpha=0.6)
            else:
                # Empty plot with reason
                ax.text(0.5, 0.5, person['reason'], ha='center', va='center', color='gray')
                ax.set_title(person.get('name', 'N/A'), fontsize=10, color='gray')
                ax.set_axis_off()

        # Remove extra axes
        for j in range(len(cleaned_data_list), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        out_file = os.path.join(output_folder, f"Chr_{chrom}_density.png")
        plt.savefig(out_file, dpi=150)
        plt.close(fig)
        print(f"Plot saved: {out_file}")
        


if __name__ == "__main__":
    
    train_folder = "CHROMOSOME_INSTABILITY_hg38/train" 

    
    train_df_list, global_min, global_max = csr.process_folder(train_folder)
    sample_size = 30
    num_samples = 1000
    num_bars = 20
    results_list = csr.sample_dataframe_list(train_df_list, global_min, global_max, sample_size, num_samples, num_bars)    

    matrix_list = zip(*results_list)[0]
    # Stack all the matrixes vertically
    final_matrix = np.vstack(matrix_list)

    # Create Dataframe 
    bin_labels = [f"bin_{i}" for i in range(num_bars)]
    train_df = pd.DataFrame(final_matrix, columns=bin_labels)

    print(f"Final DataFrame shape: {train_df.shape}")
    print(f"Final Dataframe head:\n{train_df.head()}")

    if train_df.empty:
        print("No person data available to process.")
    else:
        #Train Isolation Forest

        print("\n IF training... ")
        
        # Initialize the model
        iso_forest = IsolationForest(n_estimators=1000, contamination=10e-4, random_state=42, n_jobs=-1)
        # Fit the model
        iso_forest.fit(train_df)
        
        test_folder = "CHROMOSOME_INSTABILITY_hg38/test"
        test_df_list= csr.process_folder(test_folder)[0] # Global min and max are not used because we use the same as training to create the bins
        results_list = csr.sample_dataframe_list(test_df_list, global_min, global_max, sample_size, num_samples, num_bars)
        df_list = zip(*results_list)[1]
        row = []

        for df in df_list:
            anomaly_scores = iso_forest.decision_function(df)
            print(f"Mean anomaly score for {df.name}: {np.mean(anomaly_scores)}")
            print(f"Median anomaly score for {df.name}: {np.median(anomaly_scores)}")
            row.append({
                'person_id' : df.name,
                'mean_anomaly_score' : np.mean(anomaly_scores),
                'median_anomaly_score' : np.median(anomaly_scores)
            })

        results_df = pd.DataFrame(row)
        print("\n First 5 rows of the result")
        print(results_df.head())
        results_df.to_csv(test_folder + "/anomaly_detection_results.csv", index=False)


