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

    # Create patients Data Frame
    bam_reader.process_bam_folder()
    joined_patients_df = pd.concat(bam_reader.patients_dfs, ignore_index=True) # Concatenate all patient DataFrames
    file_path = bam_folder + "/extracted_reads.csv"
    joined_patients_df.to_csv(file_path, index=False)
    return bam_reader.patients_dfs, joined_patients_df

def plot_cna_distributions():
    input_folder = "CHROMOSOME_INSTABILITY_hg38"
    output_folder = "cna_density_plots/tumor"
    col_value_name = "logR"

    chroms = [str(i) for i in range(1, 23)] + ['X', 'Y']

    path_to_files = input_folder + "/*.c6.csv"
    file_list = glob.glob(path_to_files)
    print(f"Found {len(file_list)} '.c6.csv' files in folder '{input_folder}'.")

    for file in file_list:

        
        df = pd.read_csv(file)
        fig, axes = plt.subplots(6, 4, figsize=(20,24))
        axes = axes.flatten()
        fig.suptitle(f"'{col_value_name}' densities for patient {file}", fontsize=20)
        for i, chrom in enumerate(chroms):
            ax = axes[i]

            subset = df[df['chr']== chrom]
            full_col_name = file.replace(".cna.seg.c6.csv", '').split("/")[-1] + "." + col_value_name 

            if not subset.empty:
                sns.kdeplot(
                    data=subset,
                    x=full_col_name,
                    fill=True,
                    ax=ax,
                    color='blue',
                    alpha=0.6,
                    linewidth=1.5
                )

                median_logR = subset[full_col_name].median()
                ax.axvline(median_logR, color='red', linestyle='--', alpha=0.5, label=f'Median: {median_logR:.2f}')

                ax.set_title(f'Chr {chrom} (n={len(subset)})', fontweight='bold')
                ax.set_xlabel('')
                ax.set_ylabel('Density')
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', color='gray')
                ax.set_title(f'Chr {chrom}', color='gray')
                ax.set_xlabel('')
                ax.set_ylabel('')

            ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        output_filename = file.split("/")[-1].replace(".cna.seg.c6.csv", f"_'{col_value_name}'_density_plot.png")
        output_path = os.path.join(output_folder, output_filename)
        plt.savefig(output_path, dpi = 150)
        plt.close()
        


if __name__ == "__main__":
    
    train_folder = "CHROMOSOME_INSTABILITY_hg38/HEALTHY" 
    
    train_patients_df = csr.process_folder(train_folder)
    #train_patients_df = pd.read_csv(train_folder + "/dataframe.csv", index_col='Patient_ID')  
    

    if train_patients_df.empty:
        print("No patient data available to process.")
    else:
        #Train Isolation Forest

        print("\n IF training... ")
        
        # Initialize the model
        iso_forest = IsolationForest(n_estimators=1000, contamination=10e-4, random_state=42, n_jobs=-1)
        
        # Patient_ID is the index: fit, predict and decision_function exclude it 
        iso_forest.fit(train_patients_df)
        
        test_folder = "CHROMOSOME_INSTABILITY_hg38"
        test_patients_df= csr.process_folder(test_folder)
        #test_patients_df = pd.read_csv(test_folder + "/dataframe.csv", index_col='Patient_ID')  

        predictions = iso_forest.predict(test_patients_df)
        anomaly_scores = iso_forest.decision_function(test_patients_df)

        results_df = pd.DataFrame({'Patient_ID': test_patients_df.index, 'Prediction': predictions, 'Anomaly_Score': anomaly_scores}) 

        print("Sum of anomalies found: " + str(sum(predictions == -1)))
        print("Sum of anomaly scores < 0: " + str(sum(x for x in anomaly_scores if x < 0)))
        print("Sum of normal scores >= 0: " + str(sum(x for x in anomaly_scores if x >= 0)))

        print("\n First 5 rows of the result")
        print(results_df.head())
        results_df.to_csv(test_folder + "/anomaly_detection_results3.csv", index=False)


