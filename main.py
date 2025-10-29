import pysam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from BamReader import BamReader
from sklearn.ensemble import IsolationForest

def plot_distributions(df: pd.DataFrame, output_filename: str = "features_histogram.png", left_quantile: float = 0.01, right_quantile: float = 0.99):

    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns

    # Calculate number of rows and columns for subplots
    n_cols = 2
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 6))

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    for i, col_name in enumerate(numeric_cols):
        ax = axes[i] # current axis
        
        # Calculate quantiles
        lower = df[col_name].quantile(left_quantile)
        upper = df[col_name].quantile(right_quantile)
        
        # Create histogram
        df[col_name].hist(
            ax=ax,
            bins=30,
            range=(lower, upper)
        )
        ax.set_title(f"'{col_name}' distribution ({left_quantile*100:.1f}%-{right_quantile*100:.1f}%)")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(train_folder + "/" + output_filename)


def create_and_save_df(bam_folder):
    # Initialize BamReader for training data
    bam_reader = BamReader(bam_folder)

    # Create patients Data Frame
    bam_reader.process_bam_folder()
    joined_patients_df = pd.concat(bam_reader.patients_dfs, ignore_index=True) # Concatenate all patient DataFrames
    file_path = bam_folder + "/extracted_reads.csv"
    joined_patients_df.to_csv(file_path, index=False)
    return bam_reader.patients_dfs, joined_patients_df

    
if __name__ == "__main__":
    
    train_folder = "BAM_Files/train" 
    
    df_list,train_patients_df = create_and_save_df(train_folder)
    
    plot_distributions(train_patients_df)
    
    train_patients_df = pd.read_csv(train_folder + "/extracted_reads.csv")  

    #NOTE: maybe set the index of the DataFrame?
    
    # Fill NaN values with 0 NOTE: must be fill with median
    train_patients_df.fillna(0, inplace=True)

    if train_patients_df.empty:
        print("No patient data available to process.")
    else:
        print("\n First 5 rows of the patients features DataFrame:")
        print(train_patients_df.head())

        # 3. Train Isolation Forest
        print("\n IF training... ")
        
        # Initialize the model
        iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42, n_jobs=-1)
        
        # Train (first column excluded because is ID)
        iso_forest.fit(train_patients_df.iloc[:, 1:])
        
        test_folder = "BAM_Files/test"
        test_patients_dfs = create_and_save_df(test_folder)[0]

        if len(test_patients_dfs) == 0:
            print("No test patient data available to process.")

        result_dfs = []
        for patient_df in test_patients_dfs:
            # Fill NaN values with 0 NOTE: must be fill with median
            patient_df.fillna(0, inplace=True)
            predictions = iso_forest.predict(patient_df.iloc[:, 1:])
            scores = iso_forest.decision_function(patient_df.iloc[:, 1:])
            row = {
                'patient_id': patient_df.iloc[0, 0],
                'n_anomalous_reads': sum(predictions == -1),
                'mean_anomaly_score': sum(scores)/len(scores)
            }
            result_dfs.append(pd.DataFrame([row]))
             
        results_df = pd.concat(result_dfs, ignore_index=True)
        print("\n First 5 rows of the result")
        print(results_df.head())
        results_df.to_csv("BAM_Files/anomaly_detection_results.csv", index=False)


