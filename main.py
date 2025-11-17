import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
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

    
if __name__ == "__main__":
    
    train_folder = "CHROMOSOME_INSTABILITY_hg38/HEALTHY" 
    
    train_patients_df = csr.process_folder(train_folder)
    #train_patients_df = pd.read_csv(train_folder + "/dataframe.csv")  
    

    if train_patients_df.empty:
        print("No patient data available to process.")
    else:
        #Train Isolation Forest

        print("\n IF training... ")
        
        # Initialize the model
        iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42, n_jobs=-1)
        
        # Patient_ID is the index: fit, predict and decision_function exclude it 
        iso_forest.fit(train_patients_df)
        
        test_folder = "CHROMOSOME_INSTABILITY_hg38"
        test_patients_df= csr.process_folder(test_folder)

        predictions = iso_forest.predict(test_patients_df)
        anomaly_scores = iso_forest.decision_function(test_patients_df)

        results_df = pd.DataFrame({'Patient_ID': test_patients_df.index, 'Prediction': predictions, 'Anomaly_Score': anomaly_scores}) 

        print("Predictions sum: " + str(sum(predictions == -1)))

        print("\n First 5 rows of the result")
        print(results_df.head())
        results_df.to_csv(test_folder + "/anomaly_detection_results.csv", index=False)


