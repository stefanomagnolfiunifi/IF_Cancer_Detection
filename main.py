import pysam
import pandas as pd
import numpy as np
from BamReader import BamReader
from sklearn.ensemble import IsolationForest


if __name__ == "__main__":
    
    train_folder = "BAM_Files/train" 
    # 1. Initialize BamReader for training data
    train_bam_reader = BamReader(train_folder)

    # 2. Create patients Data Frame
    train_bam_reader.process_bam_folder()
    train_patients_df = pd.concat(train_bam_reader.patients_dfs, ignore_index=True) # Concatenate all patient DataFrames

    #train_patients_df.to_csv("BAM_Files/extracted_reads.csv", index=False)
            
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
        iso_forest = IsolationForest(n_estimators=100, contamination=1e-6, random_state=42, n_jobs=-1)
        
        # Train (first column excluded because is ID)
        predictions = iso_forest.fit(train_patients_df.iloc[:, 1:])
        
        test_folder = "BAM_Files/test"
        test_bam_reader = BamReader(test_folder)
        test_bam_reader.process_bam_folder()
        test_patients_dfs = test_bam_reader.patients_dfs

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


