import pysam
import pandas as pd
import numpy as np
from BamReader import BamReader
from sklearn.ensemble import IsolationForest


if __name__ == "__main__":
    
    bam_folder = "BAM_Files/BAM_VERSION" 
    # 1. Initialize BamReader
    bam_reader = BamReader(bam_folder)

    # 2. Create patients Data Frame
    bam_reader.process_bam_folder()
    patients_df = bam_reader.patients_df

    if not patients_df.empty:
        print("\n First 5 rows of the patients features DataFrame:")
        print(patients_df.head())

        
        # 3. Apply Isolation Forest to detect anomalous patients
        print("\n IF execution... ")
        
        # Initialize the model
        iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42, n_jobs=-1)
        
        # Train and get predictions (first columns are ID columns)
        predictions = iso_forest.fit_predict(patients_df.iloc[:, 1:])
        
        # Add predictions to the DataFrame
        patients_df['is_anomaly'] = predictions
        
        # Show anomalous patients
        anomalous_patients = patients_df[patients_df['is_anomaly'] == -1]
        
        print(f"\nIdentified {len(anomalous_patients)} anomalous patients:")
        print(anomalous_patients)
        
    else:
        print("No patient data available to process.")

