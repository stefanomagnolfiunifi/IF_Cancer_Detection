import pysam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from BamReader import BamReader
from sklearn.ensemble import IsolationForest

def plot_distributions(df: pd.DataFrame, columns_to_plot: list = None, output_filename: str = "distributions_plot.png",max_cols_per_row: int = 3, clip_quantile: float = 0.1):

    # Select columns to plot
    if columns_to_plot is None:
        # Default: every numeric column
        columns_to_plot = df.select_dtypes(include=['number']).columns
        if len(columns_to_plot) == 0:
            print("No column to plot found.")
            return
    else:
        # Check if columns exist in DataFrame
        columns_to_plot = [col for col in columns_to_plot if col in df.columns]
        if len(columns_to_plot) == 0:
            print("No column specified is in the DataFrame.")
            return

    num_plots = len(columns_to_plot)
    
    # Set the layout of the subplots
    num_rows = math.ceil(num_plots / max_cols_per_row)
    
    # Create figure and axes
    fig_height = num_rows * 4 
    fig_width = max_cols_per_row * 5
    
    fig, axes = plt.subplots(nrows=num_rows, ncols=max_cols_per_row, figsize=(fig_width, fig_height))
    
    # Transform axes to a flat list for easy iteration
    if num_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes] # Convert single axis to list for consistency

    # Plot each column
    for i, col_name in enumerate(columns_to_plot):
        ax = axes[i] # Select the appropriate axis
        data_to_plot = df[col_name].dropna()  # Remove NaN values for plotting
        plot_title = f"Distribuzione di '{col_name}'"
        try:
            # Check if the column is numeric
            if pd.api.types.is_numeric_dtype(df[col_name]):

                if clip_quantile is not None and 0 < clip_quantile < 0.5:
                    q_low = data_to_plot.quantile(clip_quantile)
                    q_high = data_to_plot.quantile(1 - clip_quantile)
                    
                    # Filtra i dati per il plot
                    data_to_plot = data_to_plot[(data_to_plot >= q_low) & (data_to_plot <= q_high)]
                    plot_title += f"\n(Mostra {100 - 2 * (clip_quantile * 100):.0f}% centrale)"

                # Plot a histogram with KDE
                sns.histplot(data_to_plot, kde=True, ax=ax, bins=30)
                ax.set_title(f"Distribuzione di '{col_name}' (Numerica)")
                ax.set_ylabel("Frequenza")
            
            # Check if the column is categorical
            elif pd.api.types.is_object_dtype(df[col_name]) or df[col_name].nunique() < 50:
                # Plot bar chart
                sns.countplot(x=df[col_name], ax=ax, order=df[col_name].value_counts().index)
                ax.set_title(plot_title)
                ax.set_ylabel("Conteggio")
                
                # Rotate labels if too many categories
                if df[col_name].nunique() > 10:
                    ax.tick_params(axis='x', rotation=45)
            else:
                ax.text(0.5, 0.5, f"Tipo di colonna '{col_name}' non supportato per plot", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        except Exception as e:
            print(f"Errore nel plottare '{col_name}': {e}")
            ax.text(0.5, 0.5, f"Impossibile plottare '{col_name}'", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        
        ax.set_xlabel(col_name)

    # Clean up any unused axes
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j]) # Rimuove l'asse non utilizzato

    # Save the figure
    plt.tight_layout() # Impedisce alle etichette di sovrapporsi
    plt.savefig(output_filename)
    plt.close(fig) # Chiude la figura per liberare memoria
    
    print(f"Grafico delle distribuzioni salvato come '{output_filename}'")
    
if __name__ == "__main__":
    
    train_folder = "BAM_Files/train" 
    # 1. Initialize BamReader for training data
    train_bam_reader = BamReader(train_folder)

    # 2. Create patients Data Frame
    train_bam_reader.process_bam_folder()
    plot_distributions(train_bam_reader.patients_dfs[3], output_filename="BAM_Files/patient_0_HEALTY_feature_distributions.png")
    train_patients_df = pd.concat(train_bam_reader.patients_dfs, ignore_index=True) # Concatenate all patient DataFrames

    #train_patients_df.to_csv("BAM_Files/train/extracted_reads.csv", index=False)
            
    #NOTE: maybe set the index of the DataFrame?
    
    # Fill NaN values with 0 NOTE: must be fill with median
    #train_patients_df.fillna(0, inplace=True)

    if train_patients_df.empty:
        print("No patient data available to process.")
    else:
        print("\n First 5 rows of the patients features DataFrame:")
        print(train_patients_df.head())

        # 3. Train Isolation Forest
        print("\n IF training... ")
        
        # Initialize the model
        iso_forest = IsolationForest(n_estimators=20, contamination=1e-6, random_state=42, n_jobs=-1)
        
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


