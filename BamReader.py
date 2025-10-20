import pysam
import pandas as pd
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

class BamReader:

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.patients_df = None


    # Return a DataFrame where each row is a patient and columns are aggregated features.
    def process_bam_folder(self):
        
        # Search every BAM file in the specified folder
        bam_files = glob.glob(os.path.join(self.folder_path, '*.bam'))
        
        print(f"Found {len(bam_files)} file BAM in the folder: {self.folder_path}")

        feature_dfs = []

        with ProcessPoolExecutor(max_workers=32) as executor:
            futures = {
                # 1. Extract features for every read in the BAM file
                executor.submit(self.extract_features_from_bam, bam_file_path): bam_file_path
                for bam_file_path in bam_files
            }

            for future in as_completed(futures):
                bam_file_path = futures[future]
                print(f"Processing {os.path.basename(bam_file_path)}...")
                try:
                    single_patient_features_df = future.result()
                    if single_patient_features_df.empty:
                        print(f"  No valid data in {os.path.basename(bam_file_path)}.")
                        continue
                    # 2. Add fragments of the patient to the overall DataFrame
                    feature_dfs.append(single_patient_features_df)
                    print(f"Processed {os.path.basename(bam_file_path)} successfully.")
                except Exception as e:
                    print(f"Error processing {os.path.basename(bam_file_path)}: {e}")
        
        if not feature_dfs:
            print("No valid BAM files processed.")
            self.patients_df = pd.DataFrame()
            return
        
        all_patients_features_df = pd.concat(feature_dfs, ignore_index=True)
            
        #NOTE: maybe set the index of the DataFrame?
        
        # Fill NaN values with 0
        all_patients_features_df.fillna(0, inplace=True)

        all_patients_features_df.to_csv("BAM_Files/extracted_reads.csv", index=False)
        
        self.patients_df = all_patients_features_df
    

#Extract numeric features for every read of the BAM file.
def extract_features_from_bam(bam_path):
        
        bamfile = pysam.AlignmentFile(bam_path, "rb")
        file_name = (os.path.basename(bam_path)).split('_')[0]
        k = 3
        rows = []

        # Iterate over each read in the BAM file
        for read in bamfile.fetch():

            '''

            # Ignore unmapped reads and reads with low mapping quality
            if read.is_unmapped or read.mapping_quality < 10:
                continue

        
            mapq = read.mapping_quality

            insert_size = abs(read.template_length) if read.is_paired and read.is_proper_pair else 0
            
            # 3. Number of Mismatches
            try:
                mismatches = read.get_tag('NM')
            except KeyError:
                mismatches = 0 # Valore di default se il tag manca

            # 4. Soft Clipping
            soft_clipping_count = 0
            if read.cigartuples:
                # CIGAR 'S' (4) correspond to soft clipping
                for op, length in read.cigartuples:
                    if op == 4:
                        soft_clipping_count += length
            
            read_name = read.query_name

            features_list.append([
                read_name,
                mapq,
                insert_size,
                mismatches,
                soft_clipping_count
            ])
            '''
            seq = read.query_sequence
            if seq is None:
                continue

            met_cyt_ratio = calculate_methylated_cytosine_ratio(read)
            
            rows.append({
                "read_file" : file_name,
                "length": len(seq),
                "methilated_cytosine_ratio": met_cyt_ratio
            })
             
        bamfile.close()

        # Create the DataFrame
        feature_df = pd.DataFrame(rows)

        return feature_df
    
#Calculate citosine metilate ratio in given DNA sequence
def calculate_methylated_cytosine_ratio(read):
    seq = read.query_sequence
    if seq is None or "MM" not in dict(read.tags):
        return None

    tags = dict(read.tags)
    mm = tags["MM"]
    ml = tags.get("ML", [])

    n_cyt = seq.count("C")
    if n_cyt == 0:
        return 0

    i=0
    #NOTE: would be worth using numpy array?
    met_cyt_confidences = []
    # C+m indicates methylated cytosines
    for mod in mm.split(";"):
        if not mod:
            continue
        n_positions = len(mod.split(",")[1:])
        if mod.startswith("C+m"):
            met_cyt_confidences.extend(ml[i:i+n_positions])
    
        i += n_positions

    #NOTE: other approach could be sum(met_cyt_confidences) / (255 * len(met_cyt_confidences))
    return sum(met_cyt_confidences) / (255 * n_cyt) 

#Calculate aggregated features (mean and std) for a patient based on per-read features.
def aggregate_features_for_patient(per_read_df):


    if per_read_df.empty:
        return None

    aggregated_features = {}
    features_to_aggregate = ['mapq', 'insert_size', 'mismatches', 'soft_clipping']
    
    for feature in features_to_aggregate:
        # Misure di tendenza centrale e dispersione
        aggregated_features[f'{feature}_mean'] = per_read_df[feature].mean()
        aggregated_features[f'{feature}_std'] = per_read_df[feature].std()
        
    return aggregated_features