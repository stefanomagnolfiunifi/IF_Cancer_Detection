import pysam
import pandas as pd
import glob
import os

class BamReader:

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.patients_df = None


    # Return a DataFrame where each row is a patient and columns are aggregated features.
    def process_bam_folder(self):
        
        # Search every BAM file in the specified folder
        bam_files = glob.glob(os.path.join(self.folder_path, '*.bam'))
        
        all_patients_features_df = pd.DataFrame()

        print(f"Found {len(bam_files)} file BAM in the folder: {self.folder_path}")

        '''
        for bam_file_path in bam_files:
            print(f"Processing {os.path.basename(bam_file_path)}...")
            
            # 1. Extract features for every read in the BAM file
            single_patient_features_df = self.extract_features_from_bam(bam_file_path)
            
            if single_patient_features_df.empty:
                print(f"  No valid data in {os.path.basename(bam_file_path)}.")
                continue
            
            # 2. Add fragments of the patient to the overall DataFrame
            all_patients_features_df.concat(single_patient_features_df, ignore_index=True)
            
        '''
            
        # 1. Extract features for every read in the BAM file
        single_patient_features_df = self.extract_features_from_bam(bam_files[0])
        
        if single_patient_features_df.empty:
            print(f"  No valid data.")
            
        
        

        #TODO: maybe set the index of the DataFrame?
        
        # Fill NaN values with 0
        all_patients_features_df.fillna(0, inplace=True)
        
        self.patients_df = all_patients_features_df



    #Extract numeric features for every read of the BAM file.
    def extract_features_from_bam(self,bam_path):
        
        bamfile = pysam.AlignmentFile(bam_path, "rb")
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

            rows.append({
                "read_id" : read.query_name,
                "sequence" : seq
            })
             
        bamfile.close()

        # Create the DataFrame
        feature_df = pd.DataFrame(rows)

        feature_df.to_csv("extracted_reads.csv", index=False)

        return feature_df

    #Calculate aggregated features (mean and std) for a patient based on per-read features.
    def aggregate_features_for_patient(self,per_read_df):
    
        if per_read_df.empty:
            return None

        aggregated_features = {}
        features_to_aggregate = ['mapq', 'insert_size', 'mismatches', 'soft_clipping']
        
        for feature in features_to_aggregate:
            # Misure di tendenza centrale e dispersione
            aggregated_features[f'{feature}_mean'] = per_read_df[feature].mean()
            aggregated_features[f'{feature}_std'] = per_read_df[feature].std()
            
        return aggregated_features