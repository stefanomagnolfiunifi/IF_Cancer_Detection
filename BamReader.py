import pysam
import pandas as pd
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

class BamReader:

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.patients_dfs = []


    # Return a DataFrame where each row is a patient and columns are aggregated features.
    def process_bam_folder(self):
        
        # Search every BAM file in the specified folder
        bam_files = glob.glob(os.path.join(self.folder_path, '*.bam'))
        
        print(f"Found {len(bam_files)} file BAM in the folder: {self.folder_path}")

        with ProcessPoolExecutor(max_workers=32) as executor:
            futures = {
                # 1. Extract features for every read in the BAM file
                executor.submit(extract_features_from_bam, bam_file_path): bam_file_path
                for bam_file_path in bam_files
            }

            for future in as_completed(futures):
                bam_file_path = futures[future]
                #print(f"Processing {os.path.basename(bam_file_path)}...")
                try:
                    single_patient_features_df = future.result()
                    if single_patient_features_df.empty:
                        print(f"  No valid data in {os.path.basename(bam_file_path)}.")
                        continue
                    # 2. Add fragments of the patient to the overall DataFrame
                    self.patients_dfs.append(single_patient_features_df)
                    #print(f"Processed {os.path.basename(bam_file_path)} successfully.")
                except Exception as e:
                    print(f"Error processing {os.path.basename(bam_file_path)}: {e}")
    

#Extract numeric features for every read of the BAM file.
def extract_features_from_bam(bam_path):
        
        bamfile = pysam.AlignmentFile(bam_path, "rb")
        file_name = (os.path.basename(bam_path)).split('_')[0]
        k = 3
        rows = []

        # Iterate over each read in the BAM file
        for read in bamfile.fetch():

            # Ignore unmapped reads and reads with low mapping quality
            if read.is_unmapped or read.mapping_quality < 60:
                continue

            # Sequence
            seq = read.query_sequence
            if seq is None:
                continue
            
            # Mapping Quality
            mapq = read.mapping_quality
            if mapq == 255:
                continue

            # Relative Position
            midpoint = (read.reference_start + read.reference_end) // 2
            chrom = bamfile.get_reference_name(read.reference_id) # Reference chromosome
            rel_pos = midpoint / bamfile.get_reference_length(chrom) # Relative position in the chromosome

            # Number of Mismatches
            try:
                mismatches = read.get_tag('NM')
            except KeyError:
                mismatches = 0 

            # Methylated Cytosine Ratio
            met_cyt_ratio = calculate_methylated_cytosine_ratio(read)
            
            row_data = {
                "read_file" : file_name,
                "length": len(seq),
                "methilated_cytosine_ratio": met_cyt_ratio,
            }

            # CIGAR features
            cigar_features = extract_cigar_features(read) 
            row_data.update(cigar_features)
            rows.append(row_data) 
        bamfile.close()

        # Create the DataFrame
        feature_df = pd.DataFrame(rows)

        # Aggregate features for the patient
        feature_df = aggregate_features_for_patient(feature_df)

        return feature_df
    

# Extract CIGAR related features from the read
def extract_cigar_features(read):
    cigar_tuples = read.cigartuples
    if cigar_tuples is None:
        return None
    
    num_indels = sum(length for (op, length) in cigar_tuples if op in (1, 2) )  # 1: insertion, 2: deletion
    soft_clips = sum(length for (op, length) in cigar_tuples if op == 4)  # 4: soft clipping
    hard_clips = sum(length for (op, length) in cigar_tuples if op == 5)  # 5: hard clipping
    aligned_bases = sum(length for (op, length) in cigar_tuples if op in (0, 7, 8))  # 0: match or mismatch, 7: match, 8: mismatch;  both match and mismatch are aligned bases
    not_aligned_bases = len(read.query_sequence) - aligned_bases
    clip_ratio = (soft_clips + hard_clips) / len(read.query_sequence)

    return {
        "clip_ratio": clip_ratio
    }

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

    #NOTE: other approach could be sum(met_cyt_confidences) / (255 * len(met_cyt_confidences) or (255 * n_cyt))
    return sum(met_cyt_confidences) /  len(seq)

#Calculate aggregated features (mean and std) for a patient based on per-read features.
def aggregate_features_for_patient(per_read_df):

    if per_read_df.empty:
        return None

    aggregated_features = {'read_file' : per_read_df['read_file'].iloc[0]} 
    features_to_aggregate = ['length', 'methilated_cytosine_ratio', 'clip_ratio']
    
    per_read_df.fillna(0, inplace=True)
    for feature in features_to_aggregate:
        # Misure di tendenza centrale e dispersione
        aggregated_features[f'{feature}_mean'] = per_read_df[feature].mean()
        aggregated_features[f'{feature}_std'] = per_read_df[feature].std()
    
    aggregated_features_df = pd.DataFrame([aggregated_features])
    return aggregated_features_df