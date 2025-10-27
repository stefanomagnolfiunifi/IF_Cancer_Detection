import pysam
import pandas as pd
import glob
import os
import math
from collections import Counter
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

def seq_entropy(seq):
    counts = Counter(seq)
    L = len(seq)
    if L == 0: return 0.0
    ent = 0.0
    for v in counts.values():
        p = v / L
        ent -= p * math.log2(p)
    return ent

def gc_content(seq):
    if not seq: return 0.0
    g = seq.count('G') + seq.count('C')
    return g / len(seq)

#Extract numeric features for every read of the BAM file.
def extract_features_from_bam(bam_path):
        
        bamfile = pysam.AlignmentFile(bam_path, "rb")
        file_name = (os.path.basename(bam_path)).split('_')[0]
        k = 3
        rows = []

        # Iterate over each read in the BAM file
        for read in bamfile.fetch():

            # flags / basic
            flags = read.flag
            is_supp = read.is_supplementary
            is_sec = read.is_secondary
            is_dup = read.is_duplicate
            mapq = read.mapping_quality

            seq = read.query_sequence or ""
            qual = read.query_qualities or []

            # CIGAR parsing
            soft_left = 0
            soft_right = 0
            if read.cigartuples:
                # cigartuples: list of (op, length) where op 4=softclip 5=hardclip 1=ins 2=del
                if read.cigartuples[0][0] == 4:
                    soft_left = read.cigartuples[0][1]
                if read.cigartuples[-1][0] == 4:
                    soft_right = read.cigartuples[-1][1]
                num_ins = sum(l for op,l in read.cigartuples if op==1)
                num_del = sum(l for op,l in read.cigartuples if op==2)
            else:
                num_ins = num_del = 0

            nm = None
            try:
                nm = read.get_tag('NM')
            except KeyError:
                nm = None

            # methylation tags common in ONT: 'Mm' (mods) and 'Ml' confidences; adapt if different
            num_meth_cpg = 0
            mean_meth_conf = None
            try:
                mm = read.get_tag('Mm')  # format varies
                ml = read.get_tag('Ml')  # list of confidences
                # parsing Mm/Ml is nontrivial; here assume ml is list and cpG count equals length
                if isinstance(ml, (list, tuple)):
                    mean_meth_conf = sum(ml)/len(ml) if len(ml)>0 else None
                    num_meth_cpg = len(ml)
            except KeyError:
                mean_meth_conf = None

            row_data = {
                'is_supplementary': int(is_supp),
                'is_secondary': int(is_sec),
                'is_duplicate': int(is_dup),
                'mapq': mapq,
                'nm': nm if nm is not None else -1,
                'soft_left': soft_left,
                'soft_right': soft_right,
                'num_ins': num_ins,
                'num_del': num_del,
                'seq_len': len(seq),
                'gc': gc_content(seq),
                'entropy': seq_entropy(seq),
                'mean_base_q': (sum(qual)/len(qual)) if qual else -1,
                'num_methylated_calls': num_meth_cpg,
                'mean_meth_conf': mean_meth_conf if mean_meth_conf is not None else -1
            }

            # CIGAR features
            cigar_features = extract_cigar_features(read) 
            row_data.update(cigar_features)
            rows.append(row_data) 
        bamfile.close()

        # Create the DataFrame
        feature_df = pd.DataFrame(rows)

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

    aggregated_features = {}
    features_to_aggregate = ['mapq', 'insert_size', 'mismatches', 'soft_clipping']
    
    for feature in features_to_aggregate:
        # Misure di tendenza centrale e dispersione
        aggregated_features[f'{feature}_mean'] = per_read_df[feature].mean()
        aggregated_features[f'{feature}_std'] = per_read_df[feature].std()
        
    return aggregated_features