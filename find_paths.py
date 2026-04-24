import os

# Set these to your actual locations:
BASE_DATA_DIR = "/storage/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data"
FASTA_DIR = "/home/hice1/salammurillo3/scratch/openfold/examples/monomer/fasta_dir"
FASTA_FILE = os.path.join(FASTA_DIR, "6kwc.fasta")

def check_path(desc, path):
    if os.path.exists(path):
        print(f"{desc}: {path}")
    else:
        print(f"{desc}: NOT FOUND ({path})")

print("=== DATABASE FILES ===")
check_path("pdb_mmcif/mmcif_files", os.path.join(BASE_DATA_DIR, "pdb_mmcif/mmcif_files"))
check_path("uniref90/uniref90.fasta", os.path.join(BASE_DATA_DIR, "uniref90/uniref90.fasta"))
check_path("mgnify/mgy_clusters_2022_05.fa", os.path.join(BASE_DATA_DIR, "mgnify/mgy_clusters_2022_05.fa"))
check_path("pdb70/pdb70", os.path.join(BASE_DATA_DIR, "pdb70/pdb70"))
check_path("uniclust30/uniclust30_2018_08", os.path.join(BASE_DATA_DIR, "uniclust30/uniclust30_2018_08"))
check_path("bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt", os.path.join(BASE_DATA_DIR, "bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"))


print("\n=== EXAMPLE PROTEIN FASTA DIRECTORY AND FILE ===")
check_path("FASTA directory", FASTA_DIR)
check_path("FASTA file", FASTA_FILE)

print("\nIf any path is marked NOT FOUND, check your directory structure or ask your system administrator.")

