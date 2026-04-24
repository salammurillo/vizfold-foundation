python3 run_pretrained_openfold.py \
  /home/hice1/salammurillo3/scratch/openfold/examples/monomer/fasta_dir \
  /storage/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data/pdb_mmcif/mmcif_files \
  --output_dir /home/hice1/salammurillo3/scratch/openfold/outputs \
  --model_device cuda:0 \
  --config_preset model_1_ptm \
  --uniref90_database_path /storage/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data/uniref90/uniref90.fasta \
  --mgnify_database_path /storage/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data/mgnify/mgy_clusters_2022_05.fa \
  --pdb70_database_path /storage/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data/pdb70/pdb70 \
  --bfd_database_path /storage/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
  --uniclust30_database_path /storage/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data/uniclust30/uniclust30_2018_08/uniclust30_2018_08
