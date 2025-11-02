 The Correct Solution

  You need to delete ONLY the files that won't break the checkpoint chain:

  Safe to delete (forces re-clustering without breaking chain):
  - communities.parquet - output of create_communities
  - community_reports.parquet - output of create_community_reports
  - Any embedding files (generated at the end)

  DO NOT delete:
  - text_units.parquet - breaks create_final_text_units checkpoint
  - documents.parquet - breaks create_final_documents checkpoint
  - entities.parquet - breaks extract_graph/finalize_graph checkpoints
  - relationships.parquet - breaks extract_graph/finalize_graph checkpoints

  Corrected deletion command:

  # Only delete clustering and downstream outputs
  rm -f /mnt/fileintel/graphrag_index/graphrag_indices/6525aacb-55b1-4a88-aaaa-a4211d03beba/output/communities.parquet
  rm -f /mnt/fileintel/graphrag_index/graphrag_indices/6525aacb-55b1-4a88-aaaa-a4211d03beba/output/community_reports.parquet

  This will force re-run of:
  - create_communities (with new max_cluster_size=75)
  - create_community_reports
  - generate_text_embeddings (if applicable)

  But preserve all the expensive entity extraction work!