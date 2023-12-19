import numpy as np
import torch
from sentence_transformers import SentenceTransformer

def get_top_proteins_expression(cell_data, marker_names, top_n=20):
    # Filter out zero expression levels and sort
    non_zero_indices = cell_data > 0
    filtered_data = cell_data[non_zero_indices]
    marker_names = np.array(marker_names)
    filtered_markers = marker_names[non_zero_indices]
    
    # Get indices for top N expression levels
    top_indices = np.argsort(filtered_data)[-top_n:]
    # Take top N proteins
    top_proteins = filtered_markers[top_indices]
    return ', '.join(top_proteins[::-1])  # Reverse to get highest first

def get_cell_embeddings(cell_data_array, marker_names):
    # Load the SBERT model
    sbert_model = SentenceTransformer('all-mpnet-base-v2')

    embeddings = []
    for cell_data in cell_data_array:
        protein_list = get_top_proteins_expression(cell_data, marker_names)
        # Convert the protein list to SBERT embedding
        embedding = sbert_model.encode(protein_list)
        embeddings.append(embedding)

    # Convert list of embeddings to a tensor
    embeddings_tensor = torch.tensor(embeddings)
    return embeddings_tensor.detach().to('cpu').numpy()


