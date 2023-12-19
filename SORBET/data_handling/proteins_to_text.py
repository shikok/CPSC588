from Bio import Entrez
import mygene
from Bio import Entrez
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import pickle
import requests

maker_ids = ["LIPG_HUMAN", "VIME_HUMAN", "TBX21_HUMAN", "CD47_HUMAN", "K1C19_HUMAN", "PTPRC_HUMAN", "PD1L1_HUMAN", "G3P_HUMAN", "Z3H7B_HUMAN", "LAG3_HUMAN", "TIMP3_HUMAN", "FOXP3_HUMAN", "CD4_HUMAN", "VTCN1_HUMAN", "CD68_HUMAN", "PDCD1_HUMAN", "CD20_HUMAN", "CD8A_HUMAN", "IL2RA_HUMAN", "VISTA_HUMAN", "KI67_HUMAN", "B2MG_HUMAN", "CD3G_HUMAN", "I23O1_HUMAN", "PD1L2_HUMAN", "GRB7_HUMAN", "H31_HUMAN", "DNMT1_HUMAN", "DNA2_HUMAN"]
def get_protein_function(uniprot_id):
    """
    Retrieve the function of a protein from UniProt using a UniProt ID.

    Parameters:
        uniprot_id: UniProt ID of the protein

    Returns:
        String containing the function of the protein.
    """
    base_url = 'https://www.uniprot.org/uniprot/'
    response = requests.get(f"{base_url}{uniprot_id}.txt")

    if response.status_code != 200:
        return f"Failed to retrieve data for {uniprot_id}"

    function_text = ""
    recording = False
    for line in response.text.split("\n"):
        if line.startswith("CC   -!- FUNCTION"):
            recording = True
            function_text += line[17:].strip() + " "
        elif recording and line.startswith("CC       "):
            function_text += line[9:].strip() + " "
        elif recording and not line.startswith("CC       "):
            break

    return function_text.strip()

def get_protein_embeddings(protein_names):    
    # Load the SBERT model
    sbert_model = SentenceTransformer('all-mpnet-base-v2')

    # Dictionary to store embeddings
    embeddings = {}
    protein_name_to_uniprot_id = {}
    # use marker ids same order as markers
    for i, marker in enumerate(protein_names):
        protein_name_to_uniprot_id[marker] = maker_ids[i]
    # Get the function of each protein
    for protein in protein_names:
        if protein not in embeddings:  # Skip if already processed
            uniprot_id = protein_name_to_uniprot_id[protein]
            if uniprot_id:
                function = get_protein_function(uniprot_id)
                embeddings[protein] = sbert_model.encode([function])
            else:
                embeddings[protein] =  np.zeros(sbert_model.get_sentence_embedding_dimension())

    # Convert dictionary of embeddings to a matrix
    embeddings_matrix = torch.tensor(list(embeddings.values()))

    # Optionally, save embeddings matrix
    with open('protein_embeddings.pkl', 'wb') as file:
        pickle.dump(embeddings_matrix, file)

    return embeddings_matrix

def create_summary_embedding(protein_names, node_data):
    
    protein_embeddings = get_protein_embeddings(protein_names)
    protein_embeddings.detach().to('cpu').numpy()
    protein_embedding_reshaped = protein_embeddings.reshape(29, 768)
    cell_embeddings = np.dot(node_data, protein_embedding_reshaped)
    return cell_embeddings
    



