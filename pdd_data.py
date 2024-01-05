"""Data Module for loading PDF documents"""
from transformers import DonutProcessor, VisionEncoderDecoderModel
from torch.utils.data import Dataset
from utils import *
import pandas as pd
import os
from typing import List, Dict, Any

def load_and_preprocess_csv(csv_path : str) -> pd.DataFrame:
    """
    Load and preprocess the CSV file containing the metadata for PDF documents.

    Args:
    csv_path (str): The file path to the CSV file.

    Returns:
    DataFrame: A Pandas DataFrame with preprocessed data.

    The function performs the following preprocessing steps:
    - Loads data from the specified CSV file.
    - Merges specific columns into arrays for easier access and manipulation.
    - Drops the original year columns after merging.
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    # Drop specified columns
    # columns_to_drop = ['Unnamed: 0', 'ARB Project', 'State', 'Project Site Location',
    #                    'Reversals Covered by Buffer Pool', 'Reversals Not Covered by Buffer']
    # df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Merge specified columns into arrays
    vintage_issue_cols = [str(year) + '.0' for year in range(2009, 2024)]
    retired_credits_cols = [str(year) + '.0.1' for year in range(2009, 2024)]

    df['vintage_issue'] = df[vintage_issue_cols].values.tolist()
    df['retired_credits'] = df[retired_credits_cols].values.tolist()

    # Drop the original year columns
    df.drop(columns=vintage_issue_cols + retired_credits_cols, inplace=True, errors='ignore')

    return df

class PDFDocumentDataset(Dataset):
    """
    A custom Dataset class for handling PDF documents.

    This class is designed to work with PDF files, extracting information using a Donut model,
    and generating text summaries as targets. It supports processing of multiple pages of a PDF,
    unlike standard approaches which often focus on single-page processing.

    Attributes:
    csv_path (str): Path to the CSV file containing document metadata.
    pdf_folder_path (str): Folder path containing PDF documents.
    processor (DonutProcessor): Instance of DonutProcessor for processing documents.
    model (VisionEncoderDecoderModel): Pretrained model for generating embeddings.
    dataframe (DataFrame): DataFrame holding the preprocessed data.
    """

    def __init__(self, csv_path : str, pdf_folder_path : str, processor : DonutProcessor, model : VisionEncoderDecoderModel) -> None:
        """
        Initialize the dataset with CSV and PDF paths, and the processor and model.

        Args:
        csv_path (str): Path to the CSV file containing document metadata.
        pdf_folder_path (str): Folder path containing PDF documents.
        processor (DonutProcessor): Instance of DonutProcessor.
        model (VisionEncoderDecoderModel): Instance of a pretrained model.
        """
        self.pdf_folder_path = pdf_folder_path
        self.processor = processor
        self.model = model
        self.dataframe = self.check_data(load_and_preprocess_csv(csv_path))

    def check_data(self, dataframe : pd.DataFrame) -> List[pd.DataFrame]:
        """
        Validate and filter the DataFrame based on the existence of corresponding PDF files.

        Args:
        dataframe (DataFrame): The preprocessed DataFrame.

        Returns:
        list: A filtered list of DataFrame rows where the PDF files exist.
        """
        df2 = []

        # Check for the existence of each PDF and append the row to df2 if the PDF exists
        for _, row in dataframe.iterrows():
            pdf_path = f"{self.pdf_folder_path}/{row['Project ID']}.pdf"
            if os.path.exists(pdf_path):
                df2.append(row)

        return df2

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns:
        int: The number of items in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx : int) -> Dict[str, Any]:
        """
        Retrieve a single item from the dataset by its index.

        Args:
        idx (int): The index of the item to retrieve.

        Returns:
        dict: A dictionary containing the data for the item, including embeddings and tokenized text.
        """
        # Get the row and its corresponding PDF path
        row = self.dataframe[idx]
        pdf_path = f"{self.pdf_folder_path}/{row['Project ID']}.pdf"

        # Generate embeddings from the PDF
        embeddings = get_concatenated_representation(pdf_path, self.processor, self.model)

        # Create a comma-separated text from the row, excluding embeddings-related data
        text_data = '\n'.join([f"{k}:{v}" for k,v in row.drop(['Project ID', 'vintage_issue', 'retired_credits']).to_dict().items()])

        # Construct the data dictionary
        data = {
            "project_id": row['Project ID'],
            "description": text_data,
            "vintage_issue": row['vintage_issue'],
            "retired_credits": row['retired_credits'],
            "embeddings": embeddings
        }

        # Concatenate data fields into a single text string, excluding embeddings
        text = ""
        for k,v in data.items():
            if k != 'embeddings':
                text += f"{k} : {v},"

        # Tokenize the text
        out = self.processor.tokenizer(text, padding="max_length", max_length=1024, add_special_tokens=False, return_tensors="pt")
        labels, attn_mask = out.input_ids, out.attention_mask

        # Add labels and attention mask to the data dictionary
        data["labels"] = labels
        data["attn_mask"] = attn_mask

        # Return the prepared data dictionary
        return data