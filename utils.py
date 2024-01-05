import torch
from pdf2image import convert_from_path
import pandas as pd
from typing import List
import os


# Function to convert a PDF file to images
def convert_pdf_to_images(pdf_path):
    return convert_from_path(pdf_path)


def load_and_preprocess_csv(csv_path: str) -> pd.DataFrame:
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
    vintage_issue_cols = [str(year) + ".0" for year in range(2009, 2024)]
    retired_credits_cols = [str(year) + ".0.1" for year in range(2009, 2024)]

    df["vintage_issue"] = df[vintage_issue_cols].values.tolist()
    df["retired_credits"] = df[retired_credits_cols].values.tolist()

    # Drop the original year columns
    df.drop(
        columns=vintage_issue_cols + retired_credits_cols, inplace=True, errors="ignore"
    )

    return check_data(df)


def check_data(self, dataframe: pd.DataFrame) -> List[pd.DataFrame]:
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


# Function to get concatenated representation of a document
@torch.no_grad()
def get_concatenated_representation(pdf_path, processor, model):
    image_array = convert_pdf_to_images(pdf_path)
    concatenated_outputs = []
    for image in image_array:
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(
            model.device
        )
        outputs = model.encoder(pixel_values)
        concatenated_outputs.append(outputs.pooler_output)
    return torch.cat(concatenated_outputs, dim=1)


def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError(
            "Make sure to set the decoder_start_token_id attribute of the model's configuration."
        )
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError(
            "Make sure to set the pad_token_id attribute of the model's configuration."
        )
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids
