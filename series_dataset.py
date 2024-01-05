"""Module for loading time series data for risk evaluation"""
import torch
from torch.utils.data import Dataset
from utils import *

class TimeSeriesDataset(Dataset):
    """
    A custom Dataset for handling time-series data for a Variational Autoencoder.

    The dataset takes a pandas DataFrame as input. Each row of the DataFrame should
    contain two fields, each field being a list of time-series data. The two lists
    from each row are concatenated to form a T x 2 tensor, where T is the time dimension.
    """

    def __init__(self, csv_path : str) -> None:
        """
        Initialize the dataset with a pandas DataFrame.

        Args:
        csv_path (str): Path to the CSV file containing document metadata.
        field1 (str): The name of the first field in the DataFrame containing time-series data.
        field2 (str): The name of the second field in the DataFrame containing time-series data.
        """
        self.field1 = 'vintage_issue'
        self.field2 = 'retired_credits'
        self.dataframe = load_and_preprocess_csv(csv_path)

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
        int: The number of rows in the DataFrame.
        """
        return len(self.dataframe)

    def __getitem__(self, idx : int) -> torch.Tensor:
        """
        Retrieve a single item from the dataset by its index.

        Args:
        idx (int): The index of the item to retrieve.

        Returns:
        torch.Tensor: A T x 2 tensor representing concatenated time-series data.
        """
        # Retrieve the time-series data from the two fields
        series1 = self.dataframe[idx][self.field1]
        series2 = self.dataframe[idx][self.field2]

        # Ensure the time-series data is in list format
        if not isinstance(series1, list) or not isinstance(series2, list):
            raise TypeError("The dataframe fields should contain lists of time-series data.")

        # Concatenate the two series along the second dimension
        concatenated_series = torch.tensor([series1, series2]).T

        return concatenated_series
