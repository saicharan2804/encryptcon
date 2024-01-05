"""Module for loading time series data for risk evaluation"""
import torch
from torch.utils.data import Dataset
from utils import *
import numpy as np


class TimeSeriesDataset(Dataset):
    """
    A custom Dataset for handling time-series data for a Variational Autoencoder.

    The dataset takes a pandas DataFrame as input. Each row of the DataFrame should
    contain two fields, each field being a list of time-series data. The two lists
    from each row are concatenated to form a T x 2 tensor, where T is the time dimension.
    """

    def __init__(self, csv_path: str, desired_length : int) -> None:
        """
        Initialize the dataset with a pandas DataFrame.

        Args:
        csv_path (str): Path to the CSV file containing document metadata.
        desired_length (int) : Length of time series data
        """
        self.field1 = "vintage_issue"
        self.field2 = "retired_credits"
        self.dataframe = load_and_preprocess_csv(csv_path)
        self.desired_length = desired_length

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
        int: The number of rows in the DataFrame.
        """
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieve a single item from the dataset by its index.

        Args:
        idx (int): The index of the item to retrieve.

        Returns:
        torch.Tensor: A T x 2 tensor representing concatenated time-series data.
        """
        # Retrieve the time-series data from the two fields

        series1 = self.dataframe.iloc[idx][self.field1]
        series2 = self.dataframe.iloc[idx][self.field2]

        # Ensure the time-series data is in list format
        if not isinstance(series1, list) or not isinstance(series2, list):
            raise TypeError("The dataframe fields should contain lists of time-series data.")

        # Pad or truncate the series to be exactly 10 values
        series1 = self.pad_or_truncate(series1)
        series2 = self.pad_or_truncate(series2)

        # Normalize the series using Z-score normalization
        series1_normalized = (series1 - np.mean(series1)) / np.std(series1)
        series2_normalized = (series2 - np.mean(series2)) / np.std(series2)

        # Concatenate the two normalized series along the second dimension
        concatenated_series = torch.tensor([series1_normalized, series2_normalized]).T

        return concatenated_series

    def pad_or_truncate(self, series):
        """
        Pad or truncate the series to be exactly 10 values.

        Args:
        series (list): The time-series data.

        Returns:
        list: The modified time-series data.
        """
        current_length = len(series)

        if current_length < self.desired_length:
            # Pad the series with zeros if it's shorter than the desired length
            series += [0] * (self.desired_length - current_length)
        elif current_length > self.desired_length:
            # Truncate the series if it's longer than the desired length
            series = series[:self.desired_length]

        return series