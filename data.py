from torch.utils.data import IterableDataset
from datasets import load_dataset
from PIL import Image
import io

class CustomDataset(IterableDataset):
    def __init__(self, dataset_name, split, feature_extractor, samples_per_class, subset='train'):
        self.dataset_name = dataset_name
        self.split = split
        self.feature_extractor = feature_extractor
        self.samples_per_class = samples_per_class
        self.subset = subset
        self.samples_collected = {label: 0 for label in range(num_labels)}
        self.dataset_stream = self.create_stream()

    def create_stream(self):
        return load_dataset(self.dataset_name, split=self.split, streaming=True)

    def reset_stream(self):
        self.dataset_stream = self.create_stream()
        self.samples_collected = {label: 0 for label in range(num_labels)}

    def preprocess(self, example):
        if isinstance(example['image'], bytes):
            image = Image.open(io.BytesIO(example['image']))
        else:
            image = example['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')

        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze()
        return pixel_values, example['label']

    def __iter__(self):
        for example in self.dataset_stream:
            label = example['label']
            if self.samples_collected[label] < self.samples_per_class:
                self.samples_collected[label] += 1
                yield self.preprocess(example)
            if all(value == self.samples_per_class for value in self.samples_collected.values()):
                break