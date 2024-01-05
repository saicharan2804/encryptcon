from transformers import DonutProcessor, VisionEncoderDecoderModel
from torch.utils.data import Dataset, DataLoader
from utils import *
import pandas as pd
import os

def load_and_preprocess_csv(csv_path):
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

# Custom Dataset
class PDFDocumentDataset(Dataset):
    def __init__(self, csv_path, pdf_folder_path, processor, model):
        self.pdf_folder_path = pdf_folder_path
        self.processor = processor
        self.model = model
        self.dataframe = self.check_data(load_and_preprocess_csv(csv_path))

    def check_data(self, dataframe):

        df2 = []

        for _, row in dataframe.iterrows():
            pdf_path = f"{self.pdf_folder_path}/{row['Project ID']}.pdf"
            if os.path.exists(pdf_path):
                df2.append(row)

        return df2

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe[idx]
        pdf_path = f"{self.pdf_folder_path}/{row['Project ID']}.pdf"
        embeddings = get_concatenated_representation(pdf_path, self.processor, self.model)

        # Create a comma-separated text from the row, excluding embeddings-related data
        text_data = '\n'.join([f"{k}:{v}" for k,v in row.drop(['Project ID', 'vintage_issue', 'retired_credits']).to_dict().items()])#.to_csv(header=False, index=False).strip('\n')

        data = {
            "project_id": row['Project ID'],
            "description": text_data,
            "vintage_issue": row['vintage_issue'],
            "retired_credits": row['retired_credits'],
            "embeddings": embeddings
        }

        text = ""
        for k,v in data.items():
            if k!= 'embeddings':
                text += f"{k} : {v},"

        out = self.processor.tokenizer(text, padding = "max_length", max_length = 1024, add_special_tokens=False, return_tensors="pt")

        labels, attn_mask = out.input_ids, out.attention_mask

        data["labels"] = labels
        data["attn_mask"] = attn_mask
        # Prepare the output dictionary
        return data
    
if __name__ == '__main__':

    # Initialize the processor and model
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    pass

    dset = PDFDocumentDataset("clean_data.csv", "dataset", processor, model)
    print(dset[0])
