import torch
from pdf2image import convert_from_path

# Function to convert a PDF file to images
def convert_pdf_to_images(pdf_path):
    return convert_from_path(pdf_path)

# Function to get concatenated representation of a document
@torch.no_grad()
def get_concatenated_representation(pdf_path, processor, model):
    image_array = convert_pdf_to_images(pdf_path)
    concatenated_outputs = []
    for image in image_array:
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(model.device)
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
