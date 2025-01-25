import torch
from PIL import Image
import numpy as np
from typing import List
from model1 import ModelArgs, CrossAttentionTransformer
# Assuming the other necessary files are in place, i.e, encoder_utils.py, image_transform.py

def create_dummy_image(size=(224,224)):
    # Generate a random image
    image_array = np.random.randint(0, 256, size=(size[0],size[1], 3), dtype=np.uint8)
    return Image.fromarray(image_array)

def create_dummy_masks(num_images, total_len, max_num_chunks):
   masks = []
   for _ in range(num_images):
      mask_elem_1 = [np.random.randint(0, total_len), np.random.randint(0, total_len)]
      masks.append([mask_elem_1, ])
   return masks
def test_cross_attention_transformer():
    # Model Configuration
    args = ModelArgs(
        dim=512,
        n_layers=2,
        n_heads=8,
        vocab_size=1000,
        vision_chunk_size=224,
        vision_max_num_chunks=4,
        vision_num_cross_attention_layers=1,
        max_seq_len=128,
        max_batch_size=2,
    )
    
    # Initialize Model
    model = CrossAttentionTransformer(args)
    model.to("cpu")
    model.setup_cache(max_batch_size=args.max_batch_size, dtype=torch.float16)

    # Dummy Inputs
    batch_size = args.max_batch_size
    total_len = args.max_seq_len
    tokens = torch.randint(0, args.vocab_size, (batch_size, total_len)).to("cpu")
    position_ids = torch.arange(0, total_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1).to("cpu")
    
    # Create dummy images and masks
    batch_images = [[create_dummy_image()] for _ in range(batch_size)]
    batch_masks = [create_dummy_masks(len(img_list), total_len, args.vision_max_num_chunks) for img_list in batch_images]
    
    # Compute vision tokens and attention masks
    xattn_caches, cross_attention_masks, full_text_row_masked_out_mask = model.compute_vision_tokens_masks(
      batch_images=batch_images,
      batch_masks=batch_masks,
      total_len=total_len
    )
    
    # Forward Pass
    logits = model.forward(
        position_ids=position_ids,
        tokens=tokens,
        cross_attention_masks=cross_attention_masks,
        full_text_row_masked_out_mask=full_text_row_masked_out_mask,
        xattn_caches=xattn_caches,
    )
    

    # Assertions
    expected_output_shape = (batch_size, total_len, args.vocab_size)
    assert logits.shape == expected_output_shape, f"Expected output shape {expected_output_shape}, but got {logits.shape}"

    print("CrossAttentionTransformer test passed with dummy inputs!")


if __name__ == "__main__":
    test_cross_attention_transformer()