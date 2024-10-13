import argparse
import torch
from tqdm import tqdm
import os
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel, T5EncoderModel, T5Tokenizer,CLIPTextModelWithProjection
from data_sd3_embed import StatefulWebDataset, create_dataloader
import glob


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute text embeddings for SD3 Medium finetuning")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--tar_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=77)
    parser.add_argument("--num_samples", type=int, default=None)
    return parser.parse_args()

def load_models_and_tokenizers(args):
    # Load tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_two = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    tokenizer_three = T5Tokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_3")

    # Load models
    text_encoder_one = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    text_encoder_three = T5EncoderModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_3")

    return (tokenizer_one, tokenizer_two, tokenizer_three), (text_encoder_one, text_encoder_two, text_encoder_three)



def process_batch(batch, text_encoders, tokenizers, max_length, device):

    prompt_embeds_list = []
    pooled_prompt_embeds_list = []

    
    tokenized_prompts = batch['tokenized_prompts']

    
    for i, (tokenized_prompt, text_encoder) in enumerate(zip(tokenized_prompts, text_encoders)):
        input_ids = tokenized_prompt["input_ids"].to(device=device)
        attention_mask = tokenized_prompt["attention_mask"].to(device=device)
        
        if i < 2:  # CLIP text encoders
            prompt_embeds = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            prompt_embeds_list.append(prompt_embeds.hidden_states[-2])
            pooled_prompt_embeds_list.append(prompt_embeds[0])
        else:  # T5 text encoder
            prompt_embeds = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )[0]
            prompt_embeds_list.append(prompt_embeds)


    # Concatenate CLIP embeddings
    clip_prompt_embeds = torch.cat(prompt_embeds_list[:2], dim=-1)
    pooled_prompt_embeds = torch.cat(pooled_prompt_embeds_list, dim=-1)
    
    # Pad CLIP embeddings to match T5 embedding size
    t5_prompt_embeds = prompt_embeds_list[2]
    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embeds.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    
    # Concatenate all embeddings
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embeds], dim=-2)

    return prompt_embeds, pooled_prompt_embeds



def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizers, text_encoders = load_models_and_tokenizers(args)
    for encoder in text_encoders:
        encoder.to(device)
        encoder.eval()

    dataset = StatefulWebDataset(
        tar_path=args.tar_path,
        tokenizers=tokenizers,
        size=512,
        center_crop=False,
        random_flip=False,
        max_length=args.max_length,
        num_samples=args.num_samples
    )

    dataloader, _ = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=False
    )


    os.makedirs(args.output_dir, exist_ok=True)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        if batch is None:
            continue

        try:
            #breakpoint()
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = process_batch(
                    batch,
                    text_encoders,
                    tokenizers,
                    args.max_length,
                    device
                )

            # Save embeddings
            for idx, (file_name, embed, pooled_embed) in enumerate(zip(batch['file_name'], prompt_embeds, pooled_prompt_embeds)):
                # Extract tar file name and internal path
                tar_name = os.path.basename(os.path.dirname(file_name))
                internal_path = os.path.basename(file_name).replace('.jpg', '.npz')
                
                # Construct output path
                output_path = os.path.join(args.output_dir, tar_name)
                output_path=output_path[:-4] #remove .tar

                os.makedirs(output_path, exist_ok=True)
                np.savez(
                    os.path.join(output_path, internal_path),
                    prompt_embeds=embed.cpu().numpy(),
                    pooled_prompt_embeds=pooled_embed.cpu().numpy().squeeze()
                )

            # Free up memory
            del prompt_embeds, pooled_prompt_embeds
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing batch {batch_idx}: {str(e)}")
            continue

    print(f"Embeddings saved in {args.output_dir}")

if __name__ == "__main__":
    main()


