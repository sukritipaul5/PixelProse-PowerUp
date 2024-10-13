import torch
from torch.utils.data import IterableDataset
import glob
import random
import webdataset as wds
from torchvision import transforms
import os
import numpy as np
from PIL import Image
import json
import utils_sd3
from collections import defaultdict

class WebDatasetwithCachedLatents(IterableDataset):
    def __init__(self, 
                 tar_path,
                 latents_dir,
                 tokenizers,
                 size=1024,
                 center_crop=False,
                 random_flip=False,
                 max_length=77,
                 num_samples=None,
                 randomize=True):
        self.tokenizers = tokenizers #added
        self.tar_path = tar_path
        self.tar_files = glob.glob(tar_path)
        print("Total tar files: ",len(self.tar_files))
        self.latents_dir = latents_dir
        self.size = size
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.max_length = max_length
        self.num_samples = num_samples
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.tar_files = self.get_tar_files_with_latents()
        print("Tar files with latents: ",len(self.tar_files))
        if randomize:
            random.shuffle(self.tar_files)

        self.webdataset = wds.DataPipeline(
            wds.SimpleShardList(self.tar_files),
            wds.shuffle(100),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.decode('pil'),
            wds.map(self.process_sample),
            wds.to_tuple('pixel_values',
                         'prompt_embeds',
                         'pooled_prompt_embeds',
                         'tokenized_vlm_captions')
            #              'vlm_caption'),
         
        )
        

    def get_tar_files_with_latents(self):
        tar_files_with_latents = []
        for tar_file in self.tar_files:
            tar_file_name = tar_file.replace('.tar','')
            tar_file_name = tar_file_name.split('/')[-2:]
            tar_file_name = '/'.join(tar_file_name)
            tar_file_name = os.path.join(self.latents_dir,tar_file_name)
            if os.path.exists(tar_file_name):
                tar_files_with_latents.append(tar_file)
        return tar_files_with_latents

    def process_sample(self,sample):
        # do a bunch of stuff
        if sample is None:
            return None
        base_name = sample['__url__']
        key = sample['__key__']
        base_name = base_name.replace('.tar','')
        base_name = base_name.split('/')[-2:]
        base_name = '/'.join(base_name)
        fname = f"{base_name}/{key}.npz"
        fname = os.path.join(self.latents_dir,fname)
        # print(fname)
        if not os.path.exists(fname):
            return None
        try:
            latents = np.load(fname)
            image = self.image_transforms(sample['jpg'])
            #added for context window
            json_data = sample['json']
            vlm_caption = json_data.get('vlm_caption', '')

            # Tokenize the vlm_caption with each tokenizer-ADDED
            tokenized_vlm_captions = []
            for idx, tokenizer in enumerate(self.tokenizers):
                tokenized_caption = tokenizer(
                    vlm_caption,
                    padding="max_length",
                    max_length=self.max_length * 2,  # Increase max length
                    truncation=True,
                    return_tensors="pt",
                )
                
                input_ids = tokenized_caption.input_ids.squeeze()
                attention_mask = tokenized_caption.attention_mask.squeeze()
                
                # Split into two subsets
                subset1_ids = input_ids[:77]
                subset1_mask = attention_mask[:77]
                subset2_ids = input_ids[77:]
                subset2_mask = attention_mask[77:]
                
                tokenized_vlm_captions.append({
                    "subset1": {"input_ids": subset1_ids, "attention_mask": subset1_mask},
                    "subset2": {"input_ids": subset2_ids, "attention_mask": subset2_mask},
                    "tokenizer_num": idx
                })
                    

        except Exception as e:
            print(f"Error processing image: {e}")
            return None
        prompt_embeds = torch.from_numpy(latents['prompt_embeds'])
        pooled_prompt_embeds = torch.from_numpy(latents['pooled_prompt_embeds'])
        return {
            "pixel_values": image,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "tokenized_vlm_captions": tokenized_vlm_captions
        }


    def clean_sample(self,sample):
        if sample is None:
            return None
        return {
            "pixel_values": sample['pixel_values'],
            "prompt_embeds": sample['prompt_embeds'],
            "pooled_prompt_embeds": sample['pooled_prompt_embeds'],
            "tokenized_vlm_captions": sample['tokenized_vlm_captions']
        }

    def __iter__(self):
        # yield from (self.clean_sample(sample) for sample in iter(self.webdataset))    
        yield from iter(self.webdataset)


def reorganize_batch(tokenized_captions):
    reorganized = [
        {"input_ids": [], "attention_mask": []},
        {"input_ids": [], "attention_mask": []},
        {"input_ids": [], "attention_mask": []}
    ]
    
    for batch in tokenized_captions:
        for i, item in enumerate(batch):
            reorganized[i]["input_ids"].append(item["input_ids"])
            reorganized[i]["attention_mask"].append(item["attention_mask"])

    # Stack the tensors
    for item in reorganized:
        item["input_ids"] = torch.stack(item["input_ids"]).squeeze(1)
        item["attention_mask"] = torch.stack(item["attention_mask"]).squeeze(1)

    return reorganized

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    collated = {}
    
    collated['pixel_values'] = torch.stack([item[0] for item in batch])
    collated['prompt_embeds'] = torch.stack([item[1] for item in batch])
    collated['pooled_prompt_embeds'] = torch.stack([item[2] for item in batch])
    
    tokenized_captions = [item[3] for item in batch]
    collated_captions = {
        "subset1": [[], [], []],  # One list for each tokenizer
        "subset2": [[], [], []]
    }
    
    for item in tokenized_captions:
        for i, tokenizer_output in enumerate(item):
            collated_captions["subset1"][i].append(tokenizer_output["subset1"])
            collated_captions["subset2"][i].append(tokenizer_output["subset2"])
    
    # Stack tensors for each subset and tokenizer
    for subset in ["subset1", "subset2"]:
        for i in range(3):
            collated_captions[subset][i] = {
                "input_ids": torch.stack([item["input_ids"] for item in collated_captions[subset][i]]),
                "attention_mask": torch.stack([item["attention_mask"] for item in collated_captions[subset][i]])
            }
    
    collated['tokenized_vlm_captions'] = collated_captions

    return collated

# def custom_collate_fn(batch):
#     batch = [item for item in batch if item is not None]
#     if len(batch) == 0:
#         return None
#     collated = {}
#     collated['pixel_values'] = torch.stack([item[0] for item in batch])
#     collated['prompt_embeds'] = torch.stack([item[1] for item in batch])
#     collated['pooled_prompt_embeds'] = torch.stack([item[2] for item in batch])
#     #collated['vlm_caption'] = [item[3] for item in batch]     
#     return collated

# def custom_collate_fn(batch):
#     if len(batch) == 0:
#         return None
    
#     return {
#         'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
#         'prompt_embeds': torch.stack([item['prompt_embeds'] for item in batch]),
#         'pooled_prompt_embeds': torch.stack([item['pooled_prompt_embeds'] for item in batch]),
#         'vlm_caption': [item['vlm_caption'] for item in batch]
#     }
    
def create_simple_dataloader(dataset, batch_size, num_workers):
    dataloader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              shuffle=False,
                                              drop_last=True,
                                              collate_fn=custom_collate_fn)
    return dataloader
