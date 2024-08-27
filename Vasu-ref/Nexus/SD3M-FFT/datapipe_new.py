import torch
from torch.utils.data import IterableDataset
import glob
import random
import webdataset as wds
from torchvision import transforms
import os
import numpy as np



class WebDatasetwithCachedLatents(IterableDataset):
    def __init__(self, 
                 tar_path,
                 latents_dir,
                 size=1024,
                 center_crop=False,
                 random_flip=False,
                 max_length=77,
                 num_samples=None,
                 randomize=True):
        
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
            # wds.shuffle(100),
            wds.decode('pil'),
            wds.map(self.process_sample),
            # wds.decode('pil'),
            wds.to_tuple('pixel_values',
                         'prompt_embeds',
                         'pooled_prompt_embeds'),
            # wds.map(self.image_transforms),
            # wds.batched(1)
        )
        # self.iterator = iter(self.webdataset)
    
    # def __len__(self):
    #     return self.num_samples

    def get_tar_files_with_latents(self):
        tar_files_with_latents = []
        for tar_file in self.tar_files:
            tar_file_name = tar_file.replace('.tar','')
            tar_file_name = tar_file_name.split('/')[-2:]
            tar_file_name = '/'.join(tar_file_name)
            # print(tar_file_name)
            # break
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
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
        prompt_embeds = torch.from_numpy(latents['prompt_embeds'])
        pooled_prompt_embeds = torch.from_numpy(latents['pooled_prompt_embeds'])
        return {
            "pixel_values": image,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds
        }

    def clean_sample(self,sample):
        if sample is None:
            return None
        return {
            "pixel_values": sample['pixel_values'],
            "prompt_embeds": sample['prompt_embeds'],
            "pooled_prompt_embeds": sample['pooled_prompt_embeds']
        }

    def __iter__(self):
        # yield from (self.clean_sample(sample) for sample in iter(self.webdataset))    
        yield from iter(self.webdataset)

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    #return torch.utils.data.dataloader.default_collate(batch)
    # print(batch[0].keys())
    collated = {}
    collated['pixel_values'] = torch.stack([item[0] for item in batch])
    collated['prompt_embeds'] = torch.stack([item[1] for item in batch])
    collated['pooled_prompt_embeds'] = torch.stack([item[2] for item in batch])
    # collated = {}
    # for key in batch[0].keys():
    #     if key in ['prompt_embeds', 'pooled_prompt_embeds']:
    #         collated[key] = torch.stack([item[key] for item in batch])
    #     else:
    #         collated[key] = torch.utils.data.dataloader.default_collate([item[key] for item in batch])
    
    return collated

    
def create_simple_dataloader(dataset, batch_size, num_workers):
    dataloader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              shuffle=False,
                                              drop_last=True,
                                              collate_fn=custom_collate_fn)
    return dataloader

if __name__ == "__main__":
    dataset = WebDatasetwithCachedLatents(
        tar_path="/fs/cml-projects/yet-another-diffusion/pixelprose-shards/redcaps_part0/redcaps_part0_*.tar",
        latents_dir="/fs/cml-projects/yet-another-diffusion/pixelprose_sd3_latents/redcaps_part0",
        size=512,
        center_crop=False,
        random_flip=False,
        max_length=77,
        num_samples=100,
        randomize=True
    )
    ds_iter = iter(dataset)
    dataloader = create_simple_dataloader(dataset, batch_size=16, num_workers=4)
    for item in dataloader:
        print(item)
        break