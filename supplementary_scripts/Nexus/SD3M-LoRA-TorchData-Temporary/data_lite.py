import torch
from torch.utils.data import IterableDataset
import glob
import random
import webdataset as wds
from torchvision import transforms
import os
import numpy as np
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader



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
        self.randomize = randomize
        self.random_seed = random.randint(0, 2**32 - 1)
        self.current_position = 0

        print("Total tar files: ", len(self.tar_files))
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
        print("Tar files with latents: ", len(self.tar_files))
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
                         'pooled_prompt_embeds'),
        )

    def get_state(self):
        return {
            'random_seed': self.random_seed,
            'current_position': self.current_position,
        }

    # def set_state(self, state):
    #     self.random_seed = state['random_seed']
    #     self.current_position = state['current_position']
    #     random.seed(self.random_seed)
    #     if self.randomize:
    #         random.shuffle(self.tar_files)

    def set_state(self, state):
        self.random_seed = state['random_seed']
        self.current_position = state['current_position']
        if dist.is_available() and dist.is_initialized():
            self.random_seed = torch.tensor(self.random_seed).to('cuda')
            dist.broadcast(self.random_seed, src=0)
            self.random_seed = self.random_seed.item()
        random.seed(self.random_seed)
        if self.randomize:
            random.shuffle(self.tar_files)

    def get_tar_files_with_latents(self):
        tar_files_with_latents = []
        for tar_file in self.tar_files:
            tar_file_name = tar_file.replace('.tar', '')
            tar_file_name = tar_file_name.split('/')[-2:]
            tar_file_name = '/'.join(tar_file_name)
            tar_file_name = os.path.join(self.latents_dir, tar_file_name)
            if os.path.exists(tar_file_name):
                tar_files_with_latents.append(tar_file)
        return tar_files_with_latents

    def process_sample(self, sample):
        if sample is None:
            return None
        base_name = sample['__url__']
        key = sample['__key__']
        base_name = base_name.replace('.tar', '')
        base_name = base_name.split('/')[-2:]
        base_name = '/'.join(base_name)
        fname = f"{base_name}/{key}.npz"
        fname = os.path.join(self.latents_dir, fname)
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

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        # Shuffle the tar files
        if self.randomize:
            random.seed(self.random_seed + rank)
            random.shuffle(self.tar_files)

        # Distribute shards across workers and processes
        num_shards_per_worker = len(self.tar_files) // (num_workers * world_size)
        start_idx = (rank * num_workers + worker_id) * num_shards_per_worker
        end_idx = start_idx + num_shards_per_worker

        worker_tar_files = self.tar_files[start_idx:end_idx]

        dataset = wds.DataPipeline(
            wds.SimpleShardList(worker_tar_files),
            wds.shuffle(100),
            wds.tarfile_to_samples(),
            wds.decode('pil'),
            wds.map(self.process_sample),
            wds.to_tuple('pixel_values', 'prompt_embeds', 'pooled_prompt_embeds'),
        )

        return iter(dataset)

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    collated = {}
    collated['pixel_values'] = torch.stack([item[0] for item in batch])
    collated['prompt_embeds'] = torch.stack([item[1] for item in batch])
    collated['pooled_prompt_embeds'] = torch.stack([item[2] for item in batch])
    return collated

# def create_simple_dataloader(dataset, batch_size, num_workers, distributed=False):
#     sampler = torch.utils.data.distributed.DistributedSampler(dataset) if distributed else None
#     return StatefulDataLoader(
#         dataset,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         shuffle=False,  # IterableDataset doesn't support shuffle in the dataloader
#         sampler=sampler,
#         drop_last=True,
#         collate_fn=custom_collate_fn
#     )

def create_simple_dataloader(dataset, batch_size, num_workers, distributed=False):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if distributed else None
    
    dataloader_kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'shuffle': False,
        'sampler': sampler,
        'drop_last': True,
        'collate_fn': custom_collate_fn,
        'snapshot_every_n_steps': 10  # Adjust this based on your checkpointing frequency
    }
    
    if num_workers > 0:
        dataloader_kwargs['persistent_workers'] = True
    
    return StatefulDataLoader(**dataloader_kwargs)

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
