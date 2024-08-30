#SINGLE NODE RUN

import torch
from torch.utils.data import IterableDataset
import torch
from torch.utils.data import IterableDataset
from torchdata.datapipes.iter import IterableWrapper, FileOpener, TarArchiveLoader, Shuffler, Mapper, Batcher, Filter
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService, InProcessReadingService
from torchdata.datapipes.iter import Batcher

from torchvision import transforms
from PIL import Image
import io
import itertools
import glob
import math
import os
import numpy as np


class StatefulWebDataset(IterableDataset):
    def __init__(self, tar_path, tokenizers, latents_dir, size=1024, center_crop=False, random_flip=False, max_length=77, num_samples=None):
        self.tokenizers = tokenizers
        self.image_size = size
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.max_length = max_length
        self.num_samples = num_samples
        self.current_sample = 0
        self.latents_dir = latents_dir

        self.tar_files = sorted(glob.glob(tar_path))
        if not self.tar_files:
            raise FileNotFoundError(f"No files found matching the pattern: {tar_path}")

        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.datapipe = self._create_datapipe()

    def _create_datapipe(self):
        datapipe = IterableWrapper(self.tar_files)
        datapipe = FileOpener(datapipe, mode="rb")
        datapipe = TarArchiveLoader(datapipe)
        datapipe = Shuffler(datapipe, buffer_size=10000)  # Reduced buffer size
        
        def group_data(data):
            image_data = next((d for d in data if d[0].endswith('.jpg')), None)
            if image_data:
                return (image_data[0], image_data[1])
            return None

        datapipe = Batcher(datapipe, 2)
        datapipe = Mapper(datapipe, group_data)
        datapipe = Filter(datapipe, lambda x: x is not None)
        
        return datapipe

    def process_sample(self, sample):
        if sample is None:
            return None
        
        file_path, image_stream = sample
        print(file_path)
        
        # Defer image loading and transformation
        def load_and_transform_image():
            image_data = image_stream.read()
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            size = 512
            center_crop=False 
            random_flip=False
            image_transforms = transforms.Compose([
                        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                        transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                        transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ])
            return image_transforms(image)
            #return self.image_transforms(image)


        
        # Extract the base filename from the full path
        file_name = os.path.basename(file_path)
        
        # Construct the path for the latent file
        latents_dir = '/fs/cml-projects/yet-another-diffusion/pixelprose_sd3_latents/'
        fname = file_path.replace('.jpg','.npz').replace('.tar','')
        fname = '/'.join(fname.split('/')[-3:])
        latent_file = os.path.join(latents_dir,fname)

        #latent_file = os.path.join(latents_dir, file_name.replace('.jpg', '.npz'))

        if not os.path.exists(latent_file):
            return None

        # Defer latent loading
        def load_latents():
            latents = np.load(latent_file)
            prompt_embeds = torch.from_numpy(latents['prompt_embeds'])
            pooled_prompt_embeds = torch.from_numpy(latents['pooled_prompt_embeds'])
            return prompt_embeds, pooled_prompt_embeds

        #self.current_sample += 1
        return {
            "file_name": file_name,
            "pixel_values": load_and_transform_image,
            "load_latents": load_latents
        }

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = self.num_samples if self.num_samples is not None else None
        else:
            per_worker = int(math.ceil((self.num_samples or float('inf')) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.num_samples) if self.num_samples is not None else None

        iterator = itertools.islice(self.datapipe, iter_start, iter_end)
        for item in iterator:
            processed_item = self.process_sample(item)
            if processed_item is not None:
                yield processed_item

    def __len__(self):
        return self.num_samples if self.num_samples is not None else float('inf')

    def state_dict(self):
        return {"current_sample": self.current_sample}

    def load_state_dict(self, state_dict):
        self.current_sample = state_dict["current_sample"]
        self.datapipe = self._create_datapipe()
    
    def estimate_num_batches(self, batch_size):
        if self.num_samples is not None:
            return math.ceil(self.num_samples / batch_size)
        else:
            return None

def create_dataloader(dataset, batch_size, num_workers, distributed=False):
    if distributed:
        dataset.datapipe = dataset.datapipe.sharding_filter()

    dataset.datapipe = Batcher(dataset.datapipe, batch_size=batch_size)
    dataset.datapipe = Mapper(dataset.datapipe, custom_collate_fn)

    if num_workers > 0:
        reading_service = MultiProcessingReadingService(num_workers=num_workers)
    else:
        reading_service = InProcessReadingService()
    
    dataloader_kwargs = {
        'datapipe': dataset.datapipe,
        'reading_service': reading_service,
    }

    if distributed:
        dataloader_kwargs['datapipe_adapter_fn'] = lambda dp: dp.fullsync()
    
    dataloader = DataLoader2(**dataloader_kwargs)

    estimated_num_batches = dataset.estimate_num_batches(batch_size)

    return dataloader, estimated_num_batches


def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    processed_batch = []
    for sample in batch:
        processed_item = StatefulWebDataset.process_sample(None, sample)
        if processed_item is not None:
            processed_batch.append(processed_item)

    if len(processed_batch) == 0:
        return None

    collated = {}
    for key in processed_batch[0].keys():
        if key == 'pixel_values':
            collated[key] = torch.stack([item[key]() for item in processed_batch])
        elif key == 'load_latents':
            prompt_embeds, pooled_prompt_embeds = zip(*[item[key]() for item in processed_batch])
            collated['prompt_embeds'] = torch.stack(prompt_embeds)
            collated['pooled_prompt_embeds'] = torch.stack(pooled_prompt_embeds)
        else:
            collated[key] = [item[key] for item in processed_batch]
    
    return collated

