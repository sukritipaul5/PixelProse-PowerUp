import torch
from torch.utils.data import IterableDataset
from torchdata.datapipes.iter import IterableWrapper, FileOpener, TarArchiveLoader, Shuffler, Mapper, Batcher, Filter
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService, InProcessReadingService
from torchvision import transforms
from PIL import Image
import io
import itertools
import glob
import math
import os



class StatefulWebDataset(IterableDataset):
    def __init__(self, tar_path, tokenizers, size=1024, center_crop=False, random_flip=False, max_length=77, num_samples=None):
        self.tokenizers = tokenizers
        self.image_size = size
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.max_length = max_length
        self.num_samples = num_samples
        self.current_sample = 0

        # Expand the glob pattern (tar * in the dataloading snippet in temp_sd3_dataswap)
        #Also handle None entries for image-captions
        self.tar_files = sorted(glob.glob(tar_path))
        if not self.tar_files:
            raise FileNotFoundError(f"No files found matching the pattern: {tar_path}")
        print(f"Found {len(self.tar_files)} tar files: {self.tar_files}")  

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
        datapipe = Shuffler(datapipe, buffer_size=1000)
        
        def group_data(data):
            image_data = next((d for d in data if d[0].endswith('.jpg')), None)
            caption_data = next((d for d in data if d[0].endswith('.txt')), None)
            if image_data and caption_data:
                return (image_data[1], caption_data[1])
            return None

        datapipe = Batcher(datapipe, 3)  # Group by 3 as each sample has .jpg, .txt, and .json (we're not using wds webd)
        datapipe = Mapper(datapipe, group_data)
        
        # Read StreamWrapper content
        def read_stream_wrapper(sample):
            if sample is None:
                return None
            image_stream, caption_stream = sample
            image_data = image_stream.read()
            caption_data = caption_stream.read()
            return (image_data, caption_data)
        
        datapipe = Mapper(datapipe, read_stream_wrapper)
        datapipe = Filter(datapipe, lambda x: x is not None)  
        datapipe = Mapper(datapipe, self.process_sample)
        datapipe = Filter(datapipe, lambda x: x is not None)  
        
        return datapipe

    def process_sample(self, sample):
        if sample is None:
            return None
        
        image_data, caption_data = sample
        try:
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            image = self.image_transforms(image)
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
        
        try:
            caption = caption_data.decode('utf-8').strip()
        except Exception as e:
            print(f"Error processing caption: {e}")
            return None

        tokenized_prompts = []
        for tokenizer in self.tokenizers:
            tokenized_prompt = tokenizer(
                caption,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            tokenized_prompts.append({
                "input_ids": tokenized_prompt.input_ids.squeeze(),
                "attention_mask": tokenized_prompt.attention_mask.squeeze()
            })
        
        self.current_sample += 1
        return {
            "pixel_values": image,
            "tokenized_prompts": tokenized_prompts
        }

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading
            iter_start = 0
            iter_end = self.num_samples if self.num_samples is not None else None
        else:  # in a worker process
            per_worker = int(math.ceil((self.num_samples or float('inf')) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.num_samples) if self.num_samples is not None else None

        iterator = itertools.islice(self.datapipe, iter_start, iter_end)
        return iterator

    def __len__(self):
        return self.num_samples if self.num_samples is not None else float('inf')

    def state_dict(self):
        return {"current_sample": self.current_sample}

    def load_state_dict(self, state_dict):
        self.current_sample = state_dict["current_sample"]
        # IMPORTANT: Recreate the datapipe to reflect the new state
        self.datapipe = self._create_datapipe()
    
    def estimate_num_batches(self, batch_size):
        if self.num_samples is not None:
            return math.ceil(self.num_samples / batch_size)
        else:
            # If num_samples is not set, we can't estimate the number of batches gaah
            return None


def create_dataloader(dataset, batch_size, num_workers, distributed=False):
    if distributed:
        # Sharding
        dataset.datapipe = dataset.datapipe.sharding_filter()
    # Batching
    dataset.datapipe = Batcher(dataset.datapipe, batch_size=batch_size)
    # Custom collate function
    dataset.datapipe = Mapper(dataset.datapipe, custom_collate_fn)

    # Choose appropriate reading service
    if num_workers > 0:
        reading_service = MultiProcessingReadingService(num_workers=num_workers)
    else:
        reading_service = InProcessReadingService()
    
    dataloader = DataLoader2(
        datapipe=dataset.datapipe,
        reading_service=reading_service
    )


    estimated_num_batches = dataset.estimate_num_batches(batch_size)

    return dataloader, estimated_num_batches

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)
