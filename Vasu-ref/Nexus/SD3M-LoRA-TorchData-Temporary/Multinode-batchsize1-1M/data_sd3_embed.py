import torch
from torch.utils.data import IterableDataset, DistributedSampler
from torchdata.datapipes.iter import IterableWrapper, FileOpener, TarArchiveLoader, Shuffler, Mapper, Batcher, Filter
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService, InProcessReadingService
from torchvision import transforms
from torchdata.datapipes.iter import ShardingFilter
from PIL import Image
import io
import glob
import os
import numpy as np
import math

from accelerate.logging import get_logger
logger = get_logger(__name__)

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
        
        def process_tar_item(item):
            file_path, file_obj = item
            # Read the content and close the file object immediately
            content = file_obj.read()
            file_obj.close()
            return (file_path, content)
        
        datapipe = Mapper(datapipe, process_tar_item)
        datapipe = Shuffler(datapipe, buffer_size=100)
        
        def group_data(data):
            image_data = next((d for d in data if d[0].endswith('.jpg')), None)
            if image_data:
                return (image_data[0], io.BytesIO(image_data[1]))
            return None
        
        datapipe = Batcher(datapipe, 2)
        datapipe = Mapper(datapipe, group_data)
        datapipe = Filter(datapipe, lambda x: x is not None)
        
        return datapipe

    def process_sample(self, sample):
        if sample is None:
            return None

        file_path, image_stream = sample

        # Check if the latent file exists before processing
        latents_dir = '/fs/cml-projects/yet-another-diffusion/pixelprose_sd3_latents/'
        fname = file_path.replace('.jpg','.npz').replace('.tar','')
        fname = '/'.join(fname.split('/')[-3:])
        latent_file = os.path.join(latents_dir, fname)

        if not os.path.exists(latent_file):
            logger.warning(f"Latent file not found for {file_path}. Skipping this sample.")
            return None

        self.current_sample += 1
        return (file_path, image_stream)  # Return the original 

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        logger.info("IN ITERRRRR")
        
        if worker_info is None:
            iter_start = 0
            iter_end = self.num_samples if self.num_samples is not None else None
            logger.info(f"Single worker mode: processing samples from {iter_start} to {iter_end}")
        else:
            if self.num_samples is not None:
                per_worker = int(math.ceil(self.num_samples / float(worker_info.num_workers)))
                worker_id = worker_info.id
                iter_start = worker_id * per_worker
                iter_end = min(iter_start + per_worker, self.num_samples)
            else:
                # For infinite datasets, we don't set iter_end
                iter_start = worker_info.id
                iter_end = None
            
            logger.info(f"Worker {worker_info.id} of {worker_info.num_workers}: "
                        f"processing samples from {iter_start} to {iter_end}")

        if iter_end is not None:
            iterator = itertools.islice(self.datapipe, iter_start, iter_end)
        else:
            # For infinite datasets, we use itertools.count() to skip samples
            iterator = itertools.islice(self.datapipe, iter_start, None)

        samples_processed = 0
        for item in iterator:
            processed_item = self.process_sample(item)
            if processed_item is not None:
                yield processed_item
                samples_processed += 1
                if samples_processed % 1000 == 0:
                    logger.info(f"Worker {worker_info.id if worker_info else 'single'} "
                                f"processed {samples_processed} samples")
            else:
                # If a sample is skipped, try to get the next one
                continue

            if iter_end is not None and samples_processed >= (iter_end - iter_start):
                logger.info(f"Worker {worker_info.id if worker_info else 'single'} "
                            f"finished processing {samples_processed} samples")
                break

        logger.info(f"Worker {worker_info.id if worker_info else 'single'} "
                    f"completed iteration, processed {samples_processed} samples")


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
    logger.info(f"Creating dataloader with batch_size={batch_size}, num_workers={num_workers}, distributed={distributed}")
    
    try:
        # if distributed:
        #     logger.info("Applying ShardingFilter for distributed setup")
        #     #dataset.datapipe = ShardingFilter(dataset.datapipe)
        
        logger.info(f"Applying Batcher with batch_size={batch_size}")
        dataset.datapipe = Batcher(dataset.datapipe, batch_size=1)#batch_size)
        
        logger.info("Applying Mapper with custom_collate_fn")
        #dataset.datapipe = Mapper(dataset.datapipe, custom_collate_fn)
        dataset.datapipe = Mapper(dataset.datapipe, fn=lambda x: custom_collate_fn(x, dataset))

        if num_workers > 0:
            logger.info(f"Using MultiProcessingReadingService with {num_workers} workers")
            reading_service = MultiProcessingReadingService(num_workers=num_workers)
        else:
            logger.info("Using InProcessReadingService")
            reading_service = InProcessReadingService()

        dataloader_kwargs = {
            'datapipe': dataset.datapipe,
            'reading_service': reading_service,
        }

        logger.info("Creating DataLoader2")
        dataloader = DataLoader2(**dataloader_kwargs)

        estimated_num_batches = dataset.estimate_num_batches(batch_size)
        logger.info(f"Estimated number of batches: {estimated_num_batches}")

        return dataloader, estimated_num_batches

    except Exception as e:
        logger.error(f"Error in create_dataloader: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def custom_collate_fn(batch, dataset):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    # Assuming the batch items are tuples in the order: (file_path, image_stream)
    file_names = []
    pixel_values = []
    prompt_embeds = []
    pooled_prompt_embeds = []

    for item in batch:
        file_path, image_stream = item

        # Process the image using the dataset's transform
        image_data = image_stream.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        pixel_value = dataset.image_transforms(image)

        # Load latents
        latents_dir = '/fs/cml-projects/yet-another-diffusion/pixelprose_sd3_latents/'
        fname = file_path.replace('.jpg','.npz').replace('.tar','')
        fname = '/'.join(fname.split('/')[-3:])
        latent_file = os.path.join(latents_dir, fname)

        if os.path.exists(latent_file):
            latents = np.load(latent_file)
            prompt_embed = torch.from_numpy(latents['prompt_embeds'])
            pooled_prompt_embed = torch.from_numpy(latents['pooled_prompt_embeds'])

            file_names.append(os.path.basename(file_path))
            pixel_values.append(pixel_value)
            prompt_embeds.append(prompt_embed)
            pooled_prompt_embeds.append(pooled_prompt_embed)
        #else: 
            #logger.warning(f"Latent file not found for {file_path}. Skipping this sample.")

    if len(file_names) == 0:
        return None

    return {
        'file_name': file_names,
        'pixel_values': torch.stack(pixel_values),
        'prompt_embeds': torch.stack(prompt_embeds),
        'pooled_prompt_embeds': torch.stack(pooled_prompt_embeds)
    }

