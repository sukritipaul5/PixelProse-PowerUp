import webdataset as wds
from torch.utils.data import IterableDataset
from torchvision import transforms
from PIL import Image
import io
import torch

class WebDatasetAdapter(IterableDataset):
    def __init__(self, tar_path, tokenizers, size=1024, center_crop=False, random_flip=False, max_length=77, num_samples=None):
        self.dataset = wds.WebDataset(tar_path, nodesplitter=wds.split_by_node).shuffle(1000)
        self.tokenizers = tokenizers
        self.image_size = size
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

    def __iter__(self):
        count = 0
        for item in self.dataset:
            if self.num_samples is not None and count >= self.num_samples:
                break

            image = Image.open(io.BytesIO(item['jpg'])).convert('RGB')
            image = self.image_transforms(image)
            
            caption = item['txt'].decode('utf-8').strip()

            # Tokenize the prompt with each tokenizer
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
            
            yield {
                "pixel_values": image,
                "tokenized_prompts": tokenized_prompts
            }
            count += 1

    def __len__(self):
        return self.num_samples if self.num_samples is not None else float('inf')