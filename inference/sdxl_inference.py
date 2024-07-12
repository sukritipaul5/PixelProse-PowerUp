import torch
from diffusers import DiffusionPipeline
import os
import csv
import ipdb

# Read prompts and file names from a CSV file
def read_prompts(file_path):
    prompts = {}
    with open(file_path, 'r', newline='') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            prompts[row['file_name']] = row['prompt']
    return prompts

# Initialize the pipeline
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")
pipe.enable_vae_slicing()

#I/O Directories
input_file = "/lus/eagle/projects/DemocAI/sukriti/diffusers/inference/prompts.csv"
output_dir = "/lus/eagle/projects/DemocAI/sukriti/diffusers/inference/Outputs"
os.makedirs(output_dir, exist_ok=True)

prompts = read_prompts(input_file)
ipdb.set_trace()

for file_name, prompt in prompts.items():
    print(f"Generating image for: {file_name}")
    image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    output_path = os.path.join(output_dir, f"{file_name}.png")
    image.save(output_path)
    print(f"Image saved: {output_path}")

print("All images generated and saved.")



