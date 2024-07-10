from mpi4py import MPI
import os
import json
import tarfile
from PIL import Image
from streaming import MDSWriter
import argparse
import logging
import shutil
from io import BytesIO
import zstandard as zstd

# PProse Webd dir 
dataset_dirs = [
    #'/lus/eagle/projects/DemocAI/common-datasets/cc12m_gemini',
    '/lus/eagle/projects/DemocAI/common-datasets/redcaps-webdataset-merged'
]
output_path = '/lus/eagle/projects/DemocAI/common-datasets/PixelProseMDS'

# A dict to map input fields to their data types
columns = {
    'image': 'jpeg',
    'caption': 'str'
}

# Shard compression
compression = 'zstd'
compression_level = 3  # Lower compression level to reduce memory usage
# Shard size in bytes (256MB)
shard_size_bytes = 256 * 1024 * 1024

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Function to clear the output directory
# def clear_output_directory(directory):
#     if os.path.exists(directory):
#         shutil.rmtree(directory)
#     os.makedirs(directory)

# Custom compression function with adjusted settings
def custom_compress(data):
    compressor = zstd.ZstdCompressor(level=compression_level)
    return compressor.compress(data)

# Function to process each tar file and save valid samples as MDS shards
def process_tar_file(tar_path, out, current_shard_size):
    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers()
        json_files = [m for m in members if m.name.endswith('.json')]

        for member in json_files:
            f = tar.extractfile(member)
            if f is None:
                continue
            metadata = json.load(f)
            image_member_name = member.name.replace('.json', '.jpg')
            image_member = tar.getmember(image_member_name)
            caption = metadata.get('vlm_caption', '')

            if caption and image_member:
                try:
                    image_f = tar.extractfile(image_member)
                    image_data = image_f.read()
                    image = Image.open(BytesIO(image_data)).convert('RGB')
                    sample_size = len(caption.encode('utf-8')) + len(image_data)
                    if current_shard_size + sample_size > shard_size_bytes:
                        out.flush_shard()
                        current_shard_size = 0
                    out.write({'image': image, 'caption': caption})
                    current_shard_size += sample_size
                except Exception as e:
                    tempvar=0
                    #logger.error(f"Error processing {image_member_name}: {e}")

    return current_shard_size

# Function to process each directory and save valid samples as MDS shards
def process_directory(directory):
    shard_path = os.path.join(output_path, os.path.basename(directory))
    #clear_output_directory(shard_path)
    os.makedirs(shard_path)

    current_shard_size = 0

    with MDSWriter(out=shard_path, columns=columns, compression=compression, size_limit=shard_size_bytes) as out:
        out.compress = custom_compress  # Use custom compression function

        for root, _, files in os.walk(directory):
            tar_files = [os.path.join(root, file) for file in files if file.endswith('.tar')]

            for tar_file in tar_files:
                current_shard_size = process_tar_file(tar_file, out, current_shard_size)

    logger.info(f"Processed all samples from directory {directory}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert webd data to MDS shards")
    parser.add_argument('--troubleshoot', type=bool, default=False, help="Interactive Debug")
    return parser.parse_args()


def main():
    args = parse_args()
    process_args = [(directory) for directory in dataset_dirs]

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    #reduce if need be
    os.environ["OPENBLAS_NUM_THREADS"] = "4"

    # Distribute -> among nodes
    node_results = []
    for i, directory in enumerate(process_args):
        if i % size == rank:
            process_directory(directory)

    # Gather
    comm.Barrier()
    if rank == 0:
        logger.info("All directories processed. 16M dataset converted to MDS.")

if __name__ == "__main__":
    main()
