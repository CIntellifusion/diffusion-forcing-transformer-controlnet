import os
import torch
import imageio
import numpy as np
import multiprocessing
from PIL import Image
from tqdm import tqdm
from io import BytesIO
import imageio.v3 as iio  # 更快、更推荐的 imageio API（也可用 imageio）
from typing import List

def rescale_and_crop(video: np.ndarray, resolution: int) -> np.ndarray:
    """
    Rescale and center-crop a video to the specified resolution.
    Args:
        video (np.ndarray): shape (T, H, W, C), dtype=uint8
        resolution (int): target resolution
    Returns:
        np.ndarray: shape (T, resolution, resolution, C), dtype=uint8
    """
    *_, h, w, _ = video.shape
    scale = max(resolution / h, resolution / w)
    h_scaled, w_scaled = round(h * scale), round(w * scale)
    row_crop = (h_scaled - resolution) // 2
    col_crop = (w_scaled - resolution) // 2

    def process_frame(frame: np.ndarray) -> np.ndarray:
        img = Image.fromarray(frame).resize((w_scaled, h_scaled), Image.Resampling.LANCZOS)
        return np.array(img)[row_crop : row_crop + resolution, col_crop : col_crop + resolution]

    return np.stack([process_frame(frame) for frame in video])

class ImageProcessor:
    def convert_images(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Convert a list of image bytes (as numpy arrays) into a single numpy array [B, H, W, 3].
        Each item is treated as raw bytes representing an RGB image.
        """
        np_images = []
        for image in images:
            img = Image.open(BytesIO(image.numpy().tobytes())).convert("RGB")
            np_images.append(np.array(img, dtype=np.uint8))  # [H, W, 3]
        return np.stack(np_images, axis=0)  # [B, H, W, 3]

    def save_images(self, batch: np.ndarray, output_dir: str, prefix: str = "image", ext: str = "png"):
        """
        Save a batch of numpy images to disk. Shape: [B, H, W, 3]
        """
        os.makedirs(output_dir, exist_ok=True)
        for i, img in enumerate(batch):
            Image.fromarray(img).save(os.path.join(output_dir, f"{prefix}_{i:03d}.{ext}"))
        print(f"✅ Saved {len(batch)} images to {output_dir}")

    def save_video(self, batch: np.ndarray, output_path: str, fps: int = 8):
        """
        Save a video from a batch of images. Shape: [B, H, W, 3]
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        batch = rescale_and_crop(batch, resolution=256)
        iio.imwrite(output_path, batch, fps=fps, codec='libx264')  # iio 自动处理帧格式
        print(f"🎞️ Saved video to {output_path}")




def process_single_file(args):
    torch_file, input_dir, output_dir, fps = args
    processor = ImageProcessor()
    file_path = os.path.join(input_dir, torch_file)
    data = torch.load(file_path)

    for i, sample in enumerate(data):
        if 'images' not in sample or not sample['images']:
            print(f"⚠️ Skipping sample {i} in {torch_file}: no valid 'images'")
            continue

        tensor_batch = processor.convert_images(sample['images'])
        basename = sample.get('key', f"{os.path.splitext(torch_file)[0]}_{i:03d}")
        output_path = os.path.join(output_dir, f"{basename}.mp4")
        processor.save_video(tensor_batch, output_path, fps=fps)


def process_torch_folder_parallel(input_dir: str, output_dir: str, fps: int = 10, num_workers: int = None):
    os.makedirs(output_dir, exist_ok=True)
    torch_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.torch')])

    # Prepare arguments for each task
    task_args = [(f, input_dir, output_dir, fps) for f in torch_files]

    # Use all CPU cores by default
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), len(torch_files))

    print(f"🚀 Launching {num_workers} workers to process {len(torch_files)} files...")

    with multiprocessing.Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_single_file, task_args), total=len(torch_files)))

    print(f"\n✅ Finished processing all files. Videos saved to: {output_dir}")

if __name__ == "__main__":
    process_torch_folder_parallel(
        input_dir="/mnt/mldl/v-diankunwu/datasets/re10k/extracted/re10k/test/",
        output_dir="/mnt/mldl/v-haoywu/datasets/Realestate10k/dfot/real-estate-10k/test_256",
        fps=10,
        num_workers=32  # 可根据机器核数调整
    )
    process_torch_folder_parallel(
        input_dir="/mnt/mldl/v-diankunwu/datasets/re10k/extracted/re10k/train/",
        output_dir="/mnt/mldl/v-haoywu/datasets/Realestate10k/dfot/real-estate-10k/training_256",
        fps=10,
        num_workers=32  # 可根据机器核数调整
    )
