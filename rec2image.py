import os
import mxnet as mx
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def save_images_from_rec(rec_path, output_dir):
    """
    Read a .rec file and convert its contents into a multi-level directory structure of images.

    Parameters:
        rec_path (str): Path to the .rec file.
        output_dir (str): Root directory to save the images.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the .rec file
    record = mx.recordio.MXRecordIO(rec_path, 'r')
    idx = 0

    # Estimate total records (requires record count or an alternative estimation method)
    # Note: Replace `total_records` with the actual number if known, or handle dynamically if possible.
    total_records = 0  # Update this to the actual record count if known

    with tqdm(total=total_records, desc="Processing images", unit="image") as pbar:
        while True:
            # Read the next record
            item = record.read()
            if not item:
                break

            # Decode the record
            header, img = mx.recordio.unpack(item)

            # Get the label (assuming header.label is the subdirectory name)
            label = int(header.label) if isinstance(header.label, (float, int)) else header.label
            label_dir = os.path.join(output_dir, str(label))

            # Ensure the subdirectory exists
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)

            # Convert the binary image data to a NumPy array
            img_array = mx.image.imdecode(img).asnumpy()

            # Save the image
            output_path = os.path.join(label_dir, f"image_{idx}.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

            idx += 1
            pbar.update(1)

    print(f"A total of {idx} images were saved across multiple subdirectories.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read a .rec file and convert its contents into a multi-level directory structure of images.")
    parser.add_argument("--rec_path", type=str, help="Path to the .rec file")
    parser.add_argument("--output_dir", type=str, help="Root directory to save the images")

    args = parser.parse_args()

    save_images_from_rec(args.rec_path, args.output_dir)