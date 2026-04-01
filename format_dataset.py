import argparse
import os

from tqdm import tqdm

# Subdirectories containing images
SUBDIRS = ["images/train2017", "images/val2017", "images/test2017"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write image path lists (train2017.txt, val2017.txt, test2017.txt) under base_dir."
    )
    parser.add_argument(
        "--base-dir",
        "-b",
        default="./kitti_COCO",
        help="Base directory containing images/train2017, images/val2017, images/test2017",
    )
    args = parser.parse_args()
    base_dir = args.base_dir

    for subdir in SUBDIRS:
        full_path = os.path.join(base_dir, subdir)
        output_file = os.path.join(base_dir, f"{subdir.replace('images/', '')}.txt")

        relative_paths = []

        if os.path.exists(full_path):
            for filename in tqdm(os.listdir(full_path)):
                if filename.endswith((".jpg", ".png", ".jpeg")):
                    relative_paths.append(f"./{subdir}/{filename}")

        with open(output_file, "w") as f:
            f.write("\n".join(relative_paths))

        print(f"Saved {len(relative_paths)} image paths to {output_file}")


if __name__ == "__main__":
    main()
