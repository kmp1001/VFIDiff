import os
import argparse
import random
from pathlib import Path


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Consistently split matching PNG images from three folders into train/test sets with an 8:2 ratio and generate six text files.'
    )
    parser.add_argument('--folder1', type=str, required=True, help='Path to the first folder')
    parser.add_argument('--folder2', type=str, required=True, help='Path to the second folder')
    parser.add_argument('--folder3', type=str, required=True, help='Path to the third folder')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for output text files')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio, default is 0.8')
    parser.add_argument('--seed', type=int, default=42, help='Random seed, default is 42 for reproducibility')
    return parser.parse_args()


def get_png_filenames(folder):
    """
    Get the names of all PNG files in the specified folder (without paths).
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"Path {folder} is not a valid folder.")
    png_files = sorted([f.name for f in folder_path.glob('*.png') if f.is_file()])
    return png_files


def verify_filenames(filenames1, filenames2, filenames3):
    """
    Verify that the PNG filenames in the three folders are exactly the same.
    """
    set1 = set(filenames1)
    set2 = set(filenames2)
    set3 = set(filenames3)
    if set1 != set2 or set1 != set3:
        missing_in_2 = set1 - set2
        missing_in_3 = set1 - set3
        missing_in_1 = set2 - set1
        missing_in_3_from_1 = set3 - set1
        error_message = "PNG filenames in the three folders do not match exactly.\n"
        if missing_in_2:
            error_message += f"Missing in folder2: {missing_in_2}\n"
        if missing_in_3:
            error_message += f"Missing in folder3: {missing_in_3}\n"
        if missing_in_1:
            error_message += f"Missing in folder1: {missing_in_1}\n"
        if missing_in_3_from_1:
            error_message += f"Missing in folder1: {missing_in_3_from_1}\n"
        raise ValueError(error_message)


def split_filenames(filenames, train_ratio=0.8, seed=42):
    """
    Split the filename list into training and testing sets according to the specified ratio.
    """
    random.seed(seed)
    shuffled = filenames.copy()
    random.shuffle(shuffled)
    split_point = int(len(shuffled) * train_ratio)
    train_files = shuffled[:split_point]
    test_files = shuffled[split_point:]
    return train_files, test_files


def write_list_to_txt(file_list, file_path):
    """
    Write a list of file paths to a text file, one path per line.
    """
    with open(file_path, 'w') as f:
        for item in file_list:
            f.write(f"{item}\n")


def main():
    args = parse_arguments()

    # Get PNG filenames from the three folders
    filenames1 = get_png_filenames(args.folder1)
    filenames2 = get_png_filenames(args.folder2)
    filenames3 = get_png_filenames(args.folder3)

    # Verify that the filenames in the three folders are identical
    try:
        verify_filenames(filenames1, filenames2, filenames3)
    except ValueError as e:
        print(e)
        return

    # Use one folder's filenames for splitting since they are identical
    all_filenames = filenames1
    train_filenames, test_filenames = split_filenames(all_filenames, args.train_ratio, args.seed)

    # Prepare the output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate train.txt and test.txt for each folder
    folders = [args.folder1, args.folder2, args.folder3]
    for idx, folder in enumerate(folders, start=1):
        folder_path = Path(folder)
        # Training set paths
        train_paths = [str(folder_path / fname) for fname in train_filenames]
        # Testing set paths
        test_paths = [str(folder_path / fname) for fname in test_filenames]
        # Output file paths
        train_txt = output_dir / f"folder{idx}_train.txt"
        test_txt = output_dir / f"folder{idx}_test.txt"
        # Write to text files
        write_list_to_txt(train_paths, train_txt)
        write_list_to_txt(test_paths, test_txt)
        print(f"Generated {train_txt} and {test_txt} for folder{idx}")


if __name__ == "__main__":
    main()
