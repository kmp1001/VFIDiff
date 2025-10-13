import argparse
from pathlib import Path
import random


def write_path_to_txt(dir_folder, txt_path, search_key, num_files=None):
    '''
    Scans the files in the given folder and writes them into a txt file
    Input:
        dir_folder: path of the target folder
        txt_path: path to save the txt file
        search_key: e.g., '*.png'
        num_files: optional, limit the number of files to write
    '''
    txt_path = Path(txt_path) if not isinstance(txt_path, Path) else txt_path
    dir_folder = Path(dir_folder) if not isinstance(dir_folder, Path) else dir_folder

    if txt_path.exists():
        txt_path.unlink()  # Delete existing file

    path_list = [str(x) for x in dir_folder.glob(search_key)]
    random.shuffle(path_list)

    if num_files is not None:
        path_list = path_list[:num_files]

    with open(txt_path, mode='w') as ff:
        for line in path_list:
            ff.write(line + '\n')


def main():
    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Write file paths from a directory to a txt file.")

    # Adding arguments
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help="Path to the input folder that contains files."
    )

    parser.add_argument(
        '--output_txt',
        type=str,
        required=True,
        help="Path to the output txt file."
    )

    parser.add_argument(
        '--search_key',
        type=str,
        default='*.*',
        help="Search key for filtering files (e.g., '*.png'). Default is '*.*' to match all files."
    )

    parser.add_argument(
        '--num_files',
        type=int,
        default=None,
        help="Optional: Limit the number of files to write. If not specified, write all."
    )

    # Parse arguments
    args = parser.parse_args()

    # Call the function to write paths to txt
    write_path_to_txt(args.input_dir, args.output_txt, args.search_key, args.num_files)


if __name__ == "__main__":
    main()
