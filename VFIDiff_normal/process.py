import os
import argparse
import random
from pathlib import Path


def parse_arguments():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description='将三个文件夹中的对应 PNG 图片按 8:2 比例进行一致性分割，并生成六个文本文件。')
    parser.add_argument('--folder1', type=str, required=True, help='第一个文件夹的路径')
    parser.add_argument('--folder2', type=str, required=True, help='第二个文件夹的路径')
    parser.add_argument('--folder3', type=str, required=True, help='第三个文件夹的路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出文本文件的目录')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例，默认为0.8')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，默认为42以保证可重复性')
    return parser.parse_args()


def get_png_filenames(folder):
    """
    获取指定文件夹中所有 PNG 文件的名称（不包含路径）。
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"路径 {folder} 不是一个有效的文件夹。")
    png_files = sorted([f.name for f in folder_path.glob('*.png') if f.is_file()])
    return png_files


def verify_filenames(filenames1, filenames2, filenames3):
    """
    验证三个文件夹中的 PNG 文件名称是否完全一致。
    """
    set1 = set(filenames1)
    set2 = set(filenames2)
    set3 = set(filenames3)
    if set1 != set2 or set1 != set3:
        missing_in_2 = set1 - set2
        missing_in_3 = set1 - set3
        missing_in_1 = set2 - set1
        missing_in_3_from_1 = set3 - set1
        error_message = "三个文件夹中的 PNG 文件名称不完全一致。\n"
        if missing_in_2:
            error_message += f"在文件夹2中缺少: {missing_in_2}\n"
        if missing_in_3:
            error_message += f"在文件夹3中缺少: {missing_in_3}\n"
        if missing_in_1:
            error_message += f"在文件夹1中缺少: {missing_in_1}\n"
        if missing_in_3_from_1:
            error_message += f"在文件夹1中缺少: {missing_in_3_from_1}\n"
        raise ValueError(error_message)


def split_filenames(filenames, train_ratio=0.8, seed=42):
    """
    将文件名列表按指定比例分为训练集和测试集。
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
    将文件路径列表写入指定的文本文件，每行一个路径。
    """
    with open(file_path, 'w') as f:
        for item in file_list:
            f.write(f"{item}\n")


def main():
    args = parse_arguments()

    # 获取三个文件夹中的 PNG 文件名
    filenames1 = get_png_filenames(args.folder1)
    filenames2 = get_png_filenames(args.folder2)
    filenames3 = get_png_filenames(args.folder3)

    # 验证三个文件夹中的文件名是否一致
    try:
        verify_filenames(filenames1, filenames2, filenames3)
    except ValueError as e:
        print(e)
        return

    # 使用其中一个文件夹的文件名进行分割（因为它们是相同的）
    all_filenames = filenames1
    train_filenames, test_filenames = split_filenames(all_filenames, args.train_ratio, args.seed)

    # 准备输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 为每个文件夹生成 train.txt 和 test.txt
    folders = [args.folder1, args.folder2, args.folder3]
    for idx, folder in enumerate(folders, start=1):
        folder_path = Path(folder)
        # 训练集路径
        train_paths = [str(folder_path / fname) for fname in train_filenames]
        # 测试集路径
        test_paths = [str(folder_path / fname) for fname in test_filenames]
        # 输出文件路径
        train_txt = output_dir / f"folder{idx}_train.txt"
        test_txt = output_dir / f"folder{idx}_test.txt"
        # 写入文本文件
        write_list_to_txt(train_paths, train_txt)
        write_list_to_txt(test_paths, test_txt)
        print(f"已为文件夹{idx}生成 {train_txt} 和 {test_txt}")


if __name__ == "__main__":
    main()
