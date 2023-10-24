import os
import sys
import shutil
import argparse
import concurrent.futures

from tqdm import tqdm


def copy_file(src, dst):
    if os.path.exists(src):
        shutil.copy2(src, dst)

def organize_and_copy(line, data_dir, class_grouping, dst_fold_group_dir, split):
    image_filename = f'{line.strip()}.jpg'
    label_filename = f'{line.strip()}.txt'

    src_image = os.path.join(data_dir, 'images', image_filename)
    dst_image = os.path.join(dst_fold_group_dir, 'images', split, image_filename)
    copy_file(src_image, dst_image)

    src_label = os.path.join(data_dir, f'labels_{class_grouping}', label_filename)
    dst_label = os.path.join(dst_fold_group_dir, 'labels', split, label_filename)
    copy_file(src_label, dst_label)

    if class_grouping in ['3cls', '8cls']:
        src_label_size = os.path.join(data_dir, f'labels_{class_grouping}_size', label_filename)
        dst_label_size = os.path.join(dst_fold_group_dir, 'labels_size', split, label_filename)
        copy_file(src_label_size, dst_label_size)

def process_split(fold_dir, data_dir, class_grouping, dst_fold_group_dir, split):
    txt_file_path = os.path.join(fold_dir, f'{split}.txt')
    if os.path.exists(txt_file_path):
        with open(txt_file_path, 'r') as file:
            lines = file.readlines()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(organize_and_copy, line, data_dir, class_grouping, dst_fold_group_dir, split)
                           for line in lines]
                for future in concurrent.futures.as_completed(futures):
                    future.result()


def process_class_grouping(fold_dir, data_dir, datasets_dir, fold, class_grouping):
    dst_fold_group_dir = os.path.join(datasets_dir, f'{fold}', f'{class_grouping}_{fold}')

    os.makedirs(os.path.join(dst_fold_group_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dst_fold_group_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(dst_fold_group_dir, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(dst_fold_group_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dst_fold_group_dir, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(dst_fold_group_dir, 'labels', 'test'), exist_ok=True)

    if class_grouping in ['3cls', '8cls']:
        os.makedirs(os.path.join(dst_fold_group_dir, 'labels_size', 'train'), exist_ok=True)
        os.makedirs(os.path.join(dst_fold_group_dir, 'labels_size', 'val'), exist_ok=True)
        os.makedirs(os.path.join(dst_fold_group_dir, 'labels_size', 'test'), exist_ok=True)

    for split in ['train', 'val', 'test']:
        process_split(fold_dir, data_dir, class_grouping, dst_fold_group_dir, split)


def organize_datasets(data_dir, datasets_dir):
    splits_dir = os.path.join(data_dir, 'splits')

    if not os.path.exists(os.path.join(data_dir, 'images')):
        print(f'[ERROR] {os.path.join(data_dir, "images")} does not exist...')
        sys.exit()

    if not os.listdir(os.path.join(data_dir, 'images')):
        response = input('[INPUT] Image directory is empty. Continue? (y/n): ')
        if response.lower() != 'y':
            print('[EXITING]')
            sys.exit()
        print('[INFO] Continuing...')

    for fold in tqdm(os.listdir(splits_dir), desc='Organising folds...'):
        fold_dir = os.path.join(splits_dir, fold)
        class_groupings = ['1cls', '3cls', '8cls']
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_class_grouping, fold_dir, data_dir, datasets_dir, fold, class_grouping) for class_grouping in class_groupings]
            # Progress bar for class groupings
            for _ in tqdm(concurrent.futures.as_completed(futures), desc='Organising class groups...', total=len(class_groupings)):
                pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Organize datasets.')
    parser.add_argument('--data-dir', default='data', required=False, help='Directory containing the data')
    parser.add_argument('--output-dir', default='datasets', required=False, help='Directory to output the organized datasets')
    args = parser.parse_args()

    organize_datasets(args.data_dir, args.output_dir)
