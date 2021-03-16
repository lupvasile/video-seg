import os
import shutil
from os.path import exists, join

from tqdm import tqdm


class cityscapes_copy():
    def __init__(self, root_gtfine, root_leftImg8bit_sequence, new_img8b_folder, subset, nr_frames: int):
        """subset should be 'train', 'val' or 'test' """
        self.nr_frames = nr_frames
        self.images_root = root_leftImg8bit_sequence
        self.labels_root = root_gtfine
        self.dest = new_img8b_folder

        assert exists(self.images_root), f'{self.images_root} not existing'
        assert exists(self.labels_root), f'{self.labels_root} not existing'
        assert exists(self.dest), f'{self.dest} not existing'

        self.images_root += subset
        self.labels_root += subset
        self.dest += subset

        self.filenames_labels = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in
                                 fn if f.endswith('labelTrainIds.png')]
        self.filenames_labels.sort()

        print(f'Found {len(self)} sequences to copy')

    def copy_one_sequence(self, filename_label):
        parts = filename_label.replace('\\', '/').split('/')[-1].split('_')
        city, seq, frame = parts[0], parts[1], parts[2]

        for idx in range(-self.nr_frames + 1, 1):
            x = int(frame) + idx
            img_filename = join(self.images_root, city, f'{city}_{seq}_{x:06}_leftImg8bit.png')
            dest_filename = join(self.dest, city, f'{city}_{seq}_{x:06}_leftImg8bit.png')

            if not os.path.exists(join(self.dest, city)):
                os.makedirs(join(self.dest, city))
            shutil.copy2(img_filename, dest_filename)

    def do_copy(self):
        for label in tqdm(self.filenames_labels):
            self.copy_one_sequence(label)

    def __len__(self):
        return len(self.filenames_labels)


if __name__ == '__main__':
    copier = cityscapes_copy('/media/vasi/Elements/datasets/cityscapes/gtFine/',
                             '/media/vasi/Elements/datasets/cityscapes/leftImg8bit_sequence/',
                             '/media/vasi/Elements/datasets/cityscapes/leftImg8bit_sequence_only_five/',
                             subset='val', nr_frames=5)

    copier.do_copy()
