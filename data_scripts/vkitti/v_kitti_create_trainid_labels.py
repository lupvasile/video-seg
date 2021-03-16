import multiprocessing
from pathlib import Path

from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm

from config.config import Config
from dataset.dataset_metadata import vkittyClasses

color2label = {c.COLOR: c.TRAIN_ID for c in vkittyClasses}


def create_trainId_image(inputImage: Image.Image):
    width, height = inputImage.size
    img = Image.new("L", inputImage.size)

    input_load = inputImage.load()
    img_load = img.load()

    for j in range(width):
        for i in range(height):
            img_load[j, i] = color2label[input_load[j, i]]

    """
    input_np = np.array(inputImage)
    output_np = np.empty(input_np.shape[:2], dtype=np.uint8)
    for color, label in color2label.items():
        output_np[np.where((input_np==color).all(axis=2))] = label
    return Image.fromarray(output_np)
    """

    return img


def main():
    kitty_root = Path(Config.get_datadir('v_kitti'))
    assert Path.exists(kitty_root)

    files = [f for f in kitty_root.glob("Scene*/*/frames/classSegmentation/*/*") if 'trainId' not in f.name]
    files = tqdm(files)

    nrCores = multiprocessing.cpu_count()

    Parallel(n_jobs=nrCores)(delayed(convert_image)(file) for file in files)


def convert_image(file):
    assert 'trainId' not in file.name
    trainIdFile = file.with_name(file.name.replace('classgt', 'classgt_trainId'))
    # print(trainIdFile)
    original = Image.open(file)
    trainIdImage = create_trainId_image(original)
    trainIdImage.save(trainIdFile)


if __name__ == '__main__':
    main()
