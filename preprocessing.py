from PIL import Image
from torchvision import transforms
from pathlib import Path
from utils import load_configs

class preprocessing:
    def __init__(self, celeba_256_dir, celeba_128_dir, celeba_64_dir):
        self.celeba_256_dir = celeba_256_dir
        self.celeba_128_dir = celeba_128_dir
        self.celeba_64_dir = celeba_64_dir

    def preprocess_celeba(self, path):
        crop = transforms.CenterCrop((160, 160))
        resample = Image.LANCZOS
        img = Image.open(path)
        img = crop(img)

        img_256_path = self.celeba_256_dir / path.name
        img.resize((256, 256), resample=resample).save(img_256_path)

        img_128_path = self.celeba_128_dir / path.name
        img.resize((128, 128), resample=resample).save(img_128_path)

        img_64_path = self.celeba_64_dir / path.name
        img.resize((64, 64), resample=resample).save(img_64_path)
        return

if __name__ == "__main__":
    configs = load_configs()['data']
    celeba_folder = Path(configs['CELEBA_FOLDER'])
    save_root = Path(configs['CELEBA_PREPROCESSED_ROOT'])

    celeba_64_dir = save_root / "CelebA_64"
    celeba_64_dir.mkdir(parents=True, exist_ok=True)
    celeba_128_dir = save_root / "CelebA_128"
    celeba_128_dir.mkdir(parents=True, exist_ok=True)
    celeba_256_dir = save_root / "CelebA_256"
    celeba_256_dir.mkdir(parents=True, exist_ok=True)
    ppc = preprocessing(celeba_256_dir, celeba_128_dir, celeba_64_dir)
    
    paths = list(celeba_folder.glob('*.jpg'))

    from multiprocessing import Pool
    with Pool() as pool:
        pool.map(ppc.preprocess_celeba, paths)

