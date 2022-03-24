import numpy as np
from glob import glob
from os import listdir
from os.path import join
from dataset import AbstractDataset

SPLITS = ["train", "test"]


class CelebDF(AbstractDataset):
    """
    Celeb-DF v2 Dataset proposed in "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics".
    """

    def __init__(self, cfg, seed=2022, transforms=None, transform=None, target_transform=None):
        # pre-check
        if cfg['split'] not in SPLITS:
            raise ValueError(f"split should be one of {SPLITS}, but found {cfg['split']}.")
        super(CelebDF, self).__init__(cfg, seed, transforms, transform, target_transform)
        print(f"Loading data from 'Celeb-DF' of split '{cfg['split']}'"
              f"\nPlease wait patiently...")
        self.categories = ['original', 'fake']
        self.root = cfg['root']
        images_ids = self.__get_images_ids()
        test_ids = self.__get_test_ids()
        train_ids = [images_ids[0] - test_ids[0],
                     images_ids[1] - test_ids[1],
                     images_ids[2] - test_ids[2]]
        self.images, self.targets = self.__get_images(
            test_ids if cfg['split'] == "test" else train_ids, cfg['balance'])
        assert len(self.images) == len(self.targets), "The number of images and targets not consistent."
        print("Data from 'Celeb-DF' loaded.\n")
        print(f"Dataset contains {len(self.images)} images.\n")

    def __get_images_ids(self):
        youtube_real = listdir(join(self.root, 'YouTube-real', 'images'))
        celeb_real = listdir(join(self.root, 'Celeb-real', 'images'))
        celeb_fake = listdir(join(self.root, 'Celeb-synthesis', 'images'))
        return set(youtube_real), set(celeb_real), set(celeb_fake)

    def __get_test_ids(self):
        youtube_real = set()
        celeb_real = set()
        celeb_fake = set()
        with open(join(self.root, "List_of_testing_videos.txt"), "r", encoding="utf-8") as f:
            contents = f.readlines()
            for line in contents:
                name = line.split(" ")[-1]
                number = name.split("/")[-1].split(".")[0]
                if "YouTube-real" in name:
                    youtube_real.add(number)
                elif "Celeb-real" in name:
                    celeb_real.add(number)
                elif "Celeb-synthesis" in name:
                    celeb_fake.add(number)
                else:
                    raise ValueError("'List_of_testing_videos.txt' file corrupted.")
        return youtube_real, celeb_real, celeb_fake

    def __get_images(self, ids, balance=False):
        real = list()
        fake = list()
        # YouTube-real
        for _ in ids[0]:
            real.extend(glob(join(self.root, 'YouTube-real', 'images', _, '*.png')))
        # Celeb-real
        for _ in ids[1]:
            real.extend(glob(join(self.root, 'Celeb-real', 'images', _, '*.png')))
        # Celeb-synthesis
        for _ in ids[2]:
            fake.extend(glob(join(self.root, 'Celeb-synthesis', 'images', _, '*.png')))
        print(f"Real: {len(real)}, Fake: {len(fake)}")
        if balance:
            fake = np.random.choice(fake, size=len(real), replace=False)
            print(f"After Balance | Real: {len(real)}, Fake: {len(fake)}")
        real_tgt = [0] * len(real)
        fake_tgt = [1] * len(fake)
        return [*real, *fake], [*real_tgt, *fake_tgt]


if __name__ == '__main__':
    import yaml

    config_path = "../config/dataset/celeb_df.yml"
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = config["train_cfg"]
    # config = config["test_cfg"]

    def run_dataset():
        dataset = CelebDF(config)
        print(f"dataset: {len(dataset)}")
        for i, _ in enumerate(dataset):
            path, target = _
            print(f"path: {path}, target: {target}")
            if i >= 9:
                break


    def run_dataloader(display_samples=False):
        from torch.utils import data
        import matplotlib.pyplot as plt

        dataset = CelebDF(config)
        dataloader = data.DataLoader(dataset, batch_size=8, shuffle=True)
        print(f"dataset: {len(dataset)}")
        for i, _ in enumerate(dataloader):
            path, targets = _
            image = dataloader.dataset.load_item(path)
            print(f"image: {image.shape}, target: {targets}")
            if display_samples:
                plt.figure()
                img = image[0].permute([1, 2, 0]).numpy()
                plt.imshow(img)
                # plt.savefig("./img_" + str(i) + ".png")
                plt.show()
            if i >= 9:
                break


    ###########################
    # run the functions below #
    ###########################

    # run_dataset()
    run_dataloader(False)
