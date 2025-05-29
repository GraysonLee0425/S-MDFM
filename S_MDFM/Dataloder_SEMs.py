import json
from torchvision import transforms
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
import tifffile as tiff
import numpy as np


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def find_row(filename):
    filename1 = filename.split(".")
    num = filename1[0].rfind("_")
    index = filename1[0][num + 1:]
    return int(index)

class MyDataset(Dataset):
    def __init__(self, img_path0, img_path1, img_path2, img_path3, img_path4, img_path5, img_path6, label_path, excel_path, flag, stage):
        self.img_path0 = img_path0
        self.img_path1 = img_path1
        self.img_path2 = img_path2
        self.img_path3 = img_path3
        self.img_path4 = img_path4
        self.img_path5 = img_path5
        self.img_path6 = img_path6
        self.label_path = label_path
        self.excel_path = excel_path
        self.flag = flag
        self.stage = stage
        self.stage = int(stage)

        self.norm_params = [
            {"mean": [0.20349307, 0.377304, 0.36736356, 0.02483612, 0.0510145, 0.0376691, 0.17007063, 0.29588284, 13.54986917],
             "std": [0.1016542, 0.13411573, 0.14308043, 0.00936452, 0.01992511, 0.0190979, 0.06066503, 0.10615168, 5.89372402]},
            {"mean": [0.23880044, 0.36393377, 0.35665791, 0.02536611, 0.03885711, 0.02873958, 0.13677568, 0.33192751, 17.00881132],
             "std": [0.11921545, 0.14386931, 0.15207364, 0.01110813, 0.0180173, 0.01912885, 0.04744015, 0.10839566, 5.99693954]},
            {"mean": [0.26972057, 0.40768921, 0.45732763, 0.0325519, 0.06030464, 0.0573574, 0.15260207, 0.24838024, 17.29726097],
             "std": [0.12678274, 0.15147674, 0.1650054, 0.01446975, 0.02612324, 0.02935979, 0.05190934, 0.07801236, 8.57566802]},
            {"mean": [0.22858165, 0.37710989, 0.44889768, 0.03561842, 0.07428022, 0.07916684, 0.19923365, 0.31622791, 23.85511915],
             "std": [0.11724221, 0.1481664, 0.16424013, 0.01582362, 0.0330122, 0.03807144, 0.07631879, 0.11226482, 10.42767396]},
            {"mean": [0.23206066, 0.38273012, 0.48467614, 0.06352023, 0.13585463, 0.16543924, 0.31788514, 0.4576872, 29.38398136],
             "std": [0.09863325, 0.12659227, 0.14511883, 0.02259489, 0.04918073, 0.06123055, 0.10651701, 0.14556628, 15.4894949]}
        ]

        assert 0 <= self.stage < len(
            self.norm_params), f"Invalid stage: {self.stage}. Stage must be between 0 and {len(self.norm_params) - 1}."

        self.norm_mean = self.norm_params[stage]["mean"]
        self.norm_std = self.norm_params[stage]["std"]

        with open(label_path) as anno_file:
            self.anno = json.load(anno_file)


        self.excel_data = pd.read_csv(excel_path)
        hsi_features = self.excel_data.iloc[:, 0:10].values
        texture_features = self.excel_data.iloc[:, 10:17].values
        fvc_features = self.excel_data.iloc[:, 18:19].values
        self.excel_features = np.concatenate((hsi_features, texture_features, fvc_features), axis=1)

        self.excel_features = np.tile(self.excel_features, (int(np.ceil(len(self.anno) / len(self.excel_features))), 1))
        self.excel_features = self.excel_features[:len(self.anno)]

    def open_image(self, file_path, use_tifffile=False):
        if use_tifffile:
            with tiff.TiffFile(file_path) as tif:
                return tif.asarray()
        else:
            with Image.open(file_path) as img:
                return np.array(img)

    def __getitem__(self, idx):
        a = self.anno[idx]
        image_name = a['image_name']
        image_name0 = self.img_path0 + image_name
        image_name1 = self.img_path1 + image_name
        image_name2 = self.img_path2 + image_name
        image_name3 = self.img_path3 + image_name
        image_name4 = self.img_path4 + image_name
        image_name5 = self.img_path5 + image_name
        image_name6 = self.img_path6 + image_name

        img0_array = self.open_image(image_name0)
        img1_array = self.open_image(image_name1)
        img2_array = self.open_image(image_name2)
        img3_array = self.open_image(image_name3)
        img4_array = self.open_image(image_name4)
        img5_array = self.open_image(image_name5)
        img6_array = self.open_image(image_name6, use_tifffile=True)


        img6_resized = resize(img6_array, (img1_array.shape[0], img0_array.shape[1]), anti_aliasing=True)
        img6_resized_array = np.array(img6_resized)


        stacked_images = np.stack((img1_array, img2_array, img3_array, img4_array, img5_array, img6_resized_array),
                                  axis=-1)
        img0_pil = Image.fromarray(img0_array).convert('RGB')
        img0_tensor = transforms.ToTensor()(img0_pil)
        stacked_images_tensor = transforms.ToTensor()(stacked_images)
        combined_tensor = torch.cat((img0_tensor, stacked_images_tensor), dim=0)


        img_1 = transforms.Normalize(self.norm_mean, self.norm_std)(combined_tensor)

        img_1 = transforms.RandomResizedCrop(224)(img_1)
        img_1 = img_1.float()

        excel_feature = torch.tensor(self.excel_features[idx], dtype=torch.float32)

        label = a["yield"]
        return img_1, excel_feature, label

    def __len__(self):
        return len(self.anno)
