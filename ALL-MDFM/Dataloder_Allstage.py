import json
import os

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
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
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
    def __init__(self, img_paths_list, label_paths, excel_paths, flag):

        self.img_paths_list = img_paths_list
        self.label_paths = label_paths    
        self.excel_paths = excel_paths            
        self.flag = flag
        print(
            f"img_paths_list shape: {len(self.img_paths_list)}, {len(self.img_paths_list[0]) if isinstance(self.img_paths_list[0], list) else 'Not a list'}")

        with open(label_paths[-1], 'r') as anno_file:
            self.anno = json.load(anno_file)

        excel_features_list = []
        for path in excel_paths:
            excel_data = pd.read_csv(path)

            hsi_features = excel_data.iloc[:, 0:10].values
            texture_features = excel_data.iloc[:, 10:17].values
            fvc_features = excel_data.iloc[:, 18:19].values
            excel_features = np.concatenate((hsi_features, texture_features, fvc_features), axis=1)
            excel_features_list.append(excel_features)

        self.excel_features = np.concatenate(excel_features_list, axis=1)

        self.excel_features = np.tile(self.excel_features, (int(np.ceil(len(self.anno) / self.excel_features.shape[0])), 1))
        self.excel_features = self.excel_features[:len(self.anno)]

        self.norm_params = [
            {
                "mean": [0.20349307, 0.377304, 0.36736356,
                        0.02483612, 0.0510145, 0.0376691,
                         0.17007063, 0.29588284,13.54986917],
                "std": [0.1016542, 0.13411573, 0.14308043,
                        0.00936452, 0.01992511, 0.0190979,
                        0.06066503, 0.10615168, 5.89372402]
            },
            {
                "mean": [0.23880044, 0.36393377, 0.35665791,
                        0.02536611, 0.03885711, 0.02873958,
                         0.13677568, 0.33192751,17.00881132],
                "std": [0.11921545, 0.14386931, 0.15207364,
                        0.01110813, 0.0180173, 0.01912885,
                        0.04744015, 0.10839566, 5.99693954]
            },
            {
                "mean": [0.26972057, 0.40768921, 0.45732763,
                         0.0325519, 0.06030464, 0.0573574,
                         0.15260207, 0.24838024, 17.29726097],
                "std": [0.12678274, 0.15147674, 0.1650054,
                        0.01446975, 0.02612324, 0.02935979,
                        0.05190934, 0.07801236, 8.57566802]
            },
            {
                "mean": [0.22858165, 0.37710989, 0.44889768,
                         0.03561842, 0.07428022, 0.07916684,
                         0.19923365, 0.31622791, 23.85511915],
                "std": [0.11724221, 0.1481664, 0.16424013,
                        0.01582362, 0.0330122, 0.03807144,
                        0.07631879, 0.11226482, 10.42767396]
            },
            {
                "mean": [0.23206066, 0.38273012, 0.48467614,
                         0.06352023, 0.13585463, 0.16543924,
                         0.31788514, 0.4576872, 29.38398136],
                "std": [0.09863325, 0.12659227, 0.14511883,
                        0.02259489, 0.04918073, 0.06123055,
                        0.10651701, 0.14556628, 15.4894949]
            }
        ]

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

        stage_img_tensors = []

        for stage in range(4): 

            image_paths = [os.path.join(getattr(self, f'img_path{i}'), image_name) for i in range(7)]
            print(f"Stage: {stage}, Image Name: {image_name}, Image Paths: {image_paths}")
            img_arrays = [self.open_image(img_path, use_tifffile=(i == 6)) for i, img_path in enumerate(image_paths)]

            img6_resized = resize(img_arrays[6], (img_arrays[1].shape[0], img_arrays[0].shape[1]), anti_aliasing=True)
            stacked_images = np.stack(
                (img_arrays[1], img_arrays[2], img_arrays[3], img_arrays[4], img_arrays[5], img6_resized), axis=-1)

            img0_pil = Image.fromarray(img_arrays[0]).convert('RGB')
            img0_tensor = transforms.ToTensor()(img0_pil)
            stacked_images_tensor = transforms.ToTensor()(stacked_images)
            combined_tensor = torch.cat((img0_tensor, stacked_images_tensor), dim=0)

            params = self.norm_params[stage]
            normalized = transforms.Normalize(mean=params["mean"], std=params["std"])(combined_tensor)
            normalized = transforms.RandomResizedCrop(224)(normalized)
            normalized = normalized.float()

            stage_img_tensors.append(normalized)

        combined_stage_images = torch.cat(stage_img_tensors, dim=0)
        excel_feature = torch.tensor(self.excel_features[idx], dtype=torch.float32)
        label = a["yield"]

        return combined_stage_images, excel_feature, label

    def __len__(self):
        return len(self.anno)