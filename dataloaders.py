# from torch.utils.data import DataLoader
# from torch.utils.data import Dataset
# from torchvision.io import read_image
# import os
# from PIL import Image
# import SimpleITK as sitk
# # import nibabel as nib
# # import nilearn as nl


# root = r'/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'

# class BraTS2020_Dataset(Dataset): 
#     def __init__(self): 
#         self.modality_dict = {0: 'flair', 1: 't1', 2: 't1ce', 3: 't2'}
#         pass 

#     def __len__(self):
#         return len(os.listdir(root))*4

#     def __getitem__(self, idx):
#         patient_id, modality = divmod(idx, 4)
#         patient_id = str(patient_id).zfill(3)
#         modality = self.modality_dict[modality]
#         image_path = os.path.join(root, f'/BraTS20_Training_{patient_id}', f'/BraTS20_Training_{patient_id}_{modality}.nii.gz')
        
#         img = sitk.ReadImage(image_path)
#         # img = Image.open(path)
#         # img = read_image(image_path)
#         return img

import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

class MRIDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = self.get_sample_list()

    def get_sample_list(self):
        sample_list = []
        for patient_dir in os.listdir(self.data_dir):
            patient_path = os.path.join(self.data_dir, patient_dir)
            if os.path.isdir(patient_path):
                for scan_dir in os.listdir(patient_path):
                    scan_path = os.path.join(patient_path, scan_dir)
                    if os.path.isdir(scan_path):
                        sample_list.append(scan_path)
        return sample_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scan_dir = self.samples[idx]

        # Load MRI scan data
        mri_path = os.path.join(scan_dir, 'data.nii.gz')
        mri_img = nib.load(mri_path)
        mri_data = np.array(mri_img.get_fdata())
        mri_data = np.expand_dims(mri_data, axis=0)
        mri_tensor = torch.from_numpy(mri_data).float()

        # Load tumor segmentation mask
        mask_path = os.path.join(scan_dir, 'mask.nii.gz')
        mask_img = nib.load(mask_path)
        mask_data = np.array(mask_img.get_fdata())
        mask_data = np.expand_dims(mask_data, axis=0)
        mask_tensor = torch.from_numpy(mask_data).float()

        return mri_tensor, mask_tensor


# train_dataset = BraTS2020_Dataset()
# print('train_dataset',train_dataset)
# train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

# from torch.utils.data import DataLoader

# batch_size = 4
# data_dir = '/path/to/data'

# dataset = MRIDataset(data_dir)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)