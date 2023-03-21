from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
from PIL import Image
import SimpleITK as sitk
# import nibabel as nib
# import nilearn as nl


root = r'/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'

class BraTS2020_Dataset(Dataset): 
    def __init__(self): 
        self.modality_dict = {0: 'flair', 1: 't1', 2: 't1ce', 3: 't2'}
        pass 

    def __len__(self):
        return len(os.listdir(root))*4

    def __getitem__(self, idx):
        patient_id, modality = divmod(idx, 4)
        patient_id = str(patient_id).zfill(3)
        modality = self.modality_dict[modality]
        image_path = os.path.join(root, f'/BraTS20_Training_{patient_id}', f'/BraTS20_Training_{patient_id}_{modality}.nii.gz')
        
        img = sitk.ReadImage(image_path)
        # img = Image.open(path)
        # img = read_image(image_path)
        return img


# train_dataset = BraTS2020_Dataset()
# print('train_dataset',train_dataset)
# train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)