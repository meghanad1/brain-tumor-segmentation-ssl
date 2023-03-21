from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
from PIL import Image

root = './kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'

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
        path = os.path.join(root, f'BraTS20_Training_{patient_id}', f'BraTS20_Training_{patient_id}_{modality}.nii.gz')
        
        img = Image.open(path)
        return img