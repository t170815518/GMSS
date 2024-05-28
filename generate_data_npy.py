from tqdm.auto import tqdm
import numpy as np
from torcheeg.datasets import SEEDIVDataset, SEEDIVFeatureDataset
from torcheeg import transforms
from torcheeg.datasets.constants import SEED_IV_CHANNEL_LOCATION_DICT
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.datasets import DEAPDataset
from torcheeg import transforms
from torcheeg.model_selection import *
from torch.utils.data import DataLoader
from split_train_test import *


dataset = SEEDIVFeatureDataset(root_path='./SEED_IV/eeg_feature_smooth',
                        feature=['de_movingAve'],
                        label_transform=transforms.Select(key='emotion'),
                                  online_transform=transforms.ToTensor(),

                               io_path='.torcheeg\datasets_1716810209183_d2dSj'
                               )
print(dataset[0])

cv = KFoldPerSubjectCrossTrialMultiSession(n_splits=3, shuffle=False)
for i, (train_dataset, test_dataset) in tqdm(enumerate(cv.split(dataset))):
    train_loader = DataLoader(train_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset)

    train_features = []
    train_labels = []
    for sample in train_loader:
        train_features.append(sample[0].numpy())
        train_labels.append(sample[1].numpy())
    train_features = np.vstack(train_features)
    train_labels = np.concatenate(train_labels)
    np.save(f'train_dataset_{cv.subjects[i]}.npy', train_features)
    np.save(f'train_labelset_{cv.subjects[i]}.npy', train_labels)

    test_features = []
    test_labels = []
    for sample in test_loader:
        test_features.append(sample[0].numpy())
        test_labels.append(sample[1].numpy())
    test_features = np.vstack(test_features)
    test_labels = np.concatenate(test_labels)
    np.save(f'test_dataset_{cv.subjects[i]}.npy', test_features)
    np.save(f'test_labelset_{cv.subjects[i]}.npy', test_labels)
