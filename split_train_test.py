import logging
import os
import re
from copy import copy
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn import model_selection
from torcheeg.datasets.module.base_dataset import BaseDataset
from torcheeg.model_selection import *


class KFoldPerSubjectCrossTrialMultiSession(KFoldPerSubjectCrossTrial):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
    
    def split_info_constructor(self, info: pd.DataFrame) -> None:
        subjects = list(set(info['subject_id']))

        for subject in subjects:
            subject_info = info[info['subject_id'] == subject]
            trial_ids = list(set(subject_info['trial_id']))

            for fold_id, (train_index_trial_ids,
                          test_index_trial_ids) in enumerate(
                    self.k_fold.split(trial_ids)):
                if fold_id != 2:
                    continue
                if len(train_index_trial_ids) == 0 or len(
                        test_index_trial_ids) == 0:
                    raise ValueError(
                            f'The number of training or testing trials for subject {subject} is zero.'
                            )

                train_trial_ids = np.array(
                        trial_ids)[train_index_trial_ids].tolist()
                test_trial_ids = np.array(
                        trial_ids)[test_index_trial_ids].tolist()

                for session_id in range(1, 4):
                    train_info = []
                    test_info = []
                    for train_trial_id in train_trial_ids:
                        train_info.append(subject_info[(subject_info['trial_id'] == train_trial_id) &
                                                       (subject_info['session_id'] == session_id)])
                    train_info = pd.concat(train_info, ignore_index=True)

                    for test_trial_id in test_trial_ids:
                        test_info.append(
                                subject_info[(subject_info['trial_id'] == test_trial_id) &
                                             (subject_info['session_id'] == session_id)])
                    test_info = pd.concat(test_info, ignore_index=True)

                    train_info.to_csv(os.path.join(
                            self.split_path,
                            f'train_subject_{subject}_fold_{session_id - 1}.csv'),
                            index=False)
                    test_info.to_csv(os.path.join(
                            self.split_path,
                            f'test_subject_{subject}_fold_{session_id - 1}.csv'),
                            index=False)
    def split(
            self,
            dataset: BaseDataset,
            subject: Union[int,
                           None] = None) -> Tuple[BaseDataset, BaseDataset]:
        if not os.path.exists(self.split_path):
            log.info(
                f'📊 | Create the split of train and test set.'
            )
            log.info(
                f'😊 | Please set \033[92msplit_path\033[0m to \033[92m{self.split_path}\033[0m for the next run, if you want to use the same setting for the experiment.'
            )
            os.makedirs(self.split_path)
            self.split_info_constructor(dataset.info)
        else:
            log.info(
                f'📊 | Detected existing split of train and test set, use existing split from {self.split_path}.'
            )
            log.info(
                f'💡 | If the dataset is re-generated, you need to re-generate the split of the dataset instead of using the previous split.'
            )

        subjects = self.subjects
        fold_ids = self.fold_ids

        if not subject is None:
            assert subject in subjects, f'The subject should be in the subject list {subjects}.'

        for local_subject in subjects:
            if (not subject is None) and (local_subject != subject):
                continue

            for fold_id in fold_ids:
                train_info = pd.read_csv(
                    os.path.join(
                        self.split_path,
                        f'train_subject_{local_subject}_fold_{fold_id}.csv'))
                test_info = pd.read_csv(
                    os.path.join(
                        self.split_path,
                        f'test_subject_{local_subject}_fold_{fold_id}.csv'))

                train_dataset = copy(dataset)
                train_dataset.info = train_info

                test_dataset = copy(dataset)
                test_dataset.info = test_info

                yield train_dataset, test_dataset