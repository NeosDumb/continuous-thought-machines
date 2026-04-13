import torch
from torch.utils.data import Sampler
import numpy as np

class QAMNISTSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_samples = len(dataset)

    def __iter__(self):
        indices = torch.randperm(self.num_samples).tolist()
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            if self.dataset.num_images_range[0] == self.dataset.num_images_range[1]:
                batch_num_digits = self.dataset.num_images_range[0]
            else:
                batch_num_digits = np.random.randint(self.dataset.num_images_range[0], self.dataset.num_images_range[1])

            if self.dataset.num_operations_range[0] == self.dataset.num_operations_range[1]:
                batch_num_operations = self.dataset.num_operations_range[0]
            else:
                batch_num_operations = np.random.randint(self.dataset.num_operations_range[0], self.dataset.num_operations_range[1])

            self.dataset.set_num_digits(batch_num_digits)
            self.dataset.set_num_operations(batch_num_operations)
            
            yield batch_indices

    def __len__(self):
        return self.num_samples // self.batch_size