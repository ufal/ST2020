from transformer.dataset import Dataset
from transformer.model import Model
import numpy as np


dataset = Dataset()

model = Model(dataset.get_num_of_features(), dataset.label_encoders, 512, dataset.dev_x_mask, dataset.dev_x, dataset.dev_y)
train, dev = dataset.get_train_and_dev()
model.train(train, dev, 64, 1000, dataset.class_weights)
