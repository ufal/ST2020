from lang_embedding.model import Model
from lang_embedding.dataset import Dataset

dataset = Dataset() 
langs_num = dataset.train_x.shape[0] + dataset.dev_x.shape[0]
feature_val_num = dataset.global_feature_id


model = Model(langs_num, feature_val_num)

model.train(dataset.batch_generator(), 1000, 1000, dataset.feature_maps, dataset.feature_maps_int)