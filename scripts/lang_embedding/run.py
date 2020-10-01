
from lang_embedding.model import Model
from lang_embedding.dataset import Dataset

for cluster in [300]:
    for embedding_size in [512]:
        for dropout in [0.5]:

            with open("logs.txt", "a") as myfile:
                myfile.write('Embedding size {}, dropout {}, clusters {}\n'.format(embedding_size, dropout, cluster))
            dataset = Dataset(cluster) 
            langs_num = dataset.train_x.shape[0] + dataset.dev_x.shape[0] + dataset.test_x.shape[0]
            feature_val_num = dataset.global_feature_id


            model = Model(langs_num, feature_val_num, dropout, embedding_size)

            model.train(dataset.batch_generator(), 200, 1000, dataset.feature_maps, dataset.feature_maps_int)
            with open("logs.txt", "a") as myfile:
                myfile.write('-'*50)
                myfile.write('\n')