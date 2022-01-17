import _init_paths
from roi_data_layer.roidb import combined_roidb
imdb_name = 'KAIST_train_rd'
imdb, roidb, ratio_list, ratio_index = combined_roidb(imdb_name)
