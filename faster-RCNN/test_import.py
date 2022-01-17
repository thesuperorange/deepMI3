import _init_paths

from roi_data_layer.roidb import combined_roidb, rank_roidb_ratio, filter_roidb



imdb_name = "KAIST_downtown"
imdb, roidb, ratio_list, ratio_index = combined_roidb(imdb_name)

# import pickle
# from roi_data_layer.roidb import rank_roidb_ratio 
# pkl_file = 'data/cache/voc_2007_trainval_gt_roidb.pkl'
# imdb_classes =  ('__background__',  # always index 0
#                           'person',
#                           'people','cyclist'
#                          )
# with open(pkl_file, 'rb') as f:
#     roidb = pickle.load(f)
# print(len(roidb))

# roidb = filter_roidb(roidb)
# print(len(roidb))
# ratio_list, ratio_index = rank_roidb_ratio(roidb)
#dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, imdb_classes, training=True)
