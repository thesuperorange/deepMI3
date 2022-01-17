import pickle
objects = []
with (open("data/cache/voc_2007_trainval_gt_roidb.pkl.old", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

for idx,item in enumerate(objects[0]):
    print("@@@@@"+str(idx))
    print('gt_classes:')
    for cla in item['gt_classes']:
        print(cla)
    print(item['gt_ishard'])
    print(item)
