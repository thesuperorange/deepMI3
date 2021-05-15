
import torch


def rcnn_intermediate(model, x):
    """ Get intermediate results of specifc model.
        Note: This is NOT a generalized function for all torch models,
              due to the different arch. of models.
    """
    from torch.nn import MaxPool2d, AdaptiveAvgPool2d
    
    # forward the features layers of VGG.
    for l in list(model.RCNN_base.modules())[0]:
        x = l(x)
        
    x = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)(x)   
    
    x = AdaptiveAvgPool2d(output_size=(7, 7))(x)
    
    # flatten for FC layers.
    x = x.view(x.shape[0], -1)
    
#     # go through FC layers.
#     for l in list(model.RCNN_top.modules())[0]:
#         x = l(x)

    return x


def get_features(frcnn_model, imgs, batch_sz=64):
    """ Get intermediate result from FRCNN model.
    Args : frcnn_model - torch model.
           imgs - img list with proper format.
    """
    
    outputs = []
    n_batch = len(imgs)//batch_sz
    res_batch = len(imgs)%batch_sz    

    with torch.no_grad():
        for i in range(n_batch):
            in_batch = torch.cat(imgs[i*batch_sz:(i+1)*batch_sz]).to('cuda')
            x = rcnn_intermediate(frcnn_model, in_batch)
            in_batch.to('cpu')
            outputs.append(x)

        # last incomplete batch.
        in_batch = torch.cat(imgs[-res_batch:]).to('cuda')
        x = rcnn_intermediate(frcnn_model, in_batch)
        in_batch.to('cpu')
        outputs.append(x)

    outputs = torch.cat(outputs)
    X = outputs.to('cpu').numpy()
    
    return X


def within_cluster_dispersion(x, n_cluster):
    """ Calculate the pooled within-cluster sum of squares around the cluster means.
        Wk = 1/(2*n)*sum(Dr), where Dr = sum(d(xi, xj)) for xi, xj in the cluster.
        Wk is equivalent to inertia in kmeans.
    Args: x - data, of shape (n, feature_dim)
          n_cluster - number of cluster
    Return: Wk
    """
    import numpy as np
    from sklearn.cluster import KMeans

    n_x = x.shape[0] # number of data.

    kmeans = KMeans(n_cluster)
    kmeans.fit(x)
    
    return kmeans.inertia_/n_x


def gap_stats(x, n_cluster=10, n_samples=5):
    """ Calculate the pooled within-cluster sum of squares around the cluster means.
        Wk = 1/(2*n)*sum(Dr), where Dr = sum(d(xi, xj)) for xi, xj in the cluster.
    Args: x - data, of shape (n, feature_dim)
          n_cluster - number of cluster
          n_samples - to estimate the gap statistics (seems not necessary for Wk)
    Return: Wk
    """
    
    import numpy as np
    from sklearn.cluster import KMeans
    
    def calc_Dr(x):
        """ Helper for calculating Dr.
        Args: x - data of shape (n, feature_dim)
        Return: Dr (as def in the original paper)
        """
        calc_d = lambda x: np.sqrt(np.square(x).sum(axis=-1)).sum()
        Dr = np.sum([calc_d(x- xi) for xi in x])
        return Dr
    
    
    n_x = x.shape[0] # number of data.
    # kmeans for clustering.
    kmeans = KMeans(n_cluster)
    kmeans.fit(x)
    x_pred = kmeans.predict(x)
    
    Wk = []

    # accumulate the within-cluster sum of squares.
    for idx in range(n_cluster):
        x_tmp = x[x_pred == idx]
        Dr = calc_Dr(x_tmp)
        Wk.append(0.5*Dr/n_x)
    
    return np.mean(Wk)