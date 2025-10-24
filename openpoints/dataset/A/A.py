import os, sys, h5py, pickle, numpy as np, logging, os.path as osp
import torch
from torch.utils.data import Dataset
from ..build import DATASETS
from openpoints.models.layers import fps
import random,pdb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def normal_pc(pc):
    """
    normalize point cloud in range L
    :param pc: type list
    :return: type list
    """
    pc_mean = pc.mean(axis=0)
    pc = pc - pc_mean
    pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
    pc = pc / pc_L_max
    return pc

def jitter_point_cloud(pc, sigma=0.01, clip=0.05):
    """
    Randomly jitter points. jittering is per point.
    :param pc: B X N X 3 array, original batch of point clouds
    :param sigma:
    :param clip:
    :return:
    """
    jittered_data = np.clip(sigma * np.random.randn(*pc.shape), -1 * clip, clip)
    jittered_data += pc
    return jittered_data

def rotation_point_cloud(pc):
    """
    Randomly rotate the point clouds to augment the dataset
    rotation is per shape based along up direction
    :param pc: B X N X 3 array, original batch of point clouds
    :return: BxNx3 array, rotated batch of point clouds
    """

    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data

@DATASETS.register_module()
class A(Dataset):
    classes = [
        "cabinet",
        "chair",
        "lamp",
        "sofa",
        "table",
    ]
    num_classes = 5
    gravity_dim = 1

    def __init__(self, data_dir, split,
                 num_points=2048,
                 uniform_sample=True,
                 transform=None,
                 **kwargs):
        super().__init__()

        self.partition = split
        self.transform = transform
        self.num_points = num_points
        slit_name = 'train' if split == 'train' else 'test'
        
        self.points = np.load(os.path.join(data_dir, f'{slit_name}_pts.npy'))
        self.labels = np.load(os.path.join(data_dir, f'{slit_name}_label.npy'))

        if slit_name == 'test' and uniform_sample:
            precomputed_path = os.path.join(
                data_dir, f'{slit_name}_objectdataset_augmentedrot_scale75_1024_fps.pkl')
            if not os.path.exists(precomputed_path):
                points = torch.from_numpy(self.points).to(torch.float32).cuda()
                self.points = fps(points, 1024).cpu().numpy()
                with open(precomputed_path, 'wb') as f:
                    pickle.dump(self.points, f)
                    print(f"{precomputed_path} saved successfully")
            else:
                with open(precomputed_path, 'rb') as f:
                    self.points = pickle.load(f)
                    print(f"{precomputed_path} load successfully")



        logging.info(f'Successfully load ModelNet {split} '
                     f'size: {self.points.shape}, num_classes: {self.labels.max()+1}')


    @property
    def num_classes(self):
        return self.labels.max() + 1

    def __getitem__(self, idx):
        current_points = self.points[idx][:, :3]

        current_points = normal_pc(current_points)

        if self.partition == 'train':
            current_points = rotation_point_cloud(current_points)
            shuffle_index = np.arange(current_points.shape[0])
            np.random.shuffle(shuffle_index)
            current_points = current_points[shuffle_index][:self.num_points]

        label = self.labels[idx]
        if self.partition == 'train':
            np.random.shuffle(current_points)
        data = {'pos': current_points,
                'y': label
                }
        if self.transform is not None:
            data = self.transform(data)

        # height appending. @KPConv
        # TODO: remove pos here, and only use heights. 
        if 'heights' in data.keys():
            data['x'] = torch.cat((data['pos'], data['heights']), dim=1)
        else:
            data['x'] = torch.cat((data['pos'],
                                   torch.from_numpy(current_points[:, self.gravity_dim:self.gravity_dim+1] - current_points[:, self.gravity_dim:self.gravity_dim+1].min())), dim=1)
            
        data1 = data
        data2 = data
        return data1,data2, False

    def __len__(self):
        return self.points.shape[0]

    """ for visulalization
    from openpoints.dataset import vis_multi_points
    import copy
    old_points = copy.deepcopy(data['pos'])
    if self.transform is not None:
        data = self.transform(data)
    new_points = copy.deepcopy(data['pos'])
    vis_multi_points([old_points, new_points.numpy()])
    End of visulization """