import os, sys, h5py, pickle, numpy as np, logging, os.path as osp
import torch
from torch.utils.data import Dataset
from ..build import DATASETS
from openpoints.models.layers import fps
from openpoints.mixcode.emd import emd_module
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

def mix_input(virtual_partial, real_partial, lam):
    emd = emd_module.emdModule()
    _, ass = emd(virtual_partial,real_partial,1e-4,300)
    ass = ass.long()
    for i in range(real_partial.shape[0]):
        real_partial[i] = real_partial[i][ass[i]]
    lam = lam.repeat(real_partial.shape)
    mix_partial = (1 - lam) * virtual_partial + lam * real_partial
    return mix_partial

def point_mixup(point1, label1, point2, label2):
    a = 0.5
    lam = torch.tensor(np.random.beta(a,1-a)).float().cuda()
    
    point2 = torch.tensor(point2).float().cuda()
    point1 = torch.tensor(point1).float().cuda()

    mix_partial = None
    
    # <512
    batch_size = 500
    for i in range(0, point1.shape[0], batch_size):
        end = min(i + batch_size, point1.shape[0])
        virtual_partial_batch = point1[i:end]
        real_partial_batch = point2[i:end]

        result = mix_input(virtual_partial=virtual_partial_batch, real_partial=real_partial_batch, lam=lam)
        if mix_partial is None:
            mix_partial = result
        else:
            mix_partial = torch.cat([mix_partial, result], dim=0)

    mix_partial = mix_partial.cpu().numpy()

    label = label1

    return mix_partial, label

def balance_number(point_1, label_1, points2, labels2):
    point_source_1 = []
    label_source_1 = []
    point_source_2 = []
    label_source_2 = []

    for i in range(10):
        indice_1 = np.where(label_1 == i)[0]
        indice_2 = np.where(labels2 == i)[0]

        if indice_1.shape[0] > indice_2.shape[0]:
            extra_samples_sum = indice_1.shape[0] - indice_2.shape[0]
            extra_samples = np.random.choice(indice_2, extra_samples_sum, replace=True)
            indice_2 = np.concatenate((indice_2, extra_samples))
            random.shuffle(indice_2.tolist())
        else:
            extra_samples_sum = indice_2.shape[0] - indice_1.shape[0]
            extra_samples = np.random.choice(indice_1, extra_samples_sum, replace=True)
            indice_1 = np.concatenate((indice_1, extra_samples))
            random.shuffle(indice_1.tolist())

        point_source_1.append(point_1[indice_1])
        label_source_1.append(label_1[indice_1])
        point_source_2.append(points2[indice_2])
        label_source_2.append(labels2[indice_2])

    point_source_1 = np.concatenate(point_source_1, axis=0)
    label_source_1 = np.concatenate(label_source_1, axis=0)
    point_source_2 = np.concatenate(point_source_2, axis=0)
    label_source_2 = np.concatenate(label_source_2, axis=0)

    return point_source_1, label_source_1, point_source_2, label_source_2


def get_mixup_data(point_1, label_1, points2, labels2):
    point_1, label_1, points2, labels2 = balance_number(point_1, label_1, points2, labels2)
    AB_point, AB_label = point_mixup(point_1, label_1, points2, labels2)
    return AB_point, AB_label


@DATASETS.register_module()
class ACD(Dataset):
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

        # model_points
        A_point = np.load(os.path.join(data_dir[0], f'{slit_name}_pts.npy'))
        for i in range(A_point.shape[0]):
            A_point[i] = normal_pc(A_point[i])
        A_label = np.load(os.path.join(data_dir[0], f'{slit_name}_label.npy'))

        # shape_points
        C_point = np.load(os.path.join(data_dir[1], f'{slit_name}_pts.npy'))
        C_point = np.repeat(C_point, 2, axis=1)
        C_label = np.load(os.path.join(data_dir[1], f'{slit_name}_label.npy'))
        # pdb.set_trace()

        # 3d feature
        D_point = np.load(os.path.join(data_dir[2], f'{slit_name}_pts.npy'))
        for i in range(D_point.shape[0]):
            D_point[i] = normal_pc(D_point[i])
        D_label = np.load(os.path.join(data_dir[2], f'{slit_name}_label.npy'))
        

        logging.info(f'Successfully load ModelNet {split} '
                     f'size: {A_point.shape}, num_classes: {A_label.max()+1}')
        logging.info(f'Successfully load ShapeNet {split} '
                     f'size: {C_point.shape}, num_classes: {C_label.max()+1}')
        logging.info(f'Successfully load 3D feature {split} '
                     f'size: {D_point.shape}, num_classes: {D_label.max()+1}')
        
        if slit_name == 'train':
            m1_point , m1_label ,n1_point , n1_label = balance_number(C_point, C_label, D_point, D_label)
            logging.info(f'Successfully process m1:{m1_point.shape[0]}')
            m2_point , m2_label ,n2_point , n2_label = balance_number(D_point, D_label, A_point, A_label)
            logging.info(f'Successfully process m2:{m2_point.shape[0]}')

            logging.info(f'Start AD augmented.')
            AD_point, AD_label= get_mixup_data(A_point, A_label, D_point, D_label)
            logging.info(f'Successfully process AD augmented:{AD_point.shape[0]}')
            m3_point, m3_label, n3_point, n3_label = balance_number(C_point, C_label, AD_point, AD_label)
            logging.info(f'Successfully process m3:{m3_point.shape[0]}')

            self.points_1 = np.concatenate([m1_point, m2_point, m3_point], axis=0)
            self.labels_1 = np.concatenate([m1_label, m2_label, m3_label], axis=0)
            logging.info(f'Successfully process source1: {self.points_1.shape[0]}')

            self.points_2 = np.concatenate([n1_point, n2_point, n3_point], axis=0)
            self.labels_2 = np.concatenate([n1_label, n2_label, n3_label], axis=0)
            logging.info(f'Successfully process source2: {self.points_2.shape[0]}')

        if slit_name == 'test':
            point_1 = A_point
            label_1 = A_label
            point_2 = C_point
            label_2 = C_label
            point_3 = D_point
            label_3 = D_label


        if slit_name == 'test' and uniform_sample:
            precomputed_path_1 = os.path.join(data_dir[0], f'{slit_name}_objectdataset_augmentedrot_scale75_1024_fps.pkl')
            precomputed_path_2 = os.path.join(data_dir[1], f'{slit_name}_objectdataset_augmentedrot_scale75_1024_fps.pkl')
            precomputed_path_3 = os.path.join(data_dir[2], f'{slit_name}_objectdataset_augmentedrot_scale75_1024_fps.pkl')

            if not os.path.exists(precomputed_path_1):
                point_1 = torch.from_numpy(point_1).to(torch.float32).cuda()
                point_1 = fps(point_1, 1024).cpu().numpy()
                with open(precomputed_path_1, 'wb') as f:
                    pickle.dump(point_1, f)
                    print(f"{precomputed_path_1} saved successfully")
            else:
                with open(precomputed_path_1, 'rb') as f:
                    point_1 = pickle.load(f)
                    print(f"{precomputed_path_1} load successfully")

            if not os.path.exists(precomputed_path_2):
                point_2 = torch.from_numpy(point_2).to(torch.float32).cuda()
                point_2 = fps(point_2, 1024).cpu().numpy()
                with open(precomputed_path_2, 'wb') as f:
                    pickle.dump(point_2, f)
                    print(f"{precomputed_path_2} saved successfully")
            else:
                with open(precomputed_path_2, 'rb') as f:
                    point_2 = pickle.load(f)
                    print(f"{precomputed_path_2} load successfully")

            if not os.path.exists(precomputed_path_3):
                point_3 = torch.from_numpy(point_3).to(torch.float32).cuda()
                point_3 = fps(point_3, 1024).cpu().numpy()
                with open(precomputed_path_3, 'wb') as f:
                    pickle.dump(point_3, f)
                    print(f"{precomputed_path_3} saved successfully")
            else:
                with open(precomputed_path_3, 'rb') as f:
                    point_3 = pickle.load(f)
                    print(f"{precomputed_path_3} load successfully")
        

        if slit_name == 'test':
            self.points_1 = np.concatenate([point_1, point_2, point_3], axis=0)
            self.labels_1 = np.concatenate([label_1, label_2, label_3], axis=0)
            self.points_2 = self.points_1
            self.labels_2 = self.labels_1

    @property
    def num_classes(self):
        return self.labels_1.max() + 1

    def __getitem__(self, idx):
        current_points_1 = self.points_1[idx][:, :3]
        label_1          = self.labels_1[idx]
        current_points_2 = self.points_2[idx][:, :3]
        label_2          = self.labels_2[idx]

        if self.partition == 'train':
            current_points_1 = rotation_point_cloud(current_points_1)
            shuffle_index_1  = np.arange(current_points_1.shape[0])
            np.random.shuffle(shuffle_index_1)
            current_points_1 = current_points_1[shuffle_index_1][:self.num_points]

            current_points_2 = rotation_point_cloud(current_points_2)
            shuffle_index_2  = np.arange(current_points_2.shape[0])
            np.random.shuffle(shuffle_index_2)
            current_points_2 = current_points_2[shuffle_index_2][:self.num_points]

        if self.partition == 'train':
            np.random.shuffle(current_points_1)
            np.random.shuffle(current_points_2)

        data_1 = {'pos': current_points_1, 'y': label_1}
        data_2 = {'pos': current_points_2, 'y': label_2}

        if self.transform is not None:
            data_1 = self.transform(data_1)
            data_2 = self.transform(data_2)

        # height appending. @KPConv
        # TODO: remove pos here, and only use heights. 
        if 'heights' in data_1.keys():
            data_1['x'] = torch.cat((data_1['pos'], data_1['heights']), dim=1)
            data_2['x'] = torch.cat((data_2['pos'], data_2['heights']), dim=1)
        else:
            data_1['x'] = torch.cat((data_1['pos'],
                                   torch.from_numpy(current_points_1[:, self.gravity_dim:self.gravity_dim+1] - current_points_1[:, self.gravity_dim:self.gravity_dim+1].min())), dim=1)
            data_2['x'] = torch.cat((data_2['pos'],
                                   torch.from_numpy(current_points_2[:, self.gravity_dim:self.gravity_dim+1] - current_points_2[:, self.gravity_dim:self.gravity_dim+1].min())), dim=1)
        
        is_for_contrastive_learning = False

        if idx <= self.points_1.shape[0]:
            is_for_contrastive_learning = True

        return data_1, data_2 ,is_for_contrastive_learning

    def __len__(self):
        return self.points_1.shape[0]