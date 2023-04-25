# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import random
import warnings

import mmcv
import numpy as np
import torch

from mmdet3d.core.bbox import box_np_ops
from mmdet3d.datasets.pipelines import data_augment_utils
from ..builder import OBJECTSAMPLERS, PIPELINES
from dgl.geometry import farthest_point_sampler
from math import sqrt

class BatchSampler:
    """Class for sampling specific category of ground truths.

    Args:
        sample_list (list[dict]): List of samples.
        name (str, optional): The category of samples. Default: None.
        epoch (int, optional): Sampling epoch. Default: None.
        shuffle (bool, optional): Whether to shuffle indices. Default: False.
        drop_reminder (bool, optional): Drop reminder. Default: False.
    """

    def __init__(self,
                 sampled_list,
                 name=None,
                 epoch=None,
                 shuffle=True,
                 drop_reminder=False):
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        if shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def _sample(self, num):
        """Sample specific number of ground truths and return indices.

        Args:
            num (int): Sampled number.

        Returns:
            list[int]: Indices of sampled ground truths.
        """
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx:].copy()
            self._reset()
        else:
            ret = self._indices[self._idx:self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        """Reset the index of batchsampler to zero."""
        assert self._name is not None
        # print("reset", self._name)
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num):
        """Sample specific number of ground truths.

        Args:
            num (int): Sampled number.

        Returns:
            list[dict]: Sampled ground truths.
        """
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]


@OBJECTSAMPLERS.register_module()
class DataBaseSampler(object):
    """Class for sampling data from the ground truth database.

    Args:
        info_path (str): Path of groundtruth database info.
        data_root (str): Path of groundtruth database.
        rate (float): Rate of actual sampled over maximum sampled number.
        prepare (dict): Name of preparation functions and the input value.
        sample_groups (dict): Sampled classes and numbers.
        classes (list[str], optional): List of classes. Default: None.
        bbox_code_size (int, optional): The number of bbox dimensions.
            Default: None.
        points_loader(dict, optional): Config of points loader. Default:
            dict(type='LoadPointsFromFile', load_dim=4, use_dim=[0,1,2,3])
        ds_rate (dict: class -> float, optional): Downsample rate, i.e. probability for EACH object sampled to be downsampled. Default: 0.0 for all classes.
        ds_scale (dict: class -> []float, optional): Downsample scale upper and lower bound to sample from uniformly. The uniformly sampled value is the on that each downsampled objects x- and y-coordinates are multiplied by. Default: [1.0, 1.0] (multiplication of 1.0 means no change compared to actual gt database) for all clases.
        ds_flip_xy (bool, optional): Whether to multiply x and y coordinates of sampled ground truths by -1 with probability of 0.5 for evening out sampling dsitribution. Default: False.
        ds_method (str, optional): Method to use for downsampling. Default: 'Random'.
        max_range_class (dict: class -> float, optional): Maximum range of sampled objects. Default: Max detection ranges for nuScenes.
    """

    def __init__(self,
                 info_path,
                 data_root,
                 rate,
                 prepare,
                 sample_groups,
                 classes=None,
                 bbox_code_size=None,
                 points_loader=dict(
                     type='LoadPointsFromFile',
                     coord_type='LIDAR',
                     load_dim=4,
                     use_dim=[0, 1, 2, 3]),
                 file_client_args=dict(backend='disk'),
                 ds_rate={},
                 ds_target_range={},
                 long_range_fraction=1/3,
                 ds_flip_xy=False,
                 ds_method='Random',
                 max_range_class=dict(
                    car= 50, 
                    truck= 50, 
                    bus= 50, 
                    trailer= 50, 
                    construction_vehicle= 50, 
                    traffic_cone= 30, 
                    barrier= 30, 
                    motorcycle= 40, 
                    bicycle= 40, 
                    pedestrian= 40),
                 ):
        super().__init__()
        self.data_root = data_root
        self.info_path = info_path
        self.rate = rate
        self.prepare = prepare
        self.classes = classes
        self.cat2label = {name: i for i, name in enumerate(classes)}
        self.label2cat = {i: name for i, name in enumerate(classes)}
        # print(points_loader)
        self.points_loader = mmcv.build_from_cfg(points_loader, PIPELINES)
        self.file_client = mmcv.FileClient(**file_client_args)
        # set rate to 0.0 and scale to 1.0 for classes not in ds_rate and ds_scale
        for c in classes:
            if c not in ds_rate:
                ds_rate[c] = 0.0
            if c not in ds_target_range:
                ds_target_range[c] = [max_range_class[c]*(1-long_range_fraction), max_range_class[c]]
        self.ds_rate = ds_rate
        self.ds_target_range = ds_target_range
        self.ds_flip_xy = ds_flip_xy
        self.max_range_class = max_range_class
        
        assert ds_method in ['Random', 'FPS'], f"Downsample method '{ds_method}' not supported, only 'Random' and 'FPS' are supported."
        print(f"Using {ds_method} as downsample method.")
        self.ds_method = ds_method
        
        print(self.ds_rate)
        print(self.ds_target_range)
        print(self.ds_flip_xy)

        # load data base infos
        if hasattr(self.file_client, 'get_local_path'):
            with self.file_client.get_local_path(info_path) as local_path:
                # loading data from a file-like object needs file format
                db_infos = mmcv.load(open(local_path, 'rb'), file_format='pkl')
        else:
            warnings.warn(
                'The used MMCV version does not have get_local_path. '
                f'We treat the {info_path} as local paths and it '
                'might cause errors if the path is not a local path. '
                'Please use MMCV>= 1.3.16 if you meet errors.')
            db_infos = mmcv.load(info_path)

        # filter database infos
        from mmdet3d.utils import get_root_logger
        logger = get_root_logger()
        for k, v in db_infos.items():
            logger.info(f'load {len(v)} {k} database infos')
        for prep_func, val in prepare.items():
            db_infos = getattr(self, prep_func)(db_infos, val)
        logger.info('After filter database:')
        for k, v in db_infos.items():
            logger.info(f'load {len(v)} {k} database infos')

        self.db_infos = db_infos

        self.bbox_code_size = bbox_code_size
        if bbox_code_size is not None:
            for k, info_cls in self.db_infos.items():
                for info in info_cls:
                    info['box3d_lidar'] = info['box3d_lidar'][:self.
                                                              bbox_code_size]

        # load sample groups
        # TODO: more elegant way to load sample groups
        self.sample_groups = []
        for name, num in sample_groups.items():
            self.sample_groups.append({name: int(num)})

        self.group_db_infos = self.db_infos  # just use db_infos
        self.sample_classes = []
        self.sample_max_nums = []
        for group_info in self.sample_groups:
            self.sample_classes += list(group_info.keys())
            self.sample_max_nums += list(group_info.values())

        self.sampler_dict = {}
        for k, v in self.group_db_infos.items():
            self.sampler_dict[k] = BatchSampler(v, k, shuffle=True)
        # TODO: No group_sampling currently

    @staticmethod
    def filter_by_difficulty(db_infos, removed_difficulty):
        """Filter ground truths by difficulties.

        Args:
            db_infos (dict): Info of groundtruth database.
            removed_difficulty (list): Difficulties that are not qualified.

        Returns:
            dict: Info of database after filtering.
        """
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
        return new_db_infos

    @staticmethod
    def filter_by_min_points(db_infos, min_gt_points_dict):
        """Filter ground truths by number of points in the bbox.

        Args:
            db_infos (dict): Info of groundtruth database.
            min_gt_points_dict (dict): Different number of minimum points
                needed for different categories of ground truths.

        Returns:
            dict: Info of database after filtering.
        """
        for name, min_num in min_gt_points_dict.items():
            min_num = int(min_num)
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos

    @staticmethod
    def downsample_gt_sample(pcd, dss, ds_method):
        """Translate and downsample ground truth point cloud.
        
        Args:
            pcd: [] (1, c)
            dss: float for x,y-coordinates multiplication of bounding box
            ds_method: str, Type of downsampling applied; 'Random' or 'FPS'
        """
        if ds_method == 'FPS':
            num_pts_kept = int(pcd.shape[0]//(dss**3)) # Initial value (cubical scaling)
            # [ 4.35836897e-01 -3.67409854e+01  6.95919053e+02]
            # num_pts_kept = int(pcd.shape[0]*(-1.88322669*1e-2*dss**3 + 2.53214753*1e0*dss**2 - 1.02129840*1e2*dss + 1.23150913*1e3)) # 3rd grade polyfit of pedestrian
            # num_pts_kept = int(pcd.shape[0]*(4.35836897*1e-1*dss**2 - 3.67409854*1e1*dss + 6.95919053*1e2)) # 2nd grade polyfit of pedestrian
            
            # Clone tensor to not mess upp with reshape
            pcdtensor = torch.clone(pcd.tensor)
            pcdtensor = torch.clone(pcdtensor[:,:3].reshape(1, -1, 3)) # Reshape because dgl farthest_point_sampler expects a batch of point clouds
            
            # Generate indices with fps
            ind = farthest_point_sampler(pcdtensor, num_pts_kept)
            
            # Use fps indices to downsample. ind[0] once again required beacuse of batch expectation
            pcd.tensor = pcd.tensor[ind[0]][:]
            return pcd
        elif ds_method == 'Random':
            num_pts_kept = int(pcd.shape[0]//(dss**3)) # Initial value (cubical scaling)
            # [ 4.35836897e-01 -3.67409854e+01  6.95919053e+02]
            # num_pts_kept = int(pcd.shape[0]*(-1.88322669*1e-2*dss**3 + 2.53214753*1e0*dss**2 - 1.02129840*1e2*dss + 1.23150913*1e3)) # 3rd grade polyfit of pedestrian
            # num_pts_kept = int(pcd.shape[0]*(4.35836897*1e-1*dss**2 - 3.67409854*1e1*dss + 6.95919053*1e2)) # 2nd grade polyfit of pedestrian
            ind = torch.randperm(pcd.shape[0])[:num_pts_kept]
            pcd = pcd[ind]
        return pcd

    def sample_all(self, gt_bboxes, gt_labels, img=None, ground_plane=None):
        """Sampling all categories of bboxes.

        Args:
            gt_bboxes (np.ndarray): Ground truth bounding boxes.
            gt_labels (np.ndarray): Ground truth labels of boxes.

        Returns:
            dict: Dict of sampled 'pseudo ground truths'.

                - gt_labels_3d (np.ndarray): ground truths labels
                    of sampled objects.
                - gt_bboxes_3d (:obj:`BaseInstance3DBoxes`):
                    sampled ground truth 3D bounding boxes
                - points (np.ndarray): sampled points
                - group_ids (np.ndarray): ids of sampled ground truths
        """
        
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(self.sample_classes,
                                              self.sample_max_nums):
            class_label = self.cat2label[class_name]
            # sampled_num = int(max_sample_num -
            #                   np.sum([n == class_name for n in gt_names]))
            sampled_num = int(max_sample_num -
                              np.sum([n == class_label for n in gt_labels]))
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_bboxes = []
        avoid_coll_boxes = gt_bboxes
        ds_tracker = []
        ds_flip_x_tracker = []
        ds_flip_y_tracker = []
        dss = []
        flip_x = False
        flip_y = False
        for class_name, sampled_num in zip(self.sample_classes,
                                           sample_num_per_class):
            if sampled_num > 0:
                sampled_cls = self.sample_class_v2(class_name, sampled_num,
                                                   avoid_coll_boxes)
                for samp in sampled_cls:                    
                    if random.random() <= self.ds_rate[class_name]:
                        # dss += [random.uniform(self.ds_scale[class_name][0], self.ds_scale[class_name][1])]
                        # if samp["num_points_in_gt"]//(dss[-1]**3) > 5 and sqrt(samp["box3d_lidar"][0]**2+samp["box3d_lidar"][1]**2)*dss[-1] < self.max_range_class [class_name]:
                        #     samp["box3d_lidar"][0:2] *= dss[-1]
                        #     ds_tracker += [True]
                        # else:
                        #     # print("Downsampled object ends up too far away or has too few points, skipping downsampling")
                        #     dss[-1] = 1
                        #     ds_tracker += [False]
                        bev_dist = sqrt(samp["box3d_lidar"][0]**2+samp["box3d_lidar"][1]**2)
                        target_dist = random.uniform(self.ds_target_range[class_name][0], self.ds_target_range[class_name][1])
                        if target_dist > bev_dist:
                            dss += [target_dist/bev_dist]
                            if samp["num_points_in_gt"]//(dss[-1]**3) > 5:
                                samp["box3d_lidar"][0:2] *= dss[-1]
                                ds_tracker += [True]
                            else:
                                # Downsampled object has too few points, skip downsampling
                                # print("too few points")
                                dss[-1] = 1
                                ds_tracker += [False]
                        else:
                            # Target distance is smaller than current distance, skip downsampling
                            # print("original dist>target dist")
                            dss += [1]
                            ds_tracker += [False]
                    else:
                        dss += [1]
                        ds_tracker += [False]
                    if self.ds_flip_xy:
                        flip_x = random.random() <= .5
                        flip_y = random.random() <= .5
                    if flip_x:
                        samp["box3d_lidar"][0] *= -1
                        ds_flip_x_tracker += [True]
                    else:
                        ds_flip_x_tracker += [False]
                    if flip_y:
                        samp["box3d_lidar"][1] *= -1
                        ds_flip_y_tracker += [True]
                    else:
                        ds_flip_y_tracker += [False]
                    
                    # Bounding box rotation
                    if samp["box3d_lidar"][6] >= 0:
                        if flip_x:
                            if flip_y:
                                samp["box3d_lidar"][6] = samp["box3d_lidar"][6] - np.pi
                            else:
                                samp["box3d_lidar"][6] = np.pi - samp["box3d_lidar"][6]
                        else:
                            if flip_y:
                                samp["box3d_lidar"][6] = -samp["box3d_lidar"][6]
                    else:
                        if flip_x:
                            if flip_y:
                                samp["box3d_lidar"][6] = samp["box3d_lidar"][6] + np.pi
                            else:
                                samp["box3d_lidar"][6] = - np.pi - samp["box3d_lidar"][6]
                        else:
                            if flip_y:
                                samp["box3d_lidar"][6] = -samp["box3d_lidar"][6]
                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]['box3d_lidar'][
                            np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack(
                            [s['box3d_lidar'] for s in sampled_cls], axis=0)

                    sampled_gt_bboxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box], axis=0)

        ret = None
        if len(sampled) > 0:
            sampled_gt_bboxes = np.concatenate(sampled_gt_bboxes, axis=0)
            s_points_list = []
            count = 0
            for info, is_ds in zip(sampled, ds_tracker):
                file_path = os.path.join(
                    self.data_root,
                    info['path']) if self.data_root else info['path']
                results = dict(pts_filename=file_path)
                s_points = self.points_loader(results)['points']
                if is_ds:
                    s_points = self.downsample_gt_sample(s_points, dss[count], self.ds_method)
                if ds_flip_x_tracker[count]:
                    s_points.tensor[:, 0] *= -1
                if ds_flip_y_tracker[count]:
                    s_points.tensor[:, 1] *= -1
                s_points.translate(info['box3d_lidar'][:3])
                count += 1
                s_points_list.append(s_points)
            gt_labels = np.array([self.cat2label[s['name']] for s in sampled],
                                 dtype=np.long)

            if ground_plane is not None:
                xyz = sampled_gt_bboxes[:, :3]
                dz = (ground_plane[:3][None, :] *
                      xyz).sum(-1) + ground_plane[3]
                sampled_gt_bboxes[:, 2] -= dz
                for i, s_points in enumerate(s_points_list):
                    s_points.tensor[:, 2].sub_(dz[i])

            ret = {
                'gt_labels_3d':
                gt_labels,
                'gt_bboxes_3d':
                sampled_gt_bboxes,
                'points':
                s_points_list[0].cat(s_points_list),
                'group_ids':
                np.arange(gt_bboxes.shape[0],
                          gt_bboxes.shape[0] + len(sampled))
            }
        # print("final count: ", count)
        return ret
    

    def sample_class_v2(self, name, num, gt_bboxes):
        """Sampling specific categories of bounding boxes.

        Args:
            name (str): Class of objects to be sampled.
            num (int): Number of sampled bboxes.
            gt_bboxes (np.ndarray): Ground truth boxes.

        Returns:
            list[dict]: Valid samples after collision test.
        """
        sampled = self.sampler_dict[name].sample(num)
        # print("sampled: ", sampled)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_bboxes.shape[0]
        num_sampled = len(sampled)
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes[:, 0:2], gt_bboxes[:, 3:5], gt_bboxes[:, 6])

        sp_boxes = np.stack([i['box3d_lidar'] for i in sampled], axis=0)
        boxes = np.concatenate([gt_bboxes, sp_boxes], axis=0).copy()

        sp_boxes_new = boxes[gt_bboxes.shape[0]:]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6])

        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = data_augment_utils.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])
        # print("NUMBER OF VALID SAMPLES", len(valid_samples))
        return valid_samples
