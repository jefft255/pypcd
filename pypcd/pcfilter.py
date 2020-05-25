from pypcd.pcdcloud import PCDCloud

from typing import List

from sklearn import preprocessing  # easiest way to normalize normal and directions
from scipy.spatial import cKDTree

from math import ceil, floor
from time import time

import numpy as np


class PCFilter:
    def __init__(self):
        pass

    def __call__(self, cloud: PCDCloud):
        raise NotImplementedError("PCFilter is an abstract class")


class PCNullFilter(PCFilter):
    def __init__(self):
        pass

    def __call__(self, cloud: PCDCloud):
        pass


class PCFeatureFilter(PCFilter):
    def __init__(self, feature_name: str, lower_limit: float, upper_limit: float):
        super().__init__()
        assert(upper_limit >= lower_limit)
        self.feature_name = feature_name
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def __call__(self, cloud: PCDCloud):
        mask = np.logical_and(self.lower_limit <= cloud.get_field_view(self.feature_name),
                              cloud.get_field_view(self.feature_name) <= self.upper_limit)
        mask = mask.flatten()
        cloud.data = cloud.data[mask, :]


class PCComposedFilter(PCFilter):
    def __init__(self, filters: List[PCFilter]):
        super().__init__()
        self.filters = filters

    def __call__(self, cloud: PCDCloud):
        for pc_filter in self.filters:
            pc_filter(cloud)


# This class computes the incidence angle from the normal from the point cloud.
# It would probably be better to iteratively compute the incidence angle from
# the cylinder fitted iteratively. This will be done later.
class PCIncidenceAngleFilter(PCFilter):
    def __init__(self, max_angle: float):
        super().__init__()
        assert(max_angle >= 0)
        self.max_angle = max_angle

    def __call__(self, cloud: PCDCloud):
        normals = preprocessing.normalize(cloud.get_fields_view(['normal_x', 'normal_y', 'normal_z']), norm='l2')
        o_direction = preprocessing.normalize(cloud.get_field_view('observationDirections'), norm='l2')
        assert(o_direction.shape[1] == 3)
        cloud.data = cloud.data[np.arccos(np.clip(np.sum(np.multiply(normals, o_direction), axis=1),
                                                  0, 1)) <= self.max_angle, :]


# Makes the logical distiction that those filters do not actually remove any points.
class PCFeatureComputer(PCFilter):
    def __init__(self):
        pass

    def __call__(self, cloud: PCDCloud):
        raise NotImplementedError("PCFeatureComputer is an abstract class")


class PCIncidenceAngleComputer(PCFeatureComputer):
    def __init__(self):
        super().__init__()

    def __call__(self, cloud: PCDCloud):
        normals = preprocessing.normalize(cloud.get_fields_view(['normal_x', 'normal_y', 'normal_z']), norm='l2')
        o_direction = preprocessing.normalize(cloud.get_field_view('observationDirections'), norm='l2')
        assert(o_direction.shape[1] == 3)
        cloud.add_fields(['incidence_angle'])
        cloud.data[:, cloud.field_to_columns('incidence_angle')] = \
            np.arccos(np.clip(np.sum(np.multiply(normals, o_direction), axis=1), 0, 1)).reshape(-1,1)


class PCNormalComputer(PCFeatureComputer):
    def __init__(self, k_neighbours):
        super().__init__()
        self.k_neighbours = k_neighbours

    def __call__(self, cloud: PCDCloud, field_name="normal", kd_tree=None):
        if f'{field_name}_x' not in cloud.fields:  # TODO some kind of assert if some normal fields are there but not all
            cloud.add_fields([f'{field_name}_x', f'{field_name}_y', f'{field_name}_z'])

        xyz_cloud = cloud.get_xyz_view()
        normals = np.zeros(xyz_cloud.shape)
        if kd_tree is None:
            kd_tree = cKDTree(xyz_cloud)
        o_direction = preprocessing.normalize(cloud.get_field_view('observationDirections'), norm='l2')
        assert(o_direction.shape[1] == 3)

        _, neighbourhoods_indices = kd_tree.query(xyz_cloud, self.k_neighbours)
        for j in range(0, cloud.get_xyz_view().shape[0]):
            neighbourhood = xyz_cloud[neighbourhoods_indices[j]]
            cov = np.cov(neighbourhood, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            min_eigenvalue = np.inf
            normal = None
            for eigenvalue, eigenvector in zip(eigenvalues, eigenvectors):
                if eigenvalue < min_eigenvalue:
                    min_eigenvalue = eigenvalue
                    normal = eigenvector

            if np.dot(o_direction[j], normal) > np.dot(o_direction[j], -normal):  # reorientation
                normal = -normal
            normals[j] = normal

        cloud.data[:, cloud.fields_to_columns([f'{field_name}_x', f'{field_name}_y', f'{field_name}_z'])] = normals


class PCVoxelGridFilter(PCFilter):
    def __init__(self, resolution):
        super().__init__()
        self.resolution = resolution

    def __call__(self, cloud: PCDCloud):
        self.min_x = cloud.get_field_view('x').min()
        self.max_x = cloud.get_field_view('x').max()
        self.min_y = cloud.get_field_view('y').min()
        self.max_y = cloud.get_field_view('y').max()
        self.min_z = cloud.get_field_view('z').min()
        self.max_z = cloud.get_field_view('z').max()

        res_x = int(ceil((self.max_x - self.min_x) / self.resolution))
        res_y = int(ceil((self.max_y - self.min_y) / self.resolution))
        res_z = int(ceil((self.max_z - self.min_z) / self.resolution))

        grid = np.full((res_x, res_y, res_z), -1, dtype=np.int32)
        xyz_view = cloud.get_xyz_view()

        for i in range(cloud.point_count()):
            if grid[self.coords_to_index(xyz_view[i])] == -1:
                grid[self.coords_to_index(xyz_view[i])] = i

        mask = grid.flatten().tolist()
        mask = [i for i in mask if i != -1]

        cloud.data = cloud.data[mask]

    def coords_to_index(self, point):
        x_index = int(floor((point[0] - self.min_x) / self.resolution))
        y_index = int(floor((point[1] - self.min_y) / self.resolution))
        z_index = int(floor((point[2] - self.min_z) / self.resolution))
        return x_index, y_index, z_index




