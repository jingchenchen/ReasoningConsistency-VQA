import h5py
import json
import numpy as np
import os.path as osp

class ObjectsFeatureLoader:
    def __init__(self,feature_path,objects_M):
        self.feature_path = feature_path
        self.objects_M = objects_M

    def load_feature(self, imageId):
        feature = np.zeros([self.objects_M, 2048])
        h5_path = osp.join(self.feature_path, imageId + '.h5')
        with h5py.File(h5_path, 'r') as hf:
            fea = np.array(hf['features'])# bottom-up features
        num = fea.shape[0]
        if num > self.objects_M:
            fea = fea[:self.objects_M,:]
            num = fea.shape[0]
        feature[:num, :] = fea
        valid = get_valid(len(feature), num)
        return feature, valid

    def load_feature_normalized_bbox(self, imageId):
        feature = np.zeros([self.objects_M, 2048])
        bbox = np.zeros([self.objects_M, 6])
        h5_path = osp.join(self.feature_path, imageId + '.h5')
        with h5py.File(h5_path, 'r') as hf:
            fea = np.array(hf['features']) # bottom-up features
            box = np.array(hf['norm_bb']) # bounding boxes
        num = fea.shape[0]
        if num > self.objects_M:
            fea = fea[:self.objects_M,:]
            box = box[:self.objects_M,:]
            num = fea.shape[0]
        feature[:num, :] = fea
        bbox[:num, :] = box
        normalized_bbox = bbox
        valid = get_valid(len(bbox), num)
        return feature, normalized_bbox, valid

def get_valid(total_num, valid_num):
    valid = np.zeros(total_num, np.bool)
    valid[:valid_num] = True
    return valid
