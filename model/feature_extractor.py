import os
import cv2
import numpy as np
import torch
from pathlib import Path
from insightface.app import FaceAnalysis
from model.config import Config


class FeatureExtractor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        self.app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
    def extract_feature(self, img, face_bbox=None):
        faces = self.app.get(img)
        if not faces:
            return None
        
        if face_bbox is not None:
            closest_face = self._find_closest_face(faces, face_bbox)
            if closest_face is None:
                return None
            return closest_face.normed_embedding
        else:
            largest_face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
            return largest_face.normed_embedding
    
    def _find_closest_face(self, faces, target_bbox):
        best_face = None
        best_iou = 0
        
        for face in faces:
            iou = self._bbox_iou(face.bbox, target_bbox)
            if iou > best_iou:
                best_iou = iou
                best_face = face
        
        return best_face if best_iou > 0.5 else None
    
    def _bbox_iou(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        iou = intersection / min(area1, area2)
        return iou


class BatchFeatureExtractor:
    def __init__(self):
        self.extractor = FeatureExtractor()
        
    def extract_batch(self, image_paths, face_detections):
        features_list = []
        
        for idx, (img_path, faces) in enumerate(zip(image_paths, face_detections)):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    features_list.append({
                        'path': img_path,
                        'features': [],
                        'error': 'Failed to load image'
                    })
                    continue
                
                features = []
                for face in faces:
                    feature = self.extractor.extract_feature(img, face.bbox)
                    if feature is not None:
                        features.append({
                            'embedding': feature,
                            'bbox': face.bbox,
                            'det_score': face.det_score
                        })
                
                features_list.append({
                    'path': img_path,
                    'features': features,
                    'feature_count': len(features)
                })
                
            except Exception as e:
                features_list.append({
                    'path': img_path,
                    'features': [],
                    'error': str(e)
                })
        
        return features_list
