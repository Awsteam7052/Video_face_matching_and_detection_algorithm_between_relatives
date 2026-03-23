import os
import cv2
import numpy as np
import torch
from pathlib import Path
from insightface.app import FaceAnalysis
from model.config import Config


class FaceDetector:
    def __init__(self, det_thresh=Config.DETECTION_THRESHOLD):
        self.det_thresh = det_thresh
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        self.app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
    def detect_faces(self, img):
        faces = self.app.get(img)
        if not faces:
            return []
        return faces
    
    def filter_by_confidence(self, faces, min_score=Config.DETECTION_THRESHOLD):
        filtered = [f for f in faces if f.det_score >= min_score]
        return filtered
    
    def filter_by_size(self, faces, min_size=Config.FACE_SIZE_THRESHOLD):
        filtered = [f for f in faces 
                   if (f.bbox[2] - f.bbox[0]) >= min_size 
                   and (f.bbox[3] - f.bbox[1]) >= min_size]
        return filtered
    
    def detect_and_filter(self, img, min_score=Config.DETECTION_THRESHOLD, min_size=Config.FACE_SIZE_THRESHOLD):
        faces = self.detect_faces(img)
        faces = self.filter_by_confidence(faces, min_score)
        faces = self.filter_by_size(faces, min_size)
        return faces


class BatchFaceProcessor:
    def __init__(self, det_thresh=Config.DETECTION_THRESHOLD):
        self.detector = FaceDetector(det_thresh)
        
    def process_batch(self, image_paths, min_score=Config.DETECTION_THRESHOLD, min_size=Config.FACE_SIZE_THRESHOLD):
        results = []
        
        for img_path in image_paths:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    results.append({
                        'path': img_path,
                        'faces': [],
                        'error': 'Failed to load image'
                    })
                    continue
                
                faces = self.detector.detect_and_filter(img, min_score, min_size)
                
                results.append({
                    'path': img_path,
                    'faces': faces,
                    'face_count': len(faces),
                    'original_shape': img.shape
                })
                
            except Exception as e:
                results.append({
                    'path': img_path,
                    'faces': [],
                    'error': str(e)
                })
        
        return results
