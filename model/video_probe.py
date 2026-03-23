import os
import cv2
import numpy as np
import faiss
import pickle
import torch
from pathlib import Path
from model.config import Config
from model.face_detector import BatchFaceProcessor
from model.feature_extractor import BatchFeatureExtractor


class VideoFaceProbe:
    def __init__(self, gallery_index_path=Config.INDEX_PATH, 
                 id_mapping_path=Config.ID_MAPPING_PATH,
                 det_thresh=Config.DETECTION_THRESHOLD):
        self.det_thresh = det_thresh
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.detector = BatchFaceProcessor(det_thresh)
        self.feature_extractor = BatchFeatureExtractor()
        
        self.index = faiss.read_index(gallery_index_path)
        with open(id_mapping_path, 'rb') as f:
            self.id_mapping = pickle.load(f)
        
        print(f'Loaded FAISS index with {self.index.ntotal} vectors')
        print(f'Loaded ID mapping: {self.id_mapping}')
        print(f'Using device: {self.device}')
        
    def probe_frame(self, frame_path):
        results = {
            'frame_path': frame_path,
            'detections': [],
            'features': [],
            'matches': []
        }
        
        frame_results = self.detector.process_batch([frame_path], self.det_thresh)
        if not frame_results or not frame_results[0]['faces']:
            return results
        
        faces = frame_results[0]['faces']
        results['detections'] = [{
            'bbox': f.bbox.tolist(),
            'det_score': float(f.det_score),
            'face_size': (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        } for f in faces]
        
        feature_results = self.feature_extractor.extract_batch([frame_path], [faces])
        if not feature_results or not feature_results[0]['features']:
            return results
        
        features = feature_results[0]['features']
        results['features'] = features
        
        for feat in features:
            embedding = np.array([feat['embedding']]).astype('float32')
            D, I = self.index.search(embedding, k=1)
            
            similarity = float(D[0][0])
            matched_id = self.id_mapping.get(I[0][0], 'Unknown')
            
            results['matches'].append({
                'bbox': feat['bbox'],
                'matched_id': matched_id,
                'similarity': similarity,
                'det_score': feat['det_score']
            })
        
        return results
    
    def probe_frames_batch(self, frame_paths):
        all_results = []
        
        for i in range(0, len(frame_paths), Config.BATCH_SIZE):
            batch_paths = frame_paths[i:i + Config.BATCH_SIZE]
            
            batch_detections = self.detector.process_batch(
                batch_paths, self.det_thresh
            )
            
            batch_features = self.feature_extractor.extract_batch(
                batch_paths,
                [r['faces'] for r in batch_detections]
            )
            
            for det_result, feat_result in zip(batch_detections, batch_features):
                frame_path = det_result['path']
                matches = []
                
                for feat in feat_result['features']:
                    embedding = np.array([feat['embedding']]).astype('float32')
                    D, I = self.index.search(embedding, k=1)
                    
                    similarity = float(D[0][0])
                    matched_id = self.id_mapping.get(I[0][0], 'Unknown')
                    
                    matches.append({
                        'bbox': feat['bbox'],
                        'matched_id': matched_id,
                        'similarity': similarity,
                        'det_score': feat['det_score']
                    })
                
                all_results.append({
                    'frame_path': frame_path,
                    'face_count': det_result['face_count'],
                    'feature_count': feat_result['feature_count'],
                    'matches': matches
                })
            
            print(f'Processed {min(i + Config.BATCH_SIZE, len(frame_paths))}/{len(frame_paths)} frames')
        
        return all_results
    
    def save_results(self, results, output_path):
        output = []
        for r in results:
            output.append({
                'frame_path': str(r['frame_path']),
                'face_count': r['face_count'],
                'feature_count': r['feature_count'],
                'matches': r['matches']
            })
        
        np.save(output_path, np.array(output, dtype=object))
        print(f'Results saved to {output_path}')
