import os
import cv2
import numpy as np
import faiss
import pickle
import json
from pathlib import Path
from collections import defaultdict
from model.config import Config


class FaceMatcher:
    def __init__(self, gallery_index_path=Config.INDEX_PATH, 
                 id_mapping_path=Config.ID_MAPPING_PATH,
                 threshold=0.45):
        self.threshold = threshold
        
        self.index = faiss.read_index(str(gallery_index_path))
        with open(str(id_mapping_path), 'rb') as f:
            self.id_mapping = pickle.load(f)
        
        self.id_to_name = {int(k): v for k, v in self.id_mapping.items()}
        
        print(f'Loaded FAISS index with {self.index.ntotal} vectors')
        print(f'Loaded ID mapping: {self.id_to_name}')
        print(f'Matching threshold: {self.threshold}')
    
    def match_single_vector(self, embedding):
        embedding = np.array([embedding]).astype('float32')
        D, I = self.index.search(embedding, k=1)
        
        similarity = float(D[0][0])
        matched_id = self.id_to_name.get(I[0][0], 'Unknown')
        
        if similarity >= self.threshold:
            return {
                'matched_id': matched_id,
                'similarity': similarity,
                'is_match': True
            }
        else:
            return {
                'matched_id': 'Unknown',
                'similarity': similarity,
                'is_match': False
            }
    
    def match_with_voting(self, frame_results, window_size=3, vote_threshold=2):
        if not frame_results:
            return []
        
        results_with_voting = []
        
        for i, frame_data in enumerate(frame_results):
            frame_path = frame_data.get('frame_path', '')
            matches = frame_data.get('matches', [])
            
            frame_result = {
                'frame_index': i,
                'frame_path': str(frame_path),
                'faces': []
            }
            
            for match in matches:
                embedding = match.get('embedding')
                if embedding is None:
                    bbox = match.get('bbox', [])
                    matched_id = match.get('matched_id', 'Unknown')
                    similarity = match.get('similarity', 0.0)
                    
                    frame_result['faces'].append({
                        'bbox': bbox,
                        'matched_id': matched_id,
                        'similarity': similarity,
                        'is_match': similarity >= self.threshold
                    })
                    continue
                
                match_result = self.match_single_vector(embedding)
                match_result['bbox'] = match.get('bbox', [])
                match_result['det_score'] = match.get('det_score', 0.0)
                
                frame_result['faces'].append(match_result)
            
            results_with_voting.append(frame_result)
        
        return self._apply_temporal_smoothing(results_with_voting, window_size, vote_threshold)
    
    def _apply_temporal_smoothing(self, frame_results, window_size, vote_threshold):
        if len(frame_results) < window_size:
            return frame_results
        
        smoothed_results = []
        
        for i in range(len(frame_results)):
            start_idx = max(0, i - window_size + 1)
            window = frame_results[start_idx:i + 1]
            
            face_id_counts = defaultdict(int)
            face_similarities = defaultdict(list)
            
            for frame_data in window:
                for face in frame_data.get('faces', []):
                    if face.get('is_match', False):
                        face_id = face.get('matched_id', 'Unknown')
                        face_id_counts[face_id] += 1
                        face_similarities[face_id].append(face.get('similarity', 0.0))
            
            current_frame = frame_results[i].copy()
            current_frame['faces'] = []
            
            for face in frame_results[i].get('faces', []):
                if face.get('is_match', False):
                    face_id = face.get('matched_id', 'Unknown')
                    if face_id_counts[face_id] >= vote_threshold:
                        smoothed_face = face.copy()
                        smoothed_face['smoothed'] = True
                        smoothed_face['window_votes'] = face_id_counts[face_id]
                        smoothed_face['avg_similarity'] = np.mean(face_similarities[face_id])
                        current_frame['faces'].append(smoothed_face)
                    else:
                        current_frame['faces'].append(face)
                else:
                    current_frame['faces'].append(face)
            
            smoothed_results.append(current_frame)
        
        return smoothed_results
    
    def _convert_to_serializable(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        return obj
    
    def process_video_with_matching(self, probe_results_path, output_json_path, 
                                    window_size=3, vote_threshold=2):
        print(f'\nLoading probe results from {probe_results_path}')
        
        probe_data = np.load(str(probe_results_path), allow_pickle=True).tolist()
        
        print(f'Loaded {len(probe_data)} frames')
        
        final_results = []
        
        for idx, frame_result in enumerate(probe_data):
            frame_path = frame_result.get('frame_path', '')
            matches = frame_result.get('matches', [])
            
            frame_output = {
                'frame_index': idx,
                'frame_path': str(frame_path),
                'face_count': frame_result.get('face_count', 0),
                'detected_faces': []
            }
            
            for match in matches:
                bbox = match.get('bbox', [])
                embedding = match.get('embedding')
                
                if embedding is not None:
                    match_result = self.match_single_vector(embedding)
                    match_result['bbox'] = self._convert_to_serializable(bbox)
                    match_result['det_score'] = float(match.get('det_score', 0.0))
                else:
                    similarity = match.get('similarity', 0.0)
                    match_result = {
                        'bbox': self._convert_to_serializable(bbox),
                        'matched_id': match.get('matched_id', 'Unknown'),
                        'similarity': float(similarity),
                        'is_match': similarity >= self.threshold,
                        'det_score': float(match.get('det_score', 0.0))
                    }
                
                if match_result['is_match']:
                    match_result['confidence_level'] = 'high' if match_result['similarity'] >= 0.6 else 'medium' if match_result['similarity'] >= 0.5 else 'low'
                
                frame_output['detected_faces'].append(match_result)
            
            final_results.append(frame_output)
        
        if window_size > 1:
            print(f'Applying temporal smoothing (window={window_size}, threshold={vote_threshold})')
            final_results = self._apply_temporal_smoothing(final_results, window_size, vote_threshold)
        
        output_data = {
            'total_frames': len(final_results),
            'matching_threshold': float(self.threshold),
            'temporal_smoothing': {
                'enabled': window_size > 1,
                'window_size': window_size,
                'vote_threshold': vote_threshold
            },
            'frames': final_results
        }
        
        output_data = self._convert_to_serializable(output_data)
        
        with open(str(output_json_path), 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f'\nResults saved to {output_json_path}')
        
        return output_data
    
    def print_summary(self, results):
        print('\n' + '=' * 60)
        print('MATCHING SUMMARY')
        print('=' * 60)
        
        total_frames = results['total_frames']
        frames_with_faces = sum(1 for f in results['frames'] if len(f['detected_faces']) > 0)
        total_faces = sum(len(f['detected_faces']) for f in results['frames'])
        
        matched_count = 0
        unknown_count = 0
        family_member_counts = defaultdict(int)
        
        for frame in results['frames']:
            for face in frame['detected_faces']:
                if face.get('is_match', False):
                    matched_count += 1
                    family_member_counts[face['matched_id']] += 1
                else:
                    unknown_count += 1
        
        print(f'Total frames: {total_frames}')
        print(f'Frames with faces: {frames_with_faces}')
        print(f'Total faces detected: {total_faces}')
        print(f'Matches (known family): {matched_count}')
        print(f'Unknown matches: {unknown_count}')
        
        if family_member_counts:
            print(f'\nFamily Member Detection Counts:')
            for member, count in sorted(family_member_counts.items(), key=lambda x: -x[1]):
                print(f'  {member}: {count} times')
        
        smoothed_count = sum(1 for f in results['frames'] for face in f['detected_faces'] if face.get('smoothed', False))
        if smoothed_count > 0:
            print(f'\nTemporal smoothing applied: {smoothed_count} faces smoothed')
        
        print('=' * 60)
        
        return matched_count
    
    def delete_video_if_matched(self, video_path, matched_count):
        if matched_count > 0:
            try:
                video_file = Path(video_path)
                if video_file.exists():
                    video_file.unlink()
                    print(f'  已删除检测视频源文件: {video_path}')
                    return True
                else:
                    print(f'  视频文件不存在: {video_path}')
                    return False
            except Exception as e:
                print(f'  删除视频文件失败 {video_path}: {str(e)}')
                return False
        return False
