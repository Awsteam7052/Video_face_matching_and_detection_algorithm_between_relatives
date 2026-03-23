import os
import cv2
import numpy as np
import faiss
import pickle
from insightface.app import FaceAnalysis
from pathlib import Path
from model.config import Config


class FamilyGalleryBuilder:
    def __init__(self, gallery_dir, output_dir='output'):
        self.gallery_dir = Path(gallery_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        self.features = []
        self.id_mapping = {}
        self.current_id = 0
        
    def extract_largest_face(self, img):
        faces = self.app.get(img)
        if not faces:
            return None
        
        largest_face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
        return largest_face
    
    def build_gallery(self):
        image_files = list(self.gallery_dir.glob('*.png')) + list(self.gallery_dir.glob('*.jpg')) + list(self.gallery_dir.glob('*.jpeg'))
        
        for img_path in image_files:
            print(f'Processing: {img_path.name}')
            
            img = cv2.imread(str(img_path))
            if img is None:
                print(f'  Warning: Failed to load image {img_path.name}')
                continue
            
            face = self.extract_largest_face(img)
            if face is None:
                print(f'  Warning: No face detected in {img_path.name}')
                continue
            
            feature = face.normed_embedding
            family_id = img_path.stem
            
            self.features.append(feature)
            self.id_mapping[self.current_id] = family_id
            self.current_id += 1
            
            print(f'  Extracted feature vector (512-dim, L2 normalized)')
        
        print(f'\nTotal faces processed: {len(self.features)}')
    
    def build_faiss_index(self):
        features_array = np.array(self.features).astype('float32')
        dimension = features_array.shape[1]
        
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(features_array)
        self.index.add(features_array)
        
        print(f'FAISS Index built: IndexFlatIP with {self.index.ntotal} vectors')
    
    def save_gallery(self):
        index_path = self.output_dir / 'family_faiss_index.bin'
        mapping_path = self.output_dir / 'id_mapping.pkl'
        features_path = self.output_dir / 'all_features.npy'
        
        faiss.write_index(self.index, str(index_path))
        
        with open(str(mapping_path), 'wb') as f:
            pickle.dump(self.id_mapping, f)
        
        np.save(str(features_path), np.array(self.features))
        
        print(f'\nGallery saved to:')
        print(f'  FAISS Index: {index_path}')
        print(f'  ID Mapping: {mapping_path}')
        print(f'  Features: {features_path}')
    
    def verify_gallery(self):
        print('\n=== Gallery Verification ===')
        print(f'Total family members: {len(self.id_mapping)}')
        print('\nID Mapping:')
        for faiss_id, family_id in sorted(self.id_mapping.items()):
            print(f'  FAISS_ID {faiss_id}: {family_id}')
        
        test_feature = self.features[0]
        D, I = self.index.search(np.array([test_feature]).astype('float32'), k=1)
        print(f'\nSelf-retrieval test (first face):')
        print(f'  Retrieved ID: {I[0][0]} (Expected: 0)')
        print(f'  Similarity score: {D[0][0]:.4f}')
