class Config:
    DETECTION_THRESHOLD = 0.5
    FACE_SIZE_THRESHOLD = 40
    BATCH_SIZE = 16
    FEATURE_DIM = 512
    
    FAMILY_PHOTOS_DIR = r'./model/family_photos'
    
    VIDEOS_DIR = r'./videos'
    OUTPUT_DIR = r'./output'
    
    INDEX_PATH = r'./model/weights/family_faiss_index.bin'
    ID_MAPPING_PATH = r'./model/weights/id_mapping.pkl'
    FEATURES_PATH = r'./model/weights/all_features.npy'
