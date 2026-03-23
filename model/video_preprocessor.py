import os
import cv2
import numpy as np
from pathlib import Path
from scenedetect import open_video
from scenedetect import SceneManager, FrameTimecode
from scenedetect.detectors import ContentDetector


class VideoPreprocessor:
    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir) if output_dir else None
        
    def detect_scenes(self, video_path, threshold=27.0, min_scene_len=15):
        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
        
        scene_manager.detect_scenes(video=video)
        
        scene_list = scene_manager.get_scene_list()
        return scene_list

    def extract_frames_from_scene(self, video_path, scene_list, fps=1.0, min_scene_seconds=1.0):
        video = cv2.VideoCapture(str(video_path))
        fps_video = video.get(cv2.CAP_PROP_FPS)
        
        all_keyframes = []
        scene_info = []
        
        for i, (start_time, end_time) in enumerate(scene_list):
            scene_duration = (end_time - start_time).get_seconds()
            scene_info.append({
                'scene_id': i,
                'start_time': str(start_time),
                'end_time': str(end_time),
                'duration_seconds': round(scene_duration, 2)
            })
            
            if scene_duration < min_scene_seconds:
                mid_frame_num = start_time.get_frames() + (end_time.get_frames() - start_time.get_frames()) // 2
                mid_time = FrameTimecode(timecode=mid_frame_num, fps=fps_video)
                frame = self._get_frame_at_time(video, mid_time, fps_video)
                if frame is not None:
                    frame_path = self.output_dir / f'scene_{i:04d}_mid.jpg'
                    cv2.imwrite(str(frame_path), frame)
                    all_keyframes.append({
                        'scene_id': i,
                        'frame_path': str(frame_path),
                        'timestamp': str(mid_time),
                        'frame_index': mid_frame_num
                    })
            else:
                num_frames = max(1, int(scene_duration * fps))
                for j in range(num_frames):
                    relative_frame = int((scene_duration / num_frames) * j * fps_video)
                    absolute_time = FrameTimecode(timecode=start_time.get_frames() + relative_frame, fps=fps_video)
                    frame = self._get_frame_at_time(video, absolute_time, fps_video)
                    if frame is not None:
                        frame_path = self.output_dir / f'scene_{i:04d}_frame_{j:03d}.jpg'
                        cv2.imwrite(str(frame_path), frame)
                        all_keyframes.append({
                            'scene_id': i,
                            'frame_path': str(frame_path),
                            'timestamp': str(absolute_time),
                            'frame_index': absolute_time.get_frames()
                        })
        
        video.release()
        
        return all_keyframes, scene_info
    
    def _get_frame_at_time(self, video, timecode, fps):
        frame_num = timecode.get_frames()
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = video.read()
        return frame if ret else None
    
    def process_video(self, video_path, output_scenes_dir='output/scenes'):
        video_path = Path(video_path)
        scenes_dir = Path(output_scenes_dir)
        scenes_dir.mkdir(parents=True, exist_ok=True)
        
        print(f'Processing video: {video_path.name}')
        
        scene_list = self.detect_scenes(video_path)
        print(f'  Detected {len(scene_list)} scenes')
        
        if len(scene_list) == 0:
            print('  No scenes detected, extracting single frame')
            video = cv2.VideoCapture(str(video_path))
            fps = video.get(cv2.CAP_PROP_FPS)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            mid_frame_num = total_frames // 2
            mid_timecode = FrameTimecode(timecode=int(mid_frame_num), fps=fps)
            video.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_num)
            ret, frame = video.read()
            video.release()
            if ret:
                frame_path = self.output_dir / f'{video_path.stem}_mid.jpg'
                cv2.imwrite(str(frame_path), frame)
                keyframes = [{
                    'scene_id': 0,
                    'frame_path': str(frame_path),
                    'timestamp': str(mid_timecode),
                    'frame_index': mid_frame_num
                }]
                scene_info = [{
                    'scene_id': 0,
                    'start_time': '00:00:00',
                    'end_time': '00:00:00',
                    'duration_seconds': 0
                }]
        else:
            keyframes, scene_info = self.extract_frames_from_scene(video_path, scene_list)
        
        scenes_file = scenes_dir / f'{video_path.stem}_scenes.txt'
        with open(str(scenes_file), 'w', encoding='utf-8') as f:
            f.write(f'Video: {video_path.name}\n')
            f.write(f'Total scenes: {len(scene_info)}\n')
            f.write(f'Total keyframes: {len(keyframes)}\n')
            f.write('=' * 60 + '\n\n')
            for info in scene_info:
                f.write(f"Scene {info['scene_id']:04d}: {info['start_time']} -> {info['end_time']} "
                       f"({info['duration_seconds']}s)\n")
        
        print(f'  Extracted {len(keyframes)} keyframes')
        print(f'  Scenes saved to: {scenes_file}')
        
        return keyframes, scene_info
    
    def process_video_folder(self, video_dir, output_scenes_dir='output/scenes'):
        video_dir = Path(video_dir)
        scenes_dir = Path(output_scenes_dir)
        scenes_dir.mkdir(parents=True, exist_ok=True)
        
        video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi')) + \
                     list(video_dir.glob('*.mov'))
        
        print(f'Found {len(video_files)} video files')
        
        all_video_results = {}
        
        for video_path in video_files:
            keyframes, scene_info = self.process_video(video_path, scenes_dir)
            all_video_results[video_path.name] = {
                'keyframes': keyframes,
                'scene_info': scene_info
            }
        
        summary_file = scenes_dir / 'processing_summary.txt'
        with open(str(summary_file), 'w', encoding='utf-8') as f:
            f.write('Video Preprocessing Summary\n')
            f.write('=' * 60 + '\n\n')
            total_scenes = 0
            total_keyframes = 0
            for video_name, result in all_video_results.items():
                num_scenes = len(result['scene_info'])
                num_keyframes = len(result['keyframes'])
                total_scenes += num_scenes
                total_keyframes += num_keyframes
                f.write(f'{video_name}: {num_scenes} scenes, {num_keyframes} keyframes\n')
            f.write('\n' + '=' * 60 + '\n')
            f.write(f'Total: {len(all_video_results)} videos, {total_scenes} scenes, {total_keyframes} keyframes\n')
        
        print(f'\nProcessing complete!')
        print(f'Summary saved to: {summary_file}')
        print(f'Total keyframes extracted: {total_keyframes}')
        
        return all_video_results
