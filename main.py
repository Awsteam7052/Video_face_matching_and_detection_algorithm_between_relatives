import os
import sys
import cv2
import numpy as np
import faiss
import pickle
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from model.face_detector import BatchFaceProcessor
from model.feature_extractor import BatchFeatureExtractor
from model.gallery_builder import FamilyGalleryBuilder
from model.matcher import FaceMatcher
from model.video_preprocessor import VideoPreprocessor
from model.video_probe import VideoFaceProbe
from model.config import Config


class VideoFamilyMatcher:
    def __init__(self, videos_dir=None, output_dir=None):
        self.videos_dir = Path(videos_dir or Config.VIDEOS_DIR)
        self.output_dir = Path(output_dir or Config.OUTPUT_DIR)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.video_list = []
        self.all_results = {}
        
    def get_video_list(self):
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv']
        self.video_list = []
        
        for ext in video_extensions:
            self.video_list.extend(self.videos_dir.glob(ext))
        
        self.video_list = sorted(self.video_list)
        
        if not self.video_list:
            print(f"警告: 在 {self.videos_dir} 目录下未找到任何视频文件")
            return False
        
        print(f"\n找到 {len(self.video_list)} 个视频文件:")
        for i, video_path in enumerate(self.video_list, 1):
            print(f"  {i}. {video_path.name} ({video_path.stat().st_size / 1024 / 1024:.2f} MB)")
        
        return True
    
    def check_gallery_exists(self):
        index_path = Path(Config.INDEX_PATH)
        mapping_path = Path(Config.ID_MAPPING_PATH)
        
        if not index_path.exists() or not mapping_path.exists():
            print("\n警告: 亲人特征底库不存在，需要先构建底库")
            print("请先运行 gallery_builder.py 构建特征底库")
            return False
        
        return True
    
    def _cleanup_video_temp_files(self, video_path):
        video_name = Path(video_path).stem
        video_output_dir = self.output_dir / video_name
        
        try:
            if video_output_dir.exists() and video_output_dir.is_dir():
                import shutil
                shutil.rmtree(video_output_dir)
                print(f"  已删除视频临时文件夹: {video_output_dir}")
                return True
        except Exception as e:
            print(f"  删除临时文件夹失败 {video_output_dir}: {str(e)}")
            return False
        
        return False
    
    def process_single_video(self, video_path, video_index, total_videos):
        print(f"\n{'='*60}")
        print(f"[{video_index}/{total_videos}] 处理视频: {video_path.name}")
        print(f"{'='*60}")
        
        video_name = video_path.stem
        video_output_dir = self.output_dir / video_name
        
        try:
            video_output_dir.mkdir(parents=True, exist_ok=True)
            
            scenes_dir = video_output_dir / 'scenes'
            scenes_dir.mkdir(parents=True, exist_ok=True)
            
            frames_dir = video_output_dir / 'frames'
            frames_dir.mkdir(parents=True, exist_ok=True)
            
        except Exception as e:
            print(f"  错误: 创建目录失败 - {str(e)}")
            return False
        
        print(f"\n[阶段2] 视频流预处理")
        print('-' * 40)
        
        try:
            preprocessor = VideoPreprocessor(output_dir=str(frames_dir))
            keyframes, scene_info = preprocessor.process_video(str(video_path), str(scenes_dir))
            
            if not keyframes:
                print(f"  警告: 视频 {video_path.name} 未提取到任何关键帧")
                return False
            
            print(f"  提取了 {len(keyframes)} 个关键帧")
            
        except Exception as e:
            print(f"  错误: 视频预处理失败 - {str(e)}")
            return False
        
        print(f"\n[阶段3] 目标检测与特征提取")
        print('-' * 40)
        
        try:
            probe = VideoFaceProbe()
            frame_paths = [kf['frame_path'] for kf in keyframes]
            probe_results = probe.probe_frames_batch(frame_paths)
            
            probe_output_path = video_output_dir / 'probe_results.npy'
            probe.save_results(probe_results, str(probe_output_path))
            
            total_faces = sum(len(r['matches']) for r in probe_results)
            print(f"  检测到 {total_faces} 个人脸")
            
        except Exception as e:
            print(f"  错误: 目标检测与特征提取失败 - {str(e)}")
            return False
        
        print(f"\n[阶段4] 比对、投票与结果输出")
        print('-' * 40)
        
        try:
            matcher = FaceMatcher(threshold=Config.DETECTION_THRESHOLD)
            
            json_output_path = video_output_dir / f'{video_name}.json'
            results = matcher.process_video_with_matching(
                str(probe_output_path),
                str(json_output_path),
                window_size=3,
                vote_threshold=2
            )
            
            matched_count = matcher.print_summary(results)
            
            self.all_results[video_name] = {
                'video_path': str(video_path),
                'video_name': video_name,
                'total_frames': results['total_frames'],
                'total_faces': sum(len(f['detected_faces']) for f in results['frames']),
                'matches': sum(1 for f in results['frames'] for face in f['detected_faces'] if face.get('is_match', False)),
                'json_output': str(json_output_path),
                'matched_count': matched_count
            }
            
            if matched_count > 0:
                matcher.delete_video_if_matched(str(video_path), matched_count)
            
            self._cleanup_video_temp_files(video_path)
            
            return True
            
        except Exception as e:
            print(f"  错误: 比对失败 - {str(e)}")
            return False
    
    def print_final_summary(self):
        print(f"\n{'='*60}")
        print("最终结果汇总")
        print(f"{'='*60}\n")
        
        for video_name, result in self.all_results.items():
            print(f"视频: {result['video_name']}")
            print(f"  路径: {result['video_path']}")
            print(f"  总帧数: {result['total_frames']}")
            print(f"  检测到的人脸数: {result['total_faces']}")
            print(f"  匹配成功数: {result['matches']}")
            print(f"  结果文件: {result['json_output']}")
            print()
        
        total_videos = len(self.all_results)
        total_matches = sum(r['matches'] for r in self.all_results.values())
        total_faces = sum(r['total_faces'] for r in self.all_results.values())
        
        print(f"{'='*60}")
        print(f"统计汇总:")
        print(f"  处理视频数: {total_videos}")
        print(f"  总检测人脸数: {total_faces}")
        print(f"  总匹配成功数: {total_matches}")
        print(f"{'='*60}")
    
    def run(self):
        print("\n" + "="*60)
        print("亲人视频检测系统")
        print("="*60)
        
        if not self.get_video_list():
            return
        
        if not self.check_gallery_exists():
            return
        
        success_count = 0
        fail_count = 0
        
        for i, video_path in enumerate(self.video_list, 1):
            if self.process_single_video(video_path, i, len(self.video_list)):
                success_count += 1
            else:
                fail_count += 1
                print(f"\n跳过视频: {video_path.name}")
        
        self.print_final_summary()
        
        print(f"\n处理完成!")
        print(f"成功: {success_count} 个视频")
        print(f"失败: {fail_count} 个视频")


def build_gallery():
    print("\n" + "="*60)
    print("构建亲人特征底库")
    print("="*60)
    
    try:
        builder = FamilyGalleryBuilder(
            gallery_dir=Config.FAMILY_PHOTOS_DIR,
            output_dir=Config.OUTPUT_DIR
        )
        
        builder.build_gallery()
        builder.build_faiss_index()
        builder.save_gallery()
        builder.verify_gallery()
        
        print("\n" + "="*60)
        print("底库构建完成!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n底库构建失败: {str(e)}")
        return False


def interactive_menu():
    print("\n" + "="*60)
    print("亲人视频检测系统 - 主菜单")
    print("="*60)
    print("1. 构建亲人特征底库")
    print("2. 处理视频文件夹")
    print("3. 两者都执行")
    print("4. 退出")
    print("="*60)
    
    while True:
        choice = input("\n请选择操作 (1-4): ").strip()
        
        if choice == '1':
            build_gallery()
        elif choice == '2':
            matcher = VideoFamilyMatcher()
            matcher.run()
        elif choice == '3':
            if build_gallery():
                matcher = VideoFamilyMatcher()
                matcher.run()
        elif choice == '4':
            print("\n感谢使用，再见!")
            sys.exit(0)
        else:
            print("无效选择，请输入 1-4")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='亲人视频检测系统 - Family Video Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py                    # 交互式菜单模式
  python main.py --mode batch       # 批量处理模式
  python main.py --mode build       # 仅构建底库
  python main.py --mode process     # 仅处理视频
        """
    )
    
    parser.add_argument('--mode', choices=['interactive', 'batch', 'build', 'process'],
                       default='interactive',
                       help='运行模式: interactive(交互式), batch(批量), build(仅构建底库), process(仅处理视频)')
    parser.add_argument('--videos-dir', type=str, default=None,
                       help='视频文件夹路径')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出文件夹路径')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'interactive':
            interactive_menu()
        elif args.mode == 'batch':
            matcher = VideoFamilyMatcher(
                videos_dir=args.videos_dir,
                output_dir=args.output_dir
            )
            matcher.run()
        elif args.mode == 'build':
            build_gallery()
        elif args.mode == 'process':
            matcher = VideoFamilyMatcher(
                videos_dir=args.videos_dir,
                output_dir=args.output_dir
            )
            matcher.run()
    
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
