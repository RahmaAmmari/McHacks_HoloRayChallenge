"""
Batch Video Processor - Process entire dataset
"""

import cv2
import numpy as np
from pathlib import Path
import csv
from datetime import datetime


class BatchVideoProcessor:
    def __init__(self, dataset_root="D:/Projects/McHacks/Dataset/Dataset", output_root="D:/Projects/McHacks/Results"):
        self.DATASET_ROOT = Path(dataset_root)
        self.OUTPUT_ROOT = Path(output_root)

        self.RE_DETECT_INTERVAL = 10
        self.MIN_TRACKING_POINTS = 25
        self.MIN_BLOB_AREA = 200

        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        self.feature_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=6,
            blockSize=7
        )

        self.hsv_lower = np.array([90, 60, 40])   # Blue ink
        self.hsv_upper = np.array([140, 255, 255])

    def process_folder(self, folder_name, save_video=True, save_data=True):
        folder_path = self.DATASET_ROOT / folder_name
        if not folder_path.exists():
            print(f"ERROR: Folder does not exist: {folder_path}")
            return

        print("=" * 50)
        print(f"Processing folder: {folder_name}")
        print("=" * 50 + "\n")

        output_dir = self.OUTPUT_ROOT / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

        video_files = self.find_video_files(folder_path)
        print(f"Found {len(video_files)} video files\n")

        results = []
        for idx, video_path in enumerate(video_files, 1):
            print(f"\n[{idx}/{len(video_files)}] Processing: {video_path.name}")
            success, stats = self.process_video(video_path, output_dir, save_video, save_data)
            results.append((video_path.name, success, stats))
            print(f"Status: {'SUCCESS' if success else 'FAILED'}")

        successful = sum(1 for _, success, _ in results if success)
        print("\n" + "=" * 50)
        print("Processing Complete")
        print(f"Total: {len(results)} | Successful: {successful} | Failed: {len(results) - successful}")
        print("=" * 50 + "\n")

        self.save_summary_report(output_dir, results, folder_name)

    def process_video(self, video_path, output_dir, save_video, save_data):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print("  ERROR: Cannot open video")
            return False, None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f} | Total frames: {total_frames}")

        ret, frame = cap.read()
        if not ret:
            print("  ERROR: Cannot read first frame")
            cap.release()
            return False, None

        tracked_roi = self.detect_mark_roi(frame)

        if tracked_roi is None:
            print("  No mark in first frame, scanning...")
            for i in range(30):
                ret, frame = cap.read()
                if not ret:
                    break
                tracked_roi = self.detect_mark_roi(frame)
                if tracked_roi is not None:
                    print(f"  Mark found in frame {i}")
                    break

        if tracked_roi is None:
            print("  WARNING: No trackable mark found, skipping video")
            cap.release()
            return False, None

        base_name = video_path.stem
        output_video_path = output_dir / f"{base_name}_tracked.mp4"
        tracking_data_path = output_dir / f"{base_name}_data.csv"

        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

        csv_file = None
        csv_writer = None
        if save_data:
            csv_file = open(tracking_data_path, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['frame', 'roi_x', 'roi_y', 'roi_width', 'roi_height', 'num_points', 'tracking_status'])

        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = tracked_roi
        roi_gray = prev_gray[y:y + h, x:x + w]
        tracked_points = cv2.goodFeaturesToTrack(roi_gray, **self.feature_params)
        if tracked_points is not None:
            tracked_points = tracked_points.reshape(-1, 2) + np.array([x, y])

        frame_count = 0
        tracked_frames = 0
        lost_frames = 0

        print("  Tracking started...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"  Progress: {frame_count}/{total_frames}", end='\r')

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            tracking_ok = False
            if tracked_points is not None and len(tracked_points) > 0:
                tracking_ok, tracked_points, tracked_roi = self.update_tracking(prev_gray, gray, tracked_points, tracked_roi)

            if (not tracking_ok) or (frame_count % self.RE_DETECT_INTERVAL == 0):
                detected = self.detect_mark_roi(frame)
                if detected is not None:
                    tracked_roi = detected
                    x, y, w, h = tracked_roi
                    roi_gray = gray[y:y + h, x:x + w]
                    corners = cv2.goodFeaturesToTrack(roi_gray, **self.feature_params)
                    if corners is not None:
                        tracked_points = corners.reshape(-1, 2) + np.array([x, y])
                        tracking_ok = True
                elif not tracking_ok:
                    lost_frames += 1

            if tracking_ok:
                tracked_frames += 1

            display = frame.copy()
            if tracked_roi is not None:
                x, y, w, h = tracked_roi
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if tracked_points is not None:
                    for pt in tracked_points:
                        cv2.circle(display, tuple(pt.astype(int)), 2, (0, 255, 255), -1)

            if video_writer is not None:
                video_writer.write(display)

            if csv_writer is not None and tracked_roi is not None:
                x, y, w, h = tracked_roi
                num_points = len(tracked_points) if tracked_points is not None else 0
                csv_writer.writerow([frame_count, x, y, w, h, num_points, 'OK' if tracking_ok else 'LOST'])

            prev_gray = gray.copy()

        cap.release()
        if video_writer is not None:
            video_writer.release()
        if csv_file is not None:
            csv_file.close()

        print(f"\n  Tracked: {tracked_frames}/{frame_count} frames")
        print(f"  Lost tracking: {lost_frames} frames")

        if save_video:
            print(f"  Output video: {output_video_path}")
        if save_data:
            print(f"  Tracking data: {tracking_data_path}")

        stats = {
            'total_frames': frame_count,
            'tracked_frames': tracked_frames,
            'lost_frames': lost_frames,
            'tracking_rate': tracked_frames / frame_count if frame_count > 0 else 0
        }

        return tracked_frames > (frame_count * 0.5), stats

    def detect_mark_roi(self, frame):
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            return self.detect_color_mark(frame)
        return self.detect_dark_mark(frame)

    def detect_color_mark(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        glare_mask = cv2.inRange(gray, 245, 255)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(glare_mask))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return self.find_largest_blob(mask, frame.shape[1], frame.shape[0])

    def detect_dark_mark(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 5
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return self.find_largest_blob(binary, frame.shape[1], frame.shape[0])

    def find_largest_blob(self, mask, width, height):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < self.MIN_BLOB_AREA:
            return None

        x, y, w, h = cv2.boundingRect(largest)
        padding = 15
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(width - x, w + 2 * padding)
        h = min(height - y, h + 2 * padding)
        return (x, y, w, h)

    def update_tracking(self, prev_gray, curr_gray, tracked_points, tracked_roi):
        if tracked_points is None or len(tracked_points) == 0:
            return False, tracked_points, tracked_roi

        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray,
            tracked_points.astype(np.float32),
            None, **self.lk_params
        )
        if next_points is None:
            return False, tracked_points, tracked_roi

        good_new = next_points[status.flatten() == 1]
        good_old = tracked_points[status.flatten() == 1]
        if len(good_new) < self.MIN_TRACKING_POINTS:
            return False, tracked_points, tracked_roi

        dx = np.median(good_new[:, 0] - good_old[:, 0])
        dy = np.median(good_new[:, 1] - good_old[:, 1])

        x, y, w, h = tracked_roi
        x = int(np.clip(x + dx, 0, curr_gray.shape[1] - w))
        y = int(np.clip(y + dy, 0, curr_gray.shape[0] - h))

        return True, good_new, (x, y, w, h)

    def find_video_files(self, directory):
        exts = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        files = []
        for ext in exts:
            files.extend(directory.rglob(f'*{ext}'))
        return sorted(files)

    def save_summary_report(self, output_dir, results, folder_name):
        report_path = output_dir / f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"BATCH PROCESSING SUMMARY: {folder_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

            successful = sum(1 for _, success, _ in results if success)
            f.write(f"Total videos: {len(results)}\n")
            f.write(f"Successful: {successful}\n")
            f.write(f"Failed: {len(results) - successful}\n\n")

            f.write("=" * 60 + "\n")
            f.write("INDIVIDUAL RESULTS:\n")
            f.write("=" * 60 + "\n\n")

            for video_name, success, stats in results:
                f.write(f"Video: {video_name}\n")
                f.write(f"Status: {'SUCCESS' if success else 'FAILED'}\n")
                if stats:
                    f.write(f"Total frames: {stats['total_frames']}\n")
                    f.write(f"Tracked frames: {stats['tracked_frames']}\n")
                    f.write(f"Tracking rate: {stats['tracking_rate']:.2%}\n")
                f.write("\n")

        print(f"\nSummary report saved: {report_path}")
