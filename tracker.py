"""
Real-Time Drawing & Tracking Tool - Improved Accuracy
- Homography-based warping of drawing layer
- Forward-backward (FB) check for optical flow to reject bad points
- Re-seed features every N frames inside ROI
- Update ROI by transforming ROI corners with homography
- Display never upscales (prevents blurry zoom)
- Mouse coordinates map correctly when display is downscaled
"""

import cv2
import numpy as np
from pathlib import Path


class RealTimeDrawingTracker:
    def __init__(self, dataset_root="D:/Projects/McHacks/Dataset/Dataset"):
        self.cap = None
        self.current_frame = None
        self.drawing_layer = None

        # Dataset picker
        self.dataset_root = Path(dataset_root)
        self.dataset_folders = ["Echo", "Intrapartum", "Lapchole", "POCUS"]
        self.video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

        # Drawing
        self.is_drawing = False
        self.last_point = None
        self.current_color = (0, 0, 255)
        self.brush_size = 3

        # Tracking
        self.tracking_enabled = False
        self.tracked_roi = None  # (x, y, w, h)
        self.tracked_points = None  # Nx1x2 float32
        self.prev_gray = None

        # UI
        self.show_help = True
        self.paused = False

        # Display: never upscale
        self.max_display_w = 1280
        self.max_display_h = 720
        self.display_scale = 1.0

        # LK parameters (slightly stronger)
        self.lk_params = dict(
            winSize=(31, 31),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
        )

        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=400,
            qualityLevel=0.01,
            minDistance=6,
            blockSize=7
        )

        # Accuracy controls
        self.min_points = 40
        self.reseed_interval = 8  # reseed points every N frames
        self.frame_idx = 0
        self.fb_max_error = 2.0  # pixels (forward-back check threshold)
        self.ransac_thresh = 3.0  # pixels (homography RANSAC threshold)

        # ROI padding helps include texture around drawing
        self.roi_padding = 25

    # ----------------------- Startup / Loop -----------------------

    def start(self, video_source):
        print("\n" + "=" * 50)
        print("REAL-TIME DRAWING TRACKER (Improved Accuracy)")
        print("=" * 50)
        self.print_controls()

        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            print(f"ERROR: Cannot open video source: {video_source}")
            return

        ret, self.current_frame = self.cap.read()
        if not ret or self.current_frame is None:
            print("ERROR: Cannot read from video source")
            return

        self.drawing_layer = np.zeros_like(self.current_frame)

        cv2.namedWindow("Real-Time Drawing Tracker", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Real-Time Drawing Tracker", self.mouse_callback)

        self.run_loop()

    def run_loop(self):
        while True:
            if not self.paused:
                ret, self.current_frame = self.cap.read()
                if not ret or self.current_frame is None:
                    print("End of video or camera disconnected")
                    break

            display = self.process_frame()
            cv2.imshow("Real-Time Drawing Tracker", display)

            key = cv2.waitKey(30) & 0xFF
            if not self.handle_keyboard(key):
                break

        self.cleanup()

    # ----------------------- Core Processing -----------------------

    def process_frame(self):
        self.frame_idx += 1

        # 1) Warp drawing FIRST (so it visibly moves with organ)
        track_ok = False
        if self.tracking_enabled and self.tracked_roi is not None:
            gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)

            if self.prev_gray is not None and self.tracked_points is not None and len(self.tracked_points) > 0:
                track_ok = self.update_tracking(self.prev_gray, gray)
            else:
                track_ok = False

            self.prev_gray = gray.copy()

        # 2) Composite
        combined = cv2.addWeighted(self.current_frame, 1.0, self.drawing_layer, 0.7, 0)

        # 3) Overlays
        if self.tracking_enabled and self.tracked_roi is not None:
            x, y, w, h = self.tracked_roi
            cv2.rectangle(combined, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if self.tracked_points is not None:
                pts2 = self.tracked_points.reshape(-1, 2)
                for pt in pts2:
                    cv2.circle(combined, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)

            if track_ok:
                npts = 0 if self.tracked_points is None else len(self.tracked_points)
                cv2.putText(combined, f"Tracking OK | points={npts}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(combined, "TRACKING LOST - Draw again then press D", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        self.draw_ui(combined)

        # 4) Display scaling (NO UPSCALING)
        h, w = combined.shape[:2]
        scale = min(self.max_display_w / w, self.max_display_h / h, 1.0)
        self.display_scale = float(scale)

        if scale < 1.0:
            combined = cv2.resize(combined, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        return combined

    # ----------------------- Tracking Improvements -----------------------

    def update_tracking(self, prev_gray, curr_gray):
        """
        Steps:
        - LK forward prev->curr
        - LK backward curr->prev (FB check)
        - Keep only consistent points (FB error < threshold)
        - Compute homography (RANSAC)
        - Warp drawing_layer with homography
        - Update ROI by transforming ROI corners with homography
        - Optionally reseed points in ROI periodically
        """
        p0 = self.tracked_points.astype(np.float32).reshape(-1, 1, 2)

        # Forward flow
        p1, st1, err1 = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **self.lk_params)
        if p1 is None or st1 is None:
            return False

        st1 = st1.reshape(-1).astype(bool)
        p0_f = p0[st1]
        p1_f = p1[st1]

        if len(p1_f) < self.min_points:
            return False

        # Backward flow for FB check
        p0_back, st2, err2 = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, p1_f, None, **self.lk_params)
        if p0_back is None or st2 is None:
            return False

        st2 = st2.reshape(-1).astype(bool)
        p0_ff = p0_f[st2]
        p1_ff = p1_f[st2]
        p0_back = p0_back[st2]

        if len(p1_ff) < self.min_points:
            return False

        # FB error: distance between original p0 and p0_back
        fb_err = np.linalg.norm((p0_ff - p0_back).reshape(-1, 2), axis=1)
        good_fb = fb_err < self.fb_max_error

        p0_g = p0_ff[good_fb]
        p1_g = p1_ff[good_fb]

        if len(p1_g) < self.min_points:
            return False

        # Optional: keep points that remain near ROI (reduces drift)
        p1_xy = p1_g.reshape(-1, 2)
        x, y, w, h = self.tracked_roi
        pad = 20
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = x + w + pad, y + h + pad
        inside = (p1_xy[:, 0] >= x0) & (p1_xy[:, 0] <= x1) & (p1_xy[:, 1] >= y0) & (p1_xy[:, 1] <= y1)

        p0_g = p0_g[inside]
        p1_g = p1_g[inside]

        if len(p1_g) < self.min_points:
            return False

        # Homography (projective) with RANSAC
        H, inliers = cv2.findHomography(p0_g, p1_g, cv2.RANSAC, self.ransac_thresh)
        if H is None:
            return False

        # Warp drawing layer using H (so drawing "sticks" to organ surface)
        self.drawing_layer = cv2.warpPerspective(
            self.drawing_layer, H,
            (curr_gray.shape[1], curr_gray.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # Update ROI by transforming ROI corners
        self.tracked_roi = self.transform_roi_with_homography(self.tracked_roi, H, curr_gray.shape[1], curr_gray.shape[0])
        if self.tracked_roi is None:
            return False

        # Keep inlier points only (better stability)
        if inliers is not None:
            inl = inliers.reshape(-1).astype(bool)
            p1_in = p1_g[inl]
        else:
            p1_in = p1_g

        if len(p1_in) < self.min_points:
            return False

        # Update tracked points
        self.tracked_points = p1_in.reshape(-1, 1, 2).astype(np.float32)

        # Reseed points occasionally (prevents long-term drift)
        if self.frame_idx % self.reseed_interval == 0:
            self.reseed_points(curr_gray)

        return True

    def transform_roi_with_homography(self, roi, H, width, height):
        x, y, w, h = roi

        corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.float32).reshape(-1, 1, 2)

        warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

        minx = int(np.floor(np.min(warped[:, 0])))
        miny = int(np.floor(np.min(warped[:, 1])))
        maxx = int(np.ceil(np.max(warped[:, 0])))
        maxy = int(np.ceil(np.max(warped[:, 1])))

        # Clamp and validate
        minx = max(0, minx)
        miny = max(0, miny)
        maxx = min(width - 1, maxx)
        maxy = min(height - 1, maxy)

        new_w = maxx - minx
        new_h = maxy - miny

        if new_w < 10 or new_h < 10:
            return None

        return (minx, miny, new_w, new_h)

    def reseed_points(self, gray):
        """Re-detect good corners in current ROI and replace points."""
        if self.tracked_roi is None:
            return

        x, y, w, h = self.tracked_roi
        # pad ROI a little for texture
        pad = 10
        x2 = max(0, x - pad)
        y2 = max(0, y - pad)
        w2 = min(gray.shape[1] - x2, w + 2 * pad)
        h2 = min(gray.shape[0] - y2, h + 2 * pad)

        roi_gray = gray[y2:y2 + h2, x2:x2 + w2]
        corners = cv2.goodFeaturesToTrack(roi_gray, **self.feature_params)
        if corners is None or len(corners) < self.min_points:
            return

        corners = corners.astype(np.float32)
        corners[:, 0, 0] += x2
        corners[:, 0, 1] += y2
        self.tracked_points = corners

    # ----------------------- Drawing / Detection -----------------------

    def detect_drawn_mark(self):
        gray_drawing = cv2.cvtColor(self.drawing_layer, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_drawing, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("No drawing found. Draw something first!")
            return

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 150:
            print("Drawing too small. Draw a larger mark!")
            return

        x, y, w, h = cv2.boundingRect(largest)

        pad = self.roi_padding
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(self.current_frame.shape[1] - x, w + 2 * pad)
        h = min(self.current_frame.shape[0] - y, h + 2 * pad)
        self.tracked_roi = (x, y, w, h)

        gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[y:y + h, x:x + w]

        corners = cv2.goodFeaturesToTrack(roi_gray, **self.feature_params)
        if corners is None or len(corners) < self.min_points:
            print("Not enough features to track. Draw on a more textured area!")
            self.tracked_roi = None
            return

        corners = corners.astype(np.float32)
        corners[:, 0, 0] += x
        corners[:, 0, 1] += y

        self.tracked_points = corners  # Nx1x2
        self.prev_gray = gray.copy()
        self.tracking_enabled = True
        self.frame_idx = 0
        print(f"Tracking started with {len(self.tracked_points)} points")

    # ----------------------- UI / Input -----------------------

    def mouse_callback(self, event, x, y, flags, param):
        # map display coords -> original coords
        sx = int(x / self.display_scale)
        sy = int(y / self.display_scale)
        current_point = (sx, sy)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            self.last_point = current_point

        elif event == cv2.EVENT_MOUSEMOVE and self.is_drawing:
            if self.last_point is not None:
                cv2.line(self.drawing_layer, self.last_point, current_point,
                         self.current_color, self.brush_size)
                cv2.circle(self.drawing_layer, current_point, self.brush_size,
                           self.current_color, -1)
            self.last_point = current_point

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.last_point = None

    def handle_keyboard(self, key):
        if key == 27 or key == ord("q"):
            return False
        elif key == ord("h"):
            self.show_help = not self.show_help
        elif key == ord("c"):
            self.clear_drawing()
        elif key == ord("t"):
            self.toggle_tracking()
        elif key == ord("d"):
            self.detect_drawn_mark()
        elif key == ord("v"):
            self.pick_dataset_video()
        elif key == ord("s"):
            self.save_current_frame()
        elif key == ord(" "):
            self.paused = not self.paused

        elif key == ord("r"):
            self.current_color = (0, 0, 255)
        elif key == ord("g"):
            self.current_color = (0, 255, 0)
        elif key == ord("b"):
            self.current_color = (255, 0, 0)
        elif key == ord("y"):
            self.current_color = (0, 255, 255)
        elif key == ord("k"):
            self.current_color = (0, 0, 0)
        elif key == ord("w"):
            self.current_color = (255, 255, 255)

        elif ord("1") <= key <= ord("9"):
            self.brush_size = max(1, (key - ord("0")))

        return True

    def draw_ui(self, frame):
        color_name = self.get_color_name()
        status = "PAUSED" if self.paused else "PLAYING"
        mode_text = f"Color: {color_name} | Size: {self.brush_size} | {status}"
        cv2.putText(frame, mode_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if self.show_help:
            self.draw_help_overlay(frame)
        else:
            h = frame.shape[0]
            cv2.putText(frame, "Press H for help", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def draw_help_overlay(self, frame):
        help_text = [
            "=== CONTROLS ===",
            "Mouse drag - Draw",
            "D - Detect drawn mark (init tracking)",
            "T - Start/Stop tracking",
            "C - Clear drawing",
            "V - Pick dataset video",
            "SPACE - Pause/Resume",
            "R/G/B/Y/K/W - Change color",
            "1-9 - Brush size",
            "H - Toggle help",
            "ESC/Q - Quit"
        ]

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 80), (470, 340), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        y = 100
        for line in help_text:
            cv2.putText(frame, line, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 20

    def clear_drawing(self):
        self.drawing_layer = np.zeros_like(self.current_frame)
        self.tracking_enabled = False
        self.tracked_roi = None
        self.tracked_points = None
        self.prev_gray = None
        self.frame_idx = 0
        print("Drawing cleared")

    def toggle_tracking(self):
        if self.tracking_enabled:
            self.tracking_enabled = False
            self.tracked_roi = None
            self.tracked_points = None
            print("Tracking stopped")
        else:
            self.detect_drawn_mark()

    def pick_dataset_video(self):
        if not self.dataset_root.exists():
            print("Dataset root not found:", self.dataset_root)
            return

        print("\nChoose folder:")
        for i, f in enumerate(self.dataset_folders, 1):
            print(f"  {i}) {f}")
        s = input("Folder #: ").strip()
        if not s.isdigit():
            print("Cancelled.")
            return
        idx = int(s)
        if not (1 <= idx <= len(self.dataset_folders)):
            print("Cancelled.")
            return

        folder = self.dataset_root / self.dataset_folders[idx - 1]
        videos = sorted([p for p in folder.rglob("*") if p.suffix.lower() in self.video_exts])
        if not videos:
            print("No videos found in:", folder)
            return

        print("\nChoose video:")
        show = videos[:50]
        for i, p in enumerate(show, 1):
            print(f"  {i}) {p.name}")
        s2 = input("Video #: ").strip()
        if not s2.isdigit():
            print("Cancelled.")
            return
        vidx = int(s2)
        if not (1 <= vidx <= len(show)):
            print("Cancelled.")
            return

        video_path = str(show[vidx - 1])
        print("Loading:", video_path)

        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print("Failed to load video.")
            return

        ret, self.current_frame = self.cap.read()
        if not ret or self.current_frame is None:
            print("Failed to read first frame.")
            return

        self.drawing_layer = np.zeros_like(self.current_frame)
        self.clear_drawing()
        self.paused = False
        print("Loaded OK.")

    def save_current_frame(self):
        combined = cv2.addWeighted(self.current_frame, 1.0, self.drawing_layer, 0.7, 0)
        filename = f"frame_{cv2.getTickCount()}.png"
        cv2.imwrite(filename, combined)
        print(f"Saved: {filename}")

    def get_color_name(self):
        color_map = {
            (0, 0, 255): "Red",
            (0, 255, 0): "Green",
            (255, 0, 0): "Blue",
            (0, 255, 255): "Yellow",
            (0, 0, 0): "Black",
            (255, 255, 255): "White"
        }
        return color_map.get(self.current_color, "Custom")

    def print_controls(self):
        print("\nCONTROLS:")
        print("  Mouse: Draw on video")
        print("  D: Detect drawn mark (init tracking)")
        print("  T: Start/stop tracking")
        print("  C: Clear drawing")
        print("  V: Pick dataset video")
        print("  SPACE: Pause/Resume")
        print("  R/G/B/Y/K/W: Change color")
        print("  1-9: Change brush size")
        print("  H: Toggle help")
        print("  ESC/Q: Quit")
        print("=" * 50 + "\n")

    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")
