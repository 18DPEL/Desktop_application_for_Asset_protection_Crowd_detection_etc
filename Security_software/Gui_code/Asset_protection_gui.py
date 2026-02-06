"""
Asset Protection PyQt GUI
==========================
Professional security monitoring interface with:
- Left sidebar with controls
- 2x2 camera grid view
- Black and white theme
- RobustRTSPReader integration
"""

import sys
import cv2
import time
import numpy as np
import threading
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QLineEdit, QFrame, QScrollArea,
    QGroupBox, QSpacerItem, QSizePolicy, QMessageBox, QDialog,
    QFormLayout, QSpinBox, QComboBox, QCheckBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont, QPalette, QColor

# Add RTSP reader path
sys.path.insert(0, r"D:\LLM_SETUP\Scurity_software\RTSP_Reader")
sys.path.insert(0, r"D:\LLM_SETUP\Scurity_software\Gui_code")
from rtsp_reader import RobustRTSPReader, RTSPConfig, DecodingMode

# Import Asset Protection detector
YOLO_AVAILABLE = False
PolygonAssetProtector = None
DETECTION_CLASSES = {}
draw_movement_path = None

try:
    from Asset_protection import PolygonAssetProtector, DETECTION_CLASSES, draw_movement_path
    YOLO_AVAILABLE = True
    print("✓ YOLO detection module loaded from Asset_protection")
except Exception as e:
    print(f"⚠️ Asset_protection import failed: {e}")
    # Try direct YOLO import as fallback
    try:
        from ultralytics import YOLO
        import torch
        YOLO_AVAILABLE = True
        print("✓ YOLO available via ultralytics (fallback mode)")
    except:
        print("⚠️ ultralytics YOLO not available either")

# Import Line Tripwire Detector
try:
    from line_cross_trip import LineTripwireDetector
    print("✓ LineTripwireDetector module loaded")
except ImportError as e:
    print(f"⚠️ line_cross_trip import failed: {e}")
    LineTripwireDetector = None

# Import Loitering Detector
try:
    from loitering_detection import LoiteringDetector
    print("✓ LoiteringDetector module loaded")
except ImportError as e:
    print(f"⚠️ loitering_detection import failed: {e}")
    LoiteringDetector = None


# ============================================================
# DETECTION TYPES
# ============================================================
DETECTION_TYPE_ASSET = "Asset Protection"
DETECTION_TYPE_CROWD = "Crowd Detection"
DETECTION_TYPE_INTRUSION = "Intrusion Detection"
DETECTION_TYPE_LINE_CROSS = "Line Cross Tripwire"
DETECTION_TYPE_LOITERING = "Loitering Detection"
DETECTION_TYPES = [DETECTION_TYPE_ASSET, DETECTION_TYPE_CROWD, DETECTION_TYPE_INTRUSION, DETECTION_TYPE_LINE_CROSS, DETECTION_TYPE_LOITERING]

# Crowd detection settings
CROWD_LIMIT = 10  # Alert when this many people detected in zone

# ============================================================
# BLACK & WHITE THEME STYLESHEET
# ============================================================
DARK_THEME = """
QMainWindow {
    background-color: #1a1a1a;
}

QWidget {
    background-color: #1a1a1a;
    color: #ffffff;
    font-family: 'Segoe UI', Arial, sans-serif;
}

QFrame {
    background-color: #2a2a2a;
    border: 1px solid #404040;
    border-radius: 4px;
}

QLabel {
    color: #ffffff;
    background-color: transparent;
    border: none;
}

QPushButton {
    background-color: #3a3a3a;
    color: #ffffff;
    border: 1px solid #505050;
    border-radius: 4px;
    padding: 8px 16px;
    font-size: 12px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #4a4a4a;
    border-color: #606060;
}

QPushButton:pressed {
    background-color: #2a2a2a;
}

QPushButton:disabled {
    background-color: #2a2a2a;
    color: #666666;
}

QPushButton#startBtn {
    background-color: #2d5a2d;
    border-color: #3d7a3d;
}

QPushButton#startBtn:hover {
    background-color: #3d7a3d;
}

QPushButton#stopBtn {
    background-color: #5a2d2d;
    border-color: #7a3d3d;
}

QPushButton#stopBtn:hover {
    background-color: #7a3d3d;
}

QLineEdit {
    background-color: #2a2a2a;
    color: #ffffff;
    border: 1px solid #404040;
    border-radius: 4px;
    padding: 8px;
    font-size: 12px;
}

QLineEdit:focus {
    border-color: #606060;
}

QGroupBox {
    border: 1px solid #404040;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #cccccc;
}

QScrollArea {
    border: none;
    background-color: transparent;
}

QSpinBox, QComboBox {
    background-color: #2a2a2a;
    color: #ffffff;
    border: 1px solid #404040;
    border-radius: 4px;
    padding: 5px;
}

QCheckBox {
    color: #ffffff;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid #505050;
    border-radius: 3px;
    background-color: #2a2a2a;
}

QCheckBox::indicator:checked {
    background-color: #4a8a4a;
    border-color: #5a9a5a;
}
"""


# ============================================================
# CAMERA FEED WIDGET
# ============================================================
class CameraWidget(QFrame):
    """Widget to display a single camera feed"""
    
    # Signal emitted when camera is clicked
    clicked = pyqtSignal(int)  # Emits camera_id
    
    def __init__(self, camera_id: int, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.camera_name = f"Camera {camera_id + 1}"
        self.stream = None
        self.is_connected = False
        self._current_frame = None  # Store current frame for popup
        self._current_pixmap = None
        
        # Detection state - set by popup when closed
        self.detector = None  # PolygonAssetProtector instance
        self.polygon_points = []  # Polygon for visualization
        self.detection_active = False
        self._detection_result = None
        self.detection_type = DETECTION_TYPE_ASSET  # Default detection type
        
        self._setup_ui()
        self.setCursor(Qt.CursorShape.PointingHandCursor)  # Show hand cursor
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #0a0a0a;
                border: 2px solid #404040;
                border-radius: 4px;
                color: #666666;
                font-size: 18px;
            }
        """)
        self.video_label.setText(f"cam_{self.camera_id + 1}")
        layout.addWidget(self.video_label, stretch=1)
        
        # Status bar
        self.status_label = QLabel("Disconnected")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 10px;
                padding: 2px;
            }
        """)
        layout.addWidget(self.status_label)
    
    def update_frame(self, frame: np.ndarray):
        """Update the video display with a new frame"""
        if frame is None:
            return
        
        try:
            # Store original frame for popup
            self._current_frame = frame.copy()
            
            # Run detection if active
            if self.detection_active and self.detector:
                self._detection_result = self.detector.process(frame)
                # Draw detection overlays on frame
                frame = self._draw_mini_overlay(frame)
            
            # Resize to fit widget
            h, w = frame.shape[:2]
            label_w = self.video_label.width() - 4
            label_h = self.video_label.height() - 4
            
            if label_w > 0 and label_h > 0:
                # Maintain aspect ratio
                scale = min(label_w / w, label_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                if new_w > 0 and new_h > 0:
                    frame = cv2.resize(frame, (new_w, new_h))
            
            # Convert to QImage - MUST copy data to prevent flickering
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            
            # Make a contiguous copy of the data
            rgb_frame = np.ascontiguousarray(rgb_frame)
            
            # Create QImage with copy of data (crucial for preventing flicker)
            q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            
            # Store pixmap reference to prevent garbage collection
            self._current_pixmap = QPixmap.fromImage(q_image)
            self.video_label.setPixmap(self._current_pixmap)
        except Exception as e:
            pass  # Ignore frame errors silently
    
    def _draw_mini_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection overlay on small camera view based on detection type"""
        frame = frame.copy()
        
        # For Line Cross mode, check tripwires instead of polygon_points
        if self.detection_type == DETECTION_TYPE_LINE_CROSS:
            if not self._detection_result:
                 # Just draw lines if no detection yet
                 if self.detector and hasattr(self.detector, 'tripwires'):
                      for line in self.detector.tripwires.values():
                           cv2.line(frame, line['p1'], line['p2'], (0, 255, 255), 1)
                 return frame

            # Draw lines (red if crossed?)
            alerts = self._detection_result.get('alerts', [])
            crossed_lines = {a['line'] for a in alerts}
            
            tripwires = self.detector.tripwires if self.detector else {}
            for name, line in tripwires.items():
                p1, p2 = line['p1'], line['p2']
                color = (0, 0, 255) if name in crossed_lines else (0, 255, 255)
                # Check persistent flash
                if hasattr(self.detector, 'tripwires') and 'last_cross_time' in self.detector.tripwires[name]:
                     if time.time() - self.detector.tripwires[name]['last_cross_time'] < 1.0:
                         color = (0, 0, 255)
                
                cv2.line(frame, p1, p2, color, 1)

            # Draw detected objects
            for obj in self._detection_result.get('detected_objects', []):
                 b = obj['bbox']
                 cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)
            
            # Status
            cv2.putText(frame, "TRIPWIRE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            return frame

        # =========================================================
        # POLYGON-BASED MODES: Require polygon_points
        # =========================================================
        if len(self.polygon_points) == 0:
            return frame

        # =========================================================
        # CROWD DETECTION MINI OVERLAY
        # =========================================================
        if self.detection_type == DETECTION_TYPE_CROWD:
            # Count people inside polygon
            crowd_count = 0
            polygon_np = np.array(self.polygon_points, dtype=np.int32)
            
            # Draw polygon
            alert = False
            if self._detection_result:
                for obj in self._detection_result.get('detected_objects', []):
                    b = obj['bbox']
                    cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
                    if cv2.pointPolygonTest(polygon_np, (cx, cy), False) >= 0:
                        crowd_count += 1
                        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 1)
            
            if crowd_count >= CROWD_LIMIT:
                alert = True
                
            color = (0, 0, 255) if alert else (0, 255, 0)
            
            # Draw Polygon
            if len(self.polygon_points) >= 3:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [polygon_np], color)
                frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
                cv2.polylines(frame, [polygon_np], True, color, 2)
            
            # Draw Status
            cv2.putText(frame, f"CROWD: {crowd_count}", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                       
            return frame

        # =========================================================
        # INTRUSION DETECTION MINI OVERLAY
        # =========================================================
        if self.detection_type == DETECTION_TYPE_INTRUSION:
            # Count intruders inside polygon
            intrusion_count = 0
            polygon_np = np.array(self.polygon_points, dtype=np.int32)
            
            # Draw polygon
            alert = False
            if self._detection_result:
                for obj in self._detection_result.get('detected_objects', []):
                    # Only person class
                    if obj.get('object_type') != 'person':
                        continue
                        
                    b = obj['bbox']
                    cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
                    if cv2.pointPolygonTest(polygon_np, (cx, cy), False) >= 0:
                        intrusion_count += 1
                        # Draw RED box on intruder
                        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)
            
            if intrusion_count > 0:
                alert = True
                
            color = (0, 0, 255) if alert else (0, 255, 0)
            
            # Draw Polygon
            if len(self.polygon_points) >= 3:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [polygon_np], color)
                frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
                cv2.polylines(frame, [polygon_np], True, color, 2)
            
            # Draw Status
            status_text = f"INTRUDERS: {intrusion_count}" if alert else "SECURE"
            cv2.putText(frame, status_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                       
            return frame

        # =========================================================
        # LOITERING DETECTION MINI OVERLAY
        # =========================================================
        if self.detection_type == DETECTION_TYPE_LOITERING:
            polygon_np = np.array(self.polygon_points, dtype=np.int32)
            
            # Count loiterers
            loitering_count = 0
            alert = False
            
            if self._detection_result:
                loitering_count = self._detection_result.get('loitering_count', 0)
                if loitering_count > 0:
                    alert = True
                
                # Draw detected persons
                for obj in self._detection_result.get('detected_objects', []):
                    b = obj['bbox']
                    is_loitering = obj.get('is_loitering', False)
                    is_inside = obj.get('is_inside', False)
                    time_in_zone = obj.get('time_in_zone', 0)
                    
                    if is_loitering:
                        color = (0, 0, 255)  # Red for loitering
                        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 2)
                    elif is_inside:
                        color = (0, 255, 0)  # Green for in zone
                        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 1)
                    else:
                        color = (100, 100, 100)  # Gray outside
                        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 1)
            
            color = (0, 0, 255) if alert else (0, 255, 0)
            
            # Draw Polygon
            if len(self.polygon_points) >= 3:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [polygon_np], color)
                frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
                cv2.polylines(frame, [polygon_np], True, color, 2)
            
            # Draw Status
            status_text = f"LOITER: {loitering_count}" if alert else "LOITERING"
            cv2.putText(frame, status_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                       
            return frame

        # =========================================================
        # ASSET PROTECTION MINI OVERLAY
        # =========================================================
        
        # Determine polygon color based on alert state
        has_alert = False
        protected_objects = {}
        if self._detection_result:
            protected_objects = self._detection_result.get('protected_objects', {})
            for pobj in protected_objects.values():
                if pobj.get('status') == 'left_zone':
                    has_alert = True
                    break
        
        color = (0, 0, 255) if has_alert else (0, 255, 0)  # Red for alert, green for normal
        
        # Draw polygon
        polygon_np = np.array(self.polygon_points, dtype=np.int32)
        if len(self.polygon_points) >= 3:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon_np], color)
            frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
            cv2.polylines(frame, [polygon_np], True, color, 2)
        
        # Draw movement trails for protected objects
        for track_id, pobj in protected_objects.items():
            movement_path = pobj.get('movement_path', [])
            if len(movement_path) > 1:
                # Draw simplified trail (every 3rd point for small view)
                path_color = (0, 0, 255) if pobj.get('status') == 'left_zone' else (0, 255, 255)
                for i in range(1, len(movement_path), 2):
                    pt1 = tuple(map(int, movement_path[max(0, i-2)]))
                    pt2 = tuple(map(int, movement_path[i]))
                    cv2.line(frame, pt1, pt2, path_color, 1)
        
        # Draw detected objects (simplified for small view)
        if self._detection_result:
            for obj in self._detection_result.get('detected_objects', []):
                b = obj['bbox']
                track_id = obj['track_id']
                
                if track_id in protected_objects:
                    if protected_objects[track_id].get('status') == 'left_zone':
                        box_color = (0, 0, 255)  # Red
                    else:
                        box_color = (0, 255, 0)  # Green
                else:
                    box_color = (150, 150, 150)  # Gray
                
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), box_color, 1)
        
        # Draw monitoring indicator
        cv2.circle(frame, (20, 20), 8, color, -1)
        cv2.putText(frame, "LIVE", (35, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
    
    def set_status(self, status: str, is_connected: bool = False):
        """Update status label"""
        self.is_connected = is_connected
        self.status_label.setText(status)
        
        if is_connected:
            self.status_label.setStyleSheet("QLabel { color: #4a8a4a; font-size: 10px; padding: 2px; }")
        else:
            self.status_label.setStyleSheet("QLabel { color: #888888; font-size: 10px; padding: 2px; }")
    
    def show_waiting(self, message: str = "Connecting..."):
        """Show waiting state"""
        self.video_label.setText(message)
        self.video_label.setPixmap(QPixmap())
        self._current_frame = None
    
    def mousePressEvent(self, event):
        """Handle mouse click to open popup"""
        if event.button() == Qt.MouseButton.LeftButton and self._current_frame is not None:
            self.clicked.emit(self.camera_id)
        super().mousePressEvent(event)
    
    def get_current_frame(self):
        """Get the current frame for popup display"""
        return self._current_frame


# ============================================================
# CAMERA POPUP DIALOG (Enlarged View with Asset Protection)
# ============================================================
class ClickableVideoLabel(QLabel):
    """Video label that captures mouse clicks for polygon drawing"""
    
    clicked = pyqtSignal(int, int)  # x, y coordinates
    right_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._mouse_pos = None
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(event.position().x().__int__(), event.position().y().__int__())
        elif event.button() == Qt.MouseButton.RightButton:
            self.right_clicked.emit()
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        self._mouse_pos = (event.position().x().__int__(), event.position().y().__int__())
        super().mouseMoveEvent(event)
    
    def get_mouse_pos(self):
        return self._mouse_pos


class CameraPopupDialog(QDialog):
    """Large popup window with polygon drawing for asset protection"""
    
    def __init__(self, camera_id: int, camera_name: str, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.setWindowTitle(f"�️ {camera_name} - Asset Protection")
        self.setMinimumSize(900, 700)
        self.resize(1100, 800)
        self.setStyleSheet(DARK_THEME)
        
        # Polygon state
        self.polygon_points = []  # List of (x, y) in frame coordinates
        self.polygon_complete = False
        self.detection_active = False
        
        # Tripwire State
        self.current_drawing_start = None # point (x,y)

        
        # Frame state
        self._current_frame = None
        self._display_frame = None
        self._current_pixmap = None
        self._frame_scale = 1.0
        self._frame_offset = (0, 0)
        
        # Detection results
        self._detection_result = None
        self._alert_count = 0
        self.crowd_count = 0  # For crowd detection mode
        self.intrusion_count = 0  # For intrusion detection mode
        
        # Detection type selection
        self.detection_type = DETECTION_TYPE_ASSET  # Default
        
        # Initialize the asset protector (YOLO model) if available
        self.detector = None
        if YOLO_AVAILABLE and PolygonAssetProtector is not None:
            try:
                self.detector = PolygonAssetProtector()
                print(f"✓ YOLO detector initialized for {camera_name}")
            except Exception as e:
                print(f"⚠️ Failed to initialize detector: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                self.detector = None
        else:
            print(f"⚠️ Detector not available: YOLO_AVAILABLE={YOLO_AVAILABLE}, PolygonAssetProtector={PolygonAssetProtector}")
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Instructions label
        self.instructions_label = QLabel("📍 Click to add polygon points | Right-click to complete polygon")
        self.instructions_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.instructions_label.setStyleSheet("color: #ffcc00; font-size: 12px; font-weight: bold; padding: 5px; background-color: #333;")
        layout.addWidget(self.instructions_label)
        
        # Video display (clickable)
        self.video_label = ClickableVideoLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(880, 520)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #0a0a0a;
                border: 2px solid #505050;
                border-radius: 6px;
            }
        """)
        self.video_label.setCursor(Qt.CursorShape.CrossCursor)
        self.video_label.clicked.connect(self._on_video_clicked)
        self.video_label.right_clicked.connect(self._complete_polygon)
        layout.addWidget(self.video_label, stretch=1)
        
        # Status bar
        self.status_label = QLabel("Draw a protection zone by clicking points on the video")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #888888; font-size: 12px; padding: 5px;")
        layout.addWidget(self.status_label)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        
        # Detection type selector
        type_label = QLabel("Detection Type:")
        type_label.setStyleSheet("color: #cccccc; font-weight: bold;")
        btn_layout.addWidget(type_label)
        
        self.detection_type_combo = QComboBox()
        self.detection_type_combo.addItems(DETECTION_TYPES)
        self.detection_type_combo.setFixedWidth(150)
        self.detection_type_combo.setStyleSheet("""
            QComboBox {
                background-color: #2a2a2a;
                color: white;
                border: 1px solid #505050;
                padding: 5px;
                border-radius: 3px;
            }
            QComboBox:hover { border-color: #707070; }
            QComboBox::drop-down { border: none; }
            QComboBox::down-arrow { image: none; border: none; }
        """)
        self.detection_type_combo.currentTextChanged.connect(self._on_detection_type_changed)
        btn_layout.addWidget(self.detection_type_combo)
        
        btn_layout.addSpacing(20)
        
        # Reset polygon button
        self.reset_btn = QPushButton("🔄 Reset Zone")
        self.reset_btn.clicked.connect(self._reset_polygon)
        self.reset_btn.setFixedWidth(130)
        btn_layout.addWidget(self.reset_btn)
        
        # Complete polygon button
        self.complete_btn = QPushButton("✓ Complete Zone")
        self.complete_btn.clicked.connect(self._complete_polygon)
        self.complete_btn.setFixedWidth(130)
        self.complete_btn.setEnabled(False)
        btn_layout.addWidget(self.complete_btn)
        
        btn_layout.addStretch()
        
        # Start detection button
        self.start_detect_btn = QPushButton("▶ Start Detection")
        self.start_detect_btn.setObjectName("startBtn")
        self.start_detect_btn.clicked.connect(self._start_detection)
        self.start_detect_btn.setFixedWidth(150)
        self.start_detect_btn.setEnabled(False)
        btn_layout.addWidget(self.start_detect_btn)
        
        # Stop detection button  
        self.stop_detect_btn = QPushButton("■ Stop Detection")
        self.stop_detect_btn.setObjectName("stopBtn")
        self.stop_detect_btn.clicked.connect(self._stop_detection)
        self.stop_detect_btn.setFixedWidth(150)
        self.stop_detect_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_detect_btn)
        
        btn_layout.addStretch()
        
        # Close button
        close_btn = QPushButton("✕ Close")
        close_btn.clicked.connect(self.close)
        close_btn.setFixedWidth(100)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
    
    def _screen_to_frame(self, screen_x: int, screen_y: int):
        """Convert screen coordinates to frame coordinates"""
        if self._frame_scale == 0:
            return None
        
        # Get label geometry
        label_w = self.video_label.width()
        label_h = self.video_label.height()
        
        if self._current_frame is None:
            return None
        
        frame_h, frame_w = self._current_frame.shape[:2]
        
        # Calculate scaled frame size
        scale = min(label_w / frame_w, label_h / frame_h)
        scaled_w = int(frame_w * scale)
        scaled_h = int(frame_h * scale)
        
        # Calculate offset (centered)
        offset_x = (label_w - scaled_w) // 2
        offset_y = (label_h - scaled_h) // 2
        
        # Convert to frame coordinates
        frame_x = int((screen_x - offset_x) / scale)
        frame_y = int((screen_y - offset_y) / scale)
        
        # Clamp to frame bounds
        frame_x = max(0, min(frame_w - 1, frame_x))
        frame_y = max(0, min(frame_h - 1, frame_y))
        
        return (frame_x, frame_y)
    
    def _on_video_clicked(self, x: int, y: int):
        """Handle click on video to add polygon point"""
        if self.polygon_complete or (self.detection_type != DETECTION_TYPE_LINE_CROSS and self.detection_active):
            return
        
        frame_coords = self._screen_to_frame(x, y)
        if not frame_coords:
            return

        # TRIPWIRE MODE
        if self.detection_type == DETECTION_TYPE_LINE_CROSS:
            if not self.detection_active: # Only allow drawing when not active (or allow adding dynamically?)
                if self.current_drawing_start is None:
                    # Set start point
                    self.current_drawing_start = frame_coords
                    print(f"📍 Tripwire start: {frame_coords}")
                else:
                    # Set end point and add line
                    if self.detector:
                        self.detector.add_tripwire(self.current_drawing_start, frame_coords)
                    self.current_drawing_start = None
                    print(f"📍 Tripwire end: {frame_coords}. Line added.")
                self._update_ui_state()
            return

        # POLYGON MODE (Default)
        if frame_coords:
            self.polygon_points.append(frame_coords)
            self._update_ui_state()
            print(f"📍 Added point {len(self.polygon_points)}: {frame_coords}")
    
    def _complete_polygon(self):
        """Complete the polygon"""
        if len(self.polygon_points) >= 3 and not self.polygon_complete:
            self.polygon_complete = True
            self._update_ui_state()
            print(f"✅ Polygon completed with {len(self.polygon_points)} points")
    
    def _on_detection_type_changed(self, detection_type: str):
        """Handle detection type selection change"""
        self.detection_type = detection_type
        self.setWindowTitle(f"🛡️ {self.camera_name} - {detection_type}")
        print(f"🔄 Detection type changed to: {detection_type}")
        
        # Switch detector instance if needed
        if detection_type == DETECTION_TYPE_LINE_CROSS:
            if LineTripwireDetector:
                try:
                    self.detector = LineTripwireDetector()
                    print(f"✓ LineTripwireDetector initialized for {self.camera_name}")
                except Exception as e:
                    print(f"Error init line detector: {e}")
                    self.detector = None
            else:
                 print("LineTripwireDetector class not available")
        elif detection_type == DETECTION_TYPE_LOITERING:
            if LoiteringDetector:
                try:
                    self.detector = LoiteringDetector()
                    print(f"✓ LoiteringDetector initialized for {self.camera_name}")
                except Exception as e:
                    print(f"Error init loitering detector: {e}")
                    self.detector = None
            else:
                 print("LoiteringDetector class not available")
        elif detection_type in [DETECTION_TYPE_ASSET, DETECTION_TYPE_CROWD, DETECTION_TYPE_INTRUSION]:
            # Revert to Polygon detector if available
            if PolygonAssetProtector:
                self.detector = PolygonAssetProtector()
                print(f"✓ PolygonAssetProtector initialized for {self.camera_name}")
        
        # Reset state
        self._reset_polygon()
    
    def _start_detection(self):
        """Start detection with selected mode"""
        self.detection_type = self.detection_type_combo.currentText()
        
        # LINE CROSS TRIPWIRE MODE
        if self.detection_type == DETECTION_TYPE_LINE_CROSS:
            # Check if we have at least one tripwire
            has_lines = False
            if self.detector and hasattr(self.detector, 'tripwires'):
                has_lines = len(self.detector.tripwires) > 0
            
            if not has_lines:
                print("⚠️ Cannot start - no tripwires defined!")
                return
            
            self.detection_active = True
            if self.detector:
                self.detector.start_detection()
                print(f"▶️ Line Cross Tripwire started with {len(self.detector.tripwires)} line(s)")
            
            self.detection_type_combo.setEnabled(False)
            self._update_ui_state()
            return
        
        # POLYGON-BASED MODES (Asset, Crowd, Intrusion)
        if self.polygon_complete:
            self.detection_active = True
            print(f"📋 DEBUG: Set detection_active=True for {self.detection_type}")
            
            # Set polygon on the detector
            if self.detector:
                self.detector.protection_polygon = [tuple(p) for p in self.polygon_points]
                self.detector.polygon_complete = True
                self.detector.start_detection()
                print(f"▶️ {self.detection_type} started with {len(self.polygon_points)} point polygon")
            else:
                print(f"▶️ {self.detection_type} started (visual only - YOLO not available)")
            
            # Disable detection type change while active
            self.detection_type_combo.setEnabled(False)
            self._update_ui_state()
    
    def _stop_detection(self):
        """Stop asset protection detection"""
        self.detection_active = False
        if self.detector:
            self.detector.stop_detection()
        self._detection_result = None
        self.detection_type_combo.setEnabled(True)  # Re-enable dropdown
        self._update_ui_state()
        print("⏹️ Detection stopped")
    
    def _reset_polygon(self):
        """Reset polygon and detection state"""
        self.polygon_points = []
        self.polygon_complete = False
        self.detection_active = False
        self._detection_result = None
        self.crowd_count = 0
        self.intrusion_count = 0
        if self.detector:
            if hasattr(self.detector, 'reset_polygon'):
                self.detector.reset_polygon()
            if hasattr(self.detector, 'clear_tripwires'):
                 self.detector.clear_tripwires()
            
        self.detection_type_combo.setEnabled(True)  # Re-enable dropdown
        self._update_ui_state()
        print("🔄 Polygon/Lines reset")
    
    def _update_ui_state(self):
        """Update UI based on current state"""
        point_count = len(self.polygon_points)
        
        # Update complete button
        self.complete_btn.setEnabled(point_count >= 3 and not self.polygon_complete)
        
        # Update detection buttons
        self.start_detect_btn.setEnabled(self.polygon_complete and not self.detection_active)
        self.stop_detect_btn.setEnabled(self.detection_active)
        
        # Update instructions
        if self.detection_active:
            self.instructions_label.setText("🛡️ MONITORING - Objects in zone are protected")
            self.instructions_label.setStyleSheet("color: #00ff00; font-size: 12px; font-weight: bold; padding: 5px; background-color: #1a3a1a;")
        
        elif self.detection_type == DETECTION_TYPE_LINE_CROSS:
            # Tripwire instructions
            if self.current_drawing_start:
                self.instructions_label.setText("📍 Click endpoint to finish line")
            else:
                line_count = len(self.detector.tripwires) if self.detector and hasattr(self.detector, 'tripwires') else 0
                self.instructions_label.setText(f"📍 Click start point for new line (Lines: {line_count}) | Click 'Start Detection' to begin")
            self.instructions_label.setStyleSheet("color: #ffcc00; font-size: 12px; font-weight: bold; padding: 5px; background-color: #333;")
            
            # Enable start if at least one line exists
            has_lines = False
            if self.detector and hasattr(self.detector, 'tripwires'):
                 has_lines = len(self.detector.tripwires) > 0
            self.start_detect_btn.setEnabled(has_lines and not self.detection_active)


        elif self.polygon_complete:
            self.instructions_label.setText("✓ Zone defined - Click 'Start Detection' to begin monitoring")
            self.instructions_label.setStyleSheet("color: #00ccff; font-size: 12px; font-weight: bold; padding: 5px; background-color: #1a2a3a;")
        else:
            remaining = max(0, 3 - point_count)
            if remaining > 0:
                self.instructions_label.setText(f"📍 Click to add points ({point_count} added, need {remaining} more) | Right-click to complete")
            else:
                self.instructions_label.setText(f"📍 {point_count} points added | Right-click or click 'Complete Zone' to finish")
            self.instructions_label.setStyleSheet("color: #ffcc00; font-size: 12px; font-weight: bold; padding: 5px; background-color: #333;")
    
    def _draw_polygon_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw polygon and detection visualization on frame"""
        frame = frame.copy()

        # START: TRIPWIRE DRAWING
        if self.detection_type == DETECTION_TYPE_LINE_CROSS:
             # Draw existing tripwires
            if self.detector and hasattr(self.detector, 'tripwires'):
                 for name, line in self.detector.tripwires.items():
                     p1, p2 = line['p1'], line['p2']
                     cv2.line(frame, p1, p2, (0, 255, 255), 2)
                     cv2.circle(frame, p1, 4, (0, 200, 255), -1)
                     cv2.circle(frame, p2, 4, (0, 200, 255), -1)
                     cv2.putText(frame, name, (p1[0], p1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Draw current line being drawn
            if self.current_drawing_start:
                 cv2.circle(frame, self.current_drawing_start, 5, (0, 255, 255), -1)
                 mouse_pos = self.video_label.get_mouse_pos()
                 if mouse_pos:
                     frame_mouse = self._screen_to_frame(*mouse_pos)
                     if frame_mouse:
                         cv2.line(frame, self.current_drawing_start, frame_mouse, (200, 200, 200), 1)

            # Draw results if active
            if self.detection_active and self._detection_result:
                frame = self._draw_detection_results(frame)
            
            return frame
        # END: TRIPWIRE DRAWING

        if len(self.polygon_points) == 0:
            return frame
        
        # Determine color based on state
        if self.detection_active:
            color = (0, 255, 0)  # Green when active
            fill_alpha = 0.15
        elif self.polygon_complete:
            color = (255, 200, 0)  # Cyan when ready
            fill_alpha = 0.2
        else:
            color = (0, 200, 255)  # Orange when drawing
            fill_alpha = 0.1
        
        polygon_np = np.array(self.polygon_points, dtype=np.int32)
        
        # Fill polygon with transparency
        if len(self.polygon_points) >= 3:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon_np], color)
            frame = cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0)
            
            # Draw polygon outline
            cv2.polylines(frame, [polygon_np], self.polygon_complete, color, 2)
        
        # Draw points
        for i, pt in enumerate(self.polygon_points):
            cv2.circle(frame, pt, 6, (255, 255, 255), -1)
            cv2.circle(frame, pt, 6, color, 2)
            cv2.putText(frame, str(i + 1), (pt[0] + 8, pt[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw line from last point to indicate drawing
        if not self.polygon_complete and len(self.polygon_points) >= 1:
            mouse_pos = self.video_label.get_mouse_pos()
            if mouse_pos:
                frame_mouse = self._screen_to_frame(*mouse_pos)
                if frame_mouse:
                    cv2.line(frame, self.polygon_points[-1], frame_mouse, (150, 150, 150), 1)
        
        # Draw YOLO detection results when active
        if self.detection_active and self._detection_result:
            frame = self._draw_detection_results(frame)
        elif self.detection_active:
            # Just draw monitoring text if no detection results yet
            cv2.putText(frame, "MONITORING ACTIVE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame
    
    def update_frame(self, frame: np.ndarray):
        """Update the popup with a new frame and run detection"""
        if frame is None:
            return
        
        try:
            self._current_frame = frame.copy()
            
            # Run YOLO detection if active
            if self.detection_active and self.detector:
                self._detection_result = self.detector.process(frame)
                
                # For crowd detection, count people inside the polygon
                if self.detection_type == DETECTION_TYPE_CROWD and self._detection_result:
                    self.crowd_count = 0
                    polygon_np = np.array(self.polygon_points, dtype=np.int32)
                    for obj in self._detection_result.get('detected_objects', []):
                        # Get center of bounding box
                        b = obj['bbox']
                        cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
                        # Check if inside polygon
                        if cv2.pointPolygonTest(polygon_np, (cx, cy), False) >= 0:
                            self.crowd_count += 1
                
                # For intrusion detection, count PEOPLE inside the polygon
                elif self.detection_type == DETECTION_TYPE_INTRUSION and self._detection_result:
                    self.intrusion_count = 0
                    polygon_np = np.array(self.polygon_points, dtype=np.int32)
                    for obj in self._detection_result.get('detected_objects', []):
                        # Check if object is a person
                        if obj.get('object_type') == 'person':
                            # Get center of bounding box
                            b = obj['bbox']
                            cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
                            # Check if inside polygon
                            if cv2.pointPolygonTest(polygon_np, (cx, cy), False) >= 0:
                                self.intrusion_count += 1
            
            # Draw polygon overlay and detection results
            display_frame = self._draw_polygon_overlay(frame)
            
            # Resize to fit popup
            h, w = display_frame.shape[:2]
            label_w = self.video_label.width() - 4
            label_h = self.video_label.height() - 4
            
            if label_w > 0 and label_h > 0:
                scale = min(label_w / w, label_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                if new_w > 0 and new_h > 0:
                    display_frame = cv2.resize(display_frame, (new_w, new_h))
                    self._frame_scale = scale
            
            # Convert to QImage with copy
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            rgb_frame = np.ascontiguousarray(rgb_frame)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            
            self._current_pixmap = QPixmap.fromImage(q_image)
            self.video_label.setPixmap(self._current_pixmap)
        except Exception as e:
            print(f"Frame update error: {e}")
    
    def _draw_detection_results(self, frame: np.ndarray) -> np.ndarray:
        """Draw YOLO detection bounding boxes and alerts based on detection type"""
        if not self._detection_result:
            return frame
        
        result = self._detection_result
        frame = frame.copy()
        
        if self.detection_type == DETECTION_TYPE_LINE_CROSS:
            if not self._detection_result: return frame
            
            tripwires = self.detector.tripwires if self.detector else {}
            alerts = self._detection_result.get('alerts', [])
            crossed_lines = {a['line'] for a in alerts}
            
            # Calculate total crossings
            total_crossings = sum(line.get('count', 0) for line in tripwires.values())
            
            # Check if any line was recently crossed (for alert flash)
            alert_active = False
            for name, line in tripwires.items():
                if name in crossed_lines:
                    alert_active = True
                    break
                if 'last_cross_time' in line and time.time() - line['last_cross_time'] < 1.0:
                    alert_active = True
                    break
            
            # 1. Draw Tripwires
            for name, line in tripwires.items():
                p1, p2 = line['p1'], line['p2']
                color = (0, 255, 255)  # Yellow default
                
                # Flash red if recently crossed
                if name in crossed_lines:
                    color = (0, 0, 255)
                elif 'last_cross_time' in line and time.time() - line['last_cross_time'] < 1.0:
                    color = (0, 0, 255)

                cv2.line(frame, p1, p2, color, 3)
                cv2.putText(frame, f"{name}: {line['count']}", (p1[0], p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 2. Draw Objects
            for obj in self._detection_result.get('detected_objects', []):
                b = obj['bbox']
                label = f"{obj['object_type']}"
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                cv2.putText(frame, label, (b[0], b[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 3. Draw HUD Alert
            hud_color = (0, 0, 255) if alert_active else (0, 255, 255)
            
            # Background for HUD
            cv2.rectangle(frame, (10, 10), (280, 100), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (280, 100), hud_color, 2)
            
            cv2.putText(frame, "LINE CROSS TRIPWIRE", (25, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Count display
            cv2.putText(frame, f"Total Crossings: {total_crossings}", (25, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2)
            
            cv2.putText(frame, f"Lines: {len(tripwires)}", (25, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            if alert_active:
                # Flash alert text
                cv2.putText(frame, "LINE CROSSED!", (300, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            return frame

        # =========================================================
        # CROWD DETECTION MODE
        # =========================================================
        if self.detection_type == DETECTION_TYPE_CROWD:
            # Draw all people (or all detected objects)
            img_h, img_w = frame.shape[:2]
            polygon_np = np.array(self.polygon_points, dtype=np.int32) if len(self.polygon_points) >= 3 else None
            
            for obj in result.get('detected_objects', []):
                b = obj['bbox']
                cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
                
                # Check if inside polygon
                is_inside = False
                if polygon_np is not None:
                    is_inside = cv2.pointPolygonTest(polygon_np, (cx, cy), False) >= 0
                
                # Color based on inside/outside
                if is_inside:
                    color = (0, 255, 255)  # Yellow for people inside
                    label = "Person"
                else:
                    color = (100, 100, 100)  # Gray for outside
                    label = ""
                
                # Draw box
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 2)
                if is_inside:
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # Draw HUD for Crowd
            alert_active = self.crowd_count >= CROWD_LIMIT
            hud_color = (0, 0, 255) if alert_active else (0, 255, 0)
            
            # Background for HUD
            cv2.rectangle(frame, (10, 10), (280, 100), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (280, 100), hud_color, 2)
            
            cv2.putText(frame, "CROWD DETECTION", (25, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Count display
            cv2.putText(frame, f"Count: {self.crowd_count} / {CROWD_LIMIT}", (25, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2)
            
            if alert_active:
                cv2.putText(frame, "CROWD LIMIT EXCEEDED!", (25, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
            return frame

        # =========================================================
        # INTRUSION DETECTION MODE
        # =========================================================
        if self.detection_type == DETECTION_TYPE_INTRUSION:
            # Draw polygon context
            polygon_np = np.array(self.polygon_points, dtype=np.int32) if len(self.polygon_points) >= 3 else None
            
            for obj in result.get('detected_objects', []):
                # Only care about people for intrusion
                if obj.get('object_type') != 'person':
                    continue
                    
                b = obj['bbox']
                cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
                
                # Check if inside polygon
                is_inside = False
                if polygon_np is not None:
                    is_inside = cv2.pointPolygonTest(polygon_np, (cx, cy), False) >= 0
                
                # Color based on inside (INTRUDER) / outside (SAFE)
                if is_inside:
                    color = (0, 0, 255)  # RED for INTRUDER
                    label = "INTRUDER!"
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 3)
                    cv2.line(frame, (b[0], b[1]), (b[2], b[3]), color, 2)
                    cv2.line(frame, (b[2], b[1]), (b[0], b[3]), color, 2)
                else:
                    color = (0, 255, 0)  # Green for safe
                    label = "Person"
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 1)

                # Draw label
                cv2.putText(frame, label, (b[0], b[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw HUD for Intrusion
            alert_active = self.intrusion_count > 0
            hud_color = (0, 0, 255) if alert_active else (0, 255, 0)
            
            # Background
            cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (300, 100), hud_color, 2)
            
            cv2.putText(frame, "INTRUSION DETECTION", (25, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if alert_active:
                cv2.putText(frame, f"INTRUDERS: {self.intrusion_count}", (25, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, "RESTRICTED AREA BREACH!", (25, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "ZONE SECURE", (25, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
            return frame

        # =========================================================
        # LOITERING DETECTION MODE
        # =========================================================
        if self.detection_type == DETECTION_TYPE_LOITERING:
            polygon_np = np.array(self.polygon_points, dtype=np.int32) if len(self.polygon_points) >= 3 else None
            
            loitering_count = result.get('loitering_count', 0)
            
            for obj in result.get('detected_objects', []):
                b = obj['bbox']
                cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
                is_inside = obj.get('is_inside', False)
                is_loitering = obj.get('is_loitering', False)
                time_in_zone = obj.get('time_in_zone', 0)
                track_id = obj.get('track_id', 0)
                
                if is_loitering:
                    # Red for loitering
                    color = (0, 0, 255)
                    label = f"LOITER {time_in_zone:.0f}s"
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 3)
                elif is_inside:
                    # Green for in zone but not loitering yet
                    color = (0, 255, 0)
                    label = f"ID:{track_id} ({time_in_zone:.0f}s)"
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 2)
                else:
                    # Gray for outside zone
                    color = (100, 100, 100)
                    label = f"ID:{track_id}"
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 1)
                
                cv2.putText(frame, label, (b[0], b[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, (cx, cy), 3, color, -1)
            
            # Draw HUD for Loitering
            alert_active = loitering_count > 0
            hud_color = (0, 0, 255) if alert_active else (0, 255, 0)
            
            # Background
            cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (300, 100), hud_color, 2)
            
            cv2.putText(frame, "LOITERING DETECTION", (25, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if alert_active:
                cv2.putText(frame, f"LOITERING: {loitering_count}", (25, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "ALERT! Person staying too long!", (25, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Flash alert
                cv2.putText(frame, "LOITERING DETECTED!", (320, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            else:
                cv2.putText(frame, "Zone Clear", (25, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Monitoring for loiterers...", (25, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
            return frame

        # =========================================================
        # ASSET PROTECTION MODE (Existing Logic)
        # =========================================================
        
        protected_objects = result.get('protected_objects', {})
        
        # First, draw all movement paths for protected objects
        for track_id, pobj in protected_objects.items():
            movement_path = pobj.get('movement_path', [])
            if len(movement_path) > 1:
                # Determine path color based on status
                if pobj.get('status') == 'left_zone':
                    path_color = (0, 0, 255)  # Red for alert
                else:
                    path_color = (0, 255, 255)  # Yellow for normal tracking
                
                # Draw the path using draw_movement_path if available
                if draw_movement_path is not None:
                    draw_movement_path(frame, movement_path, path_color)
                else:
                    # Inline path drawing
                    for i in range(1, len(movement_path)):
                        pt1 = tuple(map(int, movement_path[i-1]))
                        pt2 = tuple(map(int, movement_path[i]))
                        alpha = i / len(movement_path)
                        thickness = max(1, int(3 * alpha))
                        cv2.line(frame, pt1, pt2, path_color, thickness)
                    
                    # Draw direction arrow at the end
                    if len(movement_path) >= 2:
                        end_pt = tuple(map(int, movement_path[-1]))
                        prev_pt = tuple(map(int, movement_path[-2]))
                        cv2.arrowedLine(frame, prev_pt, end_pt, path_color, 2, tipLength=0.3)
        
        # Draw detected objects
        for obj in result.get('detected_objects', []):
            b = obj['bbox']
            track_id = obj['track_id']
            is_inside = obj.get('is_inside', False)
            object_type = obj.get('object_type', 'object')
            
            # Check protection status
            if track_id in protected_objects:
                pobj = protected_objects[track_id]
                if pobj['status'] == 'left_zone':
                    # RED: LEFT ZONE - ALERT!
                    color = (0, 0, 255)
                    label = f"ALERT! {object_type}"
                else:
                    # GREEN: Protected and inside zone
                    color = (0, 255, 0)
                    label = f"Protected: {object_type}"
            elif is_inside:
                # CYAN: Inside zone but not yet tracked
                color = (255, 200, 0)
                label = f"{object_type}"
            else:
                # GRAY: Outside zone
                color = (150, 150, 150)
                label = f"{object_type}"
            
            # Draw bounding box
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 2)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, 
                         (b[0], b[1] - label_size[1] - 10),
                         (b[0] + label_size[0] + 5, b[1]),
                         color, -1)
            cv2.putText(frame, label, (b[0] + 2, b[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw track ID
            cv2.putText(frame, f"ID:{track_id}", (b[0], b[3] + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw HUD with detection stats
        protected_count = len(protected_objects)
        left_count = sum(1 for p in protected_objects.values() 
                        if p.get('status') == 'left_zone')
        
        cv2.rectangle(frame, (10, 10), (220, 80), (0, 0, 0), -1)
        cv2.putText(frame, "MONITORING ACTIVE", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Protected: {protected_count} | Tracking", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if left_count > 0:
            cv2.putText(frame, f"ALERTS: {left_count}", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return frame
    
    def set_status(self, status: str):
        """Update status text"""
        if not self.detection_active and not self.polygon_complete:
            self.status_label.setText(status)
        elif self.detection_active:
            self.status_label.setText(f"🛡️ {status} | Zone Active")
        else:
            self.status_label.setText(f"{status} | Zone Ready")
    
    def get_polygon(self):
        """Get the defined polygon points"""
        return self.polygon_points if self.polygon_complete else None
    
    def is_detection_active(self):
        """Check if detection is active"""
        return self.detection_active


# ============================================================
# SETTINGS DIALOG
# ============================================================
class SettingsDialog(QDialog):
    """Settings dialog for configuration"""
    
    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Settings")
        self.setMinimumSize(400, 300)
        self.setStyleSheet(DARK_THEME)
        
        self._setup_ui()
        self._load_settings()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Form layout
        form = QFormLayout()
        
        # Decoding mode
        self.decode_combo = QComboBox()
        self.decode_combo.addItems(["Auto", "GPU", "CPU"])
        form.addRow("Decoding Mode:", self.decode_combo)
        
        # Confidence threshold
        self.confidence_spin = QSpinBox()
        self.confidence_spin.setRange(10, 100)
        self.confidence_spin.setValue(40)
        self.confidence_spin.setSuffix("%")
        form.addRow("Confidence:", self.confidence_spin)
        
        # Enable alerts
        self.alerts_check = QCheckBox("Enable API Alerts")
        self.alerts_check.setChecked(True)
        form.addRow("", self.alerts_check)
        
        # Reconnect settings
        self.reconnect_spin = QSpinBox()
        self.reconnect_spin.setRange(5, 120)
        self.reconnect_spin.setValue(30)
        self.reconnect_spin.setSuffix(" sec")
        form.addRow("Max Reconnect Delay:", self.reconnect_spin)
        
        layout.addLayout(form)
        layout.addStretch()
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        btn_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
    
    def _load_settings(self):
        """Load current settings into UI"""
        mode_map = {"auto": 0, "gpu": 1, "cpu": 2}
        self.decode_combo.setCurrentIndex(mode_map.get(self.settings.get("decoding_mode", "auto"), 0))
        self.confidence_spin.setValue(self.settings.get("confidence", 40))
        self.alerts_check.setChecked(self.settings.get("enable_alerts", True))
        self.reconnect_spin.setValue(self.settings.get("max_reconnect_delay", 30))
    
    def get_settings(self) -> dict:
        """Get settings from UI"""
        mode_map = {0: "auto", 1: "gpu", 2: "cpu"}
        return {
            "decoding_mode": mode_map[self.decode_combo.currentIndex()],
            "confidence": self.confidence_spin.value(),
            "enable_alerts": self.alerts_check.isChecked(),
            "max_reconnect_delay": self.reconnect_spin.value()
        }


# ============================================================
# MAIN WINDOW
# ============================================================
class AssetProtectionGUI(QMainWindow):
    """Main application window"""
    
    frame_updated = pyqtSignal(int, object)  # camera_id, frame
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Asset Protection System")
        self.setMinimumSize(1200, 700)
        
        # State
        self.streams = []
        self.camera_widgets = []
        self.rtsp_urls = ["rtsp://localhost:8554/mystream", "rtsp://localhost:8554/mystream", "rtsp://localhost:8554/mystream", "rtsp://localhost:8554/mystream"]  # 4 cameras max
        self.is_running = False
        self.update_timer = None
        
        # Settings (saved across sessions)
        self.settings = {
            "decoding_mode": "auto",  # auto, gpu, cpu
            "confidence": 40,
            "enable_alerts": True,
            "max_reconnect_delay": 30
        }
        
        # Popup state
        self.active_popup = None  # Currently open popup dialog
        self.active_popup_camera_id = None
        
        # Apply theme
        self.setStyleSheet(DARK_THEME)
        
        self._setup_ui()
        self._setup_timer()
        
        # Connect signal
        self.frame_updated.connect(self._on_frame_updated)
    
    def _setup_ui(self):
        """Setup the main UI layout"""
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # ========== LEFT SIDEBAR ==========
        sidebar = QFrame()
        sidebar.setFixedWidth(200)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border: 1px solid #404040;
                border-radius: 6px;
            }
        """)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(10)
        
        # Title
        title_label = QLabel("Asset Protection")
        title_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #ffffff; padding: 10px;")
        sidebar_layout.addWidget(title_label)
        
        # Camera URL inputs
        self.url_inputs = []
        for i in range(4):
            url_input = QLineEdit()
            url_input.setPlaceholderText(f"Camera {i+1} RTSP URL")
            url_input.textChanged.connect(lambda text, idx=i: self._on_url_changed(idx, text))
            self.url_inputs.append(url_input)
            sidebar_layout.addWidget(url_input)
        
        sidebar_layout.addSpacing(10)
        
        # Start/Stop buttons
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self._start_monitoring)
        sidebar_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("■ Stop")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_monitoring)
        sidebar_layout.addWidget(self.stop_btn)
        
        sidebar_layout.addStretch()
        
        # Settings button
        settings_btn = QPushButton("⚙ Settings")
        settings_btn.clicked.connect(self._open_settings)
        sidebar_layout.addWidget(settings_btn)
        
        main_layout.addWidget(sidebar)
        
        # ========== RIGHT: CAMERA GRID ==========
        camera_area = QFrame()
        camera_area.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: none;
            }
        """)
        camera_layout = QGridLayout(camera_area)
        camera_layout.setContentsMargins(5, 5, 5, 5)
        camera_layout.setSpacing(10)
        
        # Create 4 camera widgets in 2x2 grid
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for i, (row, col) in enumerate(positions):
            cam_widget = CameraWidget(i)
            cam_widget.clicked.connect(self._on_camera_clicked)  # Connect click
            self.camera_widgets.append(cam_widget)
            camera_layout.addWidget(cam_widget, row, col)
        
        main_layout.addWidget(camera_area, stretch=1)
    
    def _setup_timer(self):
        """Setup frame update timer"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_frames)
    
    def _on_url_changed(self, idx: int, text: str):
        """Handle URL input change"""
        self.rtsp_urls[idx] = text.strip()
    
    def _start_monitoring(self):
        """Start monitoring all cameras with URLs"""
        # Get valid URLs
        valid_urls = [(i, url) for i, url in enumerate(self.rtsp_urls) if url]
        
        if not valid_urls:
            QMessageBox.warning(self, "No Cameras", "Please enter at least one RTSP URL")
            return
        
        self.is_running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Create streams for each valid URL
        for i, url in valid_urls:
            # Get decoding mode from settings
            mode_str = self.settings.get("decoding_mode", "auto")
            if mode_str == "gpu":
                decode_mode = DecodingMode.GPU
            elif mode_str == "cpu":
                decode_mode = DecodingMode.CPU
            else:
                decode_mode = DecodingMode.AUTO
            
            max_delay = float(self.settings.get("max_reconnect_delay", 30))
            
            config = RTSPConfig(
                decoding_mode=decode_mode,
                buffer_size=5,
                infinite_reconnect=True,
                max_reconnect_delay=max_delay,
            )
            
            print(f"[CAM-{i+1}] Starting with decoding_mode={mode_str.upper()}")
            
            stream = RobustRTSPReader(url, config=config)
            stream.start()
            self.streams.append((i, stream))
            
            self.camera_widgets[i].show_waiting("Connecting...")
            self.camera_widgets[i].set_status("Connecting...", False)
        
        # Start update timer
        self.update_timer.start(33)  # ~30 FPS
    
    def _stop_monitoring(self):
        """Stop all camera streams"""
        self.is_running = False
        self.update_timer.stop()
        
        # Stop all streams
        for i, stream in self.streams:
            stream.stop()
            self.camera_widgets[i].show_waiting(f"cam_{i+1}")
            self.camera_widgets[i].set_status("Disconnected", False)
        
        self.streams = []
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def _update_frames(self):
        """Update frames from all streams"""
        for i, stream in self.streams:
            frame = stream.get_frame()
            stats = stream.get_stats()
            
            if frame is not None:
                self.camera_widgets[i].update_frame(frame)
                self.camera_widgets[i].set_status(
                    f"FPS: {stats.fps:.1f} | {stats.decoding_mode.upper()}", 
                    True
                )
                
                # Update popup if open for this camera
                if self.active_popup and self.active_popup_camera_id == i:
                    self.active_popup.update_frame(frame)
                    self.active_popup.set_status(
                        f"FPS: {stats.fps:.1f} | Decode: {stats.decoding_mode.upper()} | Click to analyze"
                    )
            else:
                self.camera_widgets[i].show_waiting(stats.current_state.value.upper())
                self.camera_widgets[i].set_status(
                    f"Reconnects: {stats.reconnect_count}", 
                    False
                )
    
    def _on_frame_updated(self, camera_id: int, frame):
        """Handle frame update signal"""
        if camera_id < len(self.camera_widgets):
            self.camera_widgets[camera_id].update_frame(frame)
    
    def _on_camera_clicked(self, camera_id: int):
        """Handle camera widget click - open enlarged popup"""
        # Close existing popup if any
        if self.active_popup:
            self._transfer_popup_state_to_widget()
            self.active_popup.close()
            self.active_popup = None
        
        # Get camera widget
        cam_widget = self.camera_widgets[camera_id]
        frame = cam_widget.get_current_frame()
        
        if frame is None:
            return
        
        # Create new popup
        self.active_popup = CameraPopupDialog(camera_id, f"Camera {camera_id + 1}", self)
        self.active_popup_camera_id = camera_id
        
        # Restore state from camera widget if detection was already active
        if cam_widget.detection_active and cam_widget.detector:
            # Block signals to prevent _on_detection_type_changed from being triggered
            self.active_popup.detection_type_combo.blockSignals(True)
            
            self.active_popup.detector = cam_widget.detector
            self.active_popup.detection_active = True
            
            # Restore detection type
            self.active_popup.detection_type = cam_widget.detection_type
            self.active_popup.detection_type_combo.setCurrentText(cam_widget.detection_type)
            self.active_popup.detection_type_combo.setEnabled(False)  # Locked while active
            
            # Re-enable signals
            self.active_popup.detection_type_combo.blockSignals(False)
            
            # For polygon-based modes, restore polygon state
            if cam_widget.detection_type != DETECTION_TYPE_LINE_CROSS:
                self.active_popup.polygon_points = list(cam_widget.polygon_points)
                self.active_popup.polygon_complete = True
                
                # Also ensure detector has polygon state set
                if hasattr(self.active_popup.detector, 'protection_polygon'):
                    self.active_popup.detector.protection_polygon = [tuple(p) for p in cam_widget.polygon_points]
                if hasattr(self.active_popup.detector, 'polygon_complete'):
                    self.active_popup.detector.polygon_complete = True
            
            # Ensure detector's internal detection is active
            if hasattr(self.active_popup.detector, 'detection_active'):
                self.active_popup.detector.detection_active = True
            
            self.active_popup._update_ui_state()
            print(f"✓ Restored detection state ({cam_widget.detection_type}) to popup for Camera {camera_id + 1}")
        
        # Update with current frame
        self.active_popup.update_frame(frame)
        
        # Handle popup close
        self.active_popup.finished.connect(self._on_popup_closed)
        
        # Show popup (non-modal so main window keeps updating)
        self.active_popup.show()
    
    def _transfer_popup_state_to_widget(self):
        """Transfer detection state from popup back to camera widget"""
        if self.active_popup and self.active_popup_camera_id is not None:
            popup = self.active_popup
            cam_widget = self.camera_widgets[self.active_popup_camera_id]
            
            print(f"📋 Transfer check: detection_active={popup.detection_active}, detector={popup.detector is not None}, type={popup.detection_type}")
            
            # Transfer detector and state
            if popup.detection_active and popup.detector:
                cam_widget.detector = popup.detector
                cam_widget.detection_active = True
                cam_widget.detection_type = popup.detection_type  # Transfer detection type
                
                # For polygon-based modes, transfer polygon points
                if popup.detection_type != DETECTION_TYPE_LINE_CROSS:
                    cam_widget.polygon_points = list(popup.polygon_points)
                # For Line Cross, tripwires are already in the detector - no extra action needed
                
                print(f"✓ Detection ({popup.detection_type}) transferred to Camera {self.active_popup_camera_id + 1} widget")
            elif popup.polygon_complete and not popup.detection_active:
                # Polygon drawn but detection not started yet - don't transfer
                print(f"📋 Polygon complete but detection not active - not transferring")
            else:
                print(f"📋 Nothing to transfer")
    
    def _on_popup_closed(self):
        """Handle popup close - transfer state to camera widget"""
        self._transfer_popup_state_to_widget()
        self.active_popup = None
        self.active_popup_camera_id = None
    
    def _open_settings(self):
        """Open settings dialog"""
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            # Save settings when user clicks Save
            self.settings = dialog.get_settings()
            print(f"Settings saved: Decoding={self.settings['decoding_mode'].upper()}")
            
            # Show confirmation
            QMessageBox.information(
                self, 
                "Settings Saved", 
                f"Decoding Mode: {self.settings['decoding_mode'].upper()}\n"
                f"Confidence: {self.settings['confidence']}%\n"
                f"Max Reconnect: {self.settings['max_reconnect_delay']}s\n\n"
                "Changes will apply on next Start."
            )
    
    def closeEvent(self, event):
        """Handle window close"""
        self._stop_monitoring()
        event.accept()


# ============================================================
# ENTRY POINT
# ============================================================
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = AssetProtectionGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


