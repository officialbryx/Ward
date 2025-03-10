import cv2
import time
import os
import numpy as np
from collections import deque
from threading import Thread
from queue import Queue, Empty
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("camera_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedStreamHandler:
    def __init__(
        self, 
        rtsp_url="rtsp://buth:4ytkfe@192.168.1.210/live/ch00_1", 
        resolution=(1280, 720),
        fps=20,
        buffer_seconds=20,
        visibility_threshold=80,
        recovery_threshold=100,
        highlight_gap=10,
        post_record_seconds=20,
        save_dir="recordings/"
    ):
        # Stream settings
        self.rtsp_url = rtsp_url
        self.width, self.height = resolution
        self.fps = fps
        
        # Visibility settings
        self.visibility_threshold = visibility_threshold
        self.recovery_threshold = recovery_threshold
        self.min_highlight_gap = highlight_gap
        self.post_record_duration = post_record_seconds
        
        # Recording settings
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize stream components
        self.buffer_size = buffer_seconds * fps
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.frame_queue = Queue(maxsize=30)  # Buffer for threaded reading
        
        # Face recognition components
        self.detector = FaceDetector(min_face_size=40, temporal_smoothing=True)
        self.recognizer = FaceRecognizer(database_path="known_faces")
        self.detector.face_recognizer = self.recognizer
        
        # Recognition settings
        self.detection_interval = 3  # Process every 3rd frame
        self.processing_frame_count = 0
        self.last_faces = []
        
        # Recording state
        self.highlight_triggered = False
        self.highlight_writer = None
        self.last_highlight_time = 0
        self.post_record_frames = 0
        self.session_writer = None
        
        # Runtime stats
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.processing_time = 0
        self.running = False
        self.display_frame = None
        
        # Initialize stream thread
        self.stream_thread = None
        self.processing_thread = None
    
    def start(self):
        """Start all threads and initialize recording"""
        # Enable hardware acceleration
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'hw_decoders_list=h264_videotoolbox'
        
        # Initialize session recording
        session_filename = os.path.join(self.save_dir, f"session_{int(time.time())}.mp4")
        self.session_writer = cv2.VideoWriter(
            session_filename, 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            self.fps, 
            (self.width, self.height)
        )
        
        # Start threads
        self.running = True
        self.stream_thread = Thread(target=self._stream_worker, daemon=True)
        self.processing_thread = Thread(target=self._processing_worker, daemon=True)
        self.stream_thread.start()
        self.processing_thread.start()
        
        logger.info(f"Started camera system with resolution {self.width}x{self.height}")
    
    def stop(self):
        """Stop all threads and clean up resources"""
        self.running = False
        
        # Wait for threads to finish
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=2.0)
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # Close video writers
        if self.session_writer:
            self.session_writer.release()
        if self.highlight_writer:
            self.highlight_writer.release()
            
        logger.info("Camera system stopped")
    
    def _stream_worker(self):
        """Thread for reading frames from the camera"""
        # Open camera stream
        cap = cv2.VideoCapture(self.rtsp_url)
        
        # Set optimal camera properties
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if not cap.isOpened():
            logger.error("Failed to open RTSP stream")
            self.running = False
            return
            
        logger.info("Stream opened successfully")
        reconnect_delay = 1.0
        reconnect_attempts = 0
        
        while self.running:
            ret, frame = cap.read()
            
            if not ret:
                reconnect_attempts += 1
                logger.warning(f"Frame retrieval failed. Reconnect attempt {reconnect_attempts}")
                
                # Increase delay with each failed attempt, max 5 seconds
                reconnect_delay = min(5.0, reconnect_delay * 1.5)
                time.sleep(reconnect_delay)
                
                # Try to reopen the camera
                cap.release()
                cap = cv2.VideoCapture(self.rtsp_url)
                continue
                
            # Reset reconnection parameters on successful read
            reconnect_delay = 1.0
            reconnect_attempts = 0
            
            # Add to queue, drop frames if queue is full (better to have latest frames)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    pass
            self.frame_queue.put(frame)
            
            # Throttle to prevent excessive CPU usage
            time.sleep(0.001)
            
        # Clean up
        cap.release()
    
    def _processing_worker(self):
        """Thread for processing frames"""
        frame_count = 0
        
        while self.running:
            try:
                # Get frame from queue with timeout
                frame = self.frame_queue.get(timeout=1.0)
                frame_count += 1
                self.fps_counter += 1
                
                # Track FPS
                if time.time() - self.fps_start_time >= 5.0:
                    self.current_fps = self.fps_counter / 5.0
                    logger.info(f"FPS: {self.current_fps:.1f}, Processing time: {self.processing_time*1000:.1f}ms")
                    self.fps_counter = 0
                    self.fps_start_time = time.time()
                
                # Store in frame buffer for highlights
                self.frame_buffer.append(frame)
                
                # Continuous recording
                if self.session_writer:
                    self.session_writer.write(frame)
                
                # Analyze visibility for highlights
                brightness = self._analyze_visibility(frame)
                
                # Process highlights
                self._handle_highlight_recording(frame, brightness)
                
                # Face detection and recognition (on subset of frames)
                if frame_count % self.detection_interval == 0:
                    self._process_face_detection(frame)
                
                # Prepare display frame
                self._prepare_display_frame(frame)
                
            except Empty:
                # No frames available
                pass
            except Exception as e:
                logger.error(f"Error in processing: {str(e)}", exc_info=True)
    
    def _analyze_visibility(self, frame):
        """Analyze frame brightness."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def _handle_highlight_recording(self, frame, brightness):
        """Handle highlight recording based on visibility changes"""
        current_time = time.time()
        
        # Highlight Trigger - when visibility drops
        if brightness < self.visibility_threshold and not self.highlight_triggered:
            if current_time - self.last_highlight_time > self.min_highlight_gap:
                logger.info(f"Visibility dropped to {brightness:.1f}! Creating highlight...")
                
                # Create highlight writer
                highlight_filename = os.path.join(
                    self.save_dir, 
                    f"highlight_{int(current_time)}.mp4"
                )
                self.highlight_writer = cv2.VideoWriter(
                    highlight_filename, 
                    cv2.VideoWriter_fourcc(*'mp4v'), 
                    self.fps, 
                    (self.width, self.height)
                )
                
                # Save past frames
                for past_frame in self.frame_buffer:
                    if self.highlight_writer:
                        self.highlight_writer.write(past_frame)
                
                self.highlight_triggered = True
                self.last_highlight_time = current_time
                self.post_record_frames = self.post_record_duration * self.fps
        
        # Continue recording highlight if active
        if self.highlight_triggered:
            if self.highlight_writer:
                self.highlight_writer.write(frame)
            self.post_record_frames -= 1
        
        # End highlight recording if brightness restored and post-record complete
        if (self.highlight_triggered and 
            brightness > self.recovery_threshold and 
            self.post_record_frames <= 0):
            
            logger.info(f"Visibility restored to {brightness:.1f}. Ending highlight.")
            if self.highlight_writer:
                self.highlight_writer.release()
                self.highlight_writer = None
            self.highlight_triggered = False
    
    def _process_face_detection(self, frame):
        """Run face detection and recognition"""
        start_time = time.time()
        
        # Detect faces
        faces = self.detector.detect_faces(frame)
        self.last_faces = faces
        
        # Save faces if recognized
        if faces:
            for face in faces:
                if hasattr(face, 'name') and face.name:
                    # Save recognized face with timestamp
                    bbox = face.bbox.astype(int)
                    if all(b >= 0 for b in bbox) and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                        face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        if face_img.size > 0:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = os.path.join(
                                self.save_dir, 
                                f"face_{face.name}_{timestamp}.jpg"
                            )
                            cv2.imwrite(filename, face_img)
        
        self.processing_time = time.time() - start_time
    
    def _prepare_display_frame(self, frame):
        """Prepare frame for display with overlays"""
        display = frame.copy()
        
        # Draw detected faces
        for face in self.last_faces:
            bbox = face.bbox.astype(int)
            
            # Use different colors for recognized vs unknown
            if hasattr(face, 'name') and face.name:
                color = (0, 255, 0)  # Green for recognized
                name_text = f"ID: {face.name}"
                cv2.putText(display, name_text,
                          (bbox[0], bbox[3] + 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                          color, 2)
            else:
                color = (0, 165, 255)  # Orange for unknown
            
            # Draw face rectangle
            cv2.rectangle(display, (bbox[0], bbox[1]), 
                        (bbox[2], bbox[3]), color, 2)
            
            # Add tracking ID if available
            if hasattr(face, 'track_id'):
                track_text = f"#{face.track_id}"
                cv2.putText(display, track_text,
                          (bbox[2] - 40, bbox[1] - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                          (255, 255, 0), 1)
        
        # Add FPS and status info
        cv2.putText(display, f"FPS: {self.current_fps:.1f}", 
                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if self.highlight_triggered:
            cv2.putText(display, "RECORDING HIGHLIGHT", 
                      (self.width - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.7, (0, 0, 255), 2)
        
        self.display_frame = display
    
    def get_display_frame(self):
        """Get the latest display frame with overlays"""
        return self.display_frame


def main():
    """Main function to run the camera system"""
    # Initialize the camera system
    camera_system = AdvancedStreamHandler(
        rtsp_url="rtsp://buth:4ytkfe@192.168.1.210/live/ch00_1",
        resolution=(1280, 720),
        fps=20,
        visibility_threshold=80,
        recovery_threshold=100
    )
    
    # Start the system
    camera_system.start()
    
    try:
        # Main display loop
        while True:
            # Get the latest processed frame
            frame = camera_system.get_display_frame()
            
            # Display the frame
            if frame is not None:
                cv2.imshow("Camera System", frame)
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping due to keyboard interrupt...")
        
    finally:
        # Clean up
        camera_system.stop()
        cv2.destroyAllWindows()
        print("System shutdown complete.")


if __name__ == "__main__":
    main()