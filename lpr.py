import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import os
import time
from datetime import datetime
import cvzone
from scipy.spatial.distance import cdist, cosine

class LicensePlateDetector:
    def __init__(self, model_path='444.pt', videos_dir='D:\Downloads\carnumberplate-main\downloaded_videos'):
        self.model = YOLO(model_path)
        self.videos_dir = videos_dir
        self.output_dir = r'D:\Downloads\carnumberplate-main\test'
        
        # Load class list
        with open("444.txt", "r") as my_file:
            self.class_list = my_file.read().split("\n")
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize variables
        self.count = 0
        self.save_count = 0
        self.tracked_riders = {}
        self.lost_riders = {}
        self.rider_features = {}
        self.tracking_confidence = {}
        
        # Tracking parameters
        self.min_confidence = 0.45
        self.iou_threshold = 0.3
        self.tracking_threshold = 200
        self.min_detection_frames = 2
        self.max_lost_time = 3.0
        self.appearance_threshold = 0.6
        self.cleanup_interval = 10.0
        self.last_cleanup_time = time.time()
        

    def setup_window(self):
        def RGB(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                print([x, y])
        
        cv2.namedWindow('RGB')
        cv2.setMouseCallback('RGB', RGB)

    def extract_features(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        
        crop = cv2.resize(crop, (64, 128))
        
        color_hist = cv2.calcHist([crop], [0, 1, 2], None, [8, 8, 8], 
                                [0, 256, 0, 256, 0, 256])
        color_hist = cv2.normalize(color_hist, color_hist).flatten()
        
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_hist = cv2.calcHist([edges], [0], None, [32], [0, 256])
        edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()
        
        return np.concatenate([color_hist, edge_hist])

    def compare_features(self, feat1, feat2):
        if feat1 is None or feat2 is None:
            return 0
        return 1 - cosine(feat1, feat2)

    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0

    def non_max_suppression(self, boxes, scores):
        indices = []
        if not boxes.size:
            return indices

        boxes = np.array(boxes)
        scores = np.array(scores)
        indices = scores.argsort()[::-1]
        keep = []

        while indices.size > 0:
            current = indices[0]
            keep.append(current)
            ious = np.array([self.calculate_iou(boxes[current], boxes[i]) 
                           for i in indices[1:]])
            indices = indices[1:][ious < self.iou_threshold]

        return keep
    
    def handle_lost_rider(self, rider_id, current_time):
        if rider_id in self.tracked_riders:
            lost_time = current_time - self.tracked_riders[rider_id]['last_seen']
            
            if lost_time > self.max_lost_time:
                if rider_id not in self.lost_riders:
                    self.lost_riders[rider_id] = self.tracked_riders[rider_id].copy()
                del self.tracked_riders[rider_id]
            else:
                self.tracking_confidence[rider_id] = max(0.2,
                    1.0 - (lost_time / self.max_lost_time))

    def find_best_match(self, frame, bbox, current_time):
        features = self.extract_features(frame, bbox)
        x1, y1, x2, y2 = bbox
        center = np.array([(x1 + x2)/2, (y1 + y2)/2])
        current_area = (x2 - x1) * (y2 - y1)
        
        best_match = None
        best_score = -float('inf')
        
        all_riders = {**self.lost_riders, **self.tracked_riders}
        
        for rid, rinfo in all_riders.items():
            if current_time - rinfo['last_seen'] > self.max_lost_time * 2:
                continue
            
            old_box = rinfo['bbox']
            old_center = np.array([(old_box[0] + old_box[2])/2,
                                 (old_box[1] + old_box[3])/2])
            
            distance = np.linalg.norm(center - old_center)
            
            old_area = (old_box[2] - old_box[0]) * (old_box[3] - old_box[1])
            size_diff = abs(current_area - old_area) / max(current_area, old_area)
            
            appearance_sim = self.compare_features(features, rinfo['features'])
            
            distance_score = max(0, 1 - distance/self.tracking_threshold)
            size_score = max(0, 1 - size_diff)
            
            total_score = (0.4 * distance_score + 
                          0.4 * appearance_sim +
                          0.2 * size_score)
            
            if (total_score > best_score and 
                total_score > 0.5 and
                distance_score > 0.3):
                
                best_score = total_score
                best_match = (rid, total_score)
        
        return best_match

    def cleanup_tracked_riders(self, current_time):
        if current_time - self.last_cleanup_time >= self.cleanup_interval:
            to_remove = []
            for rid, rinfo in self.lost_riders.items():
                if current_time - rinfo['last_seen'] > self.max_lost_time * 3:
                    to_remove.append(rid)
            
            for rid in to_remove:
                del self.lost_riders[rid]
            
            self.last_cleanup_time = current_time
            
    def calculate_image_quality(self, frame, bbox):
        """คำนวณคุณภาพของภาพ rider"""
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0

        # คำนวณขนาดเทียบกับเฟรม
        frame_area = frame.shape[0] * frame.shape[1]
        crop_area = (x2 - x1) * (y2 - y1)
        relative_size = crop_area / frame_area

        # ถ้าขนาดเล็กเกินไป ให้คะแนน 0
        if relative_size < 0.03:  # ต้องมีขนาดอย่างน้อย 3% ของเฟรม
                return 0.0
            
            # 1. ตรวจสอบความชัด
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 2. คำนวณขนาด (ให้น้ำหนักมากขึ้น)
        size_score = min(relative_size * 10, 1.0)  # ขนาด 10% ของเฟรมจะได้คะแนนเต็ม
            
            # 3. ตรวจสอบความสว่าง
        brightness = np.mean(gray)
        brightness_score = 1.0 - abs((brightness - 128) / 128)
            
            # 4. ตรวจสอบ contrast
        contrast = np.std(gray)
        contrast_score = min(contrast / 64, 1.0)
            
            # รวมคะแนน (เน้นความชัดและขนาด)
        quality_score = (
                0.35 * min(laplacian_var / 1000, 1.0) +  # ความชัด
                0.35 * size_score +                       # ขนาด
                0.15 * contrast_score +                   # contrast
                0.15 * brightness_score                   # ความสว่าง
            )

            # Debug info
        debug_info = {
                'size_ratio': relative_size * 100,
                'sharpness': min(laplacian_var / 1000, 1.0),
                'size_score': size_score,
                'brightness': brightness_score,
                'contrast': contrast_score,
                'total': quality_score
            }
        print(f"Quality scores for rider: {debug_info}")
            
        return quality_score

    def should_save_detection(self, rider_id):
        """Determine if we should save this detection"""
        if rider_id not in self.tracked_riders:
            return False

        rider = self.tracked_riders[rider_id]
        
        if not rider['is_captured'] and rider['detection_count'] >= self.min_detection_frames:
            best_bbox = rider['best_bbox']
            frame_area = rider['best_frame'].shape[0] * rider['best_frame'].shape[1]
            crop_area = (best_bbox[2] - best_bbox[0]) * (best_bbox[3] - best_bbox[1])
            size_ratio = crop_area / frame_area

            # เพิ่มเงื่อนไขการบันทึก
            quality_ok = rider['best_quality'] > 0.65  # เพิ่มเกณฑ์คุณภาพ
            size_ok = size_ratio > 0.03  # ต้องมีขนาดอย่างน้อย 3% ของเฟรม
            
            if quality_ok and size_ok:
                return True
        return False
    
    def update_tracking(self, rider_id, bbox, current_time, frame):
        """Update tracking information for a rider"""
        features = self.extract_features(frame, bbox)
        quality_score = self.calculate_image_quality(frame, bbox)
        
        # คำนวณขนาด
        frame_area = frame.shape[0] * frame.shape[1]
        crop_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        size_ratio = crop_area / frame_area

        if rider_id not in self.tracked_riders:
            self.tracked_riders[rider_id] = {
                'first_seen': current_time,
                'last_seen': current_time,
                'bbox': bbox,
                'detection_count': 1,
                'is_captured': False,
                'features': features,
                'confidence': 1.0,
                'best_frame': frame.copy(),
                'best_bbox': bbox,
                'best_quality': quality_score,
                'best_size_ratio': size_ratio
            }
            self.tracking_confidence[rider_id] = 1.0
        else:
            current_best_size = self.tracked_riders[rider_id]['best_size_ratio']
            
            # อัพเดตถ้าคุณภาพดีกว่าหรือขนาดใหญ่กว่า
            if quality_score > self.tracked_riders[rider_id]['best_quality'] or \
               (size_ratio > current_best_size * 1.2):  # ขนาดใหญ่กว่าเดิม 20%
                self.tracked_riders[rider_id].update({
                    'best_frame': frame.copy(),
                    'best_bbox': bbox,
                    'best_quality': quality_score,
                    'best_size_ratio': size_ratio
                })
            
            # อัพเดต tracking info
            if features is not None:
                old_features = self.tracked_riders[rider_id]['features']
                if old_features is not None:
                    weight = 0.7
                    self.tracked_riders[rider_id]['features'] = \
                        weight * old_features + (1 - weight) * features
            
            self.tracked_riders[rider_id].update({
                'last_seen': current_time,
                'bbox': bbox,
                'detection_count': self.tracked_riders[rider_id]['detection_count'] + 1
            })

    def process_frame(self, frame, video_name):
        # Resize frame
        height, width = frame.shape[:2]
        max_width, max_height = 1600, 900
        ratio = min(max_width/width, max_height/height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
        frame = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)
        
        current_time = time.time()
        self.cleanup_tracked_riders(current_time)
        
        # Get predictions
        results = self.model.predict(frame)
        boxes_data = results[0].boxes.data
        
        # แยกการตรวจจับ rider และ license plate
        rider_detections = []
        plate_detections = []
        rider_scores = []
        
        for row in boxes_data:
            if int(row[5]) < len(self.class_list):
                c = self.class_list[int(row[5])]
                if c.lower() == 'rider' and row[4] > self.min_confidence:
                    rider_detections.append([int(row[0]), int(row[1]), 
                                          int(row[2]), int(row[3])])
                    rider_scores.append(row[4])
                elif c.lower() == 'platenumber' and row[4] > self.min_confidence:
                    plate_detections.append([int(row[0]), int(row[1]), 
                                           int(row[2]), int(row[3])])

        if rider_detections:
            keep_indices = self.non_max_suppression(np.array(rider_detections), 
                                                  np.array(rider_scores))
            
            active_riders = list(self.tracked_riders.keys())
            for rid in active_riders:
                self.handle_lost_rider(rid, current_time)
            
            used_ids = set()
            
            for idx in keep_indices:
                bbox = rider_detections[idx]
                x1, y1, x2, y2 = bbox
                
                # ตรวจสอบว่ามีป้ายทะเบียนอยู่ในบริเวณ rider หรือไม่
                has_plate = False
                plate_visible = False
                best_plate_score = 0
                best_plate_bbox = None
                
                for plate_bbox in plate_detections:
                    px1, py1, px2, py2 = plate_bbox
                    
                    # ตรวจสอบว่าป้ายทะเบียนอยู่ในบริเวณ rider หรือใกล้เคียง
                    # ขยายพื้นที่การตรวจสอบ
                    extended_x1 = max(0, x1 - int((x2 - x1) * 0.2))
                    extended_x2 = min(frame.shape[1], x2 + int((x2 - x1) * 0.2))
                    extended_y1 = max(0, y1 - int((y2 - y1) * 0.2))
                    extended_y2 = min(frame.shape[0], y2 + int((y2 - y1) * 0.2))
                    
                    if (px1 >= extended_x1 and px2 <= extended_x2 and 
                        py1 >= extended_y1 and py2 <= extended_y2):
                        has_plate = True
                        
                        # คำนวณขนาดของป้ายทะเบียน
                        plate_area = (px2 - px1) * (py2 - py1)
                        frame_area = frame.shape[0] * frame.shape[1]
                        plate_ratio = plate_area / frame_area
                        
                        # ตรวจสอบความชัดของป้ายทะเบียน
                        plate_crop = frame[py1:py2, px1:px2]
                        if plate_crop.size > 0:
                            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                            plate_score = laplacian_var * plate_ratio
                            
                            if (laplacian_var > 500 and  # ป้ายทะเบียนต้องชัด
                                plate_ratio > 0.005):    # และมีขนาดใหญ่พอ
                                plate_visible = True
                                if plate_score > best_plate_score:
                                    best_plate_score = plate_score
                                    best_plate_bbox = plate_bbox

                if not (has_plate and plate_visible):
                    continue  # ข้ามถ้าไม่เห็นป้ายทะเบียนชัด
                
                match_result = self.find_best_match(frame, bbox, current_time)
                
                if match_result is not None:
                    matched_id, match_score = match_result
                    
                    if matched_id in used_ids:
                        matched_id = max(list(self.tracked_riders.keys()) + 
                                       list(self.lost_riders.keys()) + [-1]) + 1
                        self.tracking_confidence[matched_id] = 1.0
                    else:
                        if matched_id in self.lost_riders:
                            self.tracked_riders[matched_id] = self.lost_riders[matched_id]
                            del self.lost_riders[matched_id]
                        self.tracking_confidence[matched_id] = match_score
                else:
                    matched_id = max(list(self.tracked_riders.keys()) + 
                                   list(self.lost_riders.keys()) + [-1]) + 1
                    self.tracking_confidence[matched_id] = 1.0
                
                used_ids.add(matched_id)
                
                # อัปเดต tracking และเก็บข้อมูลป้ายทะเบียน
                self.update_tracking(matched_id, bbox, current_time, frame)
                if best_plate_bbox is not None:
                    self.tracked_riders[matched_id]['plate_bbox'] = best_plate_bbox
                    self.tracked_riders[matched_id]['plate_score'] = best_plate_score
                
                # วาดกรอบ rider และป้ายทะเบียน
                conf = self.tracking_confidence.get(matched_id, 1.0)
                color = (0, int(255 * conf), int(255 * (1-conf)))
                
                if self.should_save_detection(matched_id):
                    best_frame = self.tracked_riders[matched_id]['best_frame']
                    best_bbox = self.tracked_riders[matched_id]['best_bbox']
                    x1, y1, x2, y2 = map(int, best_bbox)
                    
                    width = x2 - x1
                    height = y2 - y1
                    x1_expanded = max(0, int(x1 - width * 0.1))
                    y1_expanded = max(0, int(y1 - height * 0.1))
                    x2_expanded = min(best_frame.shape[1], int(x2 + width * 0.1))
                    y2_expanded = min(best_frame.shape[0], int(y2 + height * 0.1))
                    
                    cropped = best_frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"rider_id{matched_id}_with_plate_q{self.tracked_riders[matched_id]['best_quality']:.2f}_ps{best_plate_score:.0f}_{timestamp}.jpg"
                    save_path = os.path.join(self.output_dir, filename)
                    
                    cv2.imwrite(save_path, cropped)
                    print(f"Saved rider {matched_id} with visible plate (quality: {self.tracked_riders[matched_id]['best_quality']:.2f}, plate score: {best_plate_score:.0f}): {save_path}")
                    
                    self.tracked_riders[matched_id]['is_captured'] = True
                    self.save_count += 1
                
                # วาด bounding boxes
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                status = f"{'Captured' if self.tracked_riders[matched_id]['is_captured'] else 'Tracking'} ({conf:.2f})"
                cv2.putText(frame, f"Rider #{matched_id} with plate {status}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # วาดกรอบป้ายทะเบียน
                if best_plate_bbox is not None:
                    px1, py1, px2, py2 = map(int, best_plate_bbox)
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Plate Score: {best_plate_score:.0f}", (px1, py1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame

    def process_video(self, video_path):
        video_name = os.path.basename(video_path)
        print(f"\nProcessing video: {video_name}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return
        
        self.count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.count += 1
                if self.count % 2 != 0:
                    continue
                
                processed_frame = self.process_frame(frame, video_name)
                cv2.imshow("RGB", processed_frame)
                
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                
        except Exception as e:
            print(f"Error processing video {video_name}: {e}")
        finally:
            cap.release()

    def run(self):
        try:
            self.setup_window()
            
            video_files = [f for f in os.listdir(self.videos_dir) 
                         if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            
            if not video_files:
                print(f"No video files found in {self.videos_dir}")
                return
            
            print(f"Found {len(video_files)} videos to process")
            
            for video_file in video_files:
                video_path = os.path.join(self.videos_dir, video_file)
                self.process_video(video_path)
                
        except Exception as e:
            print(f"Detection error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        self.tracked_riders.clear()
        self.lost_riders.clear()
        self.rider_features.clear()
        self.tracking_confidence.clear()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = LicensePlateDetector()
    detector.run()