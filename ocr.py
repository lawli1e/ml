import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import os
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore, storage
import json

class HelmetDetector:
    def __init__(self, detect_model_path='444.pt'):
        # Load settings before other initializations
        self.settings = self.load_settings()
        
        # Initialize counters
        self.daily_counts = self.load_daily_counts()
        
        # โหลดโมเดลสำหรับตรวจจับ None-helmet
        try:
            print("Loading detection model...")
            self.detect_model = YOLO(detect_model_path)
            print("Detection model loaded successfully")
        except Exception as e:
            print(f"Error loading detection model: {e}")
            raise
        
        # Initialize Firebase
        if not len(firebase_admin._apps):
            try:
                cred = credentials.Certificate("mdapp-d52ba-firebase-adminsdk-npmf6-4f0614c6a1.json")
                firebase_admin.initialize_app(cred, {
                    'storageBucket': 'mdapp-d52ba.appspot.com'
                })
            except Exception as e:
                print(f"Error initializing Firebase: {e}")
        
        self.db = firestore.client()
        self.bucket = storage.bucket()

    def load_daily_counts(self):
        """Load daily processing counts from file."""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            if os.path.exists('daily_counts.json'):
                with open('daily_counts.json', 'r') as f:
                    counts = json.load(f)
                    if today not in counts:
                        counts[today] = {
                            'total_processed': 0,
                            'violations_detected': 0
                        }
            else:
                counts = {
                    today: {
                        'total_processed': 0,
                        'violations_detected': 0
                    }
                }
            return counts
        except Exception as e:
            print(f"Error loading daily counts: {e}")
            today = datetime.now().strftime('%Y-%m-%d')
            return {
                today: {
                    'total_processed': 0,
                    'violations_detected': 0
                }
            }

    def save_daily_counts(self):
        """Save daily processing counts to file."""
        try:
            with open('daily_counts.json', 'w') as f:
                json.dump(self.daily_counts, f, indent=4)
            print("Daily counts saved successfully")
        except Exception as e:
            print(f"Error saving daily counts: {e}")

    def update_counts(self, violation_detected=False):
        """Update processing counts."""
        today = datetime.now().strftime('%Y-%m-%d')
        if today not in self.daily_counts:
            self.daily_counts[today] = {
                'total_processed': 0,
                'violations_detected': 0
            }
        
        self.daily_counts[today]['total_processed'] += 1
        if violation_detected:
            self.daily_counts[today]['violations_detected'] += 1
        
        self.save_daily_counts()
        self.send_counts_to_firebase(today)

    def send_counts_to_firebase(self, date):
        """Send daily counts to Firebase."""
        try:
            if not self.db:
                print("Firestore client not initialized")
                return False

            doc_ref = self.db.collection('daily_counts').document(date)
            doc_data = {
                'date': date,
                'total_processed': self.daily_counts[date]['total_processed'],
                'violations_detected': self.daily_counts[date]['violations_detected'],
                'address': self.settings.get('address', ''),
                'latitude': self.settings.get('latitude', ''),
                'longitude': self.settings.get('longitude', ''),
                'uid': self.settings.get('uid', ''),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            doc_ref.set(doc_data, merge=True)
            print(f"Successfully sent counts to Firestore for date: {date}")
            return True
        except Exception as e:
            print(f"Error sending counts to Firestore: {e}")
            return False

    def load_settings(self):
        """Load settings from settings.json file."""
        try:
            if os.path.exists('settings.json'):
                with open('settings.json', 'r') as f:
                    settings = json.load(f)
                    print("Successfully loaded settings:", settings)
                    return settings
            else:
                default_settings = {
                    "address": "",
                    "latitude": "",
                    "longitude": "",
                    "uid": "",
                }
                print("Settings file not found, using defaults:", default_settings)
                return default_settings
        except Exception as e:
            print(f"Error loading settings: {e}")
            default_settings = {
                "video_name": "",
                "address": "",
                "latitude": "",
                "longitude": "",
                "uid": "",
            }
            print("Error loading settings, using defaults:", default_settings)
            return default_settings

    def upload_image(self, image_path):
        """Upload image to Firebase Storage."""
        try:
            if not self.bucket:
                print("Firebase Storage bucket not initialized")
                return None
            
            blob = self.bucket.blob(f"test/{os.path.basename(image_path)}")
            blob.upload_from_filename(image_path)
            blob.make_public()
            print(f"Image uploaded: {blob.public_url}")
            return blob.public_url
        except Exception as e:
            print(f"Error uploading image to Firebase Storage: {e}")
            return None

    def send_to_firestore(self, image_url):
        """Send detection data to Firestore."""
        try:
            if not self.db:
                print("Firestore client not initialized")
                return False

            doc_ref = self.db.collection('test').document()
            doc_data = {
                'original_image_url': image_url,
                'address': self.settings.get('address', ''),
                'latitude': self.settings.get('latitude', ''),
                'longitude': self.settings.get('longitude', ''),
                'uid': self.settings.get('uid', ''),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            doc_ref.set(doc_data)
            print(f"Successfully sent to Firestore")
            return True
        except Exception as e:
            print(f"Error sending to Firestore: {e}")
            return False

    def detect_no_helmet(self, image):
        """ตรวจจับ None-helmet จากภาพ"""
        try:
            results = self.detect_model.predict(image)
            boxes_data = results[0].boxes.data
            px = pd.DataFrame(boxes_data).astype("float")
            
            for index, row in px.iterrows():
                confidence = float(row[4])
                class_id = int(row[5])
                
                if confidence > 0.5 and class_id == 0:  # class None-helmet
                    print("Found person without helmet")
                    return True
            return False
            
        except Exception as e:
            print(f"Error in detect_no_helmet: {e}")
            return False

    def process_directory(self, input_folder):
        """Process all images in a directory."""
        try:
            if not os.path.exists(input_folder):
                print(f"Input folder does not exist: {input_folder}")
                return

            image_files = [f for f in os.listdir(input_folder) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                print("No image files found in the input folder")
                return

            for image_file in image_files:
                try:
                    image_path = os.path.join(input_folder, image_file)
                    if not os.path.exists(image_path):
                        continue

                    print(f"Processing image: {image_file}")
                    
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    # ตรวจจับคนที่ไม่สวมหมวกกันน็อค
                    violation_detected = self.detect_no_helmet(image)
                    if violation_detected:
                        # อัพโหลดภาพต้นฉบับไป Firebase
                        image_url = self.upload_image(image_path)
                        
                        if image_url:
                            # ส่งข้อมูลไป Firestore
                            self.send_to_firestore(image_url)
                    
                    # Update counts regardless of detection result
                    self.update_counts(violation_detected)

                    if os.path.exists(image_path):
                        os.remove(image_path)

                except Exception as e:
                    print(f"Error processing image {image_file}: {e}")
                    continue

        except Exception as e:
            print(f"Error in process_directory: {e}")

if __name__ == "__main__":
    try:
        detector = HelmetDetector()
        input_folder = 'D:\\Downloads\\carnumberplate-main\\test'
        detector.process_directory(input_folder)
    except Exception as e:
        print(f"Main execution error: {e}")