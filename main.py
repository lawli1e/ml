import time
import os
import json
import firebase_admin
from firebase_admin import credentials
from lpr import LicensePlateDetector
from ocr import TextRecognizer  # Make sure this is correctly imported
from firebase_admin import firestore
from firebase_admin import storage
import cv2

class LicensePlateSystem:
    def __init__(self):
        self.initialize_firebase()
        self.db = firestore.client()
        self.lpr = LicensePlateDetector()  # License plate recognition
        self.recognizer = TextRecognizer()  # Text recognizer initialization
        self.bucket = storage.bucket('mdapp-d52ba.appspot.com')
        self.settings_path = os.path.join(os.path.dirname(__file__), 'settings.json')
        self.downloaded_paths_path = os.path.join(os.path.dirname(__file__), 'downloaded_paths.json')
        print(f"Settings file will be saved at: {self.settings_path}")
        print(f"Downloaded paths file will be saved at: {self.downloaded_paths_path}")

    def initialize_firebase(self):
        cred = credentials.Certificate('mdapp-d52ba-firebase-adminsdk-npmf6-4f0614c6a1.json')
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'mdapp-d52ba.appspot.com'
        })
        print("Firebase connected successfully.")
        
        

    def load_downloaded_paths(self):
        """Load the paths of already downloaded videos from downloaded_paths.json."""
        if os.path.exists(self.downloaded_paths_path):
            try:
                with open(self.downloaded_paths_path, 'r') as f:
                    return json.load(f)  # Should directly return the list
            except json.JSONDecodeError:
                return []  # Return empty list if file is invalid
        return []

    def save_downloaded_path(self, path):
        """Save the new file path to the downloaded_paths.json."""
        try:
            downloaded_paths = self.load_downloaded_paths()
            if not isinstance(downloaded_paths, list):
                downloaded_paths = []  # Ensure it's a list
            downloaded_paths.append(path)
            with open(self.downloaded_paths_path, 'w') as f:
                json.dump(downloaded_paths, f, indent=4)  # Save as direct list
                print(f"Saved {path} to downloaded_paths.json")
        except Exception as e:
            print(f"Error saving downloaded path: {e}")

    def load_settings(self):
        """Load the settings (address, latitude, longitude) from settings.json."""
        if os.path.exists(self.settings_path):
            with open(self.settings_path, 'r') as f:
                return json.load(f)
        return {}

    def save_settings(self, address, latitude, longitude):
        """Save the address, latitude, and longitude to settings.json."""
        settings = {
            "address": address,
            "latitude": latitude,
            "longitude": longitude
        }
        with open(self.settings_path, 'w') as f:
            json.dump(settings, f, indent=4)
            print(f"Saved address, latitude, and longitude to settings.json")

    def get_next_undownloaded_video(self):
        """Get the next undownloaded video path from Firestore."""
        downloaded_paths = self.load_downloaded_paths()
        videos_ref = self.db.collection('videos')
        docs = videos_ref.get()

        for doc in docs:
            data = doc.to_dict()
            file_path = data.get('filePath', None)
            if file_path and file_path not in downloaded_paths:
                address = data.get('address', 'No address')
                latitude = data.get('latitude', 'No latitude')
                longitude = data.get('longitude', 'No longitude')

                # Save the address, latitude, and longitude to settings.json
                self.save_settings(address, latitude, longitude)

                print(f"Found new video path: {file_path}")
                print(f"Address: {address}")
                print(f"Latitude: {latitude}")
                print(f"Longitude: {longitude}")

                return file_path  # Return file path for download
        print("No new videos to download")
        return None

    def download_video(self, file_path, local_folder="downloaded_videos"):
        """Download video from Firebase Storage."""
        try:
            if not os.path.exists(local_folder):
                os.makedirs(local_folder)

            filename = os.path.basename(file_path)
            local_path = os.path.join(local_folder, filename)

            blob = self.bucket.blob(file_path)
            blob.download_to_filename(local_path)
            print(f"Downloaded video to: {local_path}")

            # Save the downloaded file path to downloaded_paths.json
            self.save_downloaded_path(file_path)
            return local_path
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None

    def run(self):
        """Main method to check and download the next video."""
        try:
            file_path = self.get_next_undownloaded_video()

            if file_path:
                print(f"Proceeding with video path: {file_path}")
                local_path = self.download_video(file_path)
                if local_path:
                    print(f"Video downloaded successfully to: {local_path}")
                else:
                    print(f"Failed to download video: {file_path}")
            else:
                print("No new videos to download")

            # Run license plate detection (if needed in the future)
            print("Starting license plate detection...")
            self.lpr.run()  # Ensure LicensePlateDetector has a valid run method
            print("License plate detection completed.")

            # Ensure the directory exists before processing text recognition
            text_recognition_dir = "D:\\Downloads\\carnumberplate-main\\test"
            if os.path.exists(text_recognition_dir):
                print("Starting text recognition...")
                self.recognizer.process_directory(text_recognition_dir)  # Process the folder if it exists
                print("Text recognition completed.")
            else:
                print(f"Error: The input folder '{text_recognition_dir}' does not exist.")

        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            print("Processing completed.")

if __name__ == "__main__":
    system = LicensePlateSystem()
    
    while True:
        try:
            print("\n" + "="*50)
            print(f"Starting new run at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*50 + "\n")
            
            system.run()  # This will run detection and recognition
            
            print("\nAll processing complete. Waiting for 60 seconds...")
            time.sleep(3)  # Wait for 60 seconds before next run
            
        except KeyboardInterrupt:
            print("\nProgram stopped by user")
            break
        except Exception as e:
            print(f"\nError in main loop: {e}")
            print("Will retry in 60 seconds...")
            time.sleep(3)
            
