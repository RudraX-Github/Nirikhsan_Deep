"""
Test IP Camera Connection with Username/Password
Tests the RTSP stream configuration for Dahua/SmartPSS cameras
"""

import cv2
import urllib.parse

def test_ip_camera():
    """Test IP camera connection with the specified configuration"""
    
    # --- CONFIGURATION ---
    # User will be asked to provide these values in the dialog
    IP = input("Enter IP Address (e.g., 127.0.0.1): ").strip() or "127.0.0.1"
    RTSP_PORT = input("Enter RTSP Port (e.g., 29555): ").strip() or "29555"
    
    # REQUIRED: Username and Password
    USER = input("Enter Username: ").strip()
    if not USER:
        print("ERROR: Username is required!")
        return
    
    PASSWORD_RAW = input("Enter Password: ").strip()
    if not PASSWORD_RAW:
        print("ERROR: Password is required!")
        return

    # 1. ENCODE PASSWORD (handles special characters like '@')
    password_encoded = urllib.parse.quote(PASSWORD_RAW)

    # 2. CONSTRUCT RTSP URL
    # Standard Dahua/SmartPSS path: cam/realmonitor?channel=1&subtype=0
    stream_path = "cam/realmonitor?channel=1&subtype=0"
    
    rtsp_url = f"rtsp://{USER}:{password_encoded}@{IP}:{RTSP_PORT}/{stream_path}"
    
    print(f"\n{'='*60}")
    print(f"Connecting to IP Camera...")
    print(f"URL: rtsp://{USER}:***@{IP}:{RTSP_PORT}/{stream_path}")
    print(f"{'='*60}\n")

    # 3. INITIALIZE OPENCV with CAP_FFMPEG backend
    print("Opening stream with FFmpeg backend...")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    # Optional: Lower buffer size to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ ERROR: Could not open stream.")
        print("\nTroubleshooting:")
        print("1. Verify IP and Port are correct")
        print("2. Check username and password")
        print("3. Ensure the camera service is running on the specified port")
        print("4. Try alternative stream paths:")
        print("   - /live")
        print("   - /cam/realmonitor?channel=1&subtype=1  (sub-stream)")
        print("   - Empty string ''")
        return

    print("✅ SUCCESS! Stream opened.\n")
    print("Reading frames... Press 'q' to quit.\n")

    frame_count = 0
    
    # 4. READ LOOP
    while True:
        ret, frame = cap.read()

        if not ret:
            print(f"\n⚠ Frame drop or stream disconnected at frame {frame_count}")
            print("Attempting to reconnect...")
            break

        frame_count += 1
        
        # Display frame info every 30 frames (~1 second at 30fps)
        if frame_count % 30 == 0:
            h, w = frame.shape[:2]
            print(f"Frame {frame_count}: {w}x{h} | Press 'q' to quit", end='\r')

        # Display the frame
        cv2.imshow("IP Camera Test - Press 'q' to quit", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n\n{'='*60}")
    print(f"Test Complete!")
    print(f"Total frames received: {frame_count}")
    print(f"{'='*60}")

if __name__ == "__main__":
    print("="*60)
    print("IP Camera Connection Test")
    print("For Dahua/SmartPSS Cameras")
    print("="*60)
    print()
    
    try:
        test_ip_camera()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
    
    print("\nPress Enter to exit...")
    input()
