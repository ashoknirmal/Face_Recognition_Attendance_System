import cv2
import numpy as np

def test_camera():
    """Test camera access and display"""
    print("Testing camera access...")
    print("Press 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Cannot access camera!")
        print("Please check:")
        print("1. Camera is connected and working")
        print("2. Camera permissions are granted")
        print("3. No other application is using the camera")
        return False
    
    print("Camera access successful!")
    print("Press 'q' to quit the test")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Add some text overlay
        cv2.putText(frame, 'Camera Test - Press "q" to quit', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Frame shape: {frame.shape}', 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Camera Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Camera test completed.")
    return True

if __name__ == "__main__":
    test_camera()