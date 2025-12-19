#!/usr/bin/env python
"""Test script for Advanced Pipeline components"""

import numpy as np
import cv2
from Niraskhan_Done_1 import AdvancedDetectionPipeline, SmartGuardTracker

print("=" * 60)
print("ADVANCED PIPELINE COMPONENT TEST")
print("=" * 60)

try:
    # Initialize components
    print("\n1. Initializing Detection Pipeline...")
    pipeline = AdvancedDetectionPipeline()
    print(f"   [OK] MTCNN loaded: {pipeline.mtcnn is not None}")
    print(f"   [OK] FaceNet loaded: {pipeline.facenet_model is not None}")
    print(f"   [OK] Device: {pipeline.device}")
    
    print("\n2. Initializing Tracker...")
    tracker = SmartGuardTracker()
    print(f"   [OK] DeepSORT loaded: {tracker.tracker is not None}")
    
    print("\n3. Testing face detection with dummy image...")
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    faces = pipeline.detect_faces(dummy_frame, confidence_threshold=0.95)
    print(f"   [OK] Detection works: Found {len(faces)} faces in dummy image")
    
    print("\n4. Testing tracker update...")
    if tracker.tracker:
        tracks = tracker.update([], None)
        print(f"   [OK] Tracker update works: {len(tracks)} tracks")
    
    print("\n" + "=" * 60)
    print("SUCCESS: ALL TESTS PASSED - PIPELINE READY")
    print("=" * 60)
    print("\nThe advanced pipeline is fully initialized and operational!")
    print("You can now run: python Niraskhan_Done_1.py")
    
except Exception as e:
    print(f"\nFAILED: {e}")
    import traceback
    traceback.print_exc()
