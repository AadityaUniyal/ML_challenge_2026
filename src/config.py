import os

class Config:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Data paths
    RAW_TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "train")
    RAW_TEST_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "test")
    STUDENT_RESOURCE_DIR = os.path.join(PROJECT_ROOT, "..", "student_resource")
    
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
    REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
    ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
    
    # Default blocking parameters
    MAX_CANDIDATES_PER_S1 = 25
    DF_CAP = 2500
    
    # Threshold for F_0.5 decision logic
    DECISION_THRESHOLD = 0.65
