#!/bin/bash

# निराक्षण (Niraakshan) - CUDA Setup Script for Ubuntu 22.04
# Automated installation script for GPU-accelerated version

set -e  # Exit on error

echo "========================================================"
echo "  निराक्षण (Niraakshan) - CUDA GPU Setup"
echo "  Ubuntu 22.04.05 LTS | NVIDIA GTX 1080"
echo "========================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "📌 $1"
}

# Check if running on Ubuntu
if [ ! -f /etc/os-release ]; then
    print_error "Cannot detect OS. This script is for Ubuntu 22.04."
    exit 1
fi

source /etc/os-release
if [ "$VERSION_ID" != "22.04" ]; then
    print_warning "This script is designed for Ubuntu 22.04. You have $VERSION_ID"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for NVIDIA GPU
print_info "Checking for NVIDIA GPU..."
if ! lspci | grep -i nvidia > /dev/null; then
    print_error "No NVIDIA GPU detected!"
    exit 1
fi
print_success "NVIDIA GPU detected"

# Check if NVIDIA driver is installed
if ! command -v nvidia-smi &> /dev/null; then
    print_warning "NVIDIA driver not found"
    read -p "Install NVIDIA driver 535? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo apt update
        sudo apt install -y nvidia-driver-535
        print_success "NVIDIA driver installed. Please reboot and run this script again."
        exit 0
    else
        print_error "NVIDIA driver required. Exiting."
        exit 1
    fi
else
    print_success "NVIDIA driver found: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
fi

# Check CUDA installation
if ! command -v nvcc &> /dev/null; then
    print_warning "CUDA Toolkit not found"
    print_info "Please install CUDA 11.8 manually:"
    print_info "https://developer.nvidia.com/cuda-11-8-0-download-archive"
    read -p "Have you installed CUDA? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "CUDA Toolkit required. Exiting."
        exit 1
    fi
else
    print_success "CUDA found: $(nvcc --version | grep release | awk '{print $5}' | sed 's/,//')"
fi

# Setup project directory
PROJECT_DIR="$HOME/niraakshan_cuda"
print_info "Setting up project directory: $PROJECT_DIR"

if [ -d "$PROJECT_DIR" ]; then
    print_warning "Project directory already exists"
    read -p "Continue and overwrite? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create directory structure
print_info "Creating directory structure..."
mkdir -p logs guard_profiles capture_snapshots alert_snapshots audio_files
print_success "Directories created"

# Install system dependencies
print_info "Installing system dependencies..."
sudo apt update
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    build-essential \
    cmake \
    git \
    wget \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    libboost-all-dev
print_success "System dependencies installed"

# Create virtual environment
print_info "Creating Python 3.11 virtual environment..."
python3.11 -m venv niraakshan_cuda_env
source niraakshan_cuda_env/bin/activate
print_success "Virtual environment created and activated"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel
print_success "pip upgraded"

# Install Python packages
print_info "Installing Python packages (this may take a while)..."

# Core packages
print_info "Installing core packages..."
pip install \
    numpy==1.24.3 \
    Pillow==10.1.0 \
    opencv-contrib-python==4.8.1.78 \
    psutil==5.9.6

# TensorFlow with GPU
print_info "Installing TensorFlow with GPU support..."
pip install tensorflow[and-cuda]==2.15.0 tensorflow-hub==0.15.0

# CuPy (GPU array operations)
print_info "Installing CuPy for CUDA 11.x..."
pip install cupy-cuda11x==12.3.0

# MediaPipe
print_info "Installing MediaPipe..."
pip install mediapipe==0.10.8

# Face recognition
print_info "Installing face recognition..."
pip install face-recognition==1.3.0

# GUI and utilities
print_info "Installing GUI packages..."
pip install customtkinter==5.2.1

# Audio (optional)
print_info "Installing audio packages..."
pip install pygame==2.5.2 pydub==0.25.1

# Jupyter
print_info "Installing Jupyter..."
pip install jupyter notebook ipykernel
python -m ipykernel install --user --name=niraakshan_cuda_env

print_success "All Python packages installed"

# Verify installations
echo ""
print_info "Verifying installations..."
echo ""

# Check TensorFlow GPU
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('TensorFlow GPU:', 'Available' if len(gpus) > 0 else 'Not Available')"

# Check OpenCV
python -c "import cv2; print('OpenCV version:', cv2.__version__)"

# Check MediaPipe
python -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__)"

# Check CuPy
python -c "import cupy as cp; print('CuPy version:', cp.__version__)" || print_warning "CuPy not available"

# Create config file
print_info "Creating configuration file..."
cat > config.json << 'EOF'
{
    "detection": {
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
        "face_recognition_tolerance": 0.5,
        "re_detect_interval": 8,
        "use_gpu": true
    },
    "alert": {
        "default_interval_seconds": 10,
        "alert_cooldown_seconds": 2.5
    },
    "performance": {
        "gui_refresh_ms": 30,
        "pose_buffer_size": 12,
        "frame_skip_interval": 1,
        "enable_frame_skipping": false,
        "min_buffer_for_classification": 5,
        "gpu_batch_size": 4
    },
    "logging": {
        "log_directory": "logs",
        "max_log_size_mb": 10,
        "auto_flush_interval": 50
    },
    "storage": {
        "alert_snapshots_dir": "alert_snapshots",
        "snapshot_retention_days": 30,
        "guard_profiles_dir": "guard_profiles",
        "capture_snapshots_dir": "capture_snapshots",
        "audio_files_dir": "audio_files"
    },
    "monitoring": {
        "mode": "pose",
        "session_restart_prompt_hours": 8
    },
    "gpu": {
        "device_id": 0,
        "memory_fraction": 0.8,
        "allow_growth": true,
        "use_tensorrt": false
    }
}
EOF
print_success "Configuration file created"

# Create activation script
print_info "Creating activation script..."
cat > activate.sh << 'EOF'
#!/bin/bash
source niraakshan_cuda_env/bin/activate
echo "✅ निराक्षण CUDA Environment Activated"
echo "   Python: $(python --version)"
echo "   Project: $(pwd)"
echo ""
echo "To start Jupyter Notebook:"
echo "   jupyter notebook"
EOF
chmod +x activate.sh
print_success "Activation script created"

# Create launch script
print_info "Creating Jupyter launch script..."
cat > start_jupyter.sh << 'EOF'
#!/bin/bash
source niraakshan_cuda_env/bin/activate
echo "Starting Jupyter Notebook..."
jupyter notebook
EOF
chmod +x start_jupyter.sh
print_success "Launch script created"

# Final summary
echo ""
echo "========================================================"
print_success "निराक्षण CUDA Setup Complete!"
echo "========================================================"
echo ""
print_info "Next steps:"
echo "1. Copy your CUDA notebook to: $PROJECT_DIR"
echo "2. Activate environment: cd $PROJECT_DIR && source activate.sh"
echo "3. Start Jupyter: ./start_jupyter.sh"
echo ""
print_info "Project location: $PROJECT_DIR"
print_info "Virtual environment: niraakshan_cuda_env"
echo ""

# Optional: Compile dlib with CUDA
read -p "Do you want to compile dlib with CUDA support for better face detection? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Compiling dlib with CUDA support..."
    cd ~
    git clone https://github.com/davisking/dlib.git
    cd dlib
    mkdir build && cd build
    cmake .. \
        -DDLIB_USE_CUDA=1 \
        -DUSE_AVX_INSTRUCTIONS=1 \
        -DCUDA_ARCH_NAME=Pascal \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build . --config Release
    cd ..
    
    # Activate environment and install
    source "$PROJECT_DIR/niraakshan_cuda_env/bin/activate"
    python setup.py install
    
    # Verify
    python -c "import dlib; print('dlib CUDA:', 'Enabled' if dlib.DLIB_USE_CUDA else 'Disabled')"
    
    cd "$PROJECT_DIR"
    print_success "dlib compiled with CUDA support"
else
    print_info "Skipping dlib CUDA compilation"
fi

echo ""
print_success "Setup complete! You can now use निराक्षण with GPU acceleration."
echo ""
