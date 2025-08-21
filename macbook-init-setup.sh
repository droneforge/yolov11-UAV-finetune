#!/bin/bash

# =============================================================================
# YOLOV11 UAV DETECTION - ENVIRONMENT SETUP UTILITY
# Optimized for Apple M1 MacBook Pro
# =============================================================================

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ASCII Art Header
echo -e "${CYAN}"
cat << "EOF"
███████████████████████████████████████████████
█  YOLOv11 UAV DETECTION ENVIRONMENT SETUP    █
█  Optimized for Apple M1 Silicon             █
███████████████████████████████████████████████
EOF
echo -e "${NC}"

# Function to print status messages
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if running on Apple Silicon
check_apple_silicon() {
    if [[ $(uname -m) == "arm64" ]]; then
        print_success "Detected Apple Silicon (M1/M2/M3) architecture"
        return 0
    else
        print_warning "Not running on Apple Silicon - some optimizations may not apply"
        return 1
    fi
}

# Function to check Python installation
check_python() {
    print_status "Checking Python installation..."
    
    # Check for Python 3.8+ (required for ONNX Runtime on Apple Silicon)
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [[ $PYTHON_MAJOR -eq 3 ]] && [[ $PYTHON_MINOR -ge 8 ]]; then
            print_success "Python $PYTHON_VERSION detected (compatible)"
            
            # Check for Python 3.13+ compatibility issues
            if [[ $PYTHON_MINOR -ge 13 ]]; then
                print_warning "Python 3.13+ detected - using compatible package versions"
                export PYTHON_VERSION_MODERN=1
            fi
            
            return 0
        else
            print_error "Python 3.8+ required, found $PYTHON_VERSION"
            return 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8+ first."
        return 1
    fi
}

# Function to create optimized requirements.txt
create_requirements() {
    print_status "Creating optimized requirements.txt for Apple Silicon..."
    
    # Check if we're using Python 3.13+ for compatibility
    local python_minor=$(python3 -c "import sys; print(sys.version_info.minor)")
    
    if [[ $python_minor -ge 13 ]]; then
        print_status "Detected Python 3.13+ - using modern package versions"
        cat > requirements.txt << 'EOF'
# Core ML/CV libraries optimized for Apple Silicon and Python 3.13+
torch>=2.0.0
torchvision>=0.15.0
ultralytics

# Computer Vision (Python 3.13+ compatible versions)
opencv-python-headless>=4.10.0.84
Pillow>=10.0.0

# Data processing and ML (Python 3.13+ compatible)
roboflow>=1.1.0
numpy>=1.26.0
pandas>=2.1.0

# ONNX Runtime for Apple Silicon (Python 3.13+ compatible)
onnxruntime>=1.16.0

# Progress bars and utilities
tqdm>=4.65.0
pyyaml>=6.0
matplotlib>=3.7.0
scipy>=1.11.0

# For M1 optimization
accelerate>=0.20.0

# Python 3.13+ specific fixes
setuptools>=68.0.0
wheel>=0.41.0
EOF
    else
        print_status "Using exact versions for Python 3.12 and below"
        cat > requirements.txt << 'EOF'
# Core ML/CV libraries optimized for Apple Silicon
torch>=2.0.0
torchvision>=0.15.0
ultralytics

# Computer Vision (exact versions for compatibility)
opencv-python-headless==4.10.0.84
Pillow>=9.5.0

# Data processing and ML
roboflow==1.2.6
numpy==1.24.4
pandas==2.2.1

# ONNX Runtime for Apple Silicon
onnxruntime>=1.15.0

# Progress bars and utilities
tqdm>=4.65.0
pyyaml>=6.0
matplotlib>=3.7.0
scipy>=1.10.0

# For M1 optimization
accelerate>=0.20.0
EOF
    fi

    print_success "Requirements.txt created with Apple Silicon optimizations"
}

# Function to create virtual environment
create_venv() {
    print_status "Creating Python virtual environment..."
    
    if [[ -d "venv" ]]; then
        print_warning "Virtual environment already exists. Removing..."
        rm -rf venv
    fi
    
    python3 -m venv venv
    print_success "Virtual environment created"
}

# Function to activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip for better Apple Silicon support
    pip install --upgrade pip setuptools wheel
    print_success "Virtual environment activated and pip upgraded"
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing dependencies (this may take a few minutes)..."
    
    # Set environment variables for Apple Silicon optimization
    if check_apple_silicon; then
        export PYTORCH_ENABLE_MPS_FALLBACK=1
        export ACCELERATE_USE_MPS_DEVICE=1
    fi
    
    # Check Python version for compatibility handling
    local python_minor=$(python3 -c "import sys; print(sys.version_info.minor)")
    
    if [[ $python_minor -ge 13 ]]; then
        print_status "Using Python 3.13+ compatible installation strategy..."
        
        # Install build tools first for Python 3.13+
        pip install --upgrade "setuptools>=68.0.0" "wheel>=0.41.0" "pip>=23.0"
        
        # Install dependencies with relaxed constraints to avoid build issues
        pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
        pip install --no-cache-dir ultralytics
        pip install --no-cache-dir "opencv-python-headless>=4.9.0"
        pip install --no-cache-dir "roboflow>=1.1.0"
        pip install --no-cache-dir "numpy>=1.26.0"
        pip install --no-cache-dir "pandas>=2.1.0"
        pip install --no-cache-dir "onnxruntime>=1.16.0"
        pip install --no-cache-dir tqdm pyyaml matplotlib scipy accelerate
    else
        print_status "Using standard installation for Python 3.12 and below..."
        # Install dependencies with exact versions for older Python
        pip install --no-cache-dir -r requirements.txt
    fi
    
    print_success "Dependencies installed successfully"
}

# Function to run quick tests
run_tests() {
    print_status "Running quick system tests..."
    
    # Test 1: Check Python imports
    print_status "Test 1/4: Checking Python imports..."
    python3 -c "
import sys
print(f'Python: {sys.version}')
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    if torch.backends.mps.is_available():
        print('✓ MPS (Metal Performance Shaders) available')
    else:
        print('⚠ MPS not available - using CPU')
    
    import cv2
    print(f'OpenCV: {cv2.__version__}')
    
    import numpy as np
    print(f'NumPy: {np.__version__}')
    
    import onnxruntime as ort
    print(f'ONNX Runtime: {ort.__version__}')
    providers = ort.get_available_providers()
    if 'CoreMLExecutionProvider' in providers:
        print('✓ CoreML execution provider available')
    print('✓ All imports successful')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"
    
    # Test 2: Check model files
    print_status "Test 2/4: Checking model files..."
    if [[ -f "yolov11n-UAV-finetune.onnx" ]]; then
        print_success "✓ ONNX model found"
    else
        print_error "✗ ONNX model not found"
        return 1
    fi
    
    if [[ -f "yolov11n-UAV-finetune.pt" ]]; then
        print_success "✓ PyTorch model found"
    else
        print_warning "⚠ PyTorch model not found (optional)"
    fi
    
    # Test 3: Check config files
    print_status "Test 3/4: Checking configuration files..."
    if [[ -f "config.json" ]] && [[ -f "preprocessor.json" ]]; then
        print_success "✓ Configuration files found"
    else
        print_error "✗ Configuration files missing"
        return 1
    fi
    
    # Test 4: Quick ONNX model load test
    print_status "Test 4/4: Testing ONNX model loading..."
    python3 -c "
import onnxruntime as ort
import numpy as np

try:
    # Load model with optimized providers for Apple Silicon
    providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession('yolov11n-UAV-finetune.onnx', providers=providers)
    
    # Test with dummy input
    input_shape = session.get_inputs()[0].shape
    dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
    
    output = session.run(None, {session.get_inputs()[0].name: dummy_input})
    print(f'✓ Model loaded successfully')
    print(f'✓ Input shape: {input_shape}')
    print(f'✓ Output shape: {output[0].shape}')
    print(f'✓ Active providers: {session.get_providers()}')
except Exception as e:
    print(f'✗ Model loading failed: {e}')
    exit(1)
"
    
    print_success "All tests passed successfully!"
}

# Function to check disk space
check_disk_space() {
    print_status "Checking available disk space..."
    AVAILABLE_MB=$(df . | awk 'NR==2 {printf "%.0f", $4/1024}')
    
    if [[ $AVAILABLE_MB -lt 2048 ]]; then
        print_warning "Low disk space: ${AVAILABLE_MB}MB available"
        print_warning "At least 2GB recommended for full operation"
    else
        print_success "Sufficient disk space: ${AVAILABLE_MB}MB available"
    fi
}

# Function to show system info
show_system_info() {
    print_status "System Information:"
    echo "  Architecture: $(uname -m)"
    echo "  OS: $(uname -s) $(uname -r)"
    echo "  Python: $(python3 --version 2>&1)"
    
    if check_apple_silicon; then
        echo "  Metal Support: $(python3 -c 'import torch; print("Available" if torch.backends.mps.is_available() else "Not Available")' 2>/dev/null || echo "Unknown")"
    fi
}

# Main execution
main() {
    echo
    print_status "Starting environment setup for YOLOv11 UAV Detection..."
    echo
    
    # Pre-flight checks
    show_system_info
    echo
    check_disk_space
    echo
    
    # Core setup
    if ! check_python; then
        print_error "Python setup failed. Please install Python 3.8+ and try again."
        exit 1
    fi
    
    create_requirements
    create_venv
    activate_venv
    install_dependencies
    
    echo
    print_status "Running validation tests..."
    if run_tests; then
        echo
        print_success "🚁 Environment setup completed successfully!"
        echo
        print_status "Next steps:"
        echo "  1. Virtual environment is activated"
        echo "  2. Run './drone-launcher.sh' to start the application"
        echo "  3. All dependencies are installed and tested"
        echo
        print_status "To manually activate this environment later:"
        echo "  source venv/bin/activate"
        echo
    else
        print_error "Setup validation failed. Please check the errors above."
        exit 1
    fi
}

# Trap to deactivate venv on script exit (but keep it active for user)
cleanup() {
    if [[ "$1" != "0" ]]; then
        print_error "Setup failed with exit code $1"
        if [[ -d "venv" ]]; then
            print_status "Cleaning up failed virtual environment..."
            # More robust cleanup for permission issues
            chmod -R u+w venv 2>/dev/null || true
            rm -rf venv 2>/dev/null || {
                print_warning "Could not remove venv directory - you may need to remove it manually"
                print_status "Try: sudo rm -rf venv"
            }
        fi
    fi
}
trap 'cleanup $?' EXIT

# Run main function
main "$@"

# Keep the virtual environment active for the user
echo -e "${GREEN}Virtual environment remains active. You can now run drone-launcher.sh${NC}"