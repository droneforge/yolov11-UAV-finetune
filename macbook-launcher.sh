#!/bin/bash

# =============================================================================
# YOLOV11 UAV DETECTION - LAUNCHER & OPERATIONS CLI
# Terminal User Interface for Drone Detection Operations
# =============================================================================

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
WHITE='\033[1;37m'
NC='\033[0m'

# Global variables
CURRENT_PROJECT=""
LOG_FILE="drone_launcher.log"

# Functions for output
print_header() {
    clear
    echo -e "${CYAN}"
    cat << "EOF"
███████████████████████████████████████████████
█  YOLOv11 UAV DETECTION LAUNCHER             █
█  Operations Control Interface               █
███████████████████████████████████████████████
EOF
    echo -e "${NC}"
}

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] $1" >> "$LOG_FILE"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [SUCCESS] $1" >> "$LOG_FILE"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] $1" >> "$LOG_FILE"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [WARNING] $1" >> "$LOG_FILE"
}

# Function to check if virtual environment is active
check_venv() {
    if [[ -z "$VIRTUAL_ENV" ]]; then
        print_error "Virtual environment not active!"
        print_status "Please run: source venv/bin/activate"
        exit 1
    fi
    return 0
}

# Function to check required files
check_requirements() {
    local missing_files=()
    
    # Check core files
    [[ ! -f "yolov11n-UAV-finetune.onnx" ]] && missing_files+=("yolov11n-UAV-finetune.onnx")
    [[ ! -f "config.json" ]] && missing_files+=("config.json")
    [[ ! -f "preprocessor.json" ]] && missing_files+=("preprocessor.json")
    [[ ! -f "video_inference.py" ]] && missing_files+=("video_inference.py")
    
    if [[ ${#missing_files[@]} -ne 0 ]]; then
        print_error "Missing required files:"
        for file in "${missing_files[@]}"; do
            echo "  - $file"
        done
        return 1
    fi
    
    return 0
}

# Function to display main menu
show_main_menu() {
    print_header
    echo -e "${WHITE}═══ MAIN OPERATIONS MENU ═══${NC}"
    echo
    echo "  1. Video Inference - Process video with UAV detection"
    echo "  2. Synthetic Data Generation - Create training datasets"
    echo "  3. Model Testing & Validation - Test model performance"
    echo "  4. Project Management - Create/manage subprojects"
    echo "  5. System Status - Check system and model health"
    echo "  6. Batch Processing - Process multiple files"
    echo "  7. Configuration Manager - Edit settings"
    echo "  8. Logs & Reports - View operation logs"
    echo
    echo "  9. Help & Documentation"
    echo "  0. Exit"
    echo
    echo -e "${WHITE}═══════════════════════════${NC}"
    
    if [[ -n "$CURRENT_PROJECT" ]]; then
        echo -e "Current Project: ${CYAN}$CURRENT_PROJECT${NC}"
    fi
    
    echo -n "Select option [0-9]: "
}

# Function for video inference
video_inference() {
    print_header
    echo -e "${WHITE}═══ VIDEO INFERENCE ═══${NC}"
    echo
    
    # List available video files
    echo "Available video files:"
    local video_files=(*.mp4 *.avi *.mov)
    # Filter out non-existent files
    local existing_files=()
    for file in "${video_files[@]}"; do
        if [[ -f "$file" ]]; then
            existing_files+=("$file")
        fi
    done
    
    if [[ ${#existing_files[@]} -eq 0 ]]; then
        print_warning "No video files found in current directory"
        echo "Please ensure your video file is in the current directory"
        echo
        echo "Supported formats: .mp4, .avi, .mov"
        read -p "Press Enter to continue..."
        return
    fi
    
    echo
    for i in "${!existing_files[@]}"; do
        echo "  $((i+1)). ${existing_files[i]}"
    done
    echo "  0. Go back"
    echo
    
    read -p "Select video file [0-${#existing_files[@]}]: " choice
    
    if [[ "$choice" == "0" ]]; then
        return
    elif [[ "$choice" -ge 1 ]] && [[ "$choice" -le ${#existing_files[@]} ]]; then
        local selected_file="${existing_files[$((choice-1))]}"
        
        echo
        print_status "Processing video: $selected_file"
        
        # Configuration options
        echo "Inference Configuration:"
        read -p "Confidence threshold (0.1-1.0) [0.3]: " conf_threshold
        conf_threshold=${conf_threshold:-0.3}
        
        read -p "Output filename [auto]: " output_name
        if [[ -z "$output_name" ]]; then
            output_name="${selected_file%.*}_detections.mp4"
        fi
        
        # Create temporary inference script
        cat > temp_inference.py << EOF
import onnxruntime
import numpy as np
import cv2
import json
import sys
from pathlib import Path

# Copy the inference code but with dynamic input
input_video = '$selected_file'
output_video = '$output_name'
confidence = $conf_threshold

def load_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    with open('preprocessor.json', 'r') as f:
        preprocess_config = json.load(f)
    return config, preprocess_config

# [Include the full preprocessing and postprocessing functions from video_inference.py]
exec(open('video_inference.py').read().replace("'anduril_swarm.mp4'", "input_video").replace("'anduril_swarm_detections.mp4'", "output_video").replace("conf_threshold=0.2", f"conf_threshold={confidence}"))
EOF
        
        # Run inference
        print_status "Starting inference process..."
        if python3 temp_inference.py; then
            print_success "Video processing completed!"
            print_status "Output saved as: $output_name"
        else
            print_error "Video processing failed"
        fi
        
        # Cleanup
        rm -f temp_inference.py
        
    else
        print_error "Invalid selection"
    fi
    
    echo
    read -p "Press Enter to continue..."
}

# Function for synthetic data generation
synthetic_data_generation() {
    print_header
    echo -e "${WHITE}═══ SYNTHETIC DATA GENERATION ═══${NC}"
    echo
    
    # Check if synthetic sim directory exists
    if [[ ! -d "synthetic_drone_swarm_sim" ]]; then
        print_error "Synthetic simulation directory not found!"
        read -p "Press Enter to continue..."
        return
    fi
    
    cd synthetic_drone_swarm_sim
    
    echo "Synthetic Data Generation Options:"
    echo
    echo "  1. Quick Demo (10 drones, default settings)"
    echo "  2. Training Dataset (with ground truth)"
    echo "  3. Custom Configuration"
    echo "  4. Batch Generation (multiple scenarios)"
    echo "  5. Performance Test (benchmark generation speed)"
    echo "  0. Go back"
    echo
    
    read -p "Select option [0-5]: " choice
    
    case $choice in
        1)
            print_status "Generating quick demo with detailed reporting..."
            echo "==================== GENERATION REPORT ===================="
            echo "Start Time: $(date)"
            echo "Configuration: 10 drones, default settings"
            echo "============================================================"
            
            start_time=$(date +%s)
            python3 sim.py --num-drones 10 2>&1 | while IFS= read -r line; do
                echo "$line"
                echo "$(date '+%Y-%m-%d %H:%M:%S') [SYNTHETIC] $line" >> "../$LOG_FILE"
            done
            end_time=$(date +%s)
            
            duration=$((end_time - start_time))
            echo "============================================================"
            echo "End Time: $(date)"
            echo "Generation Duration: ${duration}s"
            echo "Output Location: outputs/"
            echo "============================================================"
            ;;
            
        2)
            print_status "Generating training dataset with ground truth..."
            read -p "Number of drones [15]: " num_drones
            num_drones=${num_drones:-15}
            
            read -p "Output filename [training_data]: " output_name
            output_name=${output_name:-training_data}
            
            echo "==================== GENERATION REPORT ===================="
            echo "Start Time: $(date)"
            echo "Configuration: $num_drones drones, ground truth enabled"
            echo "Output: outputs/${output_name}.mp4"
            echo "Annotations: outputs/labels/"
            echo "============================================================"
            
            start_time=$(date +%s)
            python3 sim.py --num-drones "$num_drones" --output "outputs/${output_name}.mp4" --gen_groundtruth 2>&1 | while IFS= read -r line; do
                echo "$line"
                echo "$(date '+%Y-%m-%d %H:%M:%S') [SYNTHETIC] $line" >> "../$LOG_FILE"
            done
            end_time=$(date +%s)
            
            duration=$((end_time - start_time))
            
            # Post-generation analysis
            if [[ -f "outputs/${output_name}.mp4" ]]; then
                video_size=$(du -h "outputs/${output_name}.mp4" | awk '{print $1}')
                frame_count=$(python3 -c "
import cv2
cap = cv2.VideoCapture('outputs/${output_name}.mp4')
print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
cap.release()
" 2>/dev/null || echo "Unknown")
                
                annotation_count=$(ls outputs/labels/*.txt 2>/dev/null | wc -l | tr -d ' ' || echo "0")
                
                echo "============================================================"
                echo "GENERATION COMPLETED SUCCESSFULLY"
                echo "End Time: $(date)"
                echo "Duration: ${duration}s"
                echo "Video Size: $video_size"
                echo "Frame Count: $frame_count"
                echo "Annotation Files: $annotation_count"
                echo "Average Speed: $(echo "scale=2; $frame_count / $duration" | bc 2>/dev/null || echo "N/A") frames/sec"
                echo "============================================================"
                
                print_success "Training dataset generated successfully!"
            else
                print_error "Generation failed - output file not created"
            fi
            ;;
            
        3)
            print_status "Custom configuration setup with detailed reporting..."
            echo "Available background videos:"
            ls assets/*.mp4 2>/dev/null || echo "No background videos found"
            echo
            echo "Available drone images:"
            ls assets/*.png 2>/dev/null || echo "No drone images found"
            echo
            
            read -p "Background video path [assets/background.mp4]: " bg_video
            bg_video=${bg_video:-assets/background.mp4}
            
            read -p "Drone image path [assets/drone.png]: " drone_img
            drone_img=${drone_img:-assets/drone.png}
            
            read -p "Number of drones [10]: " num_drones
            num_drones=${num_drones:-10}
            
            read -p "Generate ground truth? (y/n) [y]: " gen_gt
            gen_gt=${gen_gt:-y}
            
            read -p "Output filename [custom_swarm]: " output_name
            output_name=${output_name:-custom_swarm}
            
            # Validate inputs
            echo "==================== CONFIGURATION VALIDATION ==========="
            echo "Background Video: $bg_video"
            if [[ -f "$bg_video" ]]; then
                bg_size=$(du -h "$bg_video" | awk '{print $1}')
                echo "  Status: ✓ Found ($bg_size)"
            else
                echo "  Status: ✗ Not found"
                print_error "Background video not found: $bg_video"
                cd ..
                read -p "Press Enter to continue..."
                return
            fi
            
            echo "Drone Image: $drone_img"
            if [[ -f "$drone_img" ]]; then
                drone_size=$(du -h "$drone_img" | awk '{print $1}')
                echo "  Status: ✓ Found ($drone_size)"
            else
                echo "  Status: ✗ Not found"
                print_error "Drone image not found: $drone_img"
                cd ..
                read -p "Press Enter to continue..."
                return
            fi
            
            echo "Drone Count: $num_drones"
            echo "Ground Truth: $gen_gt"
            echo "Output: outputs/${output_name}.mp4"
            echo "============================================================"
            
            cmd="python3 sim.py --background '$bg_video' --drone '$drone_img' --num-drones $num_drones --output 'outputs/${output_name}.mp4'"
            
            if [[ "$gen_gt" == "y" ]]; then
                cmd="$cmd --gen_groundtruth"
            fi
            
            echo "Starting generation..."
            start_time=$(date +%s)
            
            eval "$cmd" 2>&1 | while IFS= read -r line; do
                echo "$line"
                echo "$(date '+%Y-%m-%d %H:%M:%S') [SYNTHETIC-CUSTOM] $line" >> "../$LOG_FILE"
            done
            
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            
            echo "============================================================"
            echo "CUSTOM GENERATION COMPLETED"
            echo "Duration: ${duration}s"
            echo "Configuration applied successfully"
            echo "============================================================"
            ;;
            
        4)
            print_status "Batch generation setup with comprehensive reporting..."
            read -p "Number of different scenarios [5]: " scenarios
            scenarios=${scenarios:-5}
            
            read -p "Base number of drones [10]: " base_drones
            base_drones=${base_drones:-10}
            
            read -p "Increment per scenario [5]: " increment
            increment=${increment:-5}
            
            echo "==================== BATCH GENERATION REPORT ============"
            echo "Start Time: $(date)"
            echo "Scenarios: $scenarios"
            echo "Base Drones: $base_drones"
            echo "Increment: $increment per scenario"
            echo "Total Expected Files: $((scenarios * 2)) (video + annotations)"
            echo "============================================================"
            
            batch_start_time=$(date +%s)
            successful_generations=0
            failed_generations=0
            
            for ((i=1; i<=scenarios; i++)); do
                drone_count=$((base_drones + (i-1)*increment))
                output_name="batch_scenario_${i}"
                
                echo
                echo "--- Scenario $i/$scenarios ---"
                echo "Drones: $drone_count"
                echo "Output: outputs/${output_name}.mp4"
                
                scenario_start=$(date +%s)
                
                if python3 sim.py --num-drones "$drone_count" --output "outputs/${output_name}.mp4" --gen_groundtruth 2>&1 | while IFS= read -r line; do
                    echo "  $line"
                    echo "$(date '+%Y-%m-%d %H:%M:%S') [BATCH-$i] $line" >> "../$LOG_FILE"
                done; then
                    scenario_end=$(date +%s)
                    scenario_duration=$((scenario_end - scenario_start))
                    
                    if [[ -f "outputs/${output_name}.mp4" ]]; then
                        file_size=$(du -h "outputs/${output_name}.mp4" | awk '{print $1}')
                        echo "  ✓ Success (${scenario_duration}s, $file_size)"
                        ((successful_generations++))
                    else
                        echo "  ✗ Failed - no output file"
                        ((failed_generations++))
                    fi
                else
                    echo "  ✗ Failed - generation error"
                    ((failed_generations++))
                fi
            done
            
            batch_end_time=$(date +%s)
            batch_duration=$((batch_end_time - batch_start_time))
            
            echo
            echo "============================================================"
            echo "BATCH GENERATION COMPLETED"
            echo "End Time: $(date)"
            echo "Total Duration: ${batch_duration}s"
            echo "Successful: $successful_generations/$scenarios"
            echo "Failed: $failed_generations/$scenarios"
            echo "Average per scenario: $(echo "scale=1; $batch_duration / $scenarios" | bc 2>/dev/null || echo "N/A")s"
            echo "Success Rate: $(echo "scale=1; $successful_generations * 100 / $scenarios" | bc 2>/dev/null || echo "N/A")%"
            echo "============================================================"
            
            if [[ $successful_generations -eq $scenarios ]]; then
                print_success "All scenarios generated successfully!"
            elif [[ $successful_generations -gt 0 ]]; then
                print_warning "Partial success: $successful_generations/$scenarios completed"
            else
                print_error "Batch generation failed completely"
            fi
            ;;
            
        5)
            print_status "Performance benchmark test..."
            echo "==================== PERFORMANCE BENCHMARK =============="
            echo "Test Configuration: Various drone counts"
            echo "Purpose: Measure generation speed vs complexity"
            echo "============================================================"
            
            test_counts=(5 10 15 20)
            echo "Drone Counts to Test: ${test_counts[*]}"
            echo
            
            for count in "${test_counts[@]}"; do
                echo "Testing $count drones..."
                test_start=$(date +%s)
                
                if python3 sim.py --num-drones "$count" --output "outputs/perf_test_${count}.mp4" >/dev/null 2>&1; then
                    test_end=$(date +%s)
                    test_duration=$((test_end - test_start))
                    
                    if [[ -f "outputs/perf_test_${count}.mp4" ]]; then
                        file_size=$(du -h "outputs/perf_test_${count}.mp4" | awk '{print $1}')
                        frame_count=$(python3 -c "
import cv2
cap = cv2.VideoCapture('outputs/perf_test_${count}.mp4')
print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
cap.release()
" 2>/dev/null || echo "Unknown")
                        
                        fps=$(echo "scale=2; $frame_count / $test_duration" | bc 2>/dev/null || echo "N/A")
                        
                        echo "  $count drones: ${test_duration}s ($file_size, $frame_count frames, ${fps} fps)"
                    else
                        echo "  $count drones: FAILED"
                    fi
                else
                    echo "  $count drones: ERROR"
                fi
            done
            
            echo
            echo "============================================================"
            echo "PERFORMANCE BENCHMARK COMPLETED"
            echo "All test files saved with 'perf_test_' prefix"
            echo "============================================================"
            ;;
            
        0)
            cd ..
            return
            ;;
        *)
            print_error "Invalid selection"
            ;;
    esac
    
    cd ..
    echo
    read -p "Press Enter to continue..."
}

# Function for model testing
model_testing() {
    print_header
    echo -e "${WHITE}═══ MODEL TESTING & VALIDATION ═══${NC}"
    echo
    
    echo "Testing Options:"
    echo
    echo "  1. Quick Model Health Check"
    echo "  2. Performance Benchmark"
    echo "  3. Accuracy Validation (requires validation data)"
    echo "  4. Speed Test"
    echo "  0. Go back"
    echo
    
    read -p "Select option [0-4]: " choice
    
    case $choice in
        1)
            print_status "Running model health check..."
            python3 -c "
import onnxruntime as ort
import numpy as np
import time
import warnings

# Suppress ONNX Runtime warnings
ort.set_default_logger_severity(3)

try:
    session = ort.InferenceSession('yolov11n-UAV-finetune.onnx', providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
    print('✓ Model loaded successfully')
    print(f'✓ Providers: {session.get_providers()}')
    
    # Test inference with correct input size (640x640)
    dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
    start_time = time.time()
    output = session.run(None, {session.get_inputs()[0].name: dummy_input})
    inference_time = time.time() - start_time
    
    print(f'✓ Inference test passed ({inference_time:.3f}s)')
    print(f'✓ Output shape: {output[0].shape}')
    print(f'✓ Model input size: 640x640 (required)')
except Exception as e:
    print(f'✗ Test failed: {e}')
"
            ;;
        2)
            print_status "Running performance benchmark..."
            python3 -c "
import onnxruntime as ort
import numpy as np
import time

# Suppress ONNX Runtime warnings
ort.set_default_logger_severity(3)

session = ort.InferenceSession('yolov11n-UAV-finetune.onnx', providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)

# Warmup
for _ in range(5):
    session.run(None, {session.get_inputs()[0].name: dummy_input})

# Benchmark
times = []
for i in range(50):
    start = time.time()
    session.run(None, {session.get_inputs()[0].name: dummy_input})
    times.append(time.time() - start)

import statistics
print(f'Inference Statistics (50 runs):')
print(f'  Mean: {statistics.mean(times)*1000:.2f}ms')
print(f'  Median: {statistics.median(times)*1000:.2f}ms')
print(f'  Min: {min(times)*1000:.2f}ms')
print(f'  Max: {max(times)*1000:.2f}ms')
print(f'  FPS (approx): {1/statistics.mean(times):.1f}')
"
            ;;
        3)
            if [[ -f "validation.py" ]]; then
                print_status "Running validation script..."
                python3 validation.py
            else
                print_error "validation.py not found"
            fi
            ;;
        4)
            print_status "Running speed test with model's required input size..."
            python3 -c "
import onnxruntime as ort
import numpy as np
import time

# Suppress ONNX Runtime warnings
ort.set_default_logger_severity(3)

session = ort.InferenceSession('yolov11n-UAV-finetune.onnx', providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])

# Get the actual required input shape from the model
input_shape = session.get_inputs()[0].shape
required_size = input_shape[2]  # Should be 640 for this model

print(f'Model requires input size: {required_size}x{required_size}')
print()

# Test different batch sizes with correct input dimensions
batch_sizes = [1, 2, 4]
for batch_size in batch_sizes:
    dummy_input = np.random.rand(batch_size, 3, required_size, required_size).astype(np.float32)
    
    # Warmup
    for _ in range(3):
        try:
            session.run(None, {session.get_inputs()[0].name: dummy_input})
        except Exception as e:
            print(f'Batch size {batch_size}: Not supported - {e}')
            continue
    
    # Test
    start = time.time()
    for _ in range(10):
        session.run(None, {session.get_inputs()[0].name: dummy_input})
    avg_time = (time.time() - start) / 10
    
    print(f'Batch size {batch_size}: {avg_time*1000:.2f}ms ({1/avg_time:.1f} FPS)')

# Test memory usage
print()
print('Memory efficiency test (640x640 input):')
dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
start = time.time()
for _ in range(100):
    session.run(None, {session.get_inputs()[0].name: dummy_input})
avg_time = (time.time() - start) / 100
print(f'100 iterations: {avg_time*1000:.2f}ms average ({1/avg_time:.1f} FPS)')
"
            ;;
        0)
            return
            ;;
        *)
            print_error "Invalid selection"
            ;;
    esac
    
    echo
    read -p "Press Enter to continue..."
}

# Function for project management
project_management() {
    print_header
    echo -e "${WHITE}═══ PROJECT MANAGEMENT ═══${NC}"
    echo
    
    echo "Project Options:"
    echo
    echo "  1. Create New Project Folder"
    echo "  2. List Existing Projects"
    echo "  3. Switch to Project"
    echo "  4. Archive Project"
    echo "  0. Go back"
    echo
    
    read -p "Select option [0-4]: " choice
    
    case $choice in
        1)
            read -p "Project name: " project_name
            if [[ -n "$project_name" ]]; then
                mkdir -p "projects/$project_name"
                mkdir -p "projects/$project_name/data"
                mkdir -p "projects/$project_name/outputs"
                mkdir -p "projects/$project_name/models"
                
                # Copy essential files
                cp config.json "projects/$project_name/"
                cp preprocessor.json "projects/$project_name/"
                cp yolov11n-UAV-finetune.onnx "projects/$project_name/"
                
                print_success "Project '$project_name' created"
                CURRENT_PROJECT="$project_name"
            fi
            ;;
        2)
            echo "Existing projects:"
            if [[ -d "projects" ]]; then
                ls -la projects/ | grep "^d" | awk '{print "  " $9}' | grep -v "^\.$" | grep -v "^\.\.$"
            else
                echo "  No projects found"
            fi
            ;;
        3)
            echo "Available projects:"
            if [[ -d "projects" ]]; then
                local projects=(projects/*/)
                if [[ ${#projects[@]} -gt 0 ]] && [[ -d "${projects[0]}" ]]; then
                    for i in "${!projects[@]}"; do
                        local proj_name=$(basename "${projects[i]}")
                        echo "  $((i+1)). $proj_name"
                    done
                    echo
                    read -p "Select project: " choice
                    if [[ "$choice" -ge 1 ]] && [[ "$choice" -le ${#projects[@]} ]]; then
                        CURRENT_PROJECT=$(basename "${projects[$((choice-1))]}")
                        print_success "Switched to project: $CURRENT_PROJECT"
                    fi
                else
                    echo "  No projects found"
                fi
            else
                echo "  No projects found"
            fi
            ;;
        4)
            # Archive project implementation
            print_status "Archive functionality not implemented yet"
            ;;
        0)
            return
            ;;
        *)
            print_error "Invalid selection"
            ;;
    esac
    
    echo
    read -p "Press Enter to continue..."
}

# Function to show system status
system_status() {
    print_header
    echo -e "${WHITE}═══ SYSTEM STATUS ═══${NC}"
    echo
    
    echo -e "${CYAN}Environment Status:${NC}"
    echo "  Virtual Environment: ${GREEN}✓ Active${NC}"
    echo "  Python: $(python3 --version)"
    echo "  Working Directory: $(pwd)"
    
    if [[ -n "$CURRENT_PROJECT" ]]; then
        echo "  Current Project: ${CYAN}$CURRENT_PROJECT${NC}"
    fi
    
    echo
    echo -e "${CYAN}System Resources:${NC}"
    echo "  CPU Usage: $(top -l 1 | grep "CPU usage" | awk '{print $3}' | sed 's/%//' || echo "N/A")%"
    echo "  Memory Usage: $(top -l 1 | grep "PhysMem" | awk '{print $2}' | sed 's/M//' || echo "N/A")MB used"
    echo "  Disk Space: $(df -h . | awk 'NR==2 {print $4}') available"
    
    echo
    echo -e "${CYAN}Model Status:${NC}"
    if [[ -f "yolov11n-UAV-finetune.onnx" ]]; then
        local model_size=$(du -h yolov11n-UAV-finetune.onnx | awk '{print $1}')
        echo "  ONNX Model: ${GREEN}✓ Available${NC} ($model_size)"
    else
        echo "  ONNX Model: ${RED}✗ Missing${NC}"
    fi
    
    if [[ -f "yolov11n-UAV-finetune.pt" ]]; then
        local pt_size=$(du -h yolov11n-UAV-finetune.pt | awk '{print $1}')
        echo "  PyTorch Model: ${GREEN}✓ Available${NC} ($pt_size)"
    else
        echo "  PyTorch Model: ${YELLOW}⚠ Missing${NC}"
    fi
    
    echo
    echo -e "${CYAN}File Counts:${NC}"
    echo "  Video files: $(find . -maxdepth 1 -name "*.mp4" -o -name "*.avi" -o -name "*.mov" 2>/dev/null | wc -l | tr -d ' ')"
    echo "  Output files: $(find outputs -name "*.mp4" 2>/dev/null | wc -l | tr -d ' ')"
    echo "  Log entries: $(wc -l < "$LOG_FILE" 2>/dev/null || echo "0")"
    
    echo
    echo -e "${CYAN}Recent Activity:${NC}"
    if [[ -f "$LOG_FILE" ]]; then
        echo "  Last 3 log entries:"
        tail -3 "$LOG_FILE" | while read -r line; do
            echo "    $line"
        done
    else
        echo "  No recent activity logged"
    fi
    
    echo
    read -p "Press Enter to continue..."
}

# Function for batch processing
batch_processing() {
    print_header
    echo -e "${WHITE}═══ BATCH PROCESSING ═══${NC}"
    echo
    
    echo "Batch Processing Options:"
    echo
    echo "  1. Process All Videos in Directory"
    echo "  2. Process Videos from File List"
    echo "  3. Batch Synthetic Data Generation"
    echo "  4. Process Project Folders"
    echo "  0. Go back"
    echo
    
    read -p "Select option [0-4]: " choice
    
    case $choice in
        1)
            print_status "Processing all videos in current directory..."
            local video_files=(*.mp4 *.avi *.mov)
            local processed=0
            local existing_files=()
            
            # Filter out non-existent files
            for file in "${video_files[@]}"; do
                if [[ -f "$file" ]]; then
                    existing_files+=("$file")
                fi
            done
            
            for video in "${existing_files[@]}"; do
                print_status "Processing: $video"
                output_name="${video%.*}_batch_detections.mp4"
                    
                # Create batch processing script
                cat > temp_batch.py << EOF
import sys
sys.path.append('.')
exec(open('video_inference.py').read().replace("'anduril_swarm.mp4'", "'$video'").replace("'anduril_swarm_detections.mp4'", "'$output_name'"))
EOF
                
                if python3 temp_batch.py; then
                    print_success "Completed: $output_name"
                    ((processed++))
                else
                    print_error "Failed: $video"
                fi
                
                rm -f temp_batch.py
            done
            
            print_success "Batch processing complete. Processed $processed videos."
            ;;
            
        2)
            read -p "Enter file list path (one video per line): " file_list
            if [[ -f "$file_list" ]]; then
                print_status "Processing videos from: $file_list"
                local processed=0
                
                while IFS= read -r video; do
                    if [[ -f "$video" ]]; then
                        print_status "Processing: $video"
                        output_name="${video%.*}_list_detections.mp4"
                        
                        cat > temp_batch.py << EOF
import sys
sys.path.append('.')
exec(open('video_inference.py').read().replace("'anduril_swarm.mp4'", "'$video'").replace("'anduril_swarm_detections.mp4'", "'$output_name'"))
EOF
                        
                        if python3 temp_batch.py; then
                            print_success "Completed: $output_name"
                            ((processed++))
                        else
                            print_error "Failed: $video"
                        fi
                        
                        rm -f temp_batch.py
                    else
                        print_warning "File not found: $video"
                    fi
                done < "$file_list"
                
                print_success "Batch processing complete. Processed $processed videos."
            else
                print_error "File list not found: $file_list"
            fi
            ;;
            
        3)
            print_status "Batch synthetic data generation..."
            if [[ -d "synthetic_drone_swarm_sim" ]]; then
                read -p "Number of datasets to generate [10]: " num_datasets
                num_datasets=${num_datasets:-10}
                
                read -p "Starting drone count [5]: " start_drones
                start_drones=${start_drones:-5}
                
                read -p "Max drone count [25]: " max_drones
                max_drones=${max_drones:-25}
                
                cd synthetic_drone_swarm_sim
                
                for ((i=1; i<=num_datasets; i++)); do
                    drone_count=$(( start_drones + (i-1) * (max_drones - start_drones) / (num_datasets - 1) ))
                    output_name="batch_synthetic_${i}_drones_${drone_count}"
                    
                    print_status "Generating dataset $i/$num_datasets ($drone_count drones)..."
                    python3 sim.py --num-drones "$drone_count" --output "outputs/${output_name}.mp4" --gen_groundtruth
                done
                
                cd ..
                print_success "Batch synthetic generation complete."
            else
                print_error "Synthetic simulation directory not found"
            fi
            ;;
            
        4)
            print_status "Processing project folders..."
            if [[ -d "projects" ]]; then
                for project_dir in projects/*/; do
                    if [[ -d "$project_dir" ]]; then
                        project_name=$(basename "$project_dir")
                        print_status "Processing project: $project_name"
                        
                        # Process any videos in project data folder
                        if [[ -d "${project_dir}data" ]]; then
                            for video in "${project_dir}data"/*.mp4; do
                                if [[ -f "$video" ]]; then
                                    output_name="${project_dir}outputs/$(basename "${video%.*}")_detections.mp4"
                                    mkdir -p "${project_dir}outputs"
                                    
                                    print_status "Processing: $(basename "$video")"
                                    # Implementation would go here
                                fi
                            done
                        fi
                    fi
                done
            else
                print_error "No projects directory found"
            fi
            ;;
            
        0)
            return
            ;;
        *)
            print_error "Invalid selection"
            ;;
    esac
    
    echo
    read -p "Press Enter to continue..."
}

# Function for configuration management
configuration_manager() {
    print_header
    echo -e "${WHITE}═══ CONFIGURATION MANAGER ═══${NC}"
    echo
    
    echo "Configuration Options:"
    echo
    echo "  1. View Current Configuration"
    echo "  2. Edit Detection Thresholds"
    echo "  3. Edit Preprocessing Settings"
    echo "  4. Model Provider Settings"
    echo "  5. Reset to Defaults"
    echo "  0. Go back"
    echo
    
    read -p "Select option [0-5]: " choice
    
    case $choice in
        1)
            print_status "Current Configuration:"
            echo
            echo -e "${CYAN}config.json:${NC}"
            if [[ -f "config.json" ]]; then
                cat config.json | python3 -m json.tool
            else
                print_error "config.json not found"
            fi
            
            echo
            echo -e "${CYAN}preprocessor.json:${NC}"
            if [[ -f "preprocessor.json" ]]; then
                cat preprocessor.json | python3 -m json.tool
            else
                print_error "preprocessor.json not found"
            fi
            ;;
            
        2)
            print_status "Current detection thresholds:"
            if [[ -f "config.json" ]]; then
                python3 -c "
import json
with open('config.json', 'r') as f:
    config = json.load(f)
print(f\"Confidence threshold: {config.get('confidence_threshold', 'Not set')}\")
print(f\"IoU threshold: {config.get('iou_threshold', 'Not set')}\")
"
                echo
                read -p "New confidence threshold (0.1-1.0) [current]: " new_conf
                read -p "New IoU threshold (0.1-1.0) [current]: " new_iou
                
                if [[ -n "$new_conf" ]] || [[ -n "$new_iou" ]]; then
                    python3 -c "
import json
with open('config.json', 'r') as f:
    config = json.load(f)

if '$new_conf':
    config['confidence_threshold'] = float('$new_conf')
if '$new_iou':
    config['iou_threshold'] = float('$new_iou')

with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)
"
                    print_success "Thresholds updated"
                fi
            else
                print_error "config.json not found"
            fi
            ;;
            
        3)
            print_status "Preprocessing settings editor..."
            if [[ -f "preprocessor.json" ]]; then
                print_status "Current settings:"
                cat preprocessor.json | python3 -m json.tool
                echo
                
                read -p "New pad size [640]: " new_pad_size
                read -p "New rescale factor [0.00392156862745098]: " new_rescale
                
                if [[ -n "$new_pad_size" ]] || [[ -n "$new_rescale" ]]; then
                    python3 -c "
import json
with open('preprocessor.json', 'r') as f:
    config = json.load(f)

if '$new_pad_size':
    config['pad_size'] = int('$new_pad_size')
if '$new_rescale':
    config['rescale_factor'] = float('$new_rescale')

with open('preprocessor.json', 'w') as f:
    json.dump(config, f, indent=2)
"
                    print_success "Preprocessing settings updated"
                fi
            else
                print_error "preprocessor.json not found"
            fi
            ;;
            
        4)
            print_status "Model Provider Settings..."
            echo "Available providers for Apple Silicon:"
            echo "  1. CoreMLExecutionProvider (recommended for M1/M2/M3)"
            echo "  2. CPUExecutionProvider (fallback)"
            echo "  3. Both (automatic selection)"
            echo
            read -p "Select provider preference [3]: " provider_choice
            provider_choice=${provider_choice:-3}
            
            case $provider_choice in
                1)
                    print_status "Set to CoreML only"
                    ;;
                2)
                    print_status "Set to CPU only"
                    ;;
                3)
                    print_status "Set to automatic selection"
                    ;;
            esac
            ;;
            
        5)
            print_status "Resetting configurations to defaults..."
            read -p "Are you sure? This will overwrite current settings (y/N): " confirm
            if [[ "$confirm" == "y" ]] || [[ "$confirm" == "Y" ]]; then
                # Create default config.json
                cat > config.json << 'EOF'
{
  "confidence_threshold": 0.3,
  "iou_threshold": 0.45,
  "model_path": "yolov11n-UAV-finetune.onnx",
  "classes": {
    "0": "dj-air3",
    "1": "uav",
    "2": "UAV"
  }
}
EOF
                
                # Create default preprocessor.json
                cat > preprocessor.json << 'EOF'
{
  "pad_size": 640,
  "rescale_factor": 0.00392156862745098
}
EOF
                
                print_success "Configurations reset to defaults"
            else
                print_status "Reset cancelled"
            fi
            ;;
            
        0)
            return
            ;;
        *)
            print_error "Invalid selection"
            ;;
    esac
    
    echo
    read -p "Press Enter to continue..."
}

# Function to show logs and reports
logs_reports() {
    print_header
    echo -e "${WHITE}═══ LOGS & REPORTS ═══${NC}"
    echo
    
    echo "Logs & Reports Options:"
    echo
    echo "  1. View Recent Logs"
    echo "  2. View Full Log File"
    echo "  3. Generate Activity Report"
    echo "  4. Clear Logs"
    echo "  5. Export Logs"
    echo "  0. Go back"
    echo
    
    read -p "Select option [0-5]: " choice
    
    case $choice in
        1)
            print_status "Recent log entries (last 20):"
            echo
            if [[ -f "$LOG_FILE" ]]; then
                tail -20 "$LOG_FILE"
            else
                print_warning "No log file found"
            fi
            ;;
            
        2)
            print_status "Full log file:"
            echo
            if [[ -f "$LOG_FILE" ]]; then
                less "$LOG_FILE"
            else
                print_warning "No log file found"
            fi
            ;;
            
        3)
            print_status "Generating activity report..."
            if [[ -f "$LOG_FILE" ]]; then
                echo
                echo -e "${CYAN}Activity Report - $(date)${NC}"
                echo "=================================="
                echo
                echo "Total log entries: $(wc -l < "$LOG_FILE")"
                echo "Success operations: $(grep -c "\[SUCCESS\]" "$LOG_FILE" || echo "0")"
                echo "Error operations: $(grep -c "\[ERROR\]" "$LOG_FILE" || echo "0")"
                echo "Warning operations: $(grep -c "\[WARNING\]" "$LOG_FILE" || echo "0")"
                echo
                echo "Recent errors:"
                grep "\[ERROR\]" "$LOG_FILE" | tail -5 || echo "No recent errors"
                echo
                echo "Report generated: $(date)"
            else
                print_warning "No log file found"
            fi
            ;;
            
        4)
            read -p "Clear all logs? (y/N): " confirm
            if [[ "$confirm" == "y" ]] || [[ "$confirm" == "Y" ]]; then
                > "$LOG_FILE"
                print_success "Logs cleared"
            else
                print_status "Clear cancelled"
            fi
            ;;
            
        5)
            read -p "Export filename [logs_export_$(date +%Y%m%d_%H%M%S).txt]: " export_name
            export_name=${export_name:-logs_export_$(date +%Y%m%d_%H%M%S).txt}
            
            if [[ -f "$LOG_FILE" ]]; then
                cp "$LOG_FILE" "$export_name"
                print_success "Logs exported to: $export_name"
            else
                print_warning "No log file found"
            fi
            ;;
            
        0)
            return
            ;;
        *)
            print_error "Invalid selection"
            ;;
    esac
    
    echo
    read -p "Press Enter to continue..."
}

# Function to show help
show_help() {
    print_header
    echo -e "${WHITE}═══ HELP & DOCUMENTATION ═══${NC}"
    echo
    
    echo -e "${CYAN}YOLOv11 UAV Detection System${NC}"
    echo "=============================="
    echo
    echo -e "${WHITE}Quick Start:${NC}"
    echo "  1. Place video files in the current directory"
    echo "  2. Use option 1 to run video inference"
    echo "  3. Check outputs directory for results"
    echo
    echo -e "${WHITE}Supported Formats:${NC}"
    echo "  Video: .mp4, .avi, .mov"
    echo "  Images: .jpg, .jpeg, .png"
    echo
    echo -e "${WHITE}Model Classes:${NC}"
    echo "  - dj-air3: DJI Air 3 drone"
    echo "  - uav: Generic UAV/drone"
    echo "  - UAV: Alternative UAV detection"
    echo
    echo -e "${WHITE}Performance Tips:${NC}"
    echo "  - Use confidence threshold 0.2-0.4 for best results"
    echo "  - CoreML provider optimized for Apple Silicon"
    echo "  - Process videos at original resolution for accuracy"
    echo
    echo -e "${WHITE}Synthetic Data:${NC}"
    echo "  - Generate training datasets with ground truth"
    echo "  - YOLO format annotations included"
    echo "  - Customizable drone counts and backgrounds"
    echo
    echo -e "${WHITE}Troubleshooting:${NC}"
    echo "  - Check system status for environment issues"
    echo "  - View logs for detailed error information"
    echo "  - Ensure virtual environment is activated"
    echo
    echo -e "${WHITE}Project Structure:${NC}"
    echo "  outputs/     - Processed videos and results"
    echo "  projects/    - Organized subprojects"
    echo "  logs/        - Operation logs and reports"
    echo
    read -p "Press Enter to continue..."
}

# Main program loop
main() {
    # Initialize log file
    echo "$(date '+%Y-%m-%d %H:%M:%S') [SYSTEM] Launcher started" >> "$LOG_FILE"
    
    # Check environment
    if ! check_venv; then
        exit 1
    fi
    
    if ! check_requirements; then
        print_error "Required files missing. Please run setup first."
        exit 1
    fi
    
    # Main loop
    while true; do
        show_main_menu
        read choice
        
        case $choice in
            1) video_inference ;;
            2) synthetic_data_generation ;;
            3) model_testing ;;
            4) project_management ;;
            5) system_status ;;
            6) batch_processing ;;
            7) configuration_manager ;;
            8) logs_reports ;;
            9) show_help ;;
            0) 
                print_status "Exiting launcher..."
                echo "$(date '+%Y-%m-%d %H:%M:%S') [SYSTEM] Launcher exited" >> "$LOG_FILE"
                echo -e "${GREEN}Goodbye! Virtual environment remains active.${NC}"
                exit 0
                ;;
            *)
                print_error "Invalid option. Please select 0-9."
                sleep 1
                ;;
        esac
    done
}

# Run main program
main "$@"