#!/bin/bash

# ============================================================================
# Deep Learning Environment Installation Script
# Supports: macOS (ARM64) and Linux (x86_64)
# Python Version: 3.13
# 
# AUTOMATIC GPU DETECTION:
# - Detects NVIDIA GPUs via nvidia-smi and CUDA version
# - Checks CUDA compatibility with TensorFlow (max CUDA 12.3, prompts for tf-nightly on CUDA 13)
# - Installs CUDA-enabled PyTorch (13.0/12.1/11.8) automatically when GPU is detected
# - Falls back to CPU/MPS versions when no NVIDIA GPU is found or CUDA too new
# ============================================================================

set -e

# ============================================================================
# COLOR DEFINITIONS
# ============================================================================
RESET='\033[0m'
BOLD='\033[1m'
DIM='\033[2m'

# Regular colors
BLACK='\033[0;30m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[0;37m'

# Bold colors
BRED='\033[1;31m'
BGREEN='\033[1;32m'
BYELLOW='\033[1;33m'
BBLUE='\033[1;34m'
BMAGENTA='\033[1;35m'
BCYAN='\033[1;36m'
BWHITE='\033[1;37m'

# Background colors
BG_BLACK='\033[40m'
BG_RED='\033[41m'
BG_GREEN='\033[42m'
BG_YELLOW='\033[43m'
BG_BLUE='\033[44m'
BG_MAGENTA='\033[45m'
BG_CYAN='\033[46m'
BG_WHITE='\033[47m'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_banner() {
    echo -e "${BCYAN}"
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                    ║"
    echo "║          🚀 Deep Learning Environment Installer 🚀                 ║"
    echo "║                                                                    ║"
    echo "║          Anaconda + Python 3.13 + DL Packages                      ║"
    echo "║                                                                    ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo -e "${RESET}"
    echo ""
}

print_section() {
    echo ""
    echo -e "${BMAGENTA}═══════════════════════════════════════════════════════════════════${RESET}"
    echo -e "${BMAGENTA}  $1${RESET}"
    echo -e "${BMAGENTA}═══════════════════════════════════════════════════════════════════${RESET}"
    echo ""
}

print_step() {
    echo -e "${BCYAN}▶${RESET} ${BWHITE}$1${RESET}"
}

print_success() {
    echo -e "${BGREEN}✓${RESET} ${GREEN}$1${RESET}"
}

print_warning() {
    echo -e "${BYELLOW}⚠${RESET} ${YELLOW}$1${RESET}"
}

print_error() {
    echo -e "${BRED}✗${RESET} ${RED}$1${RESET}"
}

print_info() {
    echo -e "${BBLUE}ℹ${RESET} ${BLUE}$1${RESET}"
}

# Progress bar function
show_progress() {
    local duration=$1
    local message=$2
    local steps=50
    local delay=$(awk "BEGIN {print $duration/$steps}")
    
    echo -ne "${CYAN}${message}${RESET} ["
    
    for ((i=0; i<steps; i++)); do
        echo -ne "${BGREEN}█${RESET}"
        sleep $delay
    done
    
    echo -e "] ${BGREEN}✓${RESET}"
}

# Spinner function for longer operations
spinner() {
    local pid=$1
    local message=$2
    local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    local i=0
    
    echo -ne "${CYAN}${message}${RESET} "
    
    while kill -0 $pid 2>/dev/null; do
        i=$(( (i+1) %10 ))
        echo -ne "\r${CYAN}${message}${RESET} ${BMAGENTA}${spin:$i:1}${RESET}"
        sleep 0.1
    done
    
    wait $pid
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "\r${CYAN}${message}${RESET} ${BGREEN}✓${RESET}"
        return 0
    else
        echo -e "\r${CYAN}${message}${RESET} ${BRED}✗${RESET}"
        return 1
    fi
}

# ============================================================================
# GPU DETECTION
# ============================================================================

detect_gpu() {
    print_section "GPU Detection"
    
    HAS_NVIDIA_GPU=false
    CUDA_VERSION=""
    CUDA_MAJOR=""
    CUDA_MINOR=""
    TF_COMPATIBLE_GPU=true
    
    print_step "Checking for NVIDIA GPU..."
    
    if command -v nvidia-smi &> /dev/null; then
        # nvidia-smi exists, try to get CUDA version
        if nvidia-smi &> /dev/null; then
            HAS_NVIDIA_GPU=true
            
            # Extract CUDA version from nvidia-smi output
            CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
            
            print_success "NVIDIA GPU detected!"
            
            if [ -n "$CUDA_VERSION" ]; then
                print_info "CUDA Driver Version: $CUDA_VERSION"
                
                # Parse major and minor version
                CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
                CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
                
                # TensorFlow 2.20.0 supports CUDA 12.3 (requires driver >= 525.60.13)
                # If CUDA driver is 13.0 or higher, TensorFlow may not be compatible
                if [ "$CUDA_MAJOR" -ge 13 ]; then
                    print_warning "CUDA $CUDA_VERSION detected - too new for TensorFlow CUDA support"
                    print_info "TensorFlow will be installed in CPU-only mode"
                    print_info "PyTorch will still use GPU acceleration (CUDA $CUDA_MAJOR compatible)"
                    TF_COMPATIBLE_GPU=false
                elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -gt 3 ]; then
                    print_warning "CUDA $CUDA_VERSION detected - may have limited TensorFlow support"
                    print_info "TensorFlow officially supports CUDA 12.3, attempting installation anyway"
                    TF_COMPATIBLE_GPU=true
                else
                    print_success "CUDA $CUDA_VERSION is compatible with TensorFlow"
                    TF_COMPATIBLE_GPU=true
                fi
            fi
            
            # Show GPU details
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
            if [ -n "$GPU_NAME" ]; then
                print_info "GPU: $GPU_NAME"
            fi
            
        else
            print_warning "nvidia-smi found but failed to run"
            HAS_NVIDIA_GPU=false
        fi
    else
        print_info "No NVIDIA GPU detected (nvidia-smi not found)"
        HAS_NVIDIA_GPU=false
    fi
    
    export HAS_NVIDIA_GPU
    export CUDA_VERSION
    export CUDA_MAJOR
    export TF_COMPATIBLE_GPU
}

# ============================================================================
# SYSTEM DETECTION
# ============================================================================

detect_os() {
    print_section "System Detection"
    
    OS_TYPE=$(uname -s)
    ARCH=$(uname -m)
    
    print_step "Detecting operating system..."
    sleep 0.5
    
    if [[ "$OS_TYPE" == "Darwin" ]]; then
        if [[ "$ARCH" == "arm64" ]]; then
            OS="macos"
            ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2025.06-0-MacOSX-arm64.sh"
            print_success "macOS (Apple Silicon) detected"
        else
            print_error "macOS Intel not supported by provided installer"
            exit 1
        fi
    elif [[ "$OS_TYPE" == "Linux" ]]; then
        OS="linux"
        ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh"
        print_success "Linux (x86_64) detected"
    elif [[ "$OS_TYPE" == MINGW* ]] || [[ "$OS_TYPE" == MSYS* ]] || [[ "$OS_TYPE" == CYGWIN* ]]; then
        OS="windows"
        ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Windows-x86_64.exe"
        print_success "Windows detected"
        print_info "Running in $(echo $OS_TYPE | cut -d'_' -f1) environment"
    else
        print_error "Unsupported operating system: $OS_TYPE"
        exit 1
    fi
    
    print_info "Architecture: $ARCH"
    
    # Check for NVIDIA GPU on all platforms
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
        if [ $? -eq 0 ]; then
            HAS_NVIDIA_GPU=true
            print_success "NVIDIA GPU detected: $GPU_INFO"
        else
            HAS_NVIDIA_GPU=false
        fi
    else
        HAS_NVIDIA_GPU=false
    fi
}

# ============================================================================
# ANACONDA INSTALLATION
# ============================================================================

check_anaconda() {
    print_section "Checking Anaconda Installation"
    
    if command -v conda &> /dev/null; then
        CONDA_VERSION=$(conda --version 2>/dev/null || echo "unknown")
        print_success "Anaconda is already installed: $CONDA_VERSION"
        
        # Source conda for Unix systems
        if [[ "$OS" != "windows" ]]; then
            if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
                source "$HOME/anaconda3/etc/profile.d/conda.sh"
            elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
                source "$HOME/miniconda3/etc/profile.d/conda.sh"
            fi
        fi
        
        return 0
    else
        print_warning "Anaconda not found"
        return 1
    fi
}

install_anaconda() {
    print_section "Installing Anaconda"
    
    if [[ "$OS" == "windows" ]]; then
        INSTALLER_NAME="anaconda_installer.exe"
    else
        INSTALLER_NAME="anaconda_installer.sh"
    fi
    
    print_step "Downloading Anaconda installer..."
    echo -e "${DIM}URL: $ANACONDA_URL${RESET}"
    
    if curl -L -o "$INSTALLER_NAME" "$ANACONDA_URL" 2>&1 | {
        while IFS= read -r line; do
            if [[ $line =~ ([0-9]+)\.?[0-9]* ]]; then
                percent=${BASH_REMATCH[1]}
                if [[ $percent =~ ^[0-9]+$ ]] && [ $percent -le 100 ]; then
                    echo -ne "\r${CYAN}Downloading${RESET} ["
                    filled=$((percent / 2))
                    empty=$((50 - filled))
                    printf "%${filled}s" | tr ' ' '█'
                    printf "%${empty}s" | tr ' ' '░'
                    echo -ne "] ${BGREEN}${percent}%%${RESET}"
                fi
            fi
        done
        echo ""
    }; then
        print_success "Download completed"
    else
        print_error "Download failed"
        exit 1
    fi
    
    print_step "Installing Anaconda (this may take several minutes)..."
    
    if [[ "$OS" == "windows" ]]; then
        # Windows installation
        print_warning "Please complete the Anaconda installation manually:"
        print_info "1. Run the installer: $INSTALLER_NAME"
        print_info "2. Choose 'Just Me' installation"
        print_info "3. Install to: $HOME/anaconda3"
        print_info "4. Check 'Add Anaconda to PATH' option"
        print_info "5. Press Enter here after installation completes..."
        read -p ""
        
        # Check if installation succeeded
        if [ -d "$HOME/anaconda3" ] || [ -d "$HOME/Anaconda3" ]; then
            print_success "Anaconda installation detected"
            # Try to find conda in common locations
            if [ -f "$HOME/anaconda3/Scripts/conda.exe" ]; then
                export PATH="$HOME/anaconda3/Scripts:$HOME/anaconda3:$PATH"
            elif [ -f "$HOME/Anaconda3/Scripts/conda.exe" ]; then
                export PATH="$HOME/Anaconda3/Scripts:$HOME/Anaconda3:$PATH"
            fi
        else
            print_error "Anaconda installation not found"
            exit 1
        fi
    else
        # Unix installation (macOS/Linux)
        bash "$INSTALLER_NAME" -b -p "$HOME/anaconda3" &> /tmp/anaconda_install.log &
        spinner $! "Installing Anaconda"
        
        if [ $? -eq 0 ]; then
            print_success "Anaconda installed successfully"
        else
            print_error "Anaconda installation failed"
            cat /tmp/anaconda_install.log
            exit 1
        fi
    fi
    
    # Clean up installer
    rm -f "$INSTALLER_NAME"
    
    # Initialize conda for bash and zsh
    print_step "Initializing conda for bash and zsh..."
    
    # Initialize for bash
    "$HOME/anaconda3/bin/conda" init bash &> /dev/null
    
    # Initialize for zsh (common on macOS)
    "$HOME/anaconda3/bin/conda" init zsh &> /dev/null
    
    # Source conda for current session
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
    
    print_success "Conda initialized for bash and zsh"
    
    # Detect current shell and provide specific instructions
    CURRENT_SHELL=$(basename "$SHELL")
    if [[ "$CURRENT_SHELL" == "zsh" ]]; then
        print_info "Detected zsh shell"
        print_warning "After installation completes, you MUST do ONE of:"
        echo -e "  ${BCYAN}Option 1:${RESET} Restart your terminal (recommended)"
        echo -e "  ${BCYAN}Option 2:${RESET} Run: ${BGREEN}source ~/.zshrc${RESET}"
    elif [[ "$CURRENT_SHELL" == "bash" ]]; then
        print_info "Detected bash shell"
        print_warning "After installation completes, you MUST do ONE of:"
        echo -e "  ${BCYAN}Option 1:${RESET} Restart your terminal (recommended)"
        echo -e "  ${BCYAN}Option 2:${RESET} Run: ${BGREEN}source ~/.bashrc${RESET}"
    fi
}

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

create_environment() {
    print_section "Creating Python 3.13 Environment"
    
    print_step "Checking if 'DL' environment already exists..."
    
    if conda env list | grep -q "^DL "; then
        print_warning "Environment 'DL' already exists"
        read -p "$(echo -e ${BYELLOW}Do you want to remove and recreate it? [y/N]:${RESET} )" -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_step "Removing existing environment..."
            conda env remove -n DL -y &> /dev/null
            print_success "Environment removed"
        else
            print_info "Using existing environment"
            return 0
        fi
    fi
    
    # conda tos accept --override-channels --channel defaults
    print_step "Creating new conda environment 'DL' with Python 3.13..."
    conda create -n DL python=3.13 -y &> /tmp/conda_create.log &
    spinner $! "Creating environment"
    
    if [ $? -eq 0 ]; then
        print_success "Environment 'DL' created successfully"
    else
        print_error "Failed to create environment"
        cat /tmp/conda_create.log
        exit 1
    fi
}

activate_environment() {
    print_step "Activating environment 'DL'..."
    # conda activate DL
    print_success "Environment activated"
    print_info "Python version: $(python --version)"
}

# ============================================================================
# PACKAGE INSTALLATION
# ============================================================================

install_package() {
    local package=$1
    local display_name=${2:-$1}
    
    echo -ne "${CYAN}Installing ${display_name}${RESET} "
    
    if $package &> /tmp/install_${display_name//[ \/]/_}.log; then
        echo -e "${BGREEN}✓${RESET}"
        return 0
    else
        echo -e "${BYELLOW}⚠ (skipped)${RESET}"
        print_warning "Failed to install ${display_name}, continuing..."
        return 1
    fi
}

install_packages() {
    print_section "Installing Python Packages"
    
    print_step "Upgrading pip..."
    pip install --upgrade pip &> /tmp/pip_upgrade.log
    print_success "Pip upgraded"
    
    echo ""
    print_info "Installing data science packages..."
    echo ""
    
    # Data science basics
    install_package "pip install numpy" "numpy"
    install_package "pip install pandas" "pandas"
    install_package "pip install scipy" "scipy"
    install_package "pip install matplotlib" "matplotlib"
    install_package "pip install seaborn" "seaborn"
    install_package "pip install plotly" "plotly"
    install_package "pip install scikit-learn" "scikit-learn"
    install_package "pip install ipywidgets" "ipywidgets"
    install_package "pip install ipykernel" "ipykernel"
    install_package "pip install anthropic" "anthropic"
    # install_package "pip install xgboost" "xgboost"
    install_package "pip install statsmodels" "statsmodels"
    install_package "pip install NetworkX" "NetworkX"
    install_package "pip install nbconvert" "nbconvert"
    install_package "pip install pandoc" "pandoc"
    install_package "pip install lightgbm" "lightgbm"
    install_package "pip install graphviz" "graphviz"
    install_package "pip install datasets" "datasets"
    
    echo ""
    print_info "Installing deep learning frameworks..."
    echo ""
    
    # PyTorch installation (different for macOS, Linux with GPU, Windows with GPU, and CPU-only)
    if [[ "$OS" == "macos" ]]; then
        print_step "Installing PyTorch for macOS (with MPS support)..."
        if pip3 install torch torchvision &> /tmp/install_pytorch.log; then
            print_success "PyTorch installed (MPS-enabled for Apple Silicon)"
        else
            print_warning "PyTorch installation failed, skipping..."
        fi
    elif [[ "$HAS_NVIDIA_GPU" == true ]]; then
        # NVIDIA GPU detected - install CUDA version
        print_step "Installing PyTorch with CUDA support..."
        print_info "NVIDIA GPU detected (CUDA ${CUDA_VERSION:-unknown})"

        # Check for CUDA 13.x on Linux and install appropriate version
        if [[ "$CUDA_MAJOR" -eq 13 ]] && [[ "$OS" == "linux" ]]; then
            print_step "CUDA 13.x detected on Linux - using CUDA 13.0 build..."
            if pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130 &> /tmp/install_pytorch.log; then
                print_success "PyTorch installed (CUDA 13.0 enabled)"
            else
                print_warning "CUDA 13.0 failed, falling back to CUDA 12.1..."
                if pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121 &> /tmp/install_pytorch_cu121.log; then
                    print_success "PyTorch installed (CUDA 12.1 enabled)"
                else
                    print_warning "CUDA 12.1 failed, trying CPU version..."
                    if pip3 install torch torchvision &> /tmp/install_pytorch_cpu.log; then
                        print_success "PyTorch (CPU) installed"
                    else
                        print_warning "PyTorch installation skipped"
                    fi
                fi
            fi
        # Try CUDA 12.1 first (compatible with CUDA 12.x drivers)
        elif pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121 &> /tmp/install_pytorch.log; then
            print_success "PyTorch installed (CUDA 12.1 enabled)"
        else
            print_warning "CUDA 12.1 failed, trying CUDA 11.8..."
            if pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118 &> /tmp/install_pytorch_cu118.log; then
                print_success "PyTorch installed (CUDA 11.8 enabled)"
            else
                print_warning "CUDA versions failed, trying CPU version..."
                if pip3 install torch torchvision &> /tmp/install_pytorch_cpu.log; then
                    print_success "PyTorch (CPU) installed"
                else
                    print_warning "PyTorch installation skipped"
                fi
            fi
        fi
    else
        # CPU-only installation for systems without NVIDIA GPU
        print_step "Installing PyTorch (CPU version)..."
        print_info "No NVIDIA GPU detected - installing CPU version"
        if pip3 install torch torchvision &> /tmp/install_pytorch.log; then
            print_success "PyTorch (CPU) installed"
        else
            print_warning "PyTorch installation failed, skipping..."
        fi
    fi
    
    # TensorFlow installation (based on GPU detection and CUDA version compatibility)
    if [[ "$OS" == "macos" ]]; then
        print_step "Installing TensorFlow for macOS (with Metal support)..."
        if python3 -m pip install tensorflow &> /tmp/install_tensorflow.log; then
            print_success "TensorFlow installed (Metal-enabled)"
        else
            print_warning "TensorFlow installation failed, continuing..."
        fi
    elif [[ "$HAS_NVIDIA_GPU" == true ]] && [[ "$TF_COMPATIBLE_GPU" == true ]]; then
        # GPU detected and CUDA version is compatible with TensorFlow
        print_step "Installing TensorFlow with CUDA support..."
        print_info "CUDA $CUDA_VERSION is compatible - installing GPU version"
        if [[ "$OS" == "linux" ]]; then
            # Use python3 -m pip for Linux with CUDA
            if python3 -m pip install 'tensorflow[and-cuda]' &> /tmp/install_tensorflow.log; then
                print_success "TensorFlow with CUDA installed"

                # Test if TensorFlow with CUDA works
                print_step "Testing TensorFlow CUDA installation..."
                if python3 -c "import tensorflow as tf; tf.config.list_physical_devices('GPU')" &> /tmp/test_tensorflow_cuda.log; then
                    print_success "TensorFlow CUDA test passed"
                else
                    print_warning "TensorFlow CUDA test failed - reverting to CPU version"
                    print_step "Uninstalling TensorFlow with CUDA..."
                    python3 -m pip uninstall -y tensorflow tensorflow-cuda &> /tmp/uninstall_tensorflow.log

                    print_step "Installing TensorFlow CPU version..."
                    if python3 -m pip install tensorflow &> /tmp/install_tensorflow_cpu.log; then
                        print_success "TensorFlow (CPU) installed"
                    else
                        print_warning "TensorFlow CPU installation failed, continuing..."
                    fi
                fi
            else
                print_warning "TensorFlow with CUDA failed, trying CPU version..."
                if python3 -m pip install tensorflow &> /tmp/install_tensorflow_standard.log; then
                    print_success "TensorFlow (CPU) installed"
                else
                    print_warning "TensorFlow installation failed, continuing..."
                fi
            fi
        else
            # Use regular pip for non-Linux systems
            if pip install tensorflow[and-cuda] &> /tmp/install_tensorflow.log; then
                print_success "TensorFlow with CUDA installed"

                # Test if TensorFlow with CUDA works
                print_step "Testing TensorFlow CUDA installation..."
                if python -c "import tensorflow as tf; tf.config.list_physical_devices('GPU')" &> /tmp/test_tensorflow_cuda.log; then
                    print_success "TensorFlow CUDA test passed"
                else
                    print_warning "TensorFlow CUDA test failed - reverting to CPU version"
                    print_step "Uninstalling TensorFlow with CUDA..."
                    pip uninstall -y tensorflow tensorflow-cuda &> /tmp/uninstall_tensorflow.log

                    print_step "Installing TensorFlow CPU version..."
                    if python3 -m pip install tensorflow &> /tmp/install_tensorflow_cpu.log; then
                        print_success "TensorFlow (CPU) installed"
                    else
                        print_warning "TensorFlow CPU installation failed, continuing..."
                    fi
                fi
            else
                print_warning "TensorFlow with CUDA failed, trying CPU version..."
                if python3 -m pip install tensorflow &> /tmp/install_tensorflow_standard.log; then
                    print_success "TensorFlow (CPU) installed"
                else
                    print_warning "TensorFlow installation failed, continuing..."
                fi
            fi
        fi
    elif [[ "$HAS_NVIDIA_GPU" == true ]] && [[ "$TF_COMPATIBLE_GPU" == false ]]; then
        # GPU detected but CUDA version too new for TensorFlow
        if [[ "$CUDA_MAJOR" -eq 13 ]]; then
            # CUDA 13 detected - offer tf-nightly option
            print_warning "CUDA $CUDA_VERSION detected - not officially supported by stable TensorFlow"
            print_info "You can try the nightly build which may have CUDA 13 support"
            echo ""
            read -p "$(echo -e ${BYELLOW}Do you want to try tf-nightly[and-cuda] for CUDA 13 support? [y/N]:${RESET} )" -n 1 -r
            echo

            if [[ $REPLY =~ ^[Yy]$ ]]; then
                print_step "Installing TensorFlow nightly with CUDA support..."
                if python3 -m pip install 'tf-nightly[and-cuda]' &> /tmp/install_tensorflow_nightly.log; then
                    print_success "TensorFlow nightly with CUDA installed"

                    # Test if TensorFlow nightly with CUDA works
                    print_step "Testing TensorFlow CUDA installation..."
                    if python3 -c "import tensorflow as tf; tf.config.list_physical_devices('GPU')" &> /tmp/test_tensorflow_nightly_cuda.log; then
                        print_success "TensorFlow CUDA test passed - GPU support enabled!"
                    else
                        print_warning "TensorFlow CUDA test failed - reverting to CPU version"
                        print_step "Uninstalling TensorFlow nightly..."
                        python3 -m pip uninstall -y tf-nightly tensorflow-cuda &> /tmp/uninstall_tensorflow_nightly.log

                        print_step "Installing TensorFlow CPU version..."
                        if python3 -m pip install tensorflow &> /tmp/install_tensorflow_cpu.log; then
                            print_success "TensorFlow (CPU) installed"
                        else
                            print_warning "TensorFlow CPU installation failed, continuing..."
                        fi
                    fi
                else
                    print_warning "TensorFlow nightly installation failed, falling back to CPU version..."
                    if python3 -m pip install tensorflow &> /tmp/install_tensorflow_cpu.log; then
                        print_success "TensorFlow (CPU) installed"
                    else
                        print_warning "TensorFlow installation failed, continuing..."
                    fi
                fi
            else
                print_step "Installing TensorFlow (CPU version)..."
                print_info "Note: PyTorch will still use your GPU"
                if python3 -m pip install tensorflow &> /tmp/install_tensorflow.log; then
                    print_success "TensorFlow (CPU) installed"
                else
                    print_warning "TensorFlow installation failed, continuing..."
                fi
            fi
        else
            # CUDA version too new but not CUDA 13
            print_step "Installing TensorFlow (CPU version)..."
            print_warning "CUDA $CUDA_VERSION too new for TensorFlow - using CPU version"
            print_info "Note: PyTorch will still use your GPU"
            if python3 -m pip install tensorflow &> /tmp/install_tensorflow.log; then
                print_success "TensorFlow (CPU) installed"
            else
                print_warning "TensorFlow installation failed, continuing..."
            fi
        fi
    else
        # No GPU - install CPU version
        print_step "Installing TensorFlow (CPU version)..."
        if python3 -m pip install tensorflow &> /tmp/install_tensorflow.log; then
            print_success "TensorFlow (CPU) installed"
        else
            print_warning "TensorFlow installation failed, continuing..."
        fi
    fi
    
    # Keras
    install_package "pip install keras" "keras"
    
    # Display GPU information
    echo ""
    if [[ "$HAS_NVIDIA_GPU" == true ]]; then
        print_success "NVIDIA GPU support enabled"
        print_info "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
    else
        print_info "No NVIDIA GPU detected (CPU-only or MPS on macOS)"
    fi
}

# ============================================================================
# HUGGING FACE CLI INSTALLATION
# ============================================================================

install_huggingface_cli() {
    print_section "Installing Hugging Face CLI"

    print_step "Checking if Hugging Face CLI is already installed..."

    if command -v hf &> /dev/null; then
        HF_VERSION=$(hf --version 2>/dev/null || echo "unknown")
        print_success "Hugging Face CLI is already installed: $HF_VERSION"
        return 0
    else
        print_info "Hugging Face CLI not found, installing..."
    fi

    print_step "Downloading and installing Hugging Face CLI..."
    echo -e "${DIM}Using: curl -LsSf https://hf.co/cli/install.sh | bash${RESET}"

    if curl -LsSf https://hf.co/cli/install.sh | bash &> /tmp/hf_install.log; then
        print_success "Hugging Face CLI installed successfully"

        # Add to PATH for current session
        if [[ -f "$HOME/.cargo/bin/hf" ]]; then
            export PATH="$HOME/.cargo/bin:$PATH"
            print_info "Added to PATH: $HOME/.cargo/bin"
        fi

        # Verify installation
        if command -v hf &> /dev/null; then
            HF_VERSION=$(hf --version 2>/dev/null || echo "installed")
            print_success "Verification passed: $HF_VERSION"
        else
            print_warning "Installation succeeded but 'hf' command not immediately available"
            print_info "You may need to restart your terminal or source your shell config"
        fi
    else
        print_error "Failed to install Hugging Face CLI"
        cat /tmp/hf_install.log
        print_warning "Continuing with rest of installation..."
    fi
}

# ============================================================================
# VERIFICATION
# ============================================================================

verify_installation() {
    print_section "Verification"
    
    print_step "Verifying installed packages..."
    echo ""
    
    # Test imports
    python -c "
import sys
packages = {
    'NumPy': 'numpy',
    'Pandas': 'pandas',
    'Matplotlib': 'matplotlib',
    'Seaborn': 'seaborn',
    'Plotly': 'plotly',
    'Scikit-learn': 'sklearn',
    'SciPy': 'scipy',
    'Anthropic': 'anthropic',
    'IPyWidgets': 'ipywidgets',
    'IPyKernel': 'ipykernel',
    'StatsModels': 'statsmodels',
    'NetworkX': 'networkx',
}

deep_learning = {
    'PyTorch': 'torch',
    'TorchVision': 'torchvision',
    'TensorFlow': 'tensorflow',
    'Keras': 'keras',
}

print('${BWHITE}Core Packages:${RESET}')
for name, module in packages.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f'  ${BGREEN}✓${RESET} {name:<15} {version}')
    except ImportError:
        print(f'  ${BYELLOW}✗${RESET} {name:<15} not installed')

print()
print('${BWHITE}Deep Learning Frameworks:${RESET}')
for name, module in deep_learning.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f'  ${BGREEN}✓${RESET} {name:<15} {version}')
        
        # Check CUDA/MPS for PyTorch
        if module == 'torch' and hasattr(mod, 'cuda'):
            cuda_available = mod.cuda.is_available()
            mps_available = hasattr(mod.backends, 'mps') and mod.backends.mps.is_available()
            
            if cuda_available:
                print(f'    ${BGREEN}→${RESET} CUDA available: {mod.cuda.get_device_name(0)}')
            elif mps_available:
                print(f'    ${BGREEN}→${RESET} MPS (Apple Silicon GPU) available')
            else:
                print(f'    ${BYELLOW}→${RESET} CPU only (no GPU acceleration)')
        
        # Check GPU for TensorFlow
        if module == 'tensorflow':
            try:
                gpus = mod.config.list_physical_devices('GPU')
                if gpus:
                    print(f'    ${BGREEN}→${RESET} GPU available: {len(gpus)} device(s)')
                    for gpu in gpus:
                        print(f'      ${BGREEN}•${RESET} {gpu.name}')
                else:
                    print(f'    ${BYELLOW}→${RESET} No GPU detected (CPU only)')
            except:
                print(f'    ${BYELLOW}→${RESET} GPU status unknown')
                
    except ImportError:
        print(f'  ${BYELLOW}✗${RESET} {name:<15} not installed')
"
    
    echo ""
    print_success "Verification complete"
}

# ============================================================================
# COMPLETION MESSAGE
# ============================================================================

print_completion() {
    echo ""
    print_section "Installation Complete! 🎉"
    
    echo -e "${BGREEN}"
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                    ║"
    echo "║                  ✨ Setup Successful! ✨                           ║"
    echo "║                                                                    ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo -e "${RESET}"
    
    echo ""
    print_info "Your Deep Learning environment is ready to use!"
    echo ""
    
    echo -e "${BWHITE}To activate the environment:${RESET}"
    echo -e "  ${BCYAN}conda activate DL${RESET}"
    echo ""
    
    echo -e "${BWHITE}To deactivate:${RESET}"
    echo -e "  ${BCYAN}conda deactivate${RESET}"
    echo ""
    
    echo -e "${BWHITE}To verify your setup:${RESET}"
    echo -e "  ${BCYAN}python -c 'import torch; print(torch.__version__)'${RESET}"
    echo -e "  ${BCYAN}python -c 'import tensorflow; print(tensorflow.__version__)'${RESET}"
    echo ""
    
    if [[ "$OS" == "macos" ]]; then
        echo -e "${BWHITE}To check MPS (GPU) support:${RESET}"
        echo -e "  ${BCYAN}python -c 'import torch; print(torch.backends.mps.is_available())'${RESET}"
    else
        echo -e "${BWHITE}To check CUDA (GPU) support:${RESET}"
        echo -e "  ${BCYAN}python -c 'import torch; print(torch.cuda.is_available())'${RESET}"
        echo -e "  ${BCYAN}python -c 'import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))'${RESET}"
        echo -e "  ${BCYAN}nvidia-smi${RESET}"
    fi
    echo ""
    
    # Shell-specific instructions
    CURRENT_SHELL=$(basename "$SHELL")
    echo -e "${BRED}═══════════════════════════════════════════════════════════════════${RESET}"
    echo -e "${BRED}                 ⚠️  IMPORTANT - READ THIS ⚠️                      ${RESET}"
    echo -e "${BRED}═══════════════════════════════════════════════════════════════════${RESET}"
    echo ""
    
    if [[ "$CURRENT_SHELL" == "zsh" ]]; then
        echo -e "${BWHITE}Your zsh shell has been configured, but you need to reload it:${RESET}"
        echo ""
        echo -e "${BGREEN}OPTION 1 (Recommended):${RESET}"
        echo -e "  ${BCYAN}1.${RESET} Close this terminal completely"
        echo -e "  ${BCYAN}2.${RESET} Open a NEW terminal"
        echo -e "  ${BCYAN}3.${RESET} Run: ${BGREEN}conda activate DL${RESET}"
        echo ""
        echo -e "${BGREEN}OPTION 2 (Quick - for current terminal only):${RESET}"
        echo -e "  Run this command now:"
        echo -e "  ${BG_GREEN}${BLACK} source ~/.zshrc ${RESET}"
        echo -e "  Then run: ${BGREEN}conda activate DL${RESET}"
    elif [[ "$CURRENT_SHELL" == "bash" ]]; then
        echo -e "${BWHITE}Your bash shell has been configured, but you need to reload it:${RESET}"
        echo ""
        echo -e "${BGREEN}OPTION 1 (Recommended):${RESET}"
        echo -e "  ${BCYAN}1.${RESET} Close this terminal completely"
        echo -e "  ${BCYAN}2.${RESET} Open a NEW terminal"
        echo -e "  ${BCYAN}3.${RESET} Run: ${BGREEN}conda activate DL${RESET}"
        echo ""
        echo -e "${BGREEN}OPTION 2 (Quick - for current terminal only):${RESET}"
        echo -e "  Run this command now:"
        echo -e "  ${BG_GREEN}${BLACK} source ~/.bashrc ${RESET}"
        echo -e "  Then run: ${BGREEN}conda activate DL${RESET}"
    else
        echo -e "${BYELLOW}Restart your terminal for conda to work${RESET}"
    fi
    echo ""
    
    echo -e "${BRED}═══════════════════════════════════════════════════════════════════${RESET}"
    echo ""
    
    echo -e "${DIM}═══════════════════════════════════════════════════════════════════${RESET}"
    echo ""
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    clear
    print_banner
    
    # Detect OS
    detect_os
    
    # Detect GPU
    detect_gpu
    
    # Check and install Anaconda if needed
    if ! check_anaconda; then
        install_anaconda
    fi
    
    # Create and activate environment
    create_environment
    activate_environment
    
    # Install packages
    install_packages

    # Install Hugging Face CLI
    install_huggingface_cli

    # Verify installation
    verify_installation
    
    # Print completion message
    print_completion
}

# Run main function
main