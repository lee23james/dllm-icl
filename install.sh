#!/bin/bash
# Installation script for LLaDA-ICL
# Usage: bash install.sh [cuda_version]
# Example: bash install.sh cu124

set -e

# Default CUDA version (12.4)
CUDA_VERSION=${1:-cu124}

echo "=========================================="
echo "Installing LLaDA-ICL dependencies"
echo "=========================================="
echo ""

# Initialize conda - try common locations
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    # Try to find conda and initialize
    CONDA_BASE=$(dirname $(dirname $(which conda 2>/dev/null || echo "$HOME/anaconda3/bin/conda")))
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        echo "Error: Cannot find conda. Please ensure conda is installed and initialized."
        exit 1
    fi
fi

# Step 1: Initialize conda and create Python 3.11 conda environment
echo "Step 1: Creating Python 3.11 conda environment 'llada-icl'..."

# Check if environment already exists
if conda env list | grep -q "^llada-icl "; then
    echo "Conda environment 'llada-icl' already exists. Activating it..."
    conda activate llada-icl
else
    echo "Creating new conda environment..."
    # Try to create environment with retry and better error handling
    set +e  # Temporarily disable exit on error for retry logic
    MAX_RETRIES=3
    RETRY_COUNT=0
    SUCCESS=0
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        conda create -n llada-icl python=3.11 -y --offline 2>/dev/null
        if [ $? -eq 0 ]; then
            SUCCESS=1
            break
        fi
        
        # Try without offline mode
        conda create -n llada-icl python=3.11 -y 2>&1 | grep -v "CondaHTTPError\|HTTP 000\|Failed to resolve" || true
        if [ $? -eq 0 ]; then
            SUCCESS=1
            break
        fi
        
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo "Retry $RETRY_COUNT/$MAX_RETRIES..."
            sleep 2
        fi
    done
    
    set -e  # Re-enable exit on error
    
    if [ $SUCCESS -eq 0 ]; then
        echo "Warning: Failed to create conda environment after $MAX_RETRIES attempts."
        echo "Attempting to create environment using local Python..."
        # Fallback: use system Python 3.11 if available
        if command -v python3.11 &> /dev/null; then
            conda create -n llada-icl python=3.11 -y --dry-run 2>/dev/null || \
            conda create -n llada-icl python=3.11 -y --no-channel-priority 2>/dev/null || \
            echo "Please check your network connection and conda configuration."
        fi
    fi
    
    conda activate llada-icl
    echo "Conda environment 'llada-icl' activated."
fi

# Step 2: Upgrade pip
echo ""
echo "Step 2: Upgrading pip..."
pip install --upgrade pip

# Step 3: Install PyTorch with CUDA support
echo ""
echo "Step 3: Installing PyTorch with CUDA ${CUDA_VERSION}..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

# Step 4: Install other dependencies
echo ""
echo "Step 4: Installing other dependencies..."
pip install -r requirements.txt

# Step 5: Verify installation
echo ""
echo "Step 5: Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=========================================="
echo "Installation completed!"
echo "=========================================="
echo ""
echo "To activate the conda environment in the future, run:"
echo "  conda activate llada-icl"

