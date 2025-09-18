#!/bin/bash

# Textile Pattern Matching System - Environment Setup Script
# This script sets up the complete environment for the textile matching system

echo "=========================================="
echo "Textile Pattern Matching System Setup"
echo "=========================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed. Please install Miniconda or Anaconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo ""
echo "Step 1: Creating conda environment..."
conda create -n textile_matching python=3.11 -y

# Activate environment
echo ""
echo "Step 2: Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate textile_matching

# Check CUDA availability
echo ""
echo "Step 3: Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA GPU detected. Installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "No CUDA GPU detected. Installing CPU-only PyTorch..."
    pip install torch torchvision
fi

# Install requirements
echo ""
echo "Step 4: Installing required packages..."
pip install -r requirements.txt

# Create directory structure
echo ""
echo "Step 5: Creating directory structure..."
mkdir -p source_images
mkdir -p embeddings_tta
mkdir -p mappings
mkdir -p queries

# Check if directories were created
if [ -d "source_images" ] && [ -d "embeddings_tta" ] && [ -d "mappings" ] && [ -d "queries" ]; then
    echo "✓ Directories created successfully"
else
    echo "✗ Error creating directories"
    exit 1
fi

# Set up HuggingFace token if provided
echo ""
echo "Step 6: HuggingFace Setup..."
if [ -n "$HF_TOKEN" ]; then
    echo "HF_TOKEN environment variable detected."
    huggingface-cli login --token $HF_TOKEN --add-to-git-credential
    echo "✓ Logged into HuggingFace"
else
    echo "No HF_TOKEN found. You'll need to login manually:"
    echo "  Run: huggingface-cli login"
    echo "  Or set: export HF_TOKEN='your_token_here'"
fi

# Create sample .env file
echo ""
echo "Step 7: Creating sample .env file..."
cat > .env.sample << EOL
# HuggingFace Token (required for DINOv3 model access)
HF_TOKEN=your_huggingface_token_here

# Optional: Configure paths
SOURCE_IMAGES_DIR=./source_images
EMBEDDINGS_DIR=./embeddings_tta
MAPPINGS_DIR=./mappings
QUERIES_DIR=./queries
EOL
echo "✓ Created .env.sample (copy to .env and add your token)"

# Summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: conda activate textile_matching"
echo "2. Login to HuggingFace: huggingface-cli login"
echo "3. Request DINOv3 access: https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m"
echo "4. Add textile images to: source_images/"
echo "5. Run notebook: jupyter notebook textile_matcher_dynamic_tta.ipynb"
echo ""
echo "Directory structure created:"
echo "  ./source_images/    - Place reference textile images here"
echo "  ./queries/          - Place query images here"
echo "  ./embeddings_tta/   - Auto-generated embeddings (don't modify)"
echo "  ./mappings/         - ID mappings (don't modify)"
echo ""
echo "For help, see README.md"
echo "=========================================="