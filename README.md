# Textile Pattern Matching System with DINOv3

A production-ready textile pattern matching system that uses DINOv3 vision transformers with Test-Time Augmentation (TTA) to match fabric patterns across color variations. The system automatically manages a growing database of textile images with smart caching and incremental processing.

## 🌟 Features

- **Color-Invariant Pattern Matching**: Matches identical patterns even with different colorways
- **Test-Time Augmentation (TTA)**: Rotation-based augmentation for robust matching
- **Incremental Processing**: Only processes new images added to the database
- **Automatic ID Management**: Auto-assigns persistent IDs to all textile images
- **Smart Caching**: Embeddings are cached and reused across sessions
- **Production Ready**: Drop images → Run notebook → Get matches

## 📋 Requirements

- Python 3.11+
- CUDA-capable GPU (recommended)
- ~4GB GPU memory for DINOv3-ViT-L/16
- HuggingFace account with access to DINOv3 models

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/textile-matching.git
cd textile-matching
```

### 2. Create Conda Environment

```bash
# Create environment with Python 3.11
conda create -n textile_matching python=3.11
conda activate textile_matching
```

### 3. Install Dependencies

```bash
# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install required packages
pip install transformers pillow matplotlib tqdm jupyter numpy

# Install HuggingFace CLI for model access
pip install huggingface-hub
```

### 4. Set Up HuggingFace Access

```bash
# Login to HuggingFace (you'll need an account)
huggingface-cli login

# Or set token as environment variable
export HF_TOKEN="your_huggingface_token_here"
```

### 5. Request Access to DINOv3

Visit https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m and request access to the model.

### 6. Launch Jupyter Notebook

```bash
jupyter notebook textile_matcher_dynamic_tta.ipynb
```

## 📁 Project Structure

```
textile_matching/
│
├── source_images/              # Drop textile images here (any filename)
│   ├── fabric_001.jpg
│   ├── pattern_blue.png
│   └── ...
│
├── embeddings_tta/             # Auto-generated TTA embeddings
│   ├── textile_001.pkl
│   ├── textile_002.pkl
│   └── ...
│
├── mappings/                   # ID mapping and metadata
│   └── image_mapping.json
│
├── queries/                    # Query images for matching
│   ├── query_image.jpeg
│   └── query_2.png
│
├── textile_matcher_dynamic_tta.ipynb  # Main notebook
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 📊 How to Use

### Adding Images to Database

1. **Drop images** into `source_images/` directory
   - Supports: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
   - Any filename works - no renaming needed
   - System auto-assigns IDs (textile_001, textile_002, etc.)

2. **Run the notebook** - it automatically:
   - Detects new images
   - Generates TTA embeddings only for new images
   - Updates the mapping file
   - Loads cached embeddings for existing images

### Searching for Matches

1. **Place query image** in `queries/` directory or specify direct path

2. **In the notebook**, update the query path:
   ```python
   QUERY_IMAGE = QUERIES_DIR / "your_query_image.jpg"
   ```

3. **Run the search** - returns top matches with similarity scores

### Example Workflow

```python
# First time setup - processes all images
database_embeddings = db.update_database()

# Add new image to database
new_image_path = Path("/path/to/new_textile.jpg")
database_embeddings = add_new_image_to_database(new_image_path, db)

# Search with a query
matches = search_textile(query_path, database_embeddings, model, processor)

# View results
visualize_search_results(query_path, matches, SOURCE_IMAGES_DIR)
```

## 🔧 Technical Details

### DINOv3 Model
- **Model**: facebook/dinov3-vitl16-pretrain-lvd1689m
- **Architecture**: Vision Transformer Large with 16x16 patches
- **Feature**: CLS token (1024-dimensional global representation)
- **Input**: 224x224 images (automatic resizing)

### Test-Time Augmentation (TTA)
- **Rotations**: [0°, 90°, 180°, 270°]
- **Strategy**: Average embeddings across all rotations
- **Benefit**: Rotation-invariant pattern matching
- **Cost**: ~4x processing time per image

### Embedding Storage
- **Format**: Pickle files (.pkl)
- **Size**: ~4KB per embedding
- **Naming**: textile_XXX.pkl (based on assigned ID)
- **Caching**: Persistent across sessions

## 📈 Performance

- **Database Size**: Tested with 100+ textile images
- **Processing Speed**: ~2-3 seconds per image with TTA (GPU)
- **Search Speed**: <0.1 seconds for 100 images
- **Accuracy**: Successfully matches patterns across color variations
- **Memory Usage**: ~2GB for model + embeddings

## 🎯 Use Cases

- **Textile Manufacturing**: Match fabric patterns in production
- **Quality Control**: Verify pattern consistency across batches
- **Design Library**: Search pattern database by visual similarity
- **Customer Service**: Help customers find specific patterns
- **Inventory Management**: Identify unlabeled textile samples

## 🛠️ Customization

### Adjust TTA Rotations
```python
# In extract_dinov3_embedding_tta function
rotations = [0, 45, 90, 135, 180, 225, 270, 315]  # More angles
```

### Change Embedding Model
```python
# Use smaller model for faster processing
MODEL_NAME = "facebook/dinov3-base"  # If available
```

### Modify Similarity Metric
```python
# Currently uses cosine similarity
# Can implement alternative metrics in compute_similarities()
```

## 📝 Environment Setup Script

Save as `setup_env.sh`:

```bash
#!/bin/bash

# Create conda environment
conda create -n textile_matching python=3.11 -y
conda activate textile_matching

# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install transformers==4.36.0
pip install pillow==10.0.0
pip install matplotlib==3.7.2
pip install tqdm==4.66.1
pip install jupyter==1.0.0
pip install numpy==1.24.3
pip install huggingface-hub==0.20.0

# Create directory structure
mkdir -p source_images
mkdir -p embeddings_tta
mkdir -p mappings
mkdir -p queries

echo "Setup complete! Activate environment with: conda activate textile_matching"
```

Run with: `bash setup_env.sh`

## 🐛 Troubleshooting

### "401 Client Error" when loading model
- Ensure you're logged into HuggingFace: `huggingface-cli login`
- Verify you have access to the DINOv3 model
- Check HF_TOKEN is set correctly

### CUDA out of memory
- Reduce batch size to 1 (already default)
- Use CPU instead: remove `device_map="auto"`
- Use smaller model if available

### Slow processing
- Ensure GPU is being used: `torch.cuda.is_available()`
- Reduce number of TTA rotations
- Process images in smaller batches

## 📚 References

- [DINOv3 Paper](https://arxiv.org/abs/2304.07193)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This system is designed for textile pattern matching but can be adapted for any visual similarity search task.