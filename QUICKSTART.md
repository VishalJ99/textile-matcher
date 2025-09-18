# Quick Start Guide - Textile Pattern Matching

## 🚀 5-Minute Setup

### Prerequisites
- Linux/Mac with GPU (recommended) or Windows with WSL2
- Conda/Miniconda installed
- HuggingFace account

### Step 1: Clone & Setup (2 min)
```bash
git clone https://github.com/yourusername/textile-matching.git
cd textile-matching
bash setup.sh
```

### Step 2: Activate Environment (10 sec)
```bash
conda activate textile_matching
```

### Step 3: Login to HuggingFace (1 min)
```bash
huggingface-cli login
# Enter your token when prompted
```

### Step 4: Add Your Images (30 sec)
```bash
# Add reference textiles (your database)
cp /path/to/your/textiles/* ./source_images/

# Add query image
cp /path/to/query/image.jpg ./queries/
```

### Step 5: Run Matching (1 min)
```bash
jupyter notebook textile_matcher_dynamic_tta.ipynb
# Click "Run All" in Jupyter
```

## 📊 What You'll See

1. **First Run**: Generates embeddings for all images (~2-3 sec/image)
2. **Results**: Top 5 matching textiles with similarity scores
3. **Visualization**: Query image + matched textiles side-by-side

## 🔄 Adding New Images

Just drop new images in `source_images/` and re-run the notebook!
- Only new images are processed
- IDs remain stable
- Embeddings are cached

## 💡 Tips

- **Best Results**: Use high-quality images (1000+ pixels)
- **Color Variations**: System matches patterns across different colors
- **Rotation Invariant**: Works even if textiles are photographed at angles
- **Fast Searches**: After initial processing, searches take <0.1 seconds

## 🆘 Quick Troubleshooting

**"Model not found" error?**
→ Request access at: https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m

**Out of memory?**
→ Close other applications or use CPU mode (slower but works)

**No matches found?**
→ Check if query image is similar to any database patterns

## 📝 Example Use Cases

### Fashion Designer
```python
# Search for all variations of a pattern
query = "floral_pattern_v1.jpg"
# Returns: All colorways and variations of that pattern
```

### Quality Control
```python
# Verify production sample matches original
query = "production_sample_batch_203.jpg"  
# Returns: Original design if pattern matches
```

### Customer Service
```python
# Customer sends photo asking "Do you have this?"
query = "customer_whatsapp_photo.jpg"
# Returns: Matching products from catalog
```

---

Ready to match some textiles? 🧵✨