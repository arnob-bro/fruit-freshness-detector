# Quick Start Guide - How to Run the Project

## Step 1: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install all required packages
pip install -r requirements.txt
```

## Step 2: Prepare Your Dataset

You need to organize your fruit images in the following structure:

```
data/
├── train/
│   ├── fresh/     (put fresh fruit images here)
│   └── rotten/   (put rotten fruit images here)
├── val/
│   ├── fresh/     (put fresh fruit images here)
│   └── rotten/   (put rotten fruit images here)
└── test/
    ├── fresh/     (put fresh fruit images here)
    └── rotten/   (put rotten fruit images here)
```

**Note:** If you don't have a dataset yet, you can:
- Download a fruit freshness dataset from Kaggle
- Use your own images
- Start with the notebooks to explore what's needed

## Step 3: Choose How to Run

### Option A: Run Jupyter Notebooks (Recommended for Learning)

1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open notebooks in order:**
   - `notebooks/data_exploration.ipynb` - Explore your dataset
   - `notebooks/preprocessing.ipynb` - Preprocess data
   - `notebooks/training.ipynb` - Train models
   - `notebooks/evaluation.ipynb` - Evaluate models

### Option B: Run Training Script (Recommended for Production)

1. **Configure training** (optional - defaults are set):
   ```bash
   # Edit training/config.yaml to change settings
   ```

2. **Train the model:**
   ```bash
   python training/train.py
   ```
   
   Or with custom config:
   ```bash
   python training/train.py --config training/config.yaml
   ```

3. **Evaluate the trained model:**
   ```bash
   python evaluation/evaluate.py --model models/saved/best_model.pth
   ```

### Option C: Run Web Application (For Testing/Deployment)

1. **Start Streamlit app:**
   ```bash
   streamlit run deployment/app.py
   ```
   
   The app will open in your browser automatically.

2. **Or start FastAPI server:**
   ```bash
   python deployment/api.py
   ```
   
   Then visit: http://localhost:8000
   API docs: http://localhost:8000/docs

## Step 4: Test with Sample Images

Once you have a trained model, you can test it:

### Using Streamlit App:
1. Run `streamlit run deployment/app.py`
2. Upload a fruit image
3. See the prediction

### Using API:
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/your/image.jpg" \
  -F "threshold=0.5"
```

## Common Issues & Solutions

### Issue: "No module named 'torch'"
**Solution:** Make sure you activated the virtual environment and installed requirements:
```bash
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in `training/config.yaml`:
```yaml
training:
  batch_size: 16  # Reduce from 32 to 16 or 8
```

### Issue: "No data found"
**Solution:** Make sure you have images in the data folders:
- Check `data/train/fresh/` and `data/train/rotten/` have images
- Images should be .jpg, .jpeg, or .png format

### Issue: "Model file not found"
**Solution:** Train a model first:
```bash
python training/train.py
```

## Quick Test (Without Training)

If you just want to test the code structure without training:

1. **Test data loader:**
   ```python
   python -c "from utils.dataloader import get_loaders; train, val, test = get_loaders(); print('Data loaders work!')"
   ```

2. **Test model creation:**
   ```python
   python -c "from models.resnet_transfer import get_model; model = get_model(); print('Model created!')"
   ```

## Next Steps

1. **Explore the notebooks** to understand the workflow
2. **Train your first model** using the training script
3. **Evaluate results** and compare different models
4. **Deploy** using Streamlit or FastAPI

## Need Help?

- Check the main `README.md` for detailed documentation
- Review the code comments in each module
- Run notebooks step-by-step to understand the process

