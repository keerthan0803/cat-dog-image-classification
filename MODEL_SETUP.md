# Model Setup Instructions

## Problem
The model file `best_model.h5` is too large for Git (GitHub has a 100MB file limit).

## Solution: Upload Model to Google Drive

### Step 1: Upload Model to Google Drive
1. Go to [Google Drive](https://drive.google.com)
2. Upload your `best_model.h5` file
3. Right-click the file → Share → Change to "Anyone with the link"
4. Copy the sharing link (looks like: `https://drive.google.com/file/d/XXXXXXXXXXXXX/view?usp=sharing`)

### Step 2: Extract File ID
From the URL: `https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view?usp=sharing`
The File ID is: `1a2b3c4d5e6f7g8h9i0j`

### Step 3: Add to Render Environment Variables
1. Go to your Render dashboard
2. Select your service
3. Go to **Environment** tab
4. Add new environment variable:
   - **Key**: `MODEL_GDRIVE_ID`
   - **Value**: `YOUR_FILE_ID_HERE` (paste the ID from step 2)
5. Save changes

### Step 4: Deploy
The app will automatically download the model on first startup!

---

## Alternative: Use Smaller MobileNet Model

If you want a quicker solution, use the smaller `mobilenet_model.h5` file instead:

1. Update `app.py` line 11:
   ```python
   MODEL_PATH = 'mobilenet_model.h5'
   ```

2. Remove the model from `.gitignore`:
   - Edit `.gitignore`
   - Remove the line: `mobilenet_model.h5`
   - Or remove `*.h5` if mobilenet is small enough

3. Commit and push:
   ```bash
   git add .
   git commit -m "Switch to mobilenet model"
   git push
   ```

---

## Current Setup
The app now supports automatic model downloading from Google Drive using the `gdown` package (already in requirements.txt).
