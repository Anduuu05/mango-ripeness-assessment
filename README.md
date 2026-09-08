# Mango Ripeness Assessment

This project contains the group's classical image-processing pipelines and a Streamlit user interface for mango ripeness prediction.

## Run the Streamlit UI

Open Anaconda Prompt and run:

    conda activate base
    cd /d "Image Processing assignment"
    python -m pip install -r requirements.txt
    python -m streamlit run ui.py

Then open http://localhost:8501 in a browser.

The UI accepts JPG, JPEG, PNG, and BMP images. It uses the trained hybrid_random_forest.pkl model and displays the predicted class and class probabilities for:

- Unripe
- Partially Ripe
- Ripe

The model file must remain in the same folder as ui.py.

## Project files

- Image Processing assignment/ui.py — Streamlit UI entry point.
- Image Processing assignment/hybrid_random_forest.pkl — trained Hybrid Pipeline Random Forest model.
- Image Processing assignment/Andrew_Hybrid_Pipeline.ipynb — Hybrid Pipeline notebook.
- Image Processing assignment/Lvy_lightning_enhancement_pipeline.ipynb — Lighting Enhancement Pipeline notebook.
- Image Processing assignment/Chia Jia Chen – Background Segmentation .ipynb — Background Segmentation Pipeline notebook.
- Image Processing ASSIGNMENT-LIMYUJIE PART/ — Surface Defect Handling Pipeline files.

The notebooks require the mango dataset to be available locally. The Streamlit UI does not need the dataset because it loads the trained model file directly.