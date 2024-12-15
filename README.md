# The Sweet Truth
A machine learning-powered pipeline to detect hidden sweeteners in packaged food ingredient lists. This project combines OCR, text preprocessing, and machine learning to classify and annotate sweeteners, providing clear insights for informed food choices.

## Overview
The Sweet Truth Detector works in two parts:
1. **Sweetener Detection from Text:** Classifies ingredients as sweeteners or non-sweeteners using a Random Forest Classifier trained on curated data.
2. **OCR Integration:** Extracts ingredient lists from preprocessed food packaging images using Tesseract OCR and annotates detected sweeteners.
---
## How It Works
* **Text Input:** Provide a list of ingredients (text).
  * The model classifies sweeteners with high confidence.
* **Image Input:** Provide a clear image of ingredients from any food packaging.
  * OCR extracts the text → Sweeteners are detected → The image is annotated with bounding boxes.

### **Dependencies**
Before using the Sweet Truth Detector, ensure you have the following libraries installed:
```python
pip install spacy numpy pandas opencv-python pytesseract matplotlib
```
Note: pytesseract requires Tesseract OCR to be installed on your system.

### Step 1: Import required modules
```python
from module_detector import *
import joblib
```
### Step 2: Load the pre-trained model
The `sweet_truth_detector.pkl` file contains the trained Random Forest model used for classifying the ingredients.
```python
model = joblib.load('sweet_truth_detector.pkl')
```
### Step 3: Predict sweeteners from text
If you already have a list of ingredients, the function `sweet_truth_from_list(ingredient_list, model)` can detect sweeteners and display the results as a dataframe.
```python
ingredient_list = "Glucose, Honey, Salt, Vitamins, Minerals"
sweet_truth_from_list(ingredient_list, model)
```
### Step 4: Predict sweeteners from images
You can also provide an image containing the ingredient list. The tool extracts text using OCR and highlights the detected sweeteners in two ways. An example image `children_nutrition_drink.jpg` is also provided.

#### 4.1 Display results as a dataframe
```python
image_path = 'path/to/your/image.jpg'
sweet_truth_from_image(image_path, model)
```

#### 4.2 Annotate the input image with detected sweeteners
To produce and save an annotated image with bounding boxes around detected sweeteners, use:
```python
image_path = 'path/to/your/image.jpg'
annotate_image_with_sweeteners(image_path, model)
```
* Any image with the name `image.jpg` will be annotated and saved as `image_annotated.jpg`.
* Example output:
![children_nutrition_drink_annotated](https://github.com/user-attachments/assets/281466a3-422c-4427-bff0-44b2e65361a7)


---

## Project Files
1. `final_ingredient_dataset.csv`: A list of separate ingredients, labeled as sweeteners (1) or non-sweeteners (0).
2. `ingredients_sweeteners_list.csv`: A synthetic dataset of combined ingredient lists, used for training the classification model.
3. `module_detector.py`: A custom module containing reusable functions for text preprocessing, classification, OCR integration and image annotation.
4. `sweetener_detector_from_text.ipynb`: **(Part 1)** Develops and tests the text-based classification model to detect sweeteners.
5. `ocr.ipynb`: **(Part 2)** Performs OCR on an input image and extracts the ingredient list.
6. `sweet_truth_detector.pkl`: The trained Random Forest Classifier model.
7. `the_sweet_truth_detector.ipynb`: **(Combines Part 1 & Part 2)** Tests the complete pipeline from image input to annotated sweetener detection output.

---

## Tech Stack
* Python
* scikit-learn (Random Forest Classifier and TfidfVectorizer)
* spaCy, Regular Expressions (Text Preprocessing)
* Tesseract OCR, OpenCV (Image Processing)
* Jupyter Notebooks (Model Development)

---

## Image Sources
The images used in the notebooks `the_sweet_truth_detector.ipynb` and `ocr.ipynb` for testing are:  
- **Self-Captured**: Images that I photographed for this project.  
- **Open Food Facts**: Openly available food ingredient images from [Open Food Facts](https://world.openfoodfacts.org/). These images are used solely for demonstration purposes and align with Open Food Facts' terms of use.
