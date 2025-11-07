# DSAT for 2D face alignment and Facial Paralysis Classification

This is a PyTorch implementation for face alignment with Dynamic Semantic Aggregation Transformer (DSAT). We use the normalized mean errors (NME) to measure the landmark location performance. This work has achieved outstanding performance on 300W datasets.

## 🆕 NEW FEATURE: Facial Paralysis Detection and Grading

We have successfully adapted the DSAT model from face landmark detection to **facial paralysis classification and grading**. The modified model can now:

- **Binary Classification**: Detect whether a person has facial paralysis (Normal vs Paralysis)
- **Multi-class Grading**: Classify the severity of facial paralysis (Normal, Mild, Moderate, Severe)
- **High Accuracy**: Leverages the powerful feature extraction capabilities of DSAT for medical classification

### Quick Start for Facial Paralysis Classification

```bash
# Install dependencies
pip install -r requirements_facial_paralysis.txt

# Train binary classification model
bash train_facial_paralysis.sh binary

# Train multi-class grading model
bash train_facial_paralysis.sh multiclass

# Run inference
bash inference_facial_paralysis.sh checkpoint/models/best_checkpoint.pth.tar /path/to/image.jpg
```

For detailed instructions, see [FACIAL_PARALYSIS_README.md](FACIAL_PARALYSIS_README.md).

## Original Face Alignment Features

### Install

* `Python 3`

* `Install PyTorch >= 0.4 following the official instructions (https://pytorch.org/).`

### Data

You need to download images (e.g., 300W) from official websites and then put them into `data` folder for each dataset.

Your `data` directory should look like this:

````
DSAT
-- data
   |-- afw
   |-- helen
   |-- ibug
   |-- lfpw
````  

### Training and testing

* Train

```sh
python main.py 
```

* Run evaluation to get result.

```sh
python main.py --phase test
or
python demo.py
```

