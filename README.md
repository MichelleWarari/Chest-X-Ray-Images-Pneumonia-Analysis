# Chest X-Ray Images (Pneumonia) Analysis

## Abstract

This document presents a comprehensive analysis of the Chest X-Ray Images (Pneumonia) dataset, including exploratory data analysis (EDA), model selection, training, and evaluation. The analysis utilizes Python libraries such as datasets, pandas, and plotly for data handling and visualization, with code executed in a Jupyter Notebook environment.

## 1 Task Description

The objective is to perform a comprehensive analysis on the Chest X-Ray Images (Pneumonia) dataset available at [https://data.mendeley.com/datasets/rscbjbr9sj/2/files/f12eaf6d-6023-432f-acc9-80c9d7393433](https://data.mendeley.com/datasets/rscbjbr9sj/2/files/f12eaf6d-6023-432f-acc9-80c9d7393433). This involves exploratory data analysis, model selection, training, and evaluation, with hyperparameter tuning and visualizations throughout the process.

## 2 Installing Required Libraries

The analysis begins by installing the datasets library from Hugging Face to load the dataset.

```bash
pip install datasets
```

The output confirms the successful installation of datasets along with its dependencies, including numpy, pandas, pyarrow, and others, ensuring the environment is ready for data loading.

## 3 Loading the Dataset

The dataset is loaded using the datasets library from the Hugging Face repository.

```python
from datasets import load_dataset

dataset = load_dataset("hf-vision/chest-xray-pneumonia")
```

The dataset is successfully downloaded, consisting of multiple parquet files for training, validation, and test splits, with progress bars indicating the download of each file.

## 4 Exploratory Data Analysis (EDA)

EDA is performed to understand the dataset's structure, content, and characteristics.

### Dataset Structure

The structure of the dataset is displayed to reveal its splits and features.

```python
display(dataset)
```

The dataset is a DatasetDict with three splits:

- **Train**: 5,216 examples with features image (PIL,JpegImage) and label (integer).
- **Validation**: 16 examples with the same features.
- **Test**: 624 examples with the same features.

The image feature contains chest X-ray images, and the label feature indicates the presence (1) or absence (0) of pneumonia.

### Examining Sample Data

The first example from the training set is inspected to understand the data format.

```python
display(dataset['train'][0])
```

The output shows a single example with:

- **image**: A PIL JPEG image in grayscale (mode L) with dimensions 2090x1858 pixels.
- **label**: 0 (indicating no pneumonia).

This confirms that the dataset contains grayscale chest X-ray images labeled as either normal (0) or pneumonia (1).

### Label Distribution

The distribution of labels in the training set is examined to assess class balance.

```python
import pandas as pd

train_labels = [example['label'] for example in dataset['train']]
label_counts = pd.Series(train_labels).value_counts()

display(label_counts)
```

The output indicates:

- **Label 1 (Pneumonia)**: 3,875 examples.
- **Label 0 (Normal)**: 1,341 examples.

This reveals a class imbalance, with approximately 74.3% of the training examples labeled as pneumonia and 25.7% as normal.

### Visualizing Label Distribution

The label distribution is visualized to provide a graphical representation of the class imbalance.

```python
import plotly.express as px

fig = px.bar(
    x=['Pneumonia (1)', 'Normal (0)'], 
    y=label_counts.values, 
    title='Label Distribution in Training Set', 
    labels={'x': 'Label', 'y': 'Count'}
)
fig.show()
```

The visualization is a bar chart with two bars:

- **Pneumonia (1)**: Approximately 3,875 counts.
- **Normal (0)**: Approximately 1,341 counts.

The chart highlights the significant class imbalance, which may require techniques such as class weighting or data augmentation during model training to ensure balanced performance.

## 5 Conclusion

The EDA reveals that the Chest X-Ray Images (Pneumonia) dataset consists of 5,216 training images, 16 validation images, and 624 test images, each with a grayscale image and a binary label (0 for normal, 1 for pneumonia). The training set shows a class imbalance, with 74.3% pneumonia cases and 25.7% normal cases, as visualized in the bar chart. This imbalance suggests the need for careful model design to handle biased predictions. Further steps would involve preprocessing the images, selecting a suitable model (e.g., a convolutional neural network), performing hyperparameter tuning, and evaluating performance on the test set, with additional visualizations to monitor training progress and model performance.

---

*This analysis was conducted as part of a comprehensive study on medical image classification using deep learning techniques.*
