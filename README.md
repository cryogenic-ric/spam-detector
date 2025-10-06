# Spam Detector

This project is a spam detector that uses a hybrid approach, combining a transformer-based model with a rule-based system to classify text as spam or not spam.

## Installation

1.  Clone the repository:
    ```
    git clone <repository-url>
    ```
2.  Install the dependencies:
    ```
    pip install -r requirements.txt
    ```

## Usage

### 1. Create Training Data

To train the model, you first need to create a labeled training dataset. You will need two CSV files: `spam_data.csv` and `ham_data.csv`. Each file should have a 'text' column containing the text samples.

Once you have the data, run the following command to create the training data:

```
python create_training_data.py
```

This will create a `labeled_training_data.csv` file.

### 2. Train the Model

To train the spam detection model, run the following command:

```
python train_model.py
```

This will train a new model and save it to the `trained_model` directory.

### 3. Predict Spam

To predict spam on new data, make sure you have a `summaries.csv` file with a 'title' and 'summary' column. Then, run the following command:

```
python predict.py
```

This will generate a `spam_detection_results.csv` file with the spam predictions.

## Files

-   `create_training_data.py`: Combines spam and ham data into a single labeled CSV for training.
-   `train_model.py`: Trains the spam detection model.
-   `predict.py`: Predicts spam on new data.
-   `utils.py`: Contains utility functions used by the other scripts.
-   `requirements.txt`: A list of the Python dependencies for the project.
-   `spam_data.csv`: Manually collected spam examples.
-   `ham_data.csv`: Manually collected ham examples.
-   `labeled_training_data.csv`: The combined and shuffled training data.
-   `summaries.csv`: The data to be classified.
-   `spam_detection_results.csv`: The output of the spam prediction.
-   `trained_model/`: The directory where the trained model is saved.
