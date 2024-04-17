import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

def visualize_report(report):

    # Define custom colors
    colors = ['#ffdfdf', '#dfffff', '#dfffdf', '#dfdfff']

    # Extract metrics for each class, handling potential missing values
    metrics = ['precision', 'recall', 'f1-score']
    data = {metric: [] for metric in metrics}
    labels = []
    for cls, metrics_values in report.items():
        if cls.isdigit() or cls in ['macro avg', 'weighted avg']:
            labels.append(cls)
            for metric in metrics:
                value = metrics_values.get(metric, None)
                data[metric].append(value * 100 if value is not None else 0)  # Convert to percentage and handle missing values

    # Convert data to DataFrame
    df = pd.DataFrame(data, index=labels)

    for metric in metrics:
        # Create separate plots with individual plt.figure() calls
        plt.figure(figsize=(14, 12))  # Create a new figure for each metric
        sns.barplot(x=df.index, y=metric, data=df, palette=colors)
        plt.title(f'{metric.capitalize()} by Class (%)')
        plt.xlabel('Class')  # Set x-axis label
        plt.ylabel(f'{metric.capitalize()} (%)')  # Set y-axis label
        plt.ylim(0, 100)  # Adjust y-axis limit for percentages
        plt.savefig(f'{metric}_plot.png')  # Save the figure with appropriate name
        plt.show()  # Show each plot individually

def evaluate_model(test_generator):
    
    model = load_model("model.keras")
    
    for x, y in test_generator:
        predictions = model.predict(x)
        labels = y

    y_true = np.where(labels > 0.5, 1, 0)
    y_pred = np.where(predictions > 0.5, 1, 0)
    
    # Compute confusion matrix and classification report
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=["Awake", "Drowsy"], yticklabels=["Awake", "Drowsy"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')  # Save the confusion matrix plot
    plt.show()
    
    print(classification_report(y_true, y_pred))

    report = classification_report(y_true, y_pred, output_dict=True)
    visualize_report(report)
    
def create_data_generator(directory):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        color_mode="rgb",
        batch_size=32,
        image_size=(84, 84),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False
    )

test_data_dir = 'data/test'
test_generator = create_data_generator(test_data_dir)
evaluate_model(test_generator)