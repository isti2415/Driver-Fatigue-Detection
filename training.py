import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, History
import tensorflow as tf
from keras.layers import BatchNormalization
import matplotlib.pyplot as plt

def create_drowsiness_model(input_shape):
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    # Flatten layer
    model.add(Flatten())

    # Dense layers
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model

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

def train_model():
    history = {'accuracy': [], 'val_accuracy': []}
    
    train_data_dir = 'data/train'  
    test_data_dir = 'data/test'   
    input_shape = (84, 84, 3)
    batch_size = 32

    model = create_drowsiness_model(input_shape)

    train_generator = create_data_generator(train_data_dir)

    test_generator = create_data_generator(test_data_dir)

    print("Number of training samples:", len(train_generator))
    print("Number of testing samples:", len(test_generator))
    
    # Calculate steps per epoch and validation steps
    SPE = len(train_generator) // batch_size
    VS = len(test_generator) // batch_size
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)

    early_stop = EarlyStopping(monitor="val_loss", patience=3, verbose=1)

    checkpoint = ModelCheckpoint('model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    model.fit(
        train_generator,
        steps_per_epoch=SPE,
        epochs= 20,
        validation_data=test_generator,
        validation_steps=VS,
        callbacks=[checkpoint, early_stop, reduce_lr, History()]
    )
    
    # Extract history
    history['accuracy'] = [acc * 100 for acc in model.history.history['accuracy']]
    history['val_accuracy'] = [val_acc * 100 for val_acc in model.history.history['val_accuracy']]
    history['loss'] = [loss * 100 for loss in model.history.history['loss']]
    history['val_loss'] = [val_loss * 100 for val_loss in model.history.history['val_loss']]

    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history['accuracy'], label='Training Accuracy (%)', color='blue')
    plt.plot(history['val_accuracy'], label='Validation Accuracy (%)', color='red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)  # Set y-axis range from 0 to 100
    plt.legend()
    plt.savefig('accuracy_plot.png')  # Save the figure
    plt.show()

    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss (%)', color='purple')
    plt.plot(history['val_loss'], label='Validation Loss (%)', color='orange')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (%)')
    plt.ylim(0, 100)  # Set y-axis range from 0 to 100
    plt.legend()
    plt.savefig('loss_plot.png')  # Save the figure
    plt.show()

if __name__ == "__main__":
    train_model()