# Shephard MagicianGirl PhoenixAiden 22/04/2019
import argparse
import itertools
#matplotlib.use("Agg")
import matplotlib
import numpy as np
import os
import tensorflow as tf
from skimage import data
from skimage import transform
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator

EPOCHS = 25
BS = 16
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png")
args = vars(ap.parse_args())


lego_classes = ["Brick corner", "Brick 2x2", "Brick 1x2", "Brick 1x1", "Plate 2x2", "Plate 1x2", "Plate 1x1",
                "Roof Tile", "Flat Tile 1x2", "Peg 2M",
                "Bush for Cross Axle", "Plate 1X2 with 1 Knob", "Technic Lever", "Bush 3M with Cross axle",
                "Cross Axle 2M", "half Bush"]

def load_data(data_dir):
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) 
                      if f.endswith(".png")]
        for f in file_names:
            images.append(data.imread(f))
            labels.append(int(d))
    return images, labels


ROOT_PATH = "C:/Users/Megatech/Desktop/"
train_data_dir = os.path.join(ROOT_PATH, "LEGO/train")
test_data_dir = os.path.join(ROOT_PATH, "LEGO/valid")
images, labels = load_data(train_data_dir)


# Make a histogram with 32 bins of 16 labels.
plt.hist(labels, 32)
plt.show()
plt.figure(figsize=(10,10))
legos=[300, 2250, 3650, 4000]
for i in range(len(legos)):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i], cmap=plt.cm.binary)
    plt.xlabel(lego_classes[labels[i]])
plt.show()
# scale the raw pixel intensities to the range [0, 1]
data = np.array(images, dtype="float") / 255.0
labels = np.array(labels)
#images64 = rgb2gray(np.array(images)) #Convert to Gray Scale for better perfomance

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)#Only Used the dataset marked 'Train' my pc has low amounts of RAM
 

# Define model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation=tf.nn.relu, input_shape=(200, 200, 4)),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,  activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128,  activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation=tf.nn.softmax)])

model.summary()  # Take a look at the model summary
# Compile Model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#checkpoint = ModelCheckpoint(filepath='model.hdf5', verbose=1, save_best_only=True)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='val_loss',
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=2,
        verbose=1)
]

H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=16),
    steps_per_epoch=(len(trainX) / 16),
    validation_data=[testX, testY],
    validation_steps=100,
    epochs=EPOCHS,
    verbose=1,
    callbacks=callbacks
)
# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Lego Bricks")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])


# Load the weights with the best validation accuracy
model.load_weights('model.hdf5')
# Evaluate the model on test set
score = model.evaluate(testX, testY, verbose=0)
# Print test accuracy
print('\n', 'Test accuracy:', score[1])

#Plot ROC Curve
prob = model.predict_proba(testX)
prob = pro[:,1]
auc = roc_auc_score(testY, prob)
fpr,tpr,thresholds = roc_curve(testY, prob)
plt.plot([0,1],[0,1], linestyle='*')
plt.plot(fpr, tpr, marker='.')
plt.show()
