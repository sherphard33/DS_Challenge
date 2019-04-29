import tensorflow as tf
from skimage import transform
from skimage import data
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.color import rgb2gray
import random

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

images_array = np.array(images)
labels_array = np.array(labels)
lego_classes = ["Brick corner", "Brick 2x2", "Brick 1x2", "Brick 1x1", "Plate 2x2", "Plate 1x2", "Plate 1x1",
                "Roof Tile", "Flat Tile 1x2", "Peg 2M",
                "Bush for Cross Axle", "Plate 1X2 with 1 Knob", "Technic Lever", "Bush 3M with Cross axle",
                "Cross Axle 2M", "half Bush"]

# Make a histogram with 62 bins of the `labels` data
plt.hist(labels, 32)
plt.show()

# Sample random images
lego_brick_images = [300, 2250, 3650, 4000]
for i in range(len(lego_brick_images)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images[lego_brick_images[i]])
    plt.subplots_adjust(wspace=0.5)
plt.show()


# Resize images
images64 = [transform.resize(image, (150, 150)) for image in images]
images64 = np.array(images64)

images64 = rgb2gray(np.array(images64))
print(images64.shape)


x = tf.placeholder(dtype = tf.float32, shape = [None, 150, 150])
y = tf.placeholder(dtype = tf.int32, shape = [None])
images_flat = tf.contrib.layers.flatten(x)
logits = tf.contrib.layers.fully_connected(images_flat, 16, tf.nn.relu)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_pred = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)

sess = tf.Session()
#Initialize the variables
sess.run(tf.global_variables_initializer())
# Train the model
for i in range(201):
        print('EPOCH', i)
        _, accuracy_val = sess.run((train_op,accuracy), feed_dict={x: images64, y: labels})
        if i % 10 == 0:
            print("Loss: ", loss)
        print('DONE WITH EPOCH')


# Pick 10 random images
sample_indexes = random.sample(range(len(images64)), 10)
sample_images = [images64[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

# Run the "predicted_labels" op.
predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]
                        
# Print the real and predicted labels
print(sample_labels)
print(predicted)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
             fontsize=12, color=color)
    plt.imshow(sample_images[i])

plt.show()


# Load the test data
test_images, test_labels = load_data(test_data_dir)

# Transform the images to 150 by 150 pixels
test_images64 = [transform.resize(image, (150, 150)) for image in test_images]

# Convert to grayscale
from skimage.color import rgb2gray
test_images64 = rgb2gray(np.array(test_images))

# Run predictions against the full test set.
predicted = sess.run([correct_pred], feed_dict={x: test_images64})[0]

# Calculate correct matches 
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

# Calculate the accuracy
accuracy = match_count / len(test_labels)

# Print the accuracy
print("Accuracy: {:.3f}".format(accuracy))

sess.close()
