import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import os

seed = 16
np.random.seed(seed)

# PROJECT_PATH = "/home/ubadmin/voice_rec/"
PROJECT_PATH = "C:/Users/Vlad/Desktop/voice_rec/"

ds_name = "ds1_no_aug"
model_name = "model1"

with open(PROJECT_PATH + "mydataset/", 'rb') as f:
    out = pickle.load(f)
    f.close()


(train_specs, train_labels, valid_specs, valid_labels) = out

train_ds = tf.data.Dataset.from_tensor_slices((train_specs, train_labels))
valid_ds = tf.data.Dataset.from_tensor_slices((valid_specs, valid_labels))

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 64
train_ds = train_ds.batch(batch_size)
valid_ds = valid_ds.batch(batch_size)
train_ds = train_ds.cache().prefetch(AUTOTUNE)
valid_ds = valid_ds.cache().prefetch(AUTOTUNE)

input_shape = train_specs[0].shape
num_labels = 5

norm_layer = tf.keras.layers.Normalization()
norm_layer.adapt(data=train_ds.map(map_func=lambda spec, label: spec))

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=input_shape),
    norm_layer,
    tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax'),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

EPOCHS = 20

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[tf.keras.callbacks.EarlyStopping(verbose=1, patience=3)]
)

model.save(PROJECT_PATH + "models/" + model_name)

metrics = history.history
plt.figure()
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.savefig(PROJECT_PATH + "models/" + model_name + "/loss.png")

plt.figure()
plt.plot(history.epoch, metrics['accuracy'], metrics['val_accuracy'])
plt.legend(['accuracy', 'val_accuracy'])
plt.savefig(PROJECT_PATH + "models/" + model_name + "/accuracy.png")

print("done")
