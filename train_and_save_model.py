import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data_dir = '/content/dataset-full'
rescale = 1./255.
batch_size = 256
target_size = (224, 224)
validation_split = 0.2

epochs = 10
lr = 0.0001
opt = tf.keras.optimizers.Adam(learning_rate = lr)

def generate_train_and_validation_generator(data_dir, rescale, validation_split, batch_size, target_size):
	datagen = ImageDataGenerator(rescale = rescale, validation_split = validation_split)
	train_generator = datagen.flow_from_directory(data_dir,
	                                              batch_size = batch_size,
	                                              class_mode = 'categorical',
	                                              shuffle = True,
	                                              target_size = target_size,
	                                              subset = 'training')     

	validation_generator = datagen.flow_from_directory(data_dir,
	                                                   batch_size = batch_size,
	                                                   class_mode = 'categorical',
	                                                   shuffle = False,
	                                                   target_size = target_size,
	                                                   subset = 'validation')
	return train_generator, validation_generator

def init_model():
	pre_trained_model = tf.keras.applications.MobileNet(include_top=False, weights='imagenet', input_shape=(224, 224, 3)) #Pre-trained model
	final_output = layers.Flatten()(pre_trained_model.output)

	final_output = layers.Dense(256)(final_output)
	final_output = layers.BatchNormalization()(final_output)
	final_output = layers.Activation('relu')(final_output)
	final_output = layers.Dropout(0.5)(final_output)

	final_output = layers.Dense(512)(final_output)
	final_output = layers.BatchNormalization()(final_output)
	final_output = layers.Activation('relu')(final_output)
	final_output = layers.Dropout(0.5)(final_output)

	final_output = layers.Dense(7, activation = 'softmax')(final_output)

	return keras.Model(inputs = pre_trained_model.input, outputs = final_output)

def fit_model(model, opt, train_generator, validation_generator):
	model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics=['accuracy'])

	model_name = "model_weights.h5"
	checkpoint = ModelCheckpoint(model_name, monitor = 'val_accuracy', verbose=1, save_best_only = True, mode = 'max')
	callbacks_list = [checkpoint]

	history = model.fit(train_generator,
	                    steps_per_epoch = train_generator.n // train_generator.batch_size,
	                    epochs = epochs,
	                    validation_data = validation_generator,
	                    validation_steps = validation_generator.n // validation_generator.batch_size,
	                    callbacks=callbacks_list)

	return history

def eval_model_to_png(model_history):
	fig = plt.figure(figsize=(20,10))

	plt.subplot(1, 2, 1)
	plt.ylabel('Accuracy', fontsize=16)
	plt.plot(model_history.history['accuracy'], label='Training Accuracy')
	plt.plot(model_history.history['val_accuracy'], label='Validation Accuracy')
	plt.legend(loc='lower right')

	plt.subplot(1, 2, 2)
	plt.ylabel('Loss', fontsize=16)
	plt.plot(model_history.history['loss'], label='Training Loss')
	plt.plot(model_history.history['val_loss'], label='Validation Loss')
	plt.legend(loc='upper right')

	plt.suptitle('Optimizer : Adam', fontsize=10)
	plt.show()

	fname = 'evaluation.png'
	fig.savefig(fname, transparent = True)

def eval_model_to_csv(model_history):
	columns = ['accuracy', 'loss', 'val_accuracy', 'val_loss']
	df = pd.DataFrame()
	for column in columns:
	    df[column] = ['%.4f' % elem for elem in history.history[column]]

	fname = 'evaluation.csv'
	df.to_csv(fname, header = columns, index = False)

def run():
	train_generator, validation_generator = generate_train_and_validation_generator(data_dir, rescale, validation_split, batch_size, target_size)

	model = init_model()
	model.summary()
	history = fit_model(model, opt, train_generator, validation_generator)

	eval_model_to_png(history)
	eval_model_to_csv(history)

if __name__ == '__main__':
	try:
    	run()
    except:
    	print('Error')
