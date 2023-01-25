from keras.preprocessing.image import ImageDataGenerator

# Create a data generator
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Provide the directory of the image files
dir = 'path/to/image/directory'

# Load the images
generator = datagen.flow_from_directory(
        dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

# Generate the augmented images
x_augmented, y_augmented = next(generator)
