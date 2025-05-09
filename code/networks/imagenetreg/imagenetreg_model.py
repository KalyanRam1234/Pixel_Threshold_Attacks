from networks.model import Model

import os
import json

from tqdm import tqdm
# class ImagenetModelreg(Model):
#     def __init__(self, args):
#         self.input_shape = (224,224,3)

#         self.size = 224
#         Model.__init__(self,args)
    
#     def dataset(self):
#         import tensorflow as tf
#         import os
#         import json

#         self.train_dir = '/home/kalyan/Xview/DualQualityAssessment/imagenet/SemiImagenet/train'
#         self.val_dir   = '/home/kalyan/Xview/DualQualityAssessment/imagenet/SemiImagenet/val'

#         # Load class mapping
#         with open('/home/kalyan/Xview/DualQualityAssessment/imagenet/Labels.json') as f:
#             label_map = json.load(f)

#         self.class_names = list(label_map.values())
#         self.num_classes = len(self.class_names)
#         self.dataset_name = 'SemiImagenet'

#         self.mean = [123.68, 116.78, 103.94]
#         self.std  = [1., 1., 1.]
#         self.img_rows, self.img_cols, self.img_channels = self.size, self.size, 3
#         self.input_shape = (self.img_rows, self.img_cols, self.img_channels)

#         AUTOTUNE = tf.data.experimental.AUTOTUNE

#         def preprocess_image(image, label):
#             image = tf.image.resize(image, [self.img_rows, self.img_cols])
#             image = tf.cast(image, tf.float32)
#             image = image - tf.constant(self.mean)
#             label = tf.one_hot(label, self.num_classes)
#             return image, label

#         def augment_image(image, label):
#             image = tf.image.random_flip_left_right(image)
#             return image, label

#         # Load datasets without validation split
#         train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#             self.train_dir,
#             labels='inferred',
#             label_mode='int',
#             seed=123,
#             image_size=(self.img_rows, self.img_cols),
#             batch_size=self.batch_size
#         )

#         train_ds = train_ds.take(len(train_ds) // 12)

#         val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#             self.val_dir,
#             labels='inferred',
#             label_mode='int',
#             seed=123,
#             image_size=(self.img_rows, self.img_cols),
#             batch_size=self.batch_size
#         )

#         val_ds = val_ds.take(len(val_ds)//4)

#         # Process datasets
#         self.processed_train_dataset = (
#             train_ds.map(augment_image, num_parallel_calls=AUTOTUNE)
#                     .map(preprocess_image, num_parallel_calls=AUTOTUNE)
#                     .prefetch(AUTOTUNE)
#         )

#         self.processed_test_dataset = (
#             val_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
#                 .prefetch(AUTOTUNE)
#         )

#         self.raw_train_dataset = self.processed_train_dataset
#         self.raw_test_dataset = self.processed_test_dataset

#         def extract_x_y(dataset):
#             x_list = []
#             y_list = []
#             for batch in tqdm(dataset):
#                 x_batch, y_batch = batch
#                 x_list.append(x_batch)
#                 y_list.append(y_batch)
#             x = tf.concat(x_list, axis=0)
#             y = tf.concat(y_list, axis=0)
#             return x, y

#         # Extract raw_x and raw_y from processed datasets
#         self.raw_x_train, self.raw_y_train = extract_x_y(self.raw_train_dataset)
#         self.raw_x_test, self.raw_y_test = extract_x_y(self.raw_test_dataset)

#         self.processed_x_train = self.raw_x_train
#         self.processed_x_test = self.raw_x_test

#         self.processed_y_train = self.raw_y_train
#         self.processed_y_test = self.raw_y_test

#         self.iterations_train = len(train_ds)
#         self.iterations_test = len(val_ds)

#!/usr/bin/env python
class ImagenetModelreg(Model):
    """
    ImageNet Model Loader with adjustable images per class
    """

    def __init__(self, args):
        """
        Initialize ImagenetModelreg
        """
        self.input_shape = (224, 224, 3)  # Standard ImageNet input size
        Model.__init__(self, args)

    def dataset(self):
        """
        Prepare ImageNet Dataset
        """

        from tensorflow.keras.preprocessing import image_dataset_from_directory
        from tensorflow.keras import utils
        import numpy as np
        from tqdm import tqdm
        import os

        # Set your train and validation directories
        self.train_dir = '/home/kalyan/Xview/DualQualityAssessment/imagenet/SemiImagenet/train'
        self.val_dir   = '/home/kalyan/Xview/DualQualityAssessment/imagenet/SemiImagenet/val'

        self.num_images = {'train': 0, 'test': 0}  # Will update later

        # Normalization parameters (scaled to 0-255 range)
        self.mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        self.std  = [0.229 * 255, 0.224 * 255, 0.225 * 255]

        # Create full datasets first (no batching or shuffling)
        train_ds = image_dataset_from_directory(
            self.train_dir,
            labels='inferred',
            label_mode='int',
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size
        )

        val_ds = image_dataset_from_directory(
            self.val_dir,
            labels='inferred',
            label_mode='int',
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size
        )

        # Class names
        self.class_names = train_ds.class_names
        self.num_classes = len(self.class_names)
        self.dataset_name = 'ImageNet'

        # Optional: limit number of images per class
        max_images_per_class = 40  # if self.MAX_IMAGES_PER_CLASS is set, use it

        # Initialize storage
        class_counter_train = {i: 0 for i in range(self.num_classes)}
        class_counter_val   = {i: 0 for i in range(self.num_classes)}

        train_images, train_labels = [], []
        print("Loading training images:")

        for batch_imgs, batch_labels in tqdm(train_ds, total=len(train_ds)):
            for img, label in zip(batch_imgs, batch_labels):
                label_int = int(label)  # Convert tensor to Python int
                if (max_images_per_class is None) or (class_counter_train[label_int] < max_images_per_class):
                    train_images.append(img.numpy())    # Keep as Tensor
                    train_labels.append(label)  # Keep as Tensor
                    class_counter_train[label_int] += 1

        test_images, test_labels = [], []
        print("Loading validation images:")

        for batch_imgs, batch_labels in tqdm(val_ds, total=len(val_ds)):
            for img, label in zip(batch_imgs, batch_labels):
                label_int = int(label)
                if (max_images_per_class is None) or (class_counter_val[label_int] < max_images_per_class):
                    test_images.append(img.numpy())
                    test_labels.append(label)
                    class_counter_val[label_int] += 1

        # Stack into numpy arrays
        self.raw_x_train = np.stack(train_images, axis=0)
        self.raw_y_train = np.array(train_labels)

        self.raw_x_test = np.stack(test_images, axis=0)
        self.raw_y_test = np.array(test_labels)

        self.num_images['train'] = self.raw_x_train.shape[0]
        self.num_images['test'] = self.raw_x_test.shape[0]

        # Color preprocess (normalize)
        self.processed_x_train = self.color_preprocess(self.raw_x_train, True)
        self.processed_x_test = self.color_preprocess(self.raw_x_test, False)

        # One-hot encoding of labels
        self.processed_y_train = utils.to_categorical(self.raw_y_train, self.num_classes)
        self.processed_y_test = utils.to_categorical(self.raw_y_test, self.num_classes)

        # Iterations
        self.iterations_train = (self.num_images['train'] // self.batch_size) + 1
        self.iterations_test  = (self.num_images['test']  // self.batch_size) + 1

        print(f"Done loading {self.dataset_name} dataset with {self.num_classes} classes.")