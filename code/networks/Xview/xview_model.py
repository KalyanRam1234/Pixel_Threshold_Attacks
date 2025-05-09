from networks.model import Model

import os
import json
from tqdm import tqdm
from PIL import Image, ImageEnhance  # Added this
import numpy as np
import tensorflow as tf

class XviewModel(Model):
    def __init__(self, args):
        self.input_shape = (112, 112, 3)
        self.size = 112

        Model.__init__(self, args)

    def data_generator(self,split):
        dataset_path = "/home/kalyan/Xview/datasets"
        image_dir = os.path.join(dataset_path, "images", "train")
        label_dir = os.path.join(dataset_path, "labels", "train")

        split_txt = os.path.join(dataset_path, "images", f"autosplit_{split}.txt")
        with open(split_txt, "r") as f:
            image_ids = [line.strip() for line in f.readlines()]

        

        for img_id in tqdm(image_ids):
            # print(img_id)
            img_id_split = img_id.split("/")[-1].split(".tif")[0]
            image_path = os.path.join(image_dir, f"{img_id_split}.tif")
            label_path = os.path.join(label_dir, f"{img_id_split}.txt")

            img = Image.open(image_path).convert("RGB")
            img_w, img_h = img.size

            if os.path.exists(label_path):
                with open(label_path, "r") as lf:
                    for line in lf.readlines():
                        class_id, x_center, y_center, w, h = map(float, line.strip().split())

                        x_center *= img_w
                        y_center *= img_h
                        w *= img_w
                        h *= img_h

                        x_min = max(0, int(x_center - w / 2))
                        y_min = max(0, int(y_center - h / 2))
                        x_max = min(img_w, int(x_center + w / 2))
                        y_max = min(img_h, int(y_center + h / 2))

                        cropped = img.crop((x_min, y_min, x_max, y_max))
                        cropped = cropped.resize((self.size, self.size), Image.LANCZOS)
                        enhancer = ImageEnhance.Sharpness(cropped)
                        cropped = enhancer.enhance(1.5)
                        cropped = np.asarray(cropped).astype(np.float32) / 255.0

                        yield cropped, class_id

    def create_dataset(self,split):
        dataset = tf.data.Dataset.from_generator(
            lambda: self.data_generator(split),
            output_signature=(
                tf.TensorSpec(shape=(self.size, self.size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
        dataset = dataset.map(lambda x, y: (x, tf.one_hot(y, depth=self.num_classes)), num_parallel_calls=tf.data.AUTOTUNE)

        # dataset = dataset.shuffle(1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    def dataset(self):
        import tensorflow as tf
        import numpy as np

        self.dataset_name = "xView"
        self.num_classes  = 60
        # self.num_classes = 60  # Update this based on your dataset
        self.class_names = [ "Fixed-wing Aircraft", "Small Aircraft", "Cargo Plane", "Helicopter", "Passenger Vehicle", "Small Car", "Bus", "Pickup Truck", "Utility Truck", "Truck", "Cargo Truck", "Truck w/Box", "Truck Tractor", "Trailer", "Truck w/Flatbed", "Truck w/Liquid", "Crane Truck", "Railway Vehicle", "Passenger Car", "Cargo Car", "Flat Car", "Tank car", "Locomotive", "Maritime Vessel", "Motorboat", "Sailboat", "Tugboat", "Barge", "Fishing Vessel", "Ferry", "Yacht", "Container Ship", "Oil Tanker", "Engineering Vehicle", "Tower crane", "Container Crane", "Reach Stacker", "Straddle Carrier", "Mobile Crane", "Dump Truck", "Haul Truck", "Scraper/Tractor", "Front loader/Bulldozer", "Excavator", "Cement Mixer", "Ground Grader", "Hut/Tent", "Shed", "Building", "Aircraft Hangar", "Damaged Building", "Facility", "Construction Site", "Vehicle Lot", "Helipad", "Storage Tank", "Shipping container lot", "Shipping Container", "Pylon", "Tower" ]

        self.mean = [0., 0., 0.]
        self.std = [255., 255., 255.]

        val_ds = self.create_dataset("val")
        train_ds = self.create_dataset("train")
        
        # Shuffle, batch, prefetch separately
        self.train_dataset = train_ds.shuffle(1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        self.val_dataset = val_ds.shuffle(1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        # Also assign self.processed_x_train etc. as Datasets (not numpy arrays)

        # self.raw_x_train, self.raw_y_train = train_ds.map(lambda x, y: x), train_ds.map(lambda x, y: y)
        # self.raw_x_test, self.raw_y_test = val_ds.map(lambda x, y: x), val_ds.map(lambda x, y: y)

        # self.processed_x_train, self.processed_y_train = train_ds.map(lambda x, y: x), train_ds.map(lambda x, y: y)
        # self.processed_x_test, self.processed_y_test = val_ds.map(lambda x, y: x), val_ds.map(lambda x, y: y)

        # for element in self.processed_y_train:
        #     print(element)
        # Count the number of samples efficiently
        self.num_images = {
            "train": sum(1 for _ in train_ds),
            "test": sum(1 for _ in val_ds)
        }

        self.iterations_train = (self.num_images["train"] // self.batch_size) + 1
        self.iterations_test = (self.num_images["test"] // self.batch_size) + 1

        print("Done with dataset creation")
