#!/usr/bin/env python

import numpy as np
import tensorflow as tf

class Model:
    """
    TODO: Write Comment
    """

    def __init__(self, args):
        """
        TODO: Write Comment
        """

        import os
        import tensorflow as tf
        from tensorflow.keras import losses, callbacks, optimizers, utils
        import plot_utils

        self.args = args
        
        self.VERBOSE        = self.args.verbose
        
        self.FAMILY_DATASET = self.args.family_dataset
        self.USE_DATASET    = self.args.use_dataset

        self.NUM_DEFENCE    =  self.args.defence

        self.AUGMENTATION = self.args.augmentation
        
        self.batch_size   = self.args.batch_size
        self.epochs       = self.args.epochs
        

        self.get_defence()

        self.dataset()

        if not self.AUGMENTATION: 
            self.name += '-without_augmentation'

        if self.defence_training == False: 
            try:
                import shutil
                shutil.copytree(f"./logs/models/{self.dataset_name}/{self.name}/", f"./logs/models/{self.dataset_name}/{self.name}{self.defence_name}/")
            except Exception as e:
                print(e)
                pass  

        if self.NUM_DEFENCE == 7:
            self.num_images['train'] = self.num_images['train']*2
            self.iterations_train = (self.num_images['train'] // self.batch_size) + 1 

        self.name += self.defence_name

        self.log_filepath = f"./logs/models/{self.dataset_name}/{self.name}/"
        if not os.path.exists(self.log_filepath): os.makedirs(self.log_filepath)

        self.optimizer = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

        self.cbks  = []
        # self.cbks += [plot_utils.PlotTraining(f"{self.log_filepath}")]
        # self.cbks += [callbacks.ModelCheckpoint(f"{self.log_filepath}model_weights_ckpt.h5", save_weights_only=True, period=10)]
        self.cbks += [callbacks.LearningRateScheduler(self.scheduler)]
             
        
        
    def color_preprocess(self, imgs, istrain=False):
        """
        TODO: Write Comment
        """

        if imgs.ndim < 4: imgs = np.array([imgs])
        imgs = imgs.astype('float32')

        if self.raw_defence: 
            if not self.istrain_defence: 
                imgs = self.defence(imgs)[0]
            else:
                if istrain: imgs = self.defence(imgs)[0]

        for i in range(imgs.shape[-1]): imgs[:,:,:,i] = (imgs[:,:,:,i] - self.mean[i]) / self.std[i]

        if self.processed_defence: imgs = self.defence(imgs)[0]

        return imgs
    
    def color_postprocess(self, imgs):
        """
        TODO: Write Comment
        """

        if imgs.ndim < 4: imgs = np.array([imgs])
        
        imgs = imgs.astype('float32')
        for i in range(imgs.shape[-1]): imgs[:,:,:,i] = (imgs[:,:,:,i] * self.std[i]) + self.mean[i]

        return imgs
    
    def load(self):
        """
        TODO: Write Comment
        """

        if self.VERBOSE: print(f"Loading Model...")
        
        self._model = self.build_model()
        self._model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        
        try:
            self._model.load_weights(f"{self.log_filepath}model_weights.h5")
        except Exception as e:
            print(e)
            self.train()

        if False and self.NUM_DEFENCE > 0 and self.defence_training == False:
            import pickle

            with open(f"{self.log_filepath}history.pkl", 'rb') as file: history = pickle.load(file)

            if self.FAMILY_DATASET !=2:

                acc     = np.sum(np.argmax(self._model.predict(self.processed_x_train), axis=1) == self.raw_y_train)/ self.num_images['train']
                val_acc = np.sum(np.argmax(self._model.predict(self.processed_x_test),  axis=1) == self.raw_y_test)/  self.num_images['test']

            else:
                def get_pred(dataset, steps):
                    import tensorflow as tf

                    ys   = []
                    preds = []
                    count = 0
                    for x, y in dataset.take(steps):
                        pred = self._model.predict(x)
                        preds = np.array(pred) if count == 0 else np.concatenate((preds,  np.array(pred)))
                        ys    = np.array(y)    if count == 0 else np.concatenate((ys,  np.array(y)))
                        count = ys.shape[0]
                        print(f"\rNumber of Images={count}", end = "\r")
                    print()
                    return preds, ys
                    
                pred, y = get_pred(self.processed_train_dataset, self.iterations_train)
                acc = np.sum(np.argmax(pred, axis=1)==np.argmax(y, axis=1))/ self.num_images['train']

                pred, y = get_pred(self.processed_test_dataset, self.iterations_test)
                val_acc = np.sum(np.argmax(pred, axis=1)==np.argmax(y, axis=1))/ self.num_images['test']

            print(acc, val_acc)

            history['accuracy_train'] = [acc]
            history['accuracy_test']  = [val_acc]

            with open(f"{self.log_filepath}history.pkl", 'wb') as file: pickle.dump(history, file)
     
        # utils.plot_model(self._model, show_shapes=True, to_file=f"{self.log_filepath}model.png")
        # self._model.summary()

    def save(self, history):
        """
        TODO: Write Comment
        """

        import pickle

        if self.VERBOSE: print(f"Save Model History and Weights...")
        
        with open(f"{self.log_filepath}history.pkl", 'wb') as file: pickle.dump(history, file)
        
        self._model.save(f"{self.log_filepath}model.h5")
        self._model.save_weights(f"{self.log_filepath}model_weights.h5")

        # plot_utils.plot_training_history([history], self.log_filepath)
    
    def build_model(self):
        """
        TODO: Write Comment
        """ 

        from tensorflow.keras import layers, models
        
        img_input = layers.Input(shape=self.input_shape)
        
        print("Building Model done")
        return models.Model(img_input, self.network(img_input))

    def train(self):
        """
        TODO: Write Comment
        """
        
        print("started training")
        history = self.fit_normal()
        
        self.save(history)
    
    ## OLD CODE
    # getting stuck here
    def fit_model(self, x_train, y_train, x_test, y_test, batch_size, epochs, iterations, cbks, verbose):
        """
        TODO: Write Comment
        """
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.125, height_shift_range=0.125, fill_mode='constant',cval=0.)
        
        datagen.fit(x_train)

        if self.NUM_DEFENCE != 1:
            if self.AUGMENTATION:
                history = self._model.fit_generator(
                    datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=iterations, epochs=epochs, callbacks=cbks, verbose=verbose,
                    # workers=12, use_multiprocessing=True,
                    validation_data=(x_test, y_test)
                )
            else:
                history = self._model.fit(
                    x_train, y_train, batch_size=batch_size, 
                    epochs=epochs, callbacks=cbks, verbose=1,
                    workers=4, use_multiprocessing=True,
                    validation_data=(x_test, y_test)
                )

            return history.history

        else:

            return self.fit_adversarial_traning(datagen.flow(x_train, y_train, batch_size=batch_size), x_train, y_train, x_test, y_test, batch_size, epochs, iterations, cbks, verbose)

    ## OLD Code

    def fit_adversarial_traning(self, generator, x_train, y_train, x_test, y_test, batch_size, epochs, iterations, cbks, verbose):
        
        import tensorflow as tf
        from art import attacks, classifiers
        from tensorflow.keras import losses
        from time import time

        loss_object = losses.CategoricalCrossentropy()
        
        attack = attacks.evasion.ProjectedGradientDescent(
                            classifiers.TensorFlowV2Classifier(
                                self._model, self.num_classes, self.input_shape, 
                                loss_object=loss_object, channel_index=3, 
                                clip_values=(0,255), preprocessing=(self.mean, self.std)
                            ), 
                            eps=8, eps_step=2, max_iter=10, num_random_init=True)
        
        optimizer = self.optimizer

        template = 'Epoch {}, Batch {}, Epoch Time: {:2f} Loss: {:4.2f}, Accuracy: {:4.2f}, Test Loss: {:4.2f}, Test Accuracy: {:4.2f}'
        
        history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': [], 'epoch_time': []}

        for i in range(epochs):

            x, y = [], []

            start = time()
            train_loss     = []
            train_accuracy = []

            for batch_id in range(iterations):

                print(template.format(i + 1,
                                  batch_id +1,
                                  (time()-start)/(i+1),
                                  np.mean(np.array(train_loss)),
                                  np.mean(np.array(train_accuracy)),
                                  0,
                                  0))

                self.optimizer.learning_rate = self.scheduler(i)

                if self.AUGMENTATION:
                    x_batch, y_batch = next(generator)
                    x_batch          = x_batch.copy()
                else:

                    x_batch = x_train[ batch_id * batch_size : min((batch_id + 1) * batch_size, self.num_images['train']) ].copy()
                    y_batch = y_train[ batch_id * batch_size : min((batch_id + 1) * batch_size, self.num_images['train']) ]

                fit_x = self._model.train_on_batch(self.color_preprocess(x_batch), y_batch)

                x_batch_adv = x_batch.copy()
                x_batch_adv = attack.generate(x_batch.astype(np.float32), y=y_batch)

                fit_adv_x = self._model.train_on_batch(self.color_preprocess(x_batch_adv), y_batch)

                train_loss     += [fit_x[0] + fit_adv_x[0]]
                train_accuracy += [fit_x[1] + fit_adv_x[1]]
                
            test_scores = self._model.evaluate(x_test, y_test)

            end = time()
            epoch_time = end - start

            history['epoch_time']   += [epoch_time]
            history['loss']         += [np.mean(np.array(train_loss))]
            history['accuracy']     += [np.mean(np.array(train_accuracy))]
            history['val_loss']     += [test_scores[0]]
            history['val_accuracy'] += [test_scores[1]]

            print(template.format(i + 1,
                                  batch_id +1,
                                  epoch_time,
                                  np.mean(np.array(train_loss)),
                                  np.mean(np.array(train_accuracy)),
                                  test_scores[0],
                                  test_scores[1]))

            self._model.save_weights(f"{self.log_filepath}model_weights-epoch{i+1}.h5")

        return history


    ## OLD CODE
    def fit_normal(self):
        """
        TODO: Write Comment
        """
        x_train = self.processed_x_train if self.NUM_DEFENCE != 1 else self.raw_x_train
            
        history = self.fit_model(x_train, self.processed_y_train, self.processed_x_test, self.processed_y_test,  self.batch_size, self.epochs, self.iterations_train, self.cbks, 2)

        if 'CapsNet' in self.name : return {'training_history': history, 'accuracy_train': history['output_accuracy'], 'accuracy_test':  history['val_output_accuracy']}
        else:                      return {'training_history': history, 'accuracy_train': history['accuracy'],        'accuracy_test':  history['val_accuracy']}

    
    def predict(self, img):
        """
        TODO: Write Comment
        """
        
        return self._model.predict(self.color_preprocess(img, False), batch_size=self.batch_size)

    ## OLD CODE

    def get(self, samples):
        """
        TODO: Write Comment
        """ 

        indices = list(range(10000)) 
    # np.random.randint(self.num_images['test'], size=samples)
        
        return indices, self.raw_x_test[indices].astype('float32'), self.raw_y_test[indices]

    # def get(self, samples):
    #     """
    #     Returns sample images and their labels from the test set,
    #     filtered to include only desired classes.
    #     Also returns their original indices from the unbatched dataset.
    #     """
    #     val_data = list(self.val_dataset.unbatch())
    #     desired_y = {0, 1, 55, 56, 57}

    #     x_samples = []
    #     y_samples = []
    #     selected_indices = []

    #     # Shuffle indices to simulate random sampling
    #     shuffled_indices = np.random.permutation(len(val_data))

    #     for idx in shuffled_indices:
    #         x, y = val_data[idx]
    #         y_scalar = tf.argmax(y, axis=-1).numpy()
    #         if y_scalar in desired_y:
    #             x_samples.append(x.numpy() * 255)
    #             y_samples.append(y_scalar)
    #             selected_indices.append(idx)
    #         if len(x_samples) == samples:
    #             break

    #     if len(x_samples) < samples:
    #         print(f"Warning: Only found {len(x_samples)} samples from desired classes.")

    #     x_samples = np.stack(x_samples)
    #     y_samples = np.array(y_samples)

    #     print(x_samples.shape, y_samples.shape)
    #     return selected_indices, x_samples.astype('float32'), y_samples


    # def get(self, samples):
    #     """
    #     Returns sample images and their labels from the test set.
    #     """
    #     # indices = np.random.choice(self.num_images['test'], size=samples, replace=False)
    #     # return indices, self.raw_x_test[indices].astype('float32'), self.raw_y_test[indices]
    #     val_data = list(self.val_dataset.unbatch())


    #     total = len(val_data)
    #     indices = np.random.choice(total, size=samples, replace=False)

    #     x_samples = []
    #     y_samples = []
    #     desired_y = [0,1,55,56,57]

    #     for idx in indices:
    #         x, y = val_data[idx]
    #         x_samples.append((x.numpy())*255)
    #         y_scalar = tf.argmax(y, axis=-1)  # Convert one-hot to scalar
    #         y_samples.append(y_scalar.numpy())
        
    #     x_samples = np.stack(x_samples)
    #     y_samples = np.array(y_samples)

    #     print(x_samples.shape, y_samples.shape)
    #     return indices, x_samples.astype('float32'), y_samples


    
    def get_defence(self):

        self.defence_name      = ""
        self.defence_training  = False
        self.processed_defence = False
        self.raw_defence       = False

        self.label_defence     = False
        self.istrain_defence   = False

        if self.NUM_DEFENCE != 0:
            from art import defences
    
            self.defence_name += "--"
            

            if self.NUM_DEFENCE == 1: 
                self.defence_name += "AdversarialTraining"
                self.defence_training = True

            elif self.NUM_DEFENCE == 2 or self.NUM_DEFENCE == 3:
                self.defence_name += "FeatureSqueezing" 
                self.defence = defences.preprocessor.FeatureSqueezing((0,1), bit_depth=5, apply_fit=True, apply_predict=True)
                self.processed_defence = True
                
                if self.NUM_DEFENCE == 3: 
                    self.defence_name += "Training" 
                    self.defence_training = True

            elif self.NUM_DEFENCE == 4 or self.NUM_DEFENCE == 5:
                self.defence_name += "SpatialSmoothing" 
                self.defence = defences.preprocessor.SpatialSmoothing(window_size=3, channel_index=3, clip_values=None, apply_fit=True, apply_predict=True)                    
                self.processed_defence = True
                
                if self.NUM_DEFENCE == 5: 
                    self.defence_name += "Training" 
                    self.defence_training = True

            elif self.NUM_DEFENCE == 6:
                self.defence_name += "JpegCompressionTraining" 
                self.defence_training = True
                self.defence =  defences.preprocessor.JpegCompression((0,255), quality=75, channel_index=3, apply_fit=True, apply_predict=True)
                self.raw_defence = True
                
            elif self.NUM_DEFENCE == 7:
                self.defence_name += "GausianDataAugmentedTraining"
                self.defence_training = True
                self.defence = defences.preprocessor.GaussianAugmentation(sigma=1.0, augmentation=True, ratio=1.0, clip_values=None, apply_fit=True, apply_predict=False)
                self.raw_defence = True
                self.istrain_defence =True  

            elif self.NUM_DEFENCE == 8:
                self.defence_name += "LabelSmoothingTraining"
                self.defence_training = True
                self.defence = defences.preprocessor.LabelSmoothing(max_value=0.9, apply_fit=True, apply_predict=True)
                self.label_defence = True
                
            elif self.NUM_DEFENCE == 9:
                self.defence_name += "ThermometerEncodingTraining"
                self.defence_training = True
                dim_space = 16
                self.input_shape = (32, 32, dim_space*3)
                self.defence = defences.preprocessor.ThermometerEncoding((0,1), num_space=dim_space, channel_index=3, apply_fit=True, apply_predict=True)
                self.processed_defence = True

            elif self.NUM_DEFENCE == 10 or self.NUM_DEFENCE == 11:
                self.defence_name += "TotalVarianceMinimisation" 
                defence =  defences.preprocessor.TotalVarMin(prob=0.3, norm=2, lamb=0.5, solver='L-BFGS-B', max_iter=10, clip_values=(0,1), apply_fit=True, apply_predict=True)
                self.processed_defence = True
                
                if self.NUM_DEFENCE == 11: 
                    self.defence_name += "Training" 
                    self.defence_training = True

            #To-Do Find Pixel CNN
            elif self.NUM_DEFENCE == 1010:
                self.defence_name += "PixelDefend" 
                defence =  defences.preprocessor.PixelDefend(clip_values=(0,1), eps=32, pixel_cnn=None, apply_fit=True, apply_predict=True)
                self.processed_x_train, self.processed_x_test = defence(self.processed_x_train)[0], defence(self.processed_x_test)[0]