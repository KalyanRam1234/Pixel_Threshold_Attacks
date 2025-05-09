from networks.cifar.cifar_model import CifarModel
import tensorflow as tf
import numpy as np
from tensorflow.keras import initializers, layers, regularizers
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)

class YoloV3(CifarModel):
    """
    TODO: Write Comment
    """

    def __init__(self, args):
        """
        TODO: Write Comment
        """

        self.name = 'YoloV3'
        self.weight_decay = 0.0001

        CifarModel.__init__(self, args)

    yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
    yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

    yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
                                (81, 82), (135, 169),  (344, 319)],
                                np.float32) / 416
    yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])


    def DarknetConv(self,x, filters, size, strides=1, batch_norm=True):
        if strides == 1:
            padding = 'same'
        else:
            x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
            padding = 'valid'
        x = Conv2D(filters=filters, kernel_size=size,
                strides=strides, padding=padding,
                use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
        if batch_norm:
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)
        return x


    def DarknetResidual(self,x, filters):
        prev = x
        x = self.DarknetConv(x, filters // 2, 1)
        x = self.DarknetConv(x, filters, 3)
        x = Add()([prev, x])
        return x


    def DarknetBlock(self,x, filters, blocks):
        x = self.DarknetConv(x, filters, 3, strides=2)
        for _ in range(blocks):
            x = self.DarknetResidual(x, filters)
        return x

    def DarknetTiny(self,name=None):
        x = inputs = Input([None, None, 3])
        x = self.DarknetConv(x, 16, 3)
        x = MaxPool2D(2, 2, 'same')(x)
        x = self.DarknetConv(x, 32, 3)
        x = MaxPool2D(2, 2, 'same')(x)
        x = self.DarknetConv(x, 64, 3)
        x = MaxPool2D(2, 2, 'same')(x)
        x = self.DarknetConv(x, 128, 3)
        x = MaxPool2D(2, 2, 'same')(x)
        x = x_8 = self.DarknetConv(x, 256, 3)  # skip connection
        x = MaxPool2D(2, 2, 'same')(x)
        x = self.DarknetConv(x, 512, 3)
        x = MaxPool2D(2, 1, 'same')(x)
        x = self.DarknetConv(x, 1024, 3)
        return tf.keras.Model(inputs, (x_8, x), name=name)



    def YoloConvTiny(self,filters, name=None):
        def yolo_conv(x_in):
            if isinstance(x_in, tuple):
                inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
                x, x_skip = inputs

                # concat with skip connection
                x = self.DarknetConv(x, filters, 1)
                x = UpSampling2D(2)(x)
                x = Concatenate()([x, x_skip])
            else:
                x = inputs = Input(x_in.shape[1:])
                x = self.DarknetConv(x, filters, 1)

            return Model(inputs, x, name=name)(x_in)
        return yolo_conv


    # def YoloOutput(self,filters, anchors, classes, name=None):
    #     def yolo_output(x_in):
    #         x = inputs = Input(x_in.shape[1:])
    #         x = self.DarknetConv(x, filters * 2, 3)
    #         x = self.DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
    #         x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
    #                                             anchors, classes + 5)))(x)
    #         return tf.keras.Model(inputs, x, name=name)(x_in)
    #     return yolo_output

    def ClassifierOutput(self, x_in):
        x = inputs = Input(x_in.shape[1:])

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(
            self.num_classes, 
            activation='softmax', 
            kernel_initializer=initializers.he_normal(), 
            kernel_regularizer=regularizers.l2(self.weight_decay),
            name='Output'
        )(x)

        return tf.keras.Model(inputs, x, name='classifier_output')(x_in)

    def YoloV3Tiny(self, size=None, channels=3, classes=80, training=False):
        x = inputs = Input([size, size, channels], name='input')
        
        x_8, x = self.DarknetTiny(name='yolo_darknet')(x)

        x = self.YoloConvTiny(256, name='yolo_conv_0')(x)

        # Use simple classifier head now
        output = self.ClassifierOutput(x)

        return Model(inputs, output, name='yolov3_tiny_classifier')

    # def YoloV3Tiny(self, size=None, channels=3, anchors=yolo_tiny_anchors,
    #             masks=yolo_tiny_anchor_masks, classes=80, training=False):
        
    #     x = inputs = Input([size, size, channels], name='input')

    #     x_8, x = self.DarknetTiny(name='yolo_darknet')(x)

    #     x = self.YoloConvTiny(256, name='yolo_conv_0')(x)
    #     output_0 = YoloOutput(256, len(masks[0]), classes, name='yolo_output_0')(x)

    #     x = YoloConvTiny(128, name='yolo_conv_1')((x, x_8))
    #     output_1 = YoloOutput(128, len(masks[1]), classes, name='yolo_output_1')(x)

    #     if training:
    #         return Model(inputs, (output_0, output_1), name='yolov3')

    #     boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
    #                     name='yolo_boxes_0')(output_0)
    #     boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
    #                     name='yolo_boxes_1')(output_1)
    #     outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
    #                     name='yolo_nms')((boxes_0[:3], boxes_1[:3]))
    #     return Model(inputs, outputs, name='yolov3_tiny')

    def network(self, img_input):
        """
        TODO: Write Comment
        """

        return self.YoloV3Tiny(self.input_shape[0])(img_input)

    def scheduler(self, epoch):
        """
        TODO: Write Comment
        """

        if epoch < 80: 
            return 0.1
        if epoch < 150: 
            return 0.01
        return 0.001