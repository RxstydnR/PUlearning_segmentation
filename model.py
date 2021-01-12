import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Activation, Dense, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Input, Dropout, MaxPooling2D
from tensorflow.keras.models import Model

def build_model(model_name, input_shape):
    """ Build model customized for PULearning.
    """
    if model_name=="ResNet18":
        model = ResNet18(input_shape=input_shape)

    elif model_name=="DenseNet121":
        model = DenseNet121(include_top=False, input_shape=input_shape, weights=None) # 「include_top=False」でモデルを読み込み、全結合層（Softmaxなど）を自作する。
        x = Activation('relu')(model.output)
        x = GlobalAveragePooling2D()(x) # GAP can be used instead of Flatten()(x)
        x = Dense(256, activation="relu")(x) 
        x = Dense(256, activation="relu")(x) 
        x = Dense(1)(x) 
        model = Model(model.inputs, x)

    elif model_name=="CNN_paper":
        model = CNN_paper(input_shape=input_shape)

    else: 
        model = CNN(input_shape=input_shape)

    return model


def ResNet18(input_shape):
    """ Resnet18 model 
    """
    num_filters = 64
    num_blocks = 4
    num_sub_blocks = 2

    inputs = Input(shape=input_shape)
    
    x = Conv2D(filters=num_filters, kernel_size=(7,7), padding='same', strides=2, kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='block2_pool')(x)

    for i in range(num_blocks):
        for j in range(num_sub_blocks):
            
            strides=1
            
            is_first_layer_but_not_first_block=False
            if j==0 and i>0:
                is_first_layer_but_not_first_block=True
                strides=2

            y = Conv2D(num_filters, kernel_size=3, padding='same', strides=strides, kernel_initializer='he_normal')(x)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            y = Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer='he_normal')(y)
            y = BatchNormalization()(y)
            
            # Skip structure
            if is_first_layer_but_not_first_block:
                x = Conv2D(num_filters, kernel_size=1, padding='same', strides=2, kernel_initializer='he_normal')(x)
            x = tf.keras.layers.add([x, y])
            x = Activation('relu')(x)

        num_filters *= 2

    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="tanh")(x) # It seems that "tanh" is better than relu.
    x = Dense(256, activation="tanh")(x)
    outputs = Dense(1)(x) 
    model = Model(inputs=inputs, outputs=outputs)

    return model



def CNN_paper(input_shape):
    """ CNN model based on the paper. 
        [Positive-Unlabeled Learning with Non-Negative Risk Estimator](https://arxiv.org/abs/1703.00593)
    """

    inputs = Input(shape=input_shape)
    
    x = Conv2D(96, kernel_size=(3,3), padding='same', strides=1, kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(96, kernel_size=(3,3), padding='same', strides=1, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(96, kernel_size=(3,3), padding='same', strides=2, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    for i in range(4):
        if i==2:
            x = Conv2D(192, kernel_size=(3,3), padding='same', strides=2, kernel_initializer='he_normal')(x)
        else:
            x = Conv2D(192, kernel_size=(3,3), padding='same', strides=1, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = Conv2D(192, kernel_size=(1,1), padding='same', strides=1, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(10, kernel_size=(1,1), padding='same', strides=1, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(1000, activation='relu')(x)
    x = Dense(1000, activation='relu')(x)
    outputs = Dense(1)(x)

    return Model(inputs=inputs, outputs=outputs)



def CNN(input_shape):
    """ build simple CNN model 
    """
    inputs = Input(shape=input_shape)
    
    x = Conv2D(256, kernel_size=(3,3), padding='same', strides=2, kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(128, kernel_size=(3,3), padding='same', strides=2, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, kernel_size=(3,3), padding='same', strides=1, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Flatten()(x)
    x = Dense(256, activation="tanh")(x)
    outputs = Dense(1)(x)

    return Model(inputs=inputs, outputs=outputs)


def MLP(input_shape):
    
    inputs = Input(shape=input_shape)
    x = Dense(300)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Dense(300)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Dense(300)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Dense(300)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    outputs = Dense(1)(x)
    
    return Model(inputs=inputs, outputs=outputs)