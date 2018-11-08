import os
import tensorflow as tf
from tensorflow.python.keras import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import RMSprop,Adam
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array


# define some Environmental variable
NOW_PATH = str(os.getcwd()).replace('\\', '/') + "/"
TRAIN_PATH = os.path.join(NOW_PATH ,'WEBLMT_train_divide/')
VALID_PATH = os.path.join(NOW_PATH,'WEBLMT_test_divide/')
MODEL_PATH = os.path.join(NOW_PATH ,'./model/capcha_model.h5')

DIVIDE_IMAGE_HEIGHT = 16
DIVIDE_IMAGE_WEIGHT = 16

DIVIDE_LABEL_SIZE = 36

BATCH_SIZE = 300
LEARN_RATE = 0.00025

class CaptchaTensorFlow(object):
    def __init__(self, learn_rate=LEARN_RATE, train_path=TRAIN_PATH, valid_path=VALID_PATH, model_path = MODEL_PATH):
        self.learn_rate = learn_rate
        self.train_path = train_path
        self.valid_path = valid_path
        self.model_path = MODEL_PATH
        self.model = self.build_model()
        self.captcha_class = []
        for i in range(36):
            if i < 10:
                self.captcha_class.append(str(i))
            else:
                self.captcha_class.append(chr(i + 87))

    def build_model(self):
        img_input = layers.Input(shape=(DIVIDE_IMAGE_HEIGHT,DIVIDE_IMAGE_WEIGHT,1))
        x = layers.Conv2D(64,3,strides=(1, 1),padding='same',activation='relu')(img_input)
        x = layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2),padding='SAME')(x)

        x = layers.Conv2D(128,3,strides=(1, 1),padding='same',activation='relu')(x)
        x = layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2),padding='SAME')(x)

        x = layers.Conv2D(256,3,strides=(1, 1),padding='same',activation='relu')(x)
        x = layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2),padding='SAME')(x)

        x = layers.Flatten()(x)

        x = layers.Dense(1080,activation='relu')(x)
        x = layers.Dropout(rate = 0.9)(x)

        output = layers.Dense(DIVIDE_LABEL_SIZE,activation='softmax')(x)
        model = Model(img_input,output)
        return model


    def train(self):
        #下面使用高阶语句构建模型
        model = self.build_model()
        model.compile(loss='categorical_crossentropy',
              # optimizer=RMSprop(lr=0.001),
              optimizer = Adam(lr=self.learn_rate),
            #   optimizer=tf.train.AdamOptimizer(learning_rate=self.learn_rate),
              metrics=['accuracy'])

        callbacks = [EarlyStopping(
            monitor='val_loss', patience=2)]

        # All images will be rescaled by 1./255
        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        # Flow training images in batches of 20 using train_datagen generator
        train_generator = train_datagen.flow_from_directory(
                self.train_path,  # This is the source directory for training images
                target_size=(DIVIDE_IMAGE_HEIGHT,DIVIDE_IMAGE_WEIGHT),  
                color_mode='grayscale',
                batch_size=BATCH_SIZE,
                # Since we use binary_crossentropy loss, we need binary labels
                class_mode='categorical')

        # Flow validation images in batches of 20 using test_datagen generator
        validation_generator = test_datagen.flow_from_directory(
                self.valid_path,
                target_size=(DIVIDE_IMAGE_HEIGHT,DIVIDE_IMAGE_WEIGHT),
                color_mode='grayscale',
                batch_size=BATCH_SIZE,
                class_mode='categorical')

        history = model.fit_generator(
            train_generator,
            epochs=1000,
            callbacks=callbacks,
            validation_data=validation_generator,
            verbose=2)
        self.model = model
        model.save(self.model_path)
        #########################这是分界线#################

    def loadModel(self,model_path=None):
        if model_path is not None:
            self.model_path = model_path
        self.model = load_model(self.model_path)
    
    def predict(self,pic_array):
        pred = self.model.predict(pic_array)
        return pred

    def bitsToResult(self,pred):
        pred_list = list(pred.reshape(DIVIDE_LABEL_SIZE))
        return self.captcha_class[pred_list.index(1)]

# test_img = 'WEBLMT_test_divide/2/0010-2.png'
test_img = 'WEBLMT_test_divide/f/0002-0.png'
labels = []
if __name__ == '__main__':
    train_nn = CaptchaTensorFlow()
    choice = input("1、Train\n2、Test\n")
    if choice == '1':
        train_nn.train()
    else:
        train_nn.loadModel()
        pic = load_img(test_img,grayscale=True)
        pic_array = img_to_array(pic).reshape(1,DIVIDE_IMAGE_HEIGHT,DIVIDE_IMAGE_HEIGHT,1)
        pred = train_nn.predict(pic_array)

        result = train_nn.bitsToResult(pred)
        print(result)



