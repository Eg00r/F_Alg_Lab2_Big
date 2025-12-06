""" Точка входа в приложение """
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Убираем из консоли служебную информацию от TensorFlow
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import cv2
import numpy as np

DS_TRAIN_BRAIN_DIR = '/Brain-MRI/dataset/Training'  
DS_TRAIN_HORSE_DIR = '/horse_512/data/train'  

DS_TEST_BRAIN_DIR = '/Brain-MRI/dataset/Testing'  
DS_TEST_HORSE_DIR = '/horse_512/data/validation'  
WORK_SIZE = (200, 200)  # Размер изображений для нейронки
NUM_CLASSES = 2 # Количество классов распознаваемых объектов, зависит от Вашей задачи

#Тут задаём саму модель нейронной сети
def create_conv_model():
    input_shape = (WORK_SIZE[0],WORK_SIZE[1],1) # Размерность входных данных. В данном случае изображения 200х200 пикселей 1 цвета (ЧБ).
    model = Sequential() # Создаём последовательную модель
   
    model.add(Conv2D(16, kernel_size=(16, 16), activation='relu', input_shape=input_shape)) 
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Dropout(0.3))

    model.add(Conv2D(32, kernel_size=(16, 16), activation='relu', input_shape=input_shape)) 
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.3))

    model.add(Conv2D(64, kernel_size=(16, 16), activation='relu', input_shape=input_shape)) 
    model.add(MaxPooling2D(pool_size=(2,2)))
   
    model.add(Flatten()) # Превращаем многомерный массив (у нас тут он 5х5х64 в одномерный
    model.add(Dense(64, activation='relu')) # Подсоединяем классический полносвязный слой из 64 нейронов
    model.add(Dropout(0.5)) # Данный слой будет отключать каждую связь между слоями с вер-тью 50% для эфф. обучения
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy']) # Компилируем модель
    print(model.summary()) # Печатаем в консоль структуру нашей модели с размерами данных между слоями и числом параметров
    return model

# Здесь и далее просто готовим данные (картинки) для работы с ними
def load_dataset(ds_dir):
    """ Загружаем датасеты x_train, y_train, x_test, y_test"""
    x_train, y_train = load_ds_train(ds_dir)  # Загружаем тренировочные данные
    x_test, y_test = load_ds_test(ds_dir)  # Загружаем тестовые данные
    return x_train, y_train, x_test, y_test

def load_ds_train(ds_dir):
    """ Загружаем датасеты тренировочные x_train, y_train"""
    x_train = []
    y_train = []
    
    tmp_dir = str(ds_dir + DS_TRAIN_BRAIN_DIR)
    x_train, y_train = load_ds_image(x_train, y_train, tmp_dir, 1)
    
    tmp_dir = str(ds_dir + DS_TRAIN_HORSE_DIR)
    x_train, y_train = load_ds_image(x_train, y_train, tmp_dir, 0)
    return x_train, y_train


def load_ds_test(ds_dir):
    """ Загружаем датасеты тестовые x_test, y_test"""
    x_test = []
    y_test = []
    
    tmp_dir = str(ds_dir + DS_TEST_BRAIN_DIR)
    x_test, y_test = load_ds_image(x_test, y_test, tmp_dir, 1)
    
    tmp_dir = str(ds_dir + DS_TEST_HORSE_DIR)
    x_test, y_test = load_ds_image(x_test, y_test, tmp_dir, 0)
    return x_test, y_test


def load_ds_image(x, y, dir, goodflag):
    """ Загрузка изображений из dir в x, флаг в y """
    tmp_dir = str(dir)
    filelist = os.listdir(tmp_dir)
    for i in filelist:
        tmp_img = load_img_from_file(str(tmp_dir+"/"+  i), WORK_SIZE)
        x.append(tmp_img)
        y.append(int(goodflag))  
    return x, y


def load_img_from_file(fname, imgsize=WORK_SIZE):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.resize(img, imgsize)
    img = np.expand_dims(img, axis=2) 
    return img


# Функция обучения полученной модели
def learn_conv_model(model):
    ds_dir = "dataset"
    x_train, y_train, x_test, y_test = load_dataset(ds_dir)
    # Приводим данные к нужному виду
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
   
    # В нужные массивы
    x_train = np.array(x_train, dtype=np.float64)
    x_test = np.array(x_test, dtype=np.float64)
    
    # Приводим к отрезку [0;1]
    x_train /= 255
    x_test /= 255
    
    # batch_size - размер партии для обучения, 1 - вся. epochs - число циклов обучения (чем больше - тем лучше, 1-2тыс)
    model.fit(x_train, y_train, batch_size = 1, epochs=8, verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    
    print('Потери на тесте:', score[0])
    print('Точность на тесте:', score[1])
    print("Baseline Error: %.2f%%" % (100 - score[1] * 100))
    model.save('testIT3.h5') # Сохраняем в файл
    print("Модель сохранена как testIT3.h5")


if __name__ == '__main__':

    model = create_conv_model() 
    learn_conv_model(model) 
