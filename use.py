import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Убираем из консоли служебную информацию от TensorFlow
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import cv2
import numpy as np
import sys


WORK_SIZE = (200, 200) 

if __name__ == '__main__':
    model = load_model('testIT3.h5') 
    print("Введите путь к файлу")
    tdir = input()
    try:
        img = cv2.imread(tdir)  # Открываем и переводим в градации серого
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        print("Не удалось открыть файл")
        sys.exit(0);

    img = cv2.resize(img, WORK_SIZE) # Уменьшаем до размера нейронки
    # Приводим к виду, который просит нейронка
    img = np.array(img, dtype=np.float64)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=3)
    # Переводим числа к отрезку [0;1]
    img /= 255
    res = model.predict(img) # Получаем результат нейронки к данной картинке

    print("Лошадь с вероятностью ", res[0][0], ", МРТ мозга с вероятностью ", res[0][1])