import streamlit as st
import cv2
# import tkinter as tk

from PIL import ImageTk, Image

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import datasets
from skimage import exposure

import imutils

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras 

from sklearn.metrics import classification_report
import joblib


st.markdown("# Knn Demo")

st.sidebar.header("Knn Exercise")

def preprocessing(img):
    try:
        img = img.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        img = img/255
        return img
    except Exception as e:
        img = img.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        img = img/255
        return img

def Bai01():
    np.random.seed(100)
    N = 150

    centers = [[2, 3], [5, 5], [1, 8]]
    n_classes = len(centers)
    data, labels = make_blobs(n_samples=N, 
                            centers=np.array(centers),
                            random_state=1)

    nhom_0 = []
    nhom_1 = []
    nhom_2 = []

    for i in range(0, N):
        if labels[i] == 0:
            nhom_0.append([data[i,0], data[i,1]])
        elif labels[i] == 1:
            nhom_1.append([data[i,0], data[i,1]])
        else:
            nhom_2.append([data[i,0], data[i,1]])

    nhom_0 = np.array(nhom_0)
    nhom_1 = np.array(nhom_1)
    nhom_2 = np.array(nhom_2)

    res = train_test_split(data, labels, 
                       train_size=0.8,
                       test_size=0.2,
                       random_state=1)
    
    train_data, test_data, train_labels, test_labels = res 
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(train_data, train_labels)
    predicted = knn.predict(test_data)
    sai_so = accuracy_score(test_labels, predicted)
    st.write('Sai số:', sai_so)

    my_test = np.array([[2.5, 4.0]])
    ket_qua = knn.predict(my_test)
    st.write('Kết quả nhận đang là nhóm:', ket_qua[0])

    fig = plt.figure()

    plt.plot(nhom_0[:,0], nhom_0[:,1], 'og', markersize = 2)
    plt.plot(nhom_1[:,0], nhom_1[:,1], 'or', markersize = 2)
    plt.plot(nhom_2[:,0], nhom_2[:,1], 'ob', markersize = 2)
    plt.legend(['Nhóm 0', 'Nhóm 1', 'Nhóm 2'])

    st.plotly_chart(fig)

def Bai02():
    mnist = datasets.load_digits()
    (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
        mnist.target, test_size=0.25, random_state=42)

    (trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
        test_size=0.1, random_state=84)

    st.write("Training data points: ", len(trainLabels))
    st.write("Validation data points: ", len(valLabels))
    st.write("Testing data points: ", len(testLabels))

    model = KNeighborsClassifier()
    model.fit(trainData, trainLabels)

    score = model.score(valData, valLabels)
    st.write("Accuracy = %.2f%%" % (score * 100))

    for i in list(map(int, np.random.randint(0, high=len(testLabels), size=(5,)))):
        image = testData[i]
        prediction = model.predict(image.reshape(1, -1))[0]

        image = image.reshape((8, 8)).astype("uint8")

        image = exposure.rescale_intensity(image, out_range=(0, 255))
        image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)

        st.write("Số được nhận dạng là: {}".format(prediction))
        cv2.imshow("Image", image)
        st.image(image, caption='Enter any caption here', clamp=True)

def Bai03():
    mnist = keras.datasets.mnist 
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data() 

    # 784 = 28x28
    RESHAPED = 784
    X_train = X_train.reshape(60000, RESHAPED)
    X_test = X_test.reshape(10000, RESHAPED) 

    (trainData, valData, trainLabels, valLabels) = train_test_split(X_train, Y_train,
        test_size=0.1, random_state=84)

    model = KNeighborsClassifier()
    model.fit(trainData, trainLabels)

    joblib.dump(model, "knn_mnist.pkl")

    predicted = model.predict(valData)
    do_chinh_xac = accuracy_score(valLabels, predicted)
    st.write('Độ chính xác trên tập validation: %.0f%%' % (do_chinh_xac*100))

    predicted = model.predict(X_test)
    do_chinh_xac = accuracy_score(Y_test, predicted)
    st.write('Độ chính xác trên tập test: %.0f%%' % (do_chinh_xac*100))

def Bai03a():
    mnist = keras.datasets.mnist 
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data() 


    index = np.random.randint(0, 9999, 100)
    sample = np.zeros((100,28,28), np.uint8)
    for i in range(0, 100):
        sample[i] = X_test[index[i]]


    # 784 = 28x28
    RESHAPED = 784
    sample = sample.reshape(100, RESHAPED) 
    knn = joblib.load("knn_mnist.pkl")
    predicted = knn.predict(sample)
    k = 0
    text = ''
    for x in range(0, 10):
        for y in range(0, 10):
            # st.write('%2d' % (predicted[k]), end='')
            text += str(predicted[k]) + ' '
            k = k + 1
        # st.write()
    st.write(text)
    digit = np.zeros((10*28,10*28), np.uint8)
    k = 0
    for x in range(0, 10):
        for y in range(0, 10):
            digit[x*28:(x+1)*28, y*28:(y+1)*28] = X_test[index[k]]
            k = k + 1

    cv2.imshow('Digit', digit)

def create(index):
        index = np.random.randint(0, 9999, 100)

def Bai04(index):
    st.title("Nhận diện chữ viết tay")
    st.set_option('deprecation.showfileUploaderEncoding', False)

    mnist = keras.datasets.mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data() 
    X_test = X_test
 
    knn = joblib.load("knn_mnist.pkl")

    digit = np.zeros((10*28,10*28), np.uint8)
    k = 0
    for x in range(0, 10):
        for y in range(0, 10):
            digit[x*28:(x+1)*28, y*28:(y+1)*28] = X_test[index[k]]
            k = k + 1
    cv2.imwrite('digit.jpg', digit)
    image = Image.open('digit.jpg')
    st.image(image)

    sample = np.zeros((100,28,28), np.uint8)
    for i in range(0, 100):
        sample[i] = X_test[index[i]]

    RESHAPED = 784
    sample = sample.reshape(100, RESHAPED) 
    predicted = knn.predict(sample)
    ketqua = ''
    k = 0
    for x in range(0, 10):
        for y in range(0, 10):
            ketqua = ketqua + '%3d' % (predicted[k])
            k = k + 1
        # ketqua = ketqua + '\n'
        st.write(ketqua)
        ketqua = ''
    # st.write(ketqua)

    st.button("Chạy lại")

option = st.sidebar.selectbox('Lựa chọn bài tập',
    ('Bai01', 'Bai02', 'Bai03', 'Bai03a', 'Bai04'))

if(option == 'Bai01'):
    Bai01()
if(option == 'Bai02'):
    Bai02()
if(option == 'Bai03'):
    Bai03()
if(option == 'Bai03a'):
    Bai03a()
if(option == 'Bai04'):
    index = np.random.randint(0, 9999, 100)
    Bai04(index)
# if(option == 'Bai08'):
#     Bai08()
