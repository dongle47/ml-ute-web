import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error


st.markdown("# Hồi quy Demo")

st.sidebar.header("Hồi quy Exercise")

def Bai01():
    X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]])
    one = np.ones((1, X.shape[1]))
    Xbar = np.concatenate((one, X), axis = 0) # each point is one row
    y = np.array([[ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
    A = np.dot(Xbar, Xbar.T)
    b = np.dot(Xbar, y)
    w = np.dot(np.linalg.pinv(A), b)
    # weights
    w_0, w_1 = w[0], w[1]
    st.write(w_0, w_1)
    y1 = w_1*155 + w_0
    y2 = w_1*160 + w_0
    st.write('Input 155cm, true output 52kg, predicted output %.2fkg' %(y1))
    st.write('Input 160cm, true output 56kg, predicted output %.2fkg' %(y2))

def Bai02():
    X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
    y = np.array([ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

    regr = linear_model.LinearRegression()
    regr.fit(X, y) # in scikit-learn, each sample is one row
    # Compare two results
    st.write("scikit-learn’s solution : w_1 = ", regr.coef_[0], "w_0 = ", regr.intercept_)

    X = X[:,0]

    fig = plt.figure()
    plt.plot(X, y, 'ro')
    a = regr.coef_[0]
    b = regr.intercept_
    x1 = X[0]
    y1 = a*x1 + b
    x2 = X[12]
    y2 = a*x2 + b
    x = [x1, x2]
    y = [y1, y2]


    plt.plot(x, y)
    st.plotly_chart(fig)

def Bai03():
    m = 100
    X = 6 * np.random.rand(m, 1) - 3
    y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
    X2 = X**2
    X_poly = np.hstack((X, X2))

    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(X_poly, y)

    st.write(lin_reg.intercept_)
    st.write(lin_reg.coef_)

    a = lin_reg.intercept_[0]
    b = lin_reg.coef_[0,0]
    c = lin_reg.coef_[0,1]
    st.write(a)
    st.write(b)
    st.write(c)

    x_ve = np.linspace(-3,3,m)
    y_ve = a + b*x_ve + c*x_ve**2

    fig = plt.figure()
    plt.plot(X, y, 'o')
    plt.plot(x_ve, y_ve, 'r')

    # Tinh sai so
    loss = 0 
    for i in range(0, m):
        y_mu = a + b*X_poly[i,0] + c*X_poly[i,1]
        sai_so = (y[i] - y_mu)**2 
        loss = loss + sai_so
    loss = loss/(2*m)
    st.write('loss = %.6f' % loss)

    y_train_predict = lin_reg.predict(X_poly)
    sai_so_binh_phuong_trung_binh = mean_squared_error(y, y_train_predict)
    st.write('Sai số bình phương trung bình: %.6f' % (sai_so_binh_phuong_trung_binh/2))

    st.plotly_chart(fig)

def Bai04():
    X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
    y = np.array([ 49, 50, 90, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

    regr = linear_model.LinearRegression()
    regr.fit(X, y) 

    st.write("scikit-learn’s solution : w_1 = ", regr.coef_[0], "w_0 = ", regr.intercept_)

    X = X[:,0]
    fig = plt.figure()
    plt.plot(X, y, 'ro')
    a = regr.coef_[0]
    b = regr.intercept_
    x1 = X[0]
    y1 = a*x1 + b
    x2 = X[12]
    y2 = a*x2 + b
    x = [x1, x2]
    y = [y1, y2]

    plt.plot(x, y)
    st.plotly_chart(fig)

def Bai05():
    X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
    y = np.array([ 49, 50, 90, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

    huber_reg = linear_model.HuberRegressor()
    huber_reg.fit(X, y) 

    st.write("scikit-learn’s solution : w_1 = ", huber_reg.coef_[0], "w_0 = ", huber_reg.intercept_)

    X = X[:,0]

    fig = plt.figure()

    plt.plot(X, y, 'ro')
    a = huber_reg.coef_[0]
    b = huber_reg.intercept_
    x1 = X[0]
    y1 = a*x1 + b
    x2 = X[12]
    y2 = a*x2 + b
    x = [x1, x2]
    y = [y1, y2]

    plt.plot(x, y)
    st.plotly_chart(fig)


option = st.sidebar.selectbox('Lựa chọn bài tập',
    ('Bai01', 'Bai02', 'Bai03', 'Bai04', 'Bai05'))


if(option == 'Bai01'):
    Bai01()
if(option == 'Bai02'):
    Bai02()
if(option == 'Bai03'):
    Bai03()
if(option == 'Bai04'):
    Bai04()
if(option == 'Bai05'):
    Bai05()