# -*- coding: utf-8 -*-
# http://yusuke-ujitoko.hatenablog.com/entry/2017/05/09/211400
# https://github.com/Ujitoko/keras_trial/tree/master/DCGAN
# 

import os
import  numpy  as  np
from  tqdm  import  tqdm
import  matplotlib.pyplot  as plt


import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Reshape, Flatten, Dropout
from keras.layers.convolutional import UpSampling2D, Conv2D

from keras.optimizers import SGD, Adam
from keras.datasets import mnist
from keras.regularizers import l1_l2

# noise[[examples, 100]]から生成した画像をplot_dim(例えば4x4)で表示
def plot_generated(noise, generator_model, examples=16, plot_dim=(4,4), size=(7,7), epoch=None):
    # noiseからgeneratorで画像を生成
    generated_images = generator_model.predict(noise)

    # 表示
    fig = plt.figure(figsize=size)
    for i in range(examples):
        plt.subplot(plot_dim[0], plot_dim[1], i+1)
        img = generated_images[i, :]
        img = img.reshape((28, 28))
        plt.tight_layout()
        plt.imshow(img, cmap="gray")
        plt.axis("off")
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.savefig(os.path.join("generated_figures", str(epoch) + ".png"))
    plt.close()

def plot_metrics(metrics, epoch=None):
    plt.figure(figsize=(10,8))
    plt.plot(metrics["d"], label="discriminative loss", color="b")
    plt.legend()
    plt.savefig(os.path.join("metrics", "dloss" + str(epoch) + ".png"))
    plt.close()

    plt.figure(figsize=(10,8))
    plt.plot(metrics["g"], label="generative loss", color="r")
    plt.legend()
    plt.savefig(os.path.join("metrics", "g_loss" + str(epoch) + ".png"))
    plt.close()

def Generator():
    Gen = Sequential()
    Gen.add(Dense(input_dim=100, units=1024))
    Gen.add(BatchNormalization())
    Gen.add(Activation("relu"))
    Gen.add(Dense(units=128*7*7))
    Gen.add(BatchNormalization())
    Gen.add(Activation("relu"))
    Gen.add(Reshape((7,7,128), input_shape=(128*7*7,)))
    Gen.add(UpSampling2D((2,2)))
    Gen.add(Conv2D(64, 5, padding="same"))
    Gen.add(BatchNormalization())
    Gen.add(Activation("relu"))
    Gen.add(UpSampling2D((2,2)))
    Gen.add(Conv2D(1, 5, padding="same"))
    Gen.add(Activation("tanh"))
    generator_optimizer = SGD(lr=0.1, momentum=0.3, decay=1e-5)
    # Generatorは単独で訓練しないので下のcompileはいらない
    Gen.compile(loss="binary_crossentropy", optimizer=generator_optimizer)
    return Gen

def Discriminator():
    act = keras.layers.advanced_activations.LeakyReLU(alpha=0.2)
    Dis = Sequential()
    Dis.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(2,2), padding="same", input_shape=(28,28,1)))
    Dis.add(act)
    Dis.add(Conv2D(filters=128, kernel_size=(5, 5), strides=(2,2), padding="same"))
    Dis.add(act)
    Dis.add(Flatten())
    Dis.add(Dense(units=256))
    Dis.add(act)
    Dis.add(Dropout(0.5))
    Dis.add(Dense(1))
    Dis.add(Activation("sigmoid"))
    discriminator_optimizer = Adam(lr=1e-5, beta_1=0.1)
    Dis.compile(loss="binary_crossentropy", optimizer=discriminator_optimizer)
    return Dis

def Generative_Adversarial_Network(generator_model, discriminator_model):
    GAN = Sequential()
    GAN.add(generator_model)
    discriminator_model.trainable=False
    GAN.add(discriminator_model)
    gan_optimizer = Adam(lr=1e-5, beta_1=0.1)
    GAN.compile(loss="binary_crossentropy", optimizer=gan_optimizer)
    return GAN

def main_train(z_input_size, generator_model, discriminator_model, gan_model, loss_dict, X_train, generated_figures=None, z_group=None, z_plot_freq=200, epoch=1000, plot_freq=25, batch=100):
    
    # tqdmでプログレスバー表示
    with tqdm(total=epoch) as pbar:
        for e in range(epoch):
            pbar.update(1)

            # 生成データをノイズから生成
            noise = np.random.uniform(0, 1, size=[batch, z_input_size])
            generated_images = generator_model.predict_on_batch(noise)

            # 訓練データをMNISTデータ群から抜粋
            rand_train_index = np.random.randint(0, X_train.shape[0], size=batch)
            image_batch = X_train[rand_train_index, :]

            # 訓練データと生成データを結合
            X = np.vstack((image_batch, generated_images))
            # ラベル作成
            y = np.zeros(int(2*batch))
            y[batch:] = 1
            y = y.astype(int)

            # discriminatorの学習
            discriminator_model.trainable = True
            d_loss = discriminator_model.train_on_batch(x=X, y=y)
            discriminator_model.trainable = False

            # generatorの学習
            noise = np.random.uniform(0, 1, size=[batch, z_input_size])
            y = np.zeros(batch)
            y = y.astype(int)
            g_loss = gan_model.train_on_batch(x=noise, y=y)

            loss_dict["d"].append(d_loss)
            loss_dict["g"].append(g_loss)

            # グラフ描画
            if e%plot_freq == plot_freq-1:
                plot_metrics(loss_dict, int(e/plot_freq))
                generator_model.save('./model/gen_model_' +  str(int(e/z_plot_freq)) + '.h5')
                generator_model.save_weights('./model/gen_model_weights_' +  str(int(e/z_plot_freq)) + '.h5')
                gan_model.save('./model/gan_model_' + str(int(e/z_plot_freq)) + '.h5')
                gan_model.save_weights('./model/gan_model_weights_' +  str(int(e/z_plot_freq)) + '.h5')                

            # 訓練したgeneratorによる生成画像を可視化
            if e < epoch:
                if e%z_plot_freq == z_plot_freq-1:
                    plot_generated(z_group, generator_model=generator_model, epoch=int(e/z_plot_freq))
                    #generated_figures.append(fig)



(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train.astype('float32')
X_test.astype('float32')
X_train = X_train/255
X_test = X_test/255

print("X_train shape", X_train.shape)
print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")

# モデル生成
Gen = Generator()
Dis = Discriminator()
GAN = Generative_Adversarial_Network(Gen, Dis)
#GAN.summary()
#Gen.summary()
#Dis.summary()

# パラメータ設定
gan_losses = {"d":[], "g":[], "f":[]}
epoch = 200000
batch = 64
z_plot_freq = 1000
plot_freq = 1000
z_input_vector = 100
n_train_samples = 60000
examples = 16

z_group_matrix = np.random.uniform(0,1,examples*z_input_vector)
z_group_matrix = z_group_matrix.reshape([16, z_input_vector])
print(z_group_matrix.shape)

generated_figures = []

a = Gen.predict_on_batch(z_group_matrix)
a.shape

main_train(100, Gen, Dis, GAN, loss_dict=gan_losses, X_train=X_train, generated_figures=generated_figures, z_group=z_group_matrix, z_plot_freq=z_plot_freq, epoch=epoch, plot_freq=plot_freq, batch=batch)
