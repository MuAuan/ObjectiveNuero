
from keras.layers import Lambda, Input, Dense, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import cv2

from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential
from keras.layers import Dropout, Activation, Flatten
from keras.layers import Reshape, Embedding,InputLayer

input_shape = (28, 28, 1)
latent_dim = 3

def model(latent_dim, name='decoder1'):
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(7 * 7 * 64, activation='relu')(latent_inputs)
    x = Reshape((7, 7, 64))(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same')(x)
    outputs = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)
    return Model(latent_inputs, outputs, name=name)

# instantiate decoder model

decoder=("decoder0")  #,"decoder1","decoder2","decoder3","decoder4","decoder5","decoder6","decoder7","decoder8","decoder9")
items=[]
for index, item in enumerate(decoder):
    item_model = model(latent_dim, name=item)
    item_model.load_weights('./k-means/data/decoder_mnist_weightsL2_20.h5')
    items.append(item_model)

items[0].summary()

def plot_results(z0,z1,z20,i):
    """
    z_sample=np.array([[np.array(z0),np.array(z1),np.array(z2)]])
    x_decodeds=[]
    for index, item1 in enumerate(items):  
        x_decoded = item1.predict(z_sample)
        x_decodeds.append(x_decoded)
    """
    fig=plt.figure()
    size=(280,280)
    x_decodeds=[]
    for i in range(0,20,1):
        z2=z20+5-i/2
        z_sample=np.array([[np.array(z0),np.array(z1),np.array(z2)]])
        x_decoded = items[0].predict(z_sample)
        ax=fig.add_subplot(4,5,i+1)
        img=x_decoded.reshape(28, 28)
        img=cv2.resize(img, size,interpolation = cv2.INTER_CUBIC)    
        cv2.imshow("z_sample=[{0:6.3f},{1:6.3f},{2:6.3f}]".format(z0,z1,z2),img)
        ax.imshow(img)
    #plt.savefig("./k-means/data/z/all_{}.png".format(s))  #z0,z1,z2
    plt.savefig("./k-means/data/z/all_[{0:6.3f},{1:6.3f},{2:6.3f}].png".format(z0,z1,z2))  #z0,z1
    
z0=0
z1=0
z2=0.8
s=0
while True:
    img=cv2.imread("./k-means/data/cluster3/digits_over_latent_all0.8.png")
    cv2.circle(img,(int(505+90*(z0)),int(510-80*(z1))), 10, (0,255,255), -1)
    plot_results(z0,z1,z2,s)
    cv2.imshow('image',img)
    s+=1
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
        break
    elif k == ord('j'): # 
        #cv2.imwrite('./k-means/data/z/z_{}.png'.format(s),img) #z0,z1
        cv2.imwrite('./k-means/data/z/z[{0:6.3f},{1:6.3f},{2:6.3f}].png'.format(z0,z1,z2),img) #z0,z1
        z0=z0+1
    elif k == ord('h'): # 
        #cv2.imwrite('./k-means/data/z/z_{}.png'.format(s),img)
        cv2.imwrite('./k-means/data/z/z[{0:6.3f},{1:6.3f},{2:6.3f}].png'.format(z0,z1,z2),img)
        z0=z0-1
    elif k == ord('u'): # 
        #cv2.imwrite('./k-means/data/z/z_{}.png'.format(s),img)
        cv2.imwrite('./k-means/data/z/z[{0:6.3f},{1:6.3f},{2:6.3f}].png'.format(z0,z1,z2),img)
        z1=z1+1
    elif k == ord('n'): # 
        #cv2.imwrite('./k-means/data/z/z_{}.png'.format(s),img)
        cv2.imwrite('./k-means/data/z/z[{0:6.3f},{1:6.3f},{2:6.3f}].png'.format(z0,z1,z2),img)
        z1=z1-1
        