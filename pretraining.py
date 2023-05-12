import os
import numpy as np
from scipy import ndimage  
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import gc 
from imgaug import augmenters as iaa
from sklearn import metrics
from lifelines.utils import concordance_index 

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model


# custom loss for the survival modeling using "nnet"
# refer to the following article for the nnet: A scalable discrete-time survival model for neural networks. PeerJ. 2019 Jan 25;7:e6257. doi: 10.7717/peerj.6257. eCollection 2019.
# for the following functions, "surv_likelihood" and "make_surv_array", copyrights belong to the original authors at https://github.com/MGensheimer/nnet-survival
def surv_likelihood(n_intervals):
    def loss(y_true, y_pred): 
        cens_uncens = 1. + y_true[:,0:n_intervals] * (y_pred-1.) #component for all individuals
        uncens = 1. - y_true[:,n_intervals:2*n_intervals] * y_pred #component for only uncensored individuals
        return keras.backend.sum(-keras.backend.log(keras.backend.clip(keras.backend.concatenate((cens_uncens,uncens)),keras.backend.epsilon(),None)),axis=-1) #return -log likelihood
    return loss
 
def make_surv_array(t,f,breaks): 
    n_samples=t.shape[0]
    n_intervals=len(breaks)-1
    timegap = breaks[1:] - breaks[:-1]
    breaks_midpoint = breaks[:-1] + 0.5*timegap
    y_train = np.zeros((n_samples,n_intervals*2))
    for i in range(n_samples):
        if f[i]: #if failed (not censored)
            y_train[i,0:n_intervals] = 1.0*(t[i]>=breaks[1:]) #give credit for surviving each time interval where failure time >= upper limit
            if t[i]<breaks[-1]: #if failure time is greater than end of last time interval, no time interval will have failure marked
                y_train[i,n_intervals+np.where(t[i]<breaks[1:])[0][0]]=1 #mark failure at first bin where survival time < upper break-point
        else: #if censored
            y_train[i,0:n_intervals] = 1.0*(t[i]>=breaks_midpoint) #if censored and lived more than half-way through interval, give credit for surviving the interval.
    return y_train 


# parameters
dim=42 #images were resized to 42x42x42          
batch_size= 30 
no_epoch=1000

date=20230000
nloop=1
try_no=1 
savepath = PATH 

paths= ["%s/split"%(savepath),
        "%s/log"%(savepath),
        "%s/bestmodel"%(savepath),
        "%s/finalmodel"%(savepath),
        "%s/pred"%(savepath)]

for pathway in paths:
    if os.path.isdir(pathway):
        pass
    else:
        os.makedirs(pathway)        


# number of intervals required for the nnet model, which is a discrete time survival model
breaks=np.concatenate((np.arange(0,96,24), np.arange(174, 174*2, 174)))
n_intervals=len(breaks)-1

# CT image normalization using three different window settings
MIN_BOUND = -1024.0
MAX_BOUND = 100.0
    
def normalize0(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image 

#
MIN_BOUND1 = -850.0
MAX_BOUND1 = -400.0  
    
def normalize1(image):  
    image = (image - MIN_BOUND1) / (MAX_BOUND1 - MIN_BOUND1)
    image[image>1] = 1.
    image[image<0] = 0.
    return image 

#
MIN_BOUND2 = -400.0
MAX_BOUND2 = 400.0
    
def normalize2(image):
    image = (image - MIN_BOUND2) / (MAX_BOUND2 - MIN_BOUND2)
    image[image>1] = 1.
    image[image<0] = 0.
    return image 

# normalization function for tuning and test datasets
def normcat(image, dimension):  
    
    numfiles=image.shape[0]
    image=image.reshape(numfiles, dimension, dimension, dimension)
    
    data_images0=[]
    for i in image:
        newimage0=normalize0(i)
        data_images0.append(newimage0)  

    data_images1=[]
    for i in image:
        newimage1=normalize1(i)
        data_images1.append(newimage1)  

    data_images2=[]
    for i in image:
        newimage2=normalize2(i)
        data_images2.append(newimage2)   
        
    data_images0=np.stack(data_images0,axis=0)
    data_images1=np.stack(data_images1,axis=0)
    data_images2=np.stack(data_images2,axis=0)
    
    data_images0=data_images0.reshape(numfiles, dimension, dimension, dimension, 1)
    data_images1=data_images1.reshape(numfiles, dimension, dimension, dimension, 1)
    data_images2=data_images2.reshape(numfiles, dimension, dimension, dimension, 1) 
    
    concatimage=np.concatenate((data_images0, data_images1, data_images2), axis=-1)
    
    return concatimage

# normalization function fo the generator 
def normcat1(image, dimension): 
     
    data_images0=normalize0(image) 
    data_images1=normalize1(image)
    data_images2=normalize2(image)
         
    data_images0=data_images0.reshape(dimension, dimension, dimension, 1)
    data_images1=data_images1.reshape(dimension, dimension, dimension, 1)
    data_images2=data_images2.reshape(dimension, dimension, dimension, 1) 
    
    concatimage=np.concatenate((data_images0, data_images1, data_images2), axis=-1)
    
    return concatimage 

 

### data load 


def vargen(data, order):   
    var=[]
    numfiles=len(data)
    for ii in range(numfiles):
        row=data.iloc[ii]
        var.append(row[order])
    var=np.array(var, dtype='float32')
    return var


# In[ ]:


def vargen1(data, order):   
    var=[]
    numfiles=len(data)
    for ii in range(numfiles):
        row=data.iloc[ii]
        var.append(row[order])
    var=np.array(var)
    return var


# In[ ]:


def varloader(data):
    if dim==42:
        data_path=vargen1(data, 1)
        data_w=vargen1(data, 3)
        data_h=vargen1(data, 4)
        data_s=vargen1(data, 5)
    elif dim==84:
        data_path=vargen1(data, 2)
        data_w=vargen1(data, 6)
        data_h=vargen1(data, 7)
        data_s=vargen1(data, 8)               
    VPI=vargen(data, 9)
    node=vargen(data, 11)
    LVI=vargen(data, 13)
    death=vargen(data, 14)
    OS=vargen(data, 15) 
    return data_path, data_w, data_h, data_s, VPI, node, LVI, death, OS 


# In[ ]:


def imageloader(data_path, dimension, data_w, data_h, data_s):  
    numfiles=len(data_path)
    
    data_images=[]
    for i in range(numfiles):
        a=np.fromfile(data_path[i], dtype=np.int16)
        a=a.reshape(data_s[i], data_h[i], data_w[i])
        resize_factor=[float(dimension)/data_s[i], float(dimension)/data_h[i], float(dimension)/data_w[i]]
        a=ndimage.interpolation.zoom(a, resize_factor)
        data_images.append(a) 
        
    data_images=np.stack(data_images,axis=0)
    data_images=data_images.reshape(numfiles, dimension, dimension, dimension, 1)      

    return data_images


allData = pd.read_csv("pretrain server0321.csv")
train, splitted = train_test_split(allData, test_size=0.4, shuffle=True, stratify=allData['death'] )
tune, test = train_test_split(splitted, test_size=0.5, shuffle=True, stratify=splitted['death'])

train_df=pd.DataFrame(train)
train_df.to_csv('Train.csv', index=False, header=True)

tune_df=pd.DataFrame(tune)
tune_df.to_csv('Tune.csv', index=False, header=True)

test_df=pd.DataFrame(test)
test_df.to_csv('Test.csv', index=False, header=True)

 
train=train.iloc[0:1050]


# In[ ]:


data_path1, data_w1, data_h1, data_s1, VPI1, node1, LVI1, death1, OS1  = varloader(train)
data_path2, data_w2, data_h2, data_s2, VPI2, node2, LVI2, death2, OS2  = varloader(tune)
data_path3, data_w3, data_h3, data_s3, VPI3, node3, LVI3, death3, OS3  = varloader(test)


# In[ ]:


image1 = imageloader(data_path1, dim, data_w1, data_h1, data_s1) #train
image2 = imageloader(data_path2, dim, data_w2, data_h2, data_s2) #validation
image3 = imageloader(data_path3, dim, data_w3, data_h3, data_s3) #test


# In[ ]:


# not for training, only for the inference
image1NC = normcat(image1, dim) 
image2NC = normcat(image2, dim) 
image3NC = normcat(image3, dim)  


# In[ ]:


IDs = np.arange(image1.shape[0])


# In[ ]:


surv1=make_surv_array(OS1,death1,breaks)  
surv2=make_surv_array(OS2,death2,breaks)
surv3=make_surv_array(OS3,death3,breaks)



# image augmentation 
augFlip1 = iaa.Sometimes(1, iaa.Fliplr(1))
augFlip2 = iaa.Sometimes(1, iaa.Flipud(1)) 
augBlur = iaa.Sometimes(1, iaa.GaussianBlur(sigma=(0.0, 1)))
augSharpen = iaa.Sometimes(1, iaa.Sharpen(alpha=0.2, lightness=0.7))
augNoise = iaa.Sometimes(1, iaa.AdditiveGaussianNoise(scale=(0, 100)))

def augvol(image):
    randomnum = random.randrange(1,16) 
    global imageaug
    if randomnum == 1:
        imageaug = ndimage.rotate(image, axes=(1,0), angle=90, reshape=False)
    if randomnum == 2:
        imageaug = ndimage.rotate(image, axes=(1,0), angle=180, reshape=False)
    if randomnum == 3:
        imageaug = ndimage.rotate(image, axes=(1,0), angle=270, reshape=False)
    if randomnum == 4:
        imageaug = ndimage.rotate(image, axes=(2,0), angle=90, reshape=False) 
    if randomnum == 5:
        imageaug = ndimage.rotate(image, axes=(2,0), angle=180, reshape=False)
    if randomnum == 6:
        imageaug = ndimage.rotate(image, axes=(2,0), angle=270, reshape=False)
    if randomnum == 7:
        imageaug = ndimage.rotate(image, axes=(2,1), angle=90, reshape=False)
    if randomnum == 8:
        imageaug = ndimage.rotate(image, axes=(2,1), angle=180, reshape=False)
    if randomnum == 9:
        imageaug = ndimage.rotate(image, axes=(2,1), angle=270, reshape=False) 
    if randomnum == 10:
        imageaug = np.moveaxis(image, 0, -1)
        imageaug = augFlip1.augment_images(imageaug)
        imageaug = np.moveaxis(imageaug, 2, 0)  
    if randomnum == 11:
        imageaug = np.moveaxis(image, 0, -1)
        imageaug = augFlip2.augment_images(imageaug)
        imageaug = np.moveaxis(imageaug, 2, 0)  
    if randomnum == 12:
        imageaug = augBlur.augment_images(image) 
    if randomnum == 13:
        imageaug = augSharpen.augment_images(image) 
    if randomnum == 14:
        imageaug = augNoise.augment_images(image) 
    if randomnum == 15:
        imageaug = image
    return imageaug   

 

# In[ ]:


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, inputdata1, labels1, labels2, labels3, labels4, 
                 augvol, normcat1, batch_size, inputdimension, n_channels, shuffle):
        
        'Initialization'
        self.list_IDs = list_IDs 
        self.inputdata1 = inputdata1 #image1 
        
        self.labels1 = labels1 #VPI1
        self.labels2 = labels2 #node1
        self.labels3 = labels3 #LVI1
        self.labels4 = labels4 #surv1
        
        self.batch_size = batch_size
        self.inputdimension = inputdimension
        self.n_channels = n_channels        
        self.augvol=augvol  
        self.normcat1=normcat1  
                   
        self.shuffle = shuffle
        self.on_epoch_end()    
     
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))    
    
    def __getitem__(self, index):

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs

        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        # Generate data        
        [X1], [y1, y2, y3, y4] = self.__data_generation(list_IDs_temp)
 
        return [X1], [y1, y2, y3, y4]     
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)    
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        # Initialization
        X1 = np.empty((self.batch_size, *self.inputdimension, self.n_channels))
         
        y1 = np.empty((self.batch_size))  
        y2 = np.empty((self.batch_size))  
        y3 = np.empty((self.batch_size))  
        y4 = np.empty((self.batch_size, 8)) 

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
             
            image = self.inputdata1[ID]
            image = image.reshape(dim,dim,dim)             
            image_aug = self.augvol(image) 
            X1[i,] = self.normcat1(image_aug, dim)             
             
            y1[i] = self.labels1[ID]
            y2[i] = self.labels2[ID]
            y3[i] = self.labels3[ID]
            y4[i] = self.labels4[ID]

        return [X1], [y1, y2, y3, y4]   


# In[ ]:


def dense_factor(inputs, filter2, weight_decay, kernel=1, strides=1):  
    x = layers.BatchNormalization()(inputs)
    x = layers.Activation('relu')(x) 
    x = layers.Conv3D(filter2, 
                      kernel, 
                      strides=strides, 
                      kernel_initializer='he_normal', 
                      padding='same',
                      kernel_regularizer=keras.regularizers.l2(weight_decay)
                     )(x) 
    return x


# In[ ]:


def dense_block(x, filter2, repetition, weight_decay): 
    for i in range(repetition):
        y = dense_factor(x, 4*filter2, weight_decay) 
        y = dense_factor(y, filter2, weight_decay, 3)
        x = layers.concatenate([y,x], axis=-1)
    return x


# In[ ]:


def transition_layer(x, compression_factor, weight_decay): 
 
    x = layers.BatchNormalization(axis=-1, gamma_regularizer=keras.regularizers.l2(weight_decay), beta_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = layers.Activation('relu')(x)
    num_feature_maps = x.shape[-1]
    x = layers.Conv3D( np.floor( compression_factor * num_feature_maps ).astype(int), (1, 1, 1),
               kernel_initializer="he_normal",
               padding="same",
               use_bias=False,
               kernel_regularizer=keras.regularizers.l2(weight_decay))(x) 
    x = layers.AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(x)
    return x


# In[ ]:


def dense_model(filter1, filter2, repetition0, 
                repetition11, repetition12, repetition21, repetition22, 
                repetition31, repetition32, repetition41, repetition42, 
                compression_factor, weight_decay,
                branch1, branch2, branch3, branch4,
                drop1, drop2, drop3, drop4, drop5, drop6, drop7, drop8
               ): 
    
    input1 = keras.Input((dim, dim, dim, 3), name='image')   
                          
    x = layers.BatchNormalization()(input1)
    x = layers.Activation('relu')(x)
    x = layers.Conv3D(filter1, kernel_size=3, kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)   
    
    x = dense_block(x, filter2, repetition0, weight_decay)
    x = transition_layer(x, compression_factor, weight_decay)   
    
    x1=tf.identity(x)
    x2=tf.identity(x)
    x3=tf.identity(x)
    x4=tf.identity(x)
    
    ## 1st output 
    if branch1 == 2:        
        x1 = dense_block(x1, filter2, repetition11, weight_decay)
        x1 = transition_layer(x1, compression_factor, weight_decay) 
        x1 = dense_block(x1, filter2, repetition12, weight_decay)
        x1 = transition_layer(x1, compression_factor, weight_decay)                      
    elif branch1 == 1:
        x1 = dense_block(x1, filter2, repetition11, weight_decay)
        x1 = transition_layer(x1, compression_factor, weight_decay) 
    
    x1 = layers.GlobalAveragePooling3D()(x1)     
    x1 = layers.Dropout(rate=drop1)(x1) 
    f1=layers.Dense(units=8, kernel_initializer='he_normal', bias_initializer=tf.keras.initializers.Constant(0.01))(x1)    
    
    f11 = layers.Dropout(rate=drop2)(f1)
    f11=layers.Dense(units=1, kernel_initializer='zeros', bias_initializer='zeros')(f11)
    output1=layers.Activation('sigmoid', name="VPI")(f11)    
       
    ## 2nd output 
    if branch2 ==2:        
        x2 = dense_block(x2, filter2, repetition21, weight_decay)
        x2 = transition_layer(x2, compression_factor, weight_decay)
        x2 = dense_block(x2, filter2, repetition22, weight_decay)
        x2 = transition_layer(x2, compression_factor, weight_decay)
    elif branch2 ==1:        
        x2 = dense_block(x2, filter2, repetition21, weight_decay)
        x2 = transition_layer(x2, compression_factor, weight_decay)        
    
    x2 = layers.GlobalAveragePooling3D()(x2)     
    x2 = layers.Dropout(rate=drop3)(x2) 
    f2=layers.Dense(units=8, kernel_initializer='he_normal', bias_initializer=tf.keras.initializers.Constant(0.01))(x2)
    
    f21 = layers.Dropout(rate=drop4)(f2)
    f21=layers.Dense(units=1, kernel_initializer='zeros', bias_initializer='zeros')(f21)
    output2=layers.Activation('sigmoid', name="node")(f21)   
    
    ## 3rd output 
    if branch3 ==2:
        x3 = dense_block(x3, filter2, repetition31, weight_decay)
        x3 = transition_layer(x3, compression_factor, weight_decay)
        x3 = dense_block(x3, filter2, repetition32, weight_decay)
        x3 = transition_layer(x3, compression_factor, weight_decay)    
    elif branch3 ==1:
        x3 = dense_block(x3, filter2, repetition31, weight_decay)
        x3 = transition_layer(x3, compression_factor, weight_decay)
        
    x3 = layers.GlobalAveragePooling3D()(x3)     
    x3 = layers.Dropout(rate=drop5)(x3) 
    f3=layers.Dense(units=8, kernel_initializer='he_normal', bias_initializer=tf.keras.initializers.Constant(0.01))(x3)    
    
    f31 = layers.Dropout(rate=drop6)(f3)
    f31=layers.Dense(units=1, kernel_initializer='zeros', bias_initializer='zeros')(f31)
    output3=layers.Activation('sigmoid', name="LVI")(f31) 
    
    ## 4th output 
    if branch2 ==2:
        x4 = dense_block(x4, filter2, repetition41, weight_decay)
        x4 = transition_layer(x4, compression_factor, weight_decay)
        x4 = dense_block(x4, filter2, repetition42, weight_decay)
        x4 = transition_layer(x4, compression_factor, weight_decay)
    elif branch2 ==1:        
        x4 = dense_block(x4, filter2, repetition41, weight_decay)
        x4 = transition_layer(x4, compression_factor, weight_decay)                
        
    x4 = layers.GlobalAveragePooling3D()(x4)    
    x4 = layers.Dropout(rate=drop7)(x4)     
    f4=layers.Dense(units=8, kernel_initializer='he_normal', bias_initializer=tf.keras.initializers.Constant(0.01))(x4)    
     
    
    x5 = layers.concatenate([f1, f2, f3, f4]) 
    
    x5 = layers.Dropout(rate=drop8)(x5)
    f5=layers.Dense(8, kernel_initializer='he_normal', bias_initializer=tf.keras.initializers.Constant(0.01))(x5)
    f5=layers.Dense(n_intervals, kernel_initializer='zeros', bias_initializer='zeros')(f5)   
    output4=layers.Activation('sigmoid', name="surv")(f5)  

    model = keras.Model(inputs=[input1], outputs=[output1, output2, output3, output4], name="sublobar_risk")
    return model


# In[ ]:


params = {'batch_size': batch_size, 
          'inputdimension':(dim,dim,dim),
          'n_channels': 3,
          'shuffle':True
         }

 


mirrored_strategy = tf.distribute.MirroredStrategy()


# In[ ]:


hyperparam=[]  
k=0
for i in range(nloop):  
    
    filter1=random.choice([32, 48, 64])  
    filter2=random.choice([8, 16, 20, 24])
    repetition0=random.choice([2, 3, 4])
    repetition11=random.choice([2, 3, 4])
    repetition12=random.choice([2, 3, 4])
    repetition21=random.choice([2, 3, 4])
    repetition22=random.choice([2, 3, 4])
    repetition31=random.choice([2, 3, 4])
    repetition32=random.choice([2, 3, 4])
    repetition41=random.choice([2, 3, 4])
    repetition42=random.choice([2, 3, 4])  
    branch1=random.choice([1, 2])  
    branch2=random.choice([1, 2]) 
    branch3=random.choice([1, 2]) 
    branch4=random.choice([1, 2])   

    weight_decay = random.choice([1E-5, 5*1E-5, 1E-4, 5*1E-4, 1E-3])
    compression_factor =round(random.uniform(0.6, 0.8), 1)
    initial_LR= round(random.uniform(0.001, 0.01), 3)
    surv_weight = round(random.uniform(0.5, 1.5), 1)

    drop1=random.choice([0.1, 0.15, 0.2, 0.25, 0.3])  
    drop2=random.choice([0.1, 0.15, 0.2, 0.25, 0.3])
    drop3=random.choice([0.1, 0.15, 0.2, 0.25, 0.3])
    drop4=random.choice([0.1, 0.15, 0.2, 0.25, 0.3])
    drop5=random.choice([0.1, 0.15, 0.2, 0.25, 0.3])
    drop6=random.choice([0.1, 0.15, 0.2, 0.25, 0.3])
    drop7=random.choice([0.1, 0.15, 0.2, 0.25, 0.3])
    drop8=random.choice([0.1, 0.15, 0.2, 0.25, 0.3]) 
    
    hyperparam_set =(filter1, filter2, repetition0, 
                     repetition11,repetition12,repetition21,repetition22,repetition31,repetition32,repetition41,repetition42, 
                     compression_factor, weight_decay, initial_LR, surv_weight,
                     branch1, branch2, branch3, branch4,
                     drop1, drop2, drop3, drop4, drop5, drop6, drop7, drop8 
                    )
    hyperparam.append(hyperparam_set)        
    
    cos_decay_ann = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=initial_LR, first_decay_steps=40, t_mul=2, m_mul=0.95, alpha=0.01)
    
    with mirrored_strategy.scope():
        model = dense_model(filter1, filter2, repetition0, 
                            repetition11, repetition12, repetition21, repetition22, 
                            repetition31, repetition32, repetition41, repetition42, 
                            compression_factor, weight_decay,
                            branch1, branch2, branch3, branch4,
                            drop1, drop2, drop3, drop4, drop5, drop6, drop7, drop8 
                           )
        model.compile(loss= {'VPI':'binary_crossentropy',
                             'node':'binary_crossentropy',
                             'LVI':'binary_crossentropy',
                             'surv':surv_likelihood(n_intervals)},
                      loss_weights= {'VPI':1.0,'node': 1.0,'LVI':1.0,'surv':surv_weight}, 
                      optimizer=keras.optimizers.SGD(learning_rate=cos_decay_ann)
                     )
    
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=40) 
    csv_logger = keras.callbacks.CSVLogger('./log/Model log%s.csv' %str(k), append=False, separator=';')    
    checkpointer = keras.callbacks.ModelCheckpoint(filepath='./bestmodel/bestmodel at iter%s.h5' %str(k), verbose=0, 
                                                   save_best_only=True, monitor='val_loss', mode='auto')    
    
    history=model.fit_generator(DataGenerator(IDs, image1, VPI1, node1, LVI1, surv1,
                                              augvol, normcat1, **params),
                                epochs=no_epoch,
                                validation_data=([image2NC], [VPI2, node2, LVI2, surv2]),
                                verbose=2,
                                callbacks=[early_stopping, csv_logger, checkpointer]
                               ) 
    model.save('./finalmodel/finalmodel at iter%s.h5' %str(k))    
    
    del model
    del history
    keras.backend.clear_session()
    gc.collect()
    gc.collect()
    gc.collect()
    gc.collect()
    gc.collect() 
    k+=1

hyperparam_df=pd.DataFrame(hyperparam)
hyperparam_df=hyperparam_df.rename(columns={0:"filter1",1:"filter2",2:"repetition0",
                                            3:"repetition11", 4:"repetition12", 5:"repetition21", 6:"repetition22",
                                            7:"repetition31", 8:"repetition32", 9:"repetition41", 10:"repetition42",
                                            11:"compression_factor", 12:"weight_decay", 13:"initial_LR", 14:"surv_weight",
                                            15:"branch1block", 16:"branch2block", 17:"branch3block", 18:"branch4block",
                                            19:"drop1",20:"drop2",21:"drop3",22:"drop4",23:"drop5",24:"drop6",25:"drop7",26:"drop8" 
                                           })
hyperparam_df.to_csv('./log/hyperparam.csv', index=False, header=True) 
 