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
from tensorflow.keras.utils import plot_model
 

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

 

# Parameters 
dim=42 #images were resized to 42x42x42     
batch_size= 20 

no_epoch= 1000
nloop= 1

date=20230000
 
model_no= model_no
model_path = model_path

# number of intervals required for the nnet model, which is a discrete time survival model 
breaks=np.concatenate((np.arange(0,96,24), np.arange(165, 300, 165)))
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


# In[6]:


def normcat(image, dimension): #for tune and test datasets
    
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

def normcat1(image, dimension): #normcat1 is for the generator(trainingset)
     
    data_images0=normalize0(image) 
    data_images1=normalize1(image)
    data_images2=normalize2(image)
         
    data_images0=data_images0.reshape(dimension, dimension, dimension, 1)
    data_images1=data_images1.reshape(dimension, dimension, dimension, 1)
    data_images2=data_images2.reshape(dimension, dimension, dimension, 1) 
    
    concatimage=np.concatenate((data_images0, data_images1, data_images2), axis=-1)
    
    return concatimage 


# In[7]:


### data load 

def vargen(data, order):   
    var=[]
    numfiles=len(data)
    for ii in range(numfiles):
        row=data.iloc[ii]
        var.append(row[order])
    var=np.array(var, dtype='float32')
    return var


# In[9]:


def vargen1(data, order):   
    var=[]
    numfiles=len(data)
    for ii in range(numfiles):
        row=data.iloc[ii]
        var.append(row[order])
    var=np.array(var)
    return var


# In[10]:


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
    LVI=vargen(data, 14)
    death=vargen(data, 15)
    OS=vargen(data, 16)
    CTR=vargen(data, 18)
    solid=vargen(data, 19) 
    age=vargen(data, 22) 
    sex=vargen(data, 23) 
    return data_path, data_w, data_h, data_s, VPI, node, LVI, death, OS, CTR, solid, age, sex


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


# In[ ]:


def clinical(var1,var2,var3,var4):
    var1=var1.reshape(len(var1),1)
    var2=var2.reshape(len(var2),1)
    var3=var3.reshape(len(var3),1)
    var4=var4.reshape(len(var4),1)
    varnew = np.concatenate((var1, var2, var3, var4), axis=-1)
    return varnew  


# In[ ]:


def csvload(file): 
    allData = pd.read_csv(file)
    
    os.chdir("%s/split"%(savepath))  
    train, splitted = train_test_split(allData, test_size=0.4, shuffle=True, stratify=allData['nodebinary'] )
    tune, test = train_test_split(splitted, test_size=0.5, shuffle=True, stratify=splitted['nodebinary'])
    
    train_df=pd.DataFrame(train)
    train_df.to_csv('Train.csv', index=False, header=True)
    
    tune_df=pd.DataFrame(tune)
    tune_df.to_csv('Tune.csv', index=False, header=True)

    test_df=pd.DataFrame(test)
    test_df.to_csv('Test.csv', index=False, header=True)    
    
    return train, tune, test


# In[ ]:


def varprep(data):
    data_path1, data_w1, data_h1, data_s1, VPI1, node1, LVI1, death1, OS1, CTR1, solid1, age1, sex1 = varloader(data)
    image1 = imageloader(data_path1, dim, data_w1, data_h1, data_s1)
    image1NC = normcat(image1, dim) 
    
    return VPI1, node1, LVI1, death1, OS1, CTR1, solid1, age1, sex1, image1, image1NC 


# In[ ]:


samples = 389 + 51 #total 440
order = np.arange(samples)
np.random.shuffle(order)


# In[ ]:


def choicegen(bool_var, num):
    bool_var1 = bool_var != 0
    pos = bool_var[bool_var1]
    ids = np.arange(len(pos))
    choices = np.random.choice(ids,num)
    return choices

def oversampler(targetvar, bool_var):
    bool_var1 = bool_var != 0
    
    pos_features = targetvar[bool_var1]
    neg_features = targetvar[~bool_var1] 
        
    res_pos_features = pos_features[choices]
    resampled_features = np.concatenate([res_pos_features, pos_features, neg_features])
    resampled_features = resampled_features[order] 
    return resampled_features


# In[ ]:


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
 

# 2 inputs: image1, clinical1
# 4 labels: VPI1, node1, LVI1, surv1

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, inputdata1, inputdata2, label, 
                 augvol, normcat1, batch_size, inputdimension, n_channels, shuffle):
        
        'Initialization'
        self.list_IDs = list_IDs 
        self.inputdata1 = inputdata1 #image1
        self.inputdata2 = inputdata2  
         
        self.label = label #surv1
        
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
        [X1, X2], [y1] = self.__data_generation(list_IDs_temp)
 
        return [X1, X2], [y1]     
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)    
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  
        # Initialization
        X1 = np.empty((self.batch_size, *self.inputdimension, self.n_channels))
        X2 = np.empty((self.batch_size, 4))   
        
        y1 = np.empty((self.batch_size, 8))  

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            image = self.inputdata1[ID]
            image = image.reshape(dim,dim,dim)             
            image_aug = self.augvol(image) 
            X1[i,] = self.normcat1(image_aug, dim)            
            X2[i,] = self.inputdata2[ID]             

            # Store class 
            y1[i] = self.label[ID]

        return [X1, X2], [y1]   

 

params = {'batch_size': batch_size, 
          'inputdimension':(dim,dim,dim),
          'n_channels': 3,
          'shuffle':True
         }
mirrored_strategy = tf.distribute.MirroredStrategy() 
cos_decay_ann = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=0.005, first_decay_steps=30, t_mul=2, m_mul=0.9, alpha=0.01)
  

results=[]

k=0
for i in range(nloop):
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
             
    
    train, tune, test = csvload("raw file server.csv")
    VPI1, node1, LVI1, death1, OS1, CTR1, solid1, age1, sex1, image1, image1NC = varprep(train)
    VPI2, node2, LVI2, death2, OS2, CTR2, solid2, age2, sex2, image2, image2NC = varprep(tune)
    VPI3, node3, LVI3, death3, OS3, CTR3, solid3, age3, sex3, image3, image3NC = varprep(test)
    
    #oversampling
    choices = choicegen(node1, 2)    
    VPI1o = oversampler(VPI1, node1)
    LVI1o = oversampler(LVI1, node1)
    death1o = oversampler(death1, node1)
    OS1o = oversampler(OS1, node1)
    CTR1o = oversampler(CTR1, node1)
    solid1o = oversampler(solid1, node1)
    image1o = oversampler(image1, node1) 
    age1o = oversampler(age1, node1) 
    sex1o = oversampler(sex1, node1)
    node1o = oversampler(node1, node1)

    clinical1=clinical(CTR1, solid1, age1, sex1)
    clinical1o=clinical(CTR1o, solid1o, age1o, sex1o)
    clinical2=clinical(CTR2, solid2, age2, sex2)
    clinical3=clinical(CTR3, solid3, age3, sex3)

    resampledID = np.arange(image1o.shape[0])

    surv1o=make_surv_array(OS1o,death1o,breaks)
    surv2=make_surv_array(OS2,death2,breaks)
    surv3=make_surv_array(OS3,death3,breaks)      
     
    with mirrored_strategy.scope():
        BBmodel= load_model(model_path, custom_objects={'loss':surv_likelihood(n_intervals)})
        BBmodel.trainable=False
        features = BBmodel.get_layer("dense_7").output
        encoder = keras.Model(inputs = BBmodel.input, outputs=features, name="F_extractor")
        input1 = keras.Input((dim, dim, dim, 3), name='image') 
        input2 = keras.Input(shape=(4,), name='clinical')  
        encoder_output = encoder(input1)        
        s1 = layers.Dense(units=8, kernel_initializer='he_normal', bias_initializer=tf.keras.initializers.Constant(0.01))(input2)
        s2 = layers.concatenate([s1, encoder_output])             
        s2 = layers.Dense(units=8, kernel_initializer='he_normal', bias_initializer=tf.keras.initializers.Constant(0.01))(s2)
        s2 = layers.Dropout(rate=0.1)(s2)
        s2 = layers.Dense(units=4, kernel_initializer='he_normal', bias_initializer=tf.keras.initializers.Constant(0.01))(s2)        
        s2 = layers.Dense(n_intervals, kernel_initializer='zeros', bias_initializer='zeros')(s2)  
        output=layers.Activation('sigmoid', name="FT_surv")(s2)  
        model = keras.Model(inputs=[input1, input2], outputs=output, name="FT_sublobar")
        model.compile(loss=surv_likelihood(n_intervals), 
                      optimizer=keras.optimizers.SGD(learning_rate=cos_decay_ann))
        
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=40) 
    csv_logger = keras.callbacks.CSVLogger('./log/Model log.csv', append=False, separator=';')    
    checkpointer = keras.callbacks.ModelCheckpoint(filepath='./bestmodel/surv_bestmodel.h5', verbose=0, 
                                                   save_best_only=True, monitor='val_loss', mode='auto')     
    
    history=model.fit_generator(DataGenerator(resampledID, image1o, clinical1o, surv1o,
                                              augvol, normcat1, **params),
                                epochs=no_epoch,
                                validation_data=([image2NC, clinical2], [surv2]),
                                verbose=2,
                                callbacks=[early_stopping, csv_logger, checkpointer]
                               ) 
    model.save('./finalmodel/finalmodel.h5')      
    keras.backend.clear_session() 
    gc.collect() 
    k+=1 