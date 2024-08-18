"""The implementation of LADUI(LSTM based Anomaly Detection for Unsupervised learning using Inter-metric and trend correlation embedding)
     implemented using keras API 
    Author- Mithun Mohankumar for Masters  thesis - LJMU"""

import tensorflow as tf
import tensorflow_probability as tfp
import keras
from tensorflow import keras
import tf_keras as tf_k
from keras import backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import layers
from keras.layers import LSTM
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import seaborn as sns
##from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
##from google.colab import drive
##drive.mount('/content/drive')
print(tf.__version__)
#use timeseries data get through the tfp layers
##file_path='/content/drive/MyDrive/LSTM_vae/data/shuttle-unsupervised-ad.csv'
file_path='D:/upgrad_course/LJMU_Masters_thesis/Implementation/shuttle-unsupervised-ad.csv'
df_shuttle=pd.read_csv(file_path)
df_shuttle.shape

tfd=tfp.distributions
tfpl=tfp.layers
tfb=tfp.bijectors

## to upload synthetic anomaly data (X_test)
data_path_full="D:/upgrad_course/LJMU_Masters_thesis/paper_repo/dataset/NASA_shuttle_sensor_data/synthetic_anom/df_anom_1_op.pkl"
df_anom_1hr_op=pd.read_pickle(data_path_full)
df_anom_1hr_op.head(10)
df_anom_1hr_op.shape

cols=['M-1','M-2','M-3','M-4','M-5','M-6','M-7','M-8','M-9','status']
df_shuttle.columns=cols
df_shuttle.head(10)
df_shuttle.shape
df_shuttle_training_drop=df_shuttle.iloc[879:,:].drop(['status'],axis=1)

df_shuttle_test=df_anom_1hr_op
df_shuttle_test.shape
## the data is downsampled to 1 hr worth data
shuttle_data_train=df_shuttle_training_drop.iloc[:3600,:]
shuttle_data_val=df_shuttle_training_drop.iloc[3600:7200,:]
shuttle_data_train.shape
shuttle_data_val.shape
## visualizing the Normal and anomalous data
plt.style.use('fivethirtyeight')
fig_1,ax_1=plt.subplots(9,1,figsize=(200,100))
for i in range(9):
    ax_1[i].plot(shuttle_data_train.iloc[:,i],'b')
    ax_1[i].xaxis.set_tick_params(labelsize=20,rotation=45)
plt.show()

fig_2,ax_2=plt.subplots(9,1,figsize=(200,100))
xticks=np.arange(0,3600,100)
for i in range(9):
    ax_2[i].plot(df_shuttle_test.iloc[:,i],'r')
    ax_2[i].set_xlabel('Time',fontsize=50)
    ax_2[i].set_xticks(xticks)
plt.show()
## to apply standard scaler for training  and use the same for the validation and test dataset
data_scaler=StandardScaler()
train_scaler=data_scaler.fit(shuttle_data_train)
train_data=train_scaler.transform(shuttle_data_train)
val_data=train_scaler.transform(shuttle_data_val)
test_data=train_scaler.transform(df_shuttle_test)

plt.plot(test_data[:,0],linewidth=1.0)

fig_3,ax_3=plt.subplots(9,1,figsize=(200,100))
for i in range(9):
    ax_3[i].plot(test_data[:,i],'r')
    ax_3[i].set_xlabel('Time',fontsize=50)
plt.show()
windows_size=200
## to generate the windows in the timeseries data30
def generate_sliding_windows_multivariate(data, window_size):
    windows = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        windows.append(window)
    return np.array(windows)

## create sliding window output for train and validation data
x_train=generate_sliding_windows_multivariate(train_data,windows_size)
x_train.shape
x_val=generate_sliding_windows_multivariate(val_data,windows_size)
x_val.shape
## create sliding window output for test data
x_test=generate_sliding_windows_multivariate(test_data,windows_size)
x_test.shape

## create dataframe to map between window number and ground truth labels for the test data (anomaly-1/normal-0) 
df_M1=pd.DataFrame(columns=['windows','true_labels'])
df_M1['windows']=pd.Series(range(1,3402))
df_M1['true_labels']=pd.Series(np.zeros(3401))
df_M1.loc[(df_M1['windows']>=(1500-windows_size+1))&(df_M1['windows']<=(1500+1)),'true_labels']=1.0
df_M1.loc[(df_M1['windows']>=(1750-windows_size+1)) & (df_M1['windows']<=(1750+1)),'true_labels']=1.0

df_M1.loc[(df_M1['windows'] >=(2750-windows_size+1))&(df_M1['windows']<=2750+1),'true_labels']=1.0
df_M1.loc[(df_M1['windows']>=(3250-windows_size+1)) & (df_M1['windows']<=(3250+1)),'true_labels']=1.0
df_M1.head(1753)
df_M1.set_index('windows',inplace=True)
df_M1['true_labels'].value_counts()

df_M5=pd.DataFrame(columns=['windows','true_labels'])
df_M5['windows']=pd.Series(range(1,3402))
df_M5['true_labels']=pd.Series(np.zeros(3401))
df_M5.loc[(df_M5['windows']>=(2000-windows_size+1))&(df_M5['windows']<=(2000+1)),'true_labels']=1.0
df_M5.loc[(df_M5['windows']>=(2250-windows_size+1)) & (df_M5['windows']<=(2250+1)),'true_labels']=1.0
df_M5['true_labels'].value_counts()
df_M5.set_index('windows',inplace=True)
df_M5['true_labels'].value_counts()

df_M8=pd.DataFrame(columns=['windows','true_labels'])
df_M8['windows']=pd.Series(range(1,3402))
df_M8['true_labels']=pd.Series(np.zeros(3401))
df_M8.loc[(df_M8['windows']>=(1200-windows_size+1))&(df_M8['windows']<=(1200+1)),'true_labels']=1.0
df_M8.loc[(df_M8['windows']>=(1450-windows_size+1)) & (df_M8['windows']<=(1450+1)),'true_labels']=1.0
df_M8['true_labels'].value_counts()
df_M8.set_index('windows',inplace=True)
df_M8.head(1453)

df_M9=pd.DataFrame(columns=['windows','true_labels'])
df_M9['windows']=pd.Series(range(1,3402))
df_M9['true_labels']=pd.Series(np.zeros(3401))
df_M9.loc[(df_M9['windows']>=(1800-windows_size+1))&(df_M9['windows']<=(1800+1)),'true_labels']=1.0
df_M9.loc[(df_M9['windows']>=(2000-windows_size+1)) & (df_M9['windows']<=(2000+1)),'true_labels']=1.0

df_M9.loc[(df_M9['windows'] >=(2500-windows_size+1))&(df_M9['windows']<=2500+1),'true_labels']=1.0
df_M9.loc[(df_M9['windows']>=(2510-windows_size+1)) & (df_M9['windows']<=(2510+1)),'true_labels']=1.0
df_M9['true_labels'].value_counts()
df_M9.set_index('windows',inplace=True)

## non-anomalous features
df_M3=pd.DataFrame(columns=['windows','true_labels'])
df_M3['windows']=pd.Series(range(1,3402))
df_M3['true_labels']=pd.Series(np.zeros(3401))
df_M3['true_labels'].value_counts()
df_M3.set_index('windows',inplace=True)

df_M7=pd.DataFrame(columns=['windows','true_labels'])
df_M7['windows']=pd.Series(range(1,3402))
df_M7['true_labels']=pd.Series(np.zeros(3401))
df_M7['true_labels'].value_counts()
df_M7.set_index('windows',inplace=True)

##df_ground_truth=pd.DataFrame(columns=['M_1','M_3','M_5','M_7','M_8','M_9'])
df_ground_truth=pd.concat([df_M1,df_M3,df_M5,df_M7,df_M8,df_M9],axis=1)
df_ground_truth.columns=['M_1','M_3','M_5','M_7','M_8','M_9']
df_ground_truth['M_9'].value_counts()
## to flatten so can be used metric evaluation
true_labels=df_ground_truth.to_numpy().flatten('F')
true_labels
value,counts=np.unique(true_labels,return_counts=True)
print(np.asarray((value,counts)).T)

## parameters based on input dimension of 'X' and the latent dimension size
batch=x_train.shape[0]
ts_vae=x_train.shape[1]
f_vae=x_train.shape[2]
latent_dimensions=2
## to wrap the utility class as Layer subclass
## Layer to transpose a tensor
class transpose_tensor(layers.Layer):
    def __init__(self,perm=False,*args,**kwargs):
        super(transpose_tensor, self).__init__()
        self.perm=perm

    def call(self,inputs,**kwargs):
        if self.perm:
            for k,v in kwargs.items():
                tensor=inputs
                return tf.transpose(tensor,perm=v)
        else:
            tensor=inputs
            return tf.transpose(tensor)
## Layer to remove the dimension aka squeeze a tensor along  a dimension
class squeeze_tensor(layers.Layer):
    def call(self,inputs,**kwargs):
        for k,v in kwargs.items():
            tensor=inputs
            return tf.squeeze(tensor,axis=v)
class exp(layers.Layer):
    def call(self,tensor):
        return tf.exp(tensor)

class square(layers.Layer):
    def call(self,tensor):
        return tf.square(tensor)

## sampling layer using reparametrization trick 
@tf.keras.utils.register_keras_serializable()
class Sampling_layer(layers.Layer):
    """input : z_mean , z_log_variance 
        ouput: samples z values """
    def call(self,inputs):
        z_mean, z_log_var = inputs
        batch_size = tf.shape(z_mean)[0]
        dim= tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

## layer that encapsulates the dense and distribution object
@tf.keras.utils.register_keras_serializable()
class Likelihood(layers.Layer):
    
    """ 'Likelihood layer' :to compute the log likelihood of the given observation
    inputs: x (given multivariate timeseries data) , h_t (hidden states of last decoder LSTM layer)
    output:l_2(Distribution obeject) , llh (log likelihood value)."""

    def __init__(self):
        super().__init__()
        self.Dense=layers.Dense(18,activation='tanh')
        self.dist=tfpl.DistributionLambda(make_distribution_fn=lambda t:tfd.MultivariateNormalDiag(loc=t[...,:9],
                                                             scale_diag=tf.math.softplus(t[...,9:])))
    def call(self,inputs):
        x,h_inp=inputs
        l_1=self.Dense(h_inp)
        l_2=self.dist(l_1)
        likelihood=l_2.log_prob(tf.transpose(x,perm=[1,0,2]))
        return l_1,tf.reduce_mean(tf.reduce_sum(likelihood,axis=[0,1]))

@keras.saving.register_keras_serializable()
class Concatenate(layers.Layer):
    """ Concatenate layer : to concatenate two tensors 
    parameter 1: axis along which the concatenation should happen
    input: tensor_1 and tensor_2 to be concatenated
    output: the final concatenated tensor"""
    
    def __init__(self,axis=0,**kwargs):
        super().__init__()
        self.axis=axis
    def call(self,inputs):
        return keras.layers.Concatenate(axis=self.axis)(inputs)

##Encoder Layer
@keras.saving.register_keras_serializable()
class Encoder_layer(layers.Layer):
    """Encoder layer
    parameters ts: timesteps in the current window
                f : the number of features 
                ld: latent dimension size z
                
    inputs   enc_inp_1: input-1 for conv-1D operation
             enc_inp_2:input-2 for conv-2D operation
    outputs: z_mean, z_log_var ( distribution paramters) 
                z (sampled z variables)."""
                
    def __init__(self,name="Encoder_layer",ts=200,f=9,ld=6,**kwargs):
        super().__init__(name=name,**kwargs)
        self.ts_vae=ts
        self.f_vae=f
        self.lat_dim=ld
        self.conv_1D=layers.Conv1D(1,1,activation='relu',padding='same')
        self.transpose_noperm=transpose_tensor()
        self.transpose_perm=transpose_tensor(perm=True)
        self.squeeze=squeeze_tensor()
        self.conv_2D=layers.Conv2D(1,(2,2),activation='relu',padding='same')
        self.concat=Concatenate(axis=0)
        self.enc_1=layers.LSTM(16,activation='tanh',return_sequences=True)
        ##self.bne=layers.BatchNormalization()
        self.enc_2=layers.LSTM(8,activation='tanh',return_sequences=False)
        self.enc_4=layers.Dense(self.lat_dim,activation='relu')
        self.enc_5=layers.Dense(self.lat_dim,activation='softplus')
        self.sampling=Sampling_layer()

    def call(self,inputs):
        inp_1,inp_2=inputs
        c1D=self.conv_1D(inp_1)
        c1D_trans=self.transpose_noperm(c1D)
        c_2D=self.conv_2D(inp_2)
        c_2D_trans=self.transpose_perm(c_2D,perm=[2,1,3,0])
        c_2D_drop=self.squeeze(c_2D_trans,axis=[2])
        conv_out=self.concat([c1D_trans,c_2D_drop])
        lstm_inp=self.transpose_noperm(conv_out)
        #LSTM operations
        lstm_out_1=self.enc_1(lstm_inp)
        ##bne_out=self.bne(lstm_out_1)
        lstm_out_2=self.enc_2(lstm_out_1)
        ##dense layers
        z_mean=self.enc_4(lstm_out_2)
        z_log_var=self.enc_5(lstm_out_2)
        z=self.sampling([z_mean,z_log_var])
        return z_mean,z_log_var,z

enc_inp_C1D=keras.Input(shape=(ts_vae,f_vae),name='enc_inp_1')
enc_inp_C2D=keras.Input(shape=(ts_vae,f_vae,1),name='enc_inp_2')
vae_enc=Encoder_layer(ts=ts_vae,f=f_vae,ld=latent_dimensions)
z_mean,z_log_var,z=vae_enc([enc_inp_C1D,enc_inp_C2D])
enc_block=keras.Model(inputs=[enc_inp_C1D,enc_inp_C2D],outputs=[z_mean,z_log_var,z],name='enc')
enc_block.summary()

## Decoder layer
@keras.saving.register_keras_serializable()
class Decoder_layer(layers.Layer):
    
    """Decoder layer 
    parameters : ts (timesteps in a window),f(number of features)
    inputs : enc_input_1 (input timeseries in the format batch_size*time_steps*features)
    outputs: res(the distribution object),rec_prob(the log likelihood for the given observtaion of timeseries data)
    """
    def __init__(self,name="Decoder_layer",ts=200,f=9,**kwargs):
        super().__init__(name=name,**kwargs)
        self.ts_vae=ts
        self.f_vae=f
        self.dec_1=layers.Dense(self.ts_vae*self.f_vae)
        self.dec_2=layers.Reshape((self.ts_vae,self.f_vae))
        self.dec_4=layers.LSTM(8,activation='tanh',return_sequences=True)
        ##self.bnd=layers.BatchNormalization()
        self.dec_5=layers.LSTM(16,activation='tanh',return_sequences=False)
        self.llh=Likelihood()

    def call(self,inputs):
        x,z=inputs
        dec_1_res=self.dec_1(z)
        dec_2_res=self.dec_2(dec_1_res)
        ##LSTM operations for decoder
        ##dec_3_res=self.dec_3(dec_2_res)
        ##bnd_out=self.bnd(dec_3_res)
        dec_4_res=self.dec_4(dec_2_res)
        dec_5_res=self.dec_5(dec_4_res)
        res,rec_prob=self.llh([x,dec_5_res])
        return res,rec_prob

vae_dec=Decoder_layer(ts=ts_vae,f=f_vae)
inter_res,llh=vae_dec([enc_inp_C1D,z])
inter_res.shape
dec_block=keras.Model(inputs=[enc_inp_C1D,z],outputs=[inter_res,llh],name='dec')
dec_block.summary()

## VAE class by model subclassing
@tf.keras.utils.register_keras_serializable()
class VAE(keras.Model):
    
    """VAE class subclassing the model
    parameters: encoder, decoder (layer objects)
    inputs    :data  ( two inputs corresponding to enc_1 and enc_2)
    output    :total_loss( the sum of KL loss and reconstruction loss)"""
    
    def __init__(self,encoder,decoder,latent_dim=6,**kwargs):
        super().__init__(**kwargs)
        self.encoder=encoder
        self.decoder=decoder
        self.total_loss_tracker=keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker=keras.metrics.Mean(
           name='reconstruction_loss')
        self.kl_loss_tracker=keras.metrics.Mean(name='kl_loss')

    @property
    def metric(self):
        return [self.total_loss_tracker,self.reconstruction_loss_tracker,self.kl_loss_tracker]

    @tf.function
    def train_step(self,data):
        with tf.GradientTape() as tape:
            z_mean,z_log_var,z=self.encoder(data)
            dense,llh=self.decoder([data['enc_inp_1'],z])
            kl_loss=-0.5*(1+z_log_var-tf.square(z_mean)-tf.exp(z_log_var))
            kl_loss=tf.reduce_mean(tf.reduce_sum(kl_loss,axis=1))
            ##dist_obj=tfd.MultivariateNormalDiag(x_mean,x_log_var)
            recon_loss=llh
            ##recon_loss=tf.reduce_mean(tf.reduce_sum(recon_prob,axis=[0,1]))
            total_loss=(kl_loss-recon_loss)
        grads=tape.gradient(total_loss,self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            'loss':self.total_loss_tracker.result(),
            'reconstruction_loss':self.reconstruction_loss_tracker.result(),
            'kl_loss':self.kl_loss_tracker.result()
        }

    def get_config(self):
        base_config = super().get_config()
        config={
                "encoder": keras.saving.serialize_keras_object(self.encoder),
                "decoder": keras.saving.serialize_keras_object(self.decoder)
            }
        return {**base_config,**config}

    @classmethod
    def from_config(cls, config):
        # Note that you can also use [`keras.saving.deserialize_keras_object`](/api/models/model_saving_apis/serialization_utils#deserializekerasobject-function) here
        encoder_config=config.pop('encoder')
        decoder_config=config.pop('decoder')
        encoder_vae = keras.saving.deserialize_keras_object(encoder_config)
        decoder_vae= keras.saving.deserialize_keras_object(decoder_config)
        return cls(encoder_vae,decoder_vae,**config)

## model creation stage
elbo_vae=VAE(enc_block,dec_block)
elbo_vae.layers
ADAM=keras.optimizers.Adam(learning_rate=1e-4)
elbo_vae.compile(optimizer=ADAM)
hist=elbo_vae.fit({'enc_inp_1':x_train,'enc_inp_2':x_train},epochs=10,batch_size=64)

##plot the monitoring metrics
plt.plot(hist.history['loss'],linewidth=1.0)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
plt.plot(hist.history['reconstruction_loss'],linewidth=1.0)
plt.ylabel('recon_loss')
plt.xlabel('epoch')
plt.show()
plt.plot(hist.history['kl_loss'],linewidth=1.0)
plt.ylabel('kl_loss')
plt.xlabel('epoch')
plt.show()

## Prediction stage
## predict using x_val (non-anomalous data)
z_mean_val,z_log_var_val,z_val=elbo_vae.encoder.predict([x_val,x_val])
dense_val,llh_val=elbo_vae.decoder.predict([x_val,z_val])
## predict using x_train (non-anomalous data)
z_mean_train,z_log_var_train,z_train=elbo_vae.encoder.predict([x_train,x_train])
dense_train,llh_train=elbo_vae.decoder.predict([x_train,z_train])
z_train.shape
dense_train
# predict on the test data
z_mean_test,z_log_var_test,z_test=elbo_vae.encoder.predict([x_test,x_test])
dense_test,llh_test=elbo_vae.decoder.predict([x_test,z_test])


## scatter plot to visualize the latent variables (of shape batch_size*latent_dimensions)
flag_0=np.squeeze(np.full((z_val.shape[0],1),'normal',dtype=str))
flag_1=np.squeeze(np.full((z_test.shape[0],1),'anomaly',dtype=str))
flag_3=np.squeeze(np.full((z_train.shape[0],1),'train_normal',dtype=str))

dict_z_val={"z_1":z_val[:,0],'z_2':z_val[:,1],"status":flag_0}
dict_z_test={"z_1":z_test[:,0],'z_2':z_test[:,1],"status":flag_1}
df_val=pd.DataFrame(dict_z_val)
df_test=pd.DataFrame(dict_z_test)
df_res=pd.concat([df_val,df_test],axis=0)
width=10
height=7.5
sns.color_palette("flare")
color=['b','r']
sns.set_theme(rc={'figure.figsize':(width,height)})
z_scatter=sns.scatterplot(data=df_res, x="z_1", y="z_2",alpha=0.45, hue="status",legend='full',
                          palette={'n':'blue','a':'red'})
handles, labels  =  z_scatter.get_legend_handles_labels()
new_labels = ['normal', 'anomaly']
z_scatter.legend(handles,new_labels, loc='lower right')

##t-sne plot to visualize the latent space of dimension'4'
from sklearn.manifold import TSNE
dict_z_val={"z_1":z_val[:,0],'z_2':z_val[:,1],'z_3':z_val[:,2],'z_4':z_val[:,3]}
dict_z_test={"z_1":z_test[:,0],'z_2':z_test[:,1],"z_3":z_test[:,2],'z_4':z_test[:,3]}
df_z_val=pd.DataFrame(dict_z_val)
df_z_test=pd.DataFrame(dict_z_test)
df_z_val_test=pd.concat([df_z_val,df_z_test],axis=0)

tsne = TSNE(n_components=2,perplexity=50, random_state=25,n_iter=5000,learning_rate=10.0)
z_transf = tsne.fit_transform(df_z_val_test)
status=pd.concat([pd.Series(flag_0),pd.Series(flag_1)],axis=0)
df_z_embed=pd.DataFrame({"z_1":z_transf[:,0],"z_2":z_transf[:,1],"status":status})
import seaborn as sns
width=10
height=7.5
sns.color_palette("flare")
color=['b','r']
sns.set_theme(rc={'figure.figsize':(width,height)})
z_scatter_ld_2=sns.scatterplot(data=df_z_embed, x=df_z_embed["z_1"], y=df_z_embed["z_2"],hue='status',alpha=0.45,legend='full',
                palette={'n':'blue','a':'red'})
handles_1, labels_1 =  z_scatter_ld_2.get_legend_handles_labels()
new_labels = ['normal', 'anomaly']
z_scatter_ld_2.legend(handles_1,new_labels, loc='lower right')

##t-sne plot to visualize the latent space of dimension '6'
from sklearn.manifold import TSNE
dict_z_val={"z_1":z_val[:,0],'z_2':z_val[:,1],'z_3':z_val[:,2],'z_4':z_val[:,3],'z_5':z_val[:,4],'z_6':z_val[:,5]}
dict_z_test={"z_1":z_test[:,0],'z_2':z_test[:,1],"z_3":z_test[:,2],'z_4':z_test[:,3],'z_5':z_test[:,4],'z_6':z_test[:,5]}
df_z_val=pd.DataFrame(dict_z_val)
df_z_test=pd.DataFrame(dict_z_test)

df_z=pd.concat([df_z_val,df_z_test],axis=0)
df_z.head(10)
tsne = TSNE(n_components=2,perplexity=50, random_state=25,n_iter=5000,learning_rate=10.0)
z_transf_ld_6 = tsne.fit_transform(df_z)
status=pd.concat([pd.Series(flag_0),pd.Series(flag_1)],axis=0)
df_z_embed=pd.DataFrame({"z_1":z_transf_ld_6[:,0],"z_2":z_transf_ld_6[:,1],"status":status})

import seaborn as sns
width=10
height=7.5
sns.color_palette("flare")
color=['b','r']
sns.set_theme(rc={'figure.figsize':(width,height)})
z_scatter_ld_2=sns.scatterplot(data=df_z_embed, x=df_z_embed["z_1"], y=df_z_embed["z_2"],hue='status',alpha=0.45,legend='full',
                palette={'n':'blue','a':'red'})
handles_1, labels_1 =  z_scatter_ld_2.get_legend_handles_labels()
new_labels = ['normal', 'anomaly']
z_scatter_ld_2.legend(handles_1,new_labels, loc='lower right')


from scipy.stats import multivariate_normal
class anomaly_score:
    
    """To get the anomaly score for each observation along time axis"""
    
    def __init__(self,x,dense_ouput):
        self.x=x
        self.dense=dense_ouput
        self.time_window=x.shape[1]
        
    def recon_prob(self):
        mu_x=self.dense[:,:9]
        sigma_x=self.dense[:,9:]
        batch_size=self.x.shape[0]
        t_s=self.x.shape[1]
        f_s=self.x.shape[2]
        rec_p_window=np.empty((batch_size,t_s,f_s))
        for w in range(batch_size):
            recon_prob=np.empty((t_s,f_s))
            for f in range(f_s):
                log_prob=-(multivariate_normal.logpdf(np.reshape(self.x[w,:,f],(t_s,1)),
                                                mean=mu_x[w,:],cov=np.diag(tf.math.softplus(sigma_x[w,:]))))
                recon_prob[:,f]=log_prob
            rec_p_window[w,:,:]=recon_prob
        return rec_p_window 
    
    ## to flatten the reconstrution probabilty tensor from 3D to 2D.
    def flatten(self,rec_p_window):
        flat_reconst_prob=rec_p_window[0,:,:]
        for i in range(rec_p_window.shape[0]-1):
            flat_reconst_prob=np.concatenate((flat_reconst_prob,rec_p_window[i+1,-1,:].reshape(1,rec_p_window.shape[2])),axis=0)
        return flat_reconst_prob
    
    ## to calculate the moving average
    def moving_average(self,a):
        return np.convolve(a, np.ones(self.time_window), 'valid')/self.time_window

    ## function to calculate moving average for entire feature set
    def rec_prob_moving_avg(self,flat_reconst_prob):
        res_test_rp=np.empty([3401,9])
        for i in range(9):
            test_window_avg=self.moving_average(flat_reconst_prob[:,i])
            res_test_rp[:,i]=np.vstack(test_window_avg.reshape(1,3401))
        return res_test_rp
    
val_anom=anomaly_score(x_val,dense_val)
recon_window=val_anom.recon_prob()
recon_flat=val_anom.flatten(recon_window)
val_rp_averaged_1=val_anom.rec_prob_moving_avg(recon_flat)
val_rp_averaged_1

test_anom=anomaly_score(x_test,dense_test)
recon_window_test=test_anom.recon_prob()
recon_flat_test=test_anom.flatten(recon_window_test)
test_rp_averaged_1=test_anom.rec_prob_moving_avg(recon_flat_test)
test_rp_averaged_1

##to save the numpy arrays 
file_path='D:/upgrad_course/LJMU_Masters_thesis/paper_repo/dataset/WADI_dataset/LSTM_vae_Robot/numpy_files/rp_calculated_lt_6.npy'
with open(file_path,'wb') as f:
    np.savez(f,val=val_rp_averaged_1,test=test_rp_averaged_1)

## to plot the subplots 
x = np.arange(1,3402,1)
y_1 = val_rp_averaged_1
y_2=test_rp_averaged_1
fig,(ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6, 1,figsize=(20,12.5),constrained_layout=True)
ax1.axvspan(2600,3250,color='orange',alpha=0.25)
ax1.axvspan(1300,1750,color='orange',alpha=0.25)
ax1.plot(x[:2000],val_rp_averaged_1[:2000,0],'b-',linewidth=2.0)
ax1.plot(x[:2000],test_rp_averaged_1[:2000,0],'r-',linewidth=2.0)

##ax2.axvspan(1800,2250,color='orange',alpha=0.25)
ax2.plot(x,val_rp_averaged_1[:,2],'b-',linewidth=2.0)
ax2.plot(x,test_rp_averaged_1[:,2],'r-',linewidth=2.0)

ax3.axvspan(1800,2250,color='orange',alpha=0.25)
ax3.plot(x,val_rp_averaged_1[:,4],'b-',linewidth=2.0)
ax3.plot(x,test_rp_averaged_1[:,4],'r-',linewidth=2.0)

##ax4.axvspan(1800,2250,color='orange',alpha=0.25)
ax4.plot(x,val_rp_averaged_1[:,6],'b-',linewidth=2.0)
ax4.plot(x,test_rp_averaged_1[:,6],'r-',linewidth=2.0)

ax5.axvspan(1000,1450,color='orange',alpha=0.25)
ax5.plot(x,val_rp_averaged_1[:,7],'b-',linewidth=2.0)
ax5.plot(x,test_rp_averaged_1[:,7],'r-',linewidth=2.0)

ax6.axvspan(1600,2000,color='orange',alpha=0.25)
ax6.axvspan(2300,2500,color='orange',alpha=0.25)
ax6.plot(x,val_rp_averaged_1[:,8],'b-',label='val_data',linewidth=2.0)
ax6.plot(x,test_rp_averaged_1[:,8],'r-',label='test_data',linewidth=2.0)
handles,labels = ax6.get_legend_handles_labels()

# separate subplot titles
ax1.set_title('M_1',fontsize=15)
ax2.set_title('M_3',fontsize=15)
ax3.set_title('M_5',fontsize=15)
ax4.set_title('M_7',fontsize=15)
ax5.set_title('M_8',fontsize=15)
ax6.set_title('M_9',fontsize=15)
# common axis labels
fig.supxlabel('Windows',ha='center',fontsize=25)
fig.supylabel('Anomaly_score',fontsize=25)
##labels=['val_data','test_data']
fig.legend(handles,labels,fontsize='15',loc='lower left')
plt.show()

class prediction:
    
    """ To predict the class labels and threshold calculation"""
    
    def __init__(self,test_data,val_data):
        self.test_data=test_data
        self.val_data=val_data

    ## calculate threshold based on x_val 
    def threshold(sef,rec_prob_f):
        return np.mean(rec_prob_f,axis=0)+3.25*np.std(rec_prob_f,axis=0)
    
    ## create function to create dataframe of predictions(0/1) for each features
    def prep_anomaly(self,rec_prob_f,threshold):
        df_test_res=pd.DataFrame(columns=['windows','pred_label'])
        df_test_res['windows']=pd.Series(range(1,3402))
        df_test_res['pred_label']=(rec_prob_f>=threshold)
        df_test_res.set_index('windows',inplace=True)
        return df_test_res.astype(float)
    
    ## to predict each window as anomalous or normal windows for test data.
    ## function for inference to predict the labels (1-Anomaly/0-Normal)
    def pred_label(self):
        df_res=pd.DataFrame(columns=['M_1','M_2','M_3','M_4','M_5','M_6','M_7','M_8','M_9'])
        th=self.threshold(self.val_data)
        for i in range(9):
            df_test_res=self.prep_anomaly(self.test_data[:,i],th[i])
            df_res.iloc[:,i]=pd.concat([df_test_res],axis=1)
        return df_res
    
label_pred=prediction(test_rp_averaged_1,val_rp_averaged_1)
df_test_full=label_pred.pred_label()
df_test_pred=df_test_full.drop(['M_2','M_4','M_6'],axis=1)

## to calculate the precision, recall , F1_score comparing the ground truth and predicted label
pred_labels=df_test_pred.to_numpy().flatten('F')
value_pred,counts_pred=np.unique(pred_labels,return_counts=True)
print(np.asarray((value_pred,counts_pred)).T)

f1=f1_score(true_labels.astype(float),pred_labels.astype(float))
precision=precision_score(true_labels.astype(float),pred_labels.astype(float))
recall=recall_score(true_labels.astype(float),pred_labels.astype(float))
print(f1,precision,recall)
