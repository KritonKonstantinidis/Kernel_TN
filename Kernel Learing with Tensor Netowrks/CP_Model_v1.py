import tensorflow as tf
from tensorflow import keras

class CP_Based(keras.layers.Layer):

    def __init__(self, units=1, activation=None, cp_rank=10, local_dim=2,initializer=keras.initializers.glorot_normal(seed=None),regularizer=keras.regularizers.l2(0.0), **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.cp_rank = cp_rank
        self.local_dim = local_dim
        self.initializer=initializer
        self.kernel_regularizer = regularizer

    def build(self, batch_input_shape):
        self.kernel = self.add_weight(name="kernel", shape=[self.local_dim, self.cp_rank, batch_input_shape[-1], self.units],
                                      initializer=self.initializer,regularizer=self.kernel_regularizer) 
        
        super().build(batch_input_shape)

    def call(self, X):
        
        samples = tf.shape(X)[0]
        features=self.kernel.shape[2]

        X_transformed_list = []
        for d in range(self.local_dim-1):
            X_transformed_list.append(X**(d+1))
            
        X_stand_list = [tf.ones((samples,features),dtype='float64')]
        for X_trans in X_transformed_list:
                  X_stand_list.append(X_trans)
                  
        x_processed= tf.transpose(X_stand_list,(1, 2, 0)) # NtxNxd
                
        norms = tf.sqrt(tf.reduce_sum(x_processed**2,axis=-1,keepdims=True))
        x_processed = tf.divide(x_processed,norms)   

        feat_tensor=x_processed
        
        output_list=[]
        
        for unit in range(0,self.units):                 
                    
            feat_tensor_reshaped=tf.transpose(feat_tensor,perm=[1,0,2]) # NxNtxd
            weights=tf.transpose(self.kernel[:,:,:,unit],perm=[2,0,1]) # Nxdxm
            test=tf.matmul(feat_tensor_reshaped,weights) # NxNtxm
            test_hadamard=tf.reduce_prod(test,axis=0) # Ntxm   
            output=tf.reduce_sum(test_hadamard, axis=1) # Ntx1
            output_list.append(output)
        
            to_return=tf.stack(output_list, axis=1)
        return self.activation(to_return)
    
    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units,
                "activation": keras.activations.serialize(self.activation)}

