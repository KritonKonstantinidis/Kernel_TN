import gpflow 
from typing import Optional
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import Dense 

class DeepKernel(gpflow.kernels.Kernel):
    def __init__(self, base_kernel: gpflow.kernels.Kernel,batch_size: Optional[int] = None,
                 initializer=keras.initializers.TruncatedNormal(mean=0, stddev=0.5,seed=0),
                 regularizer=keras.regularizers.l2(0.0)):
        super().__init__()
        
        self.initializer=initializer
        self.regularizer=regularizer
        
        with self.name_scope:
            self.base_kernel = base_kernel
    
    
            self.CP= tf.keras.Sequential(
    
                    [Dense(units=12,activation='relu',dtype='float64'),Dense(units=10,activation='relu',dtype='float64'),
                      Dense(units=1,activation=None,dtype='float64')]
            )

    def K(self, a_input: tf.Tensor, b_input: Optional[tf.Tensor] = None) -> tf.Tensor:
        transformed_a = self.CP(a_input)
        transformed_b = self.CP(b_input) if b_input is not None else b_input
        return self.base_kernel.K(transformed_a, transformed_b)

    def K_diag(self, a_input: tf.Tensor) -> tf.Tensor:
        transformed_a = self.CP(a_input)
        return self.base_kernel.K_diag(transformed_a)
    
class KernelSpaceInducingPoints(gpflow.inducing_variables.InducingPoints):
    pass

@gpflow.covariances.Kuu.register(KernelSpaceInducingPoints, DeepKernel)
def Kuu(inducing_variable, kernel, jitter=None):
    func = gpflow.covariances.Kuu.dispatch(
        gpflow.inducing_variables.InducingPoints, gpflow.kernels.Kernel
    )
    return func(inducing_variable, kernel.base_kernel, jitter=jitter)


@gpflow.covariances.Kuf.register(KernelSpaceInducingPoints, DeepKernel, object)
def Kuf(inducing_variable, kernel, a_input):
    return kernel.base_kernel(inducing_variable.Z, kernel.CP(a_input))