import os
import sys
import time
 
import numpy
 
import theano
import theano.tensor as T

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):

"""
隐层类，隐层输入即input，输出即隐层的神经元个数。输入层与隐层是全连接的。

假设输入是n维的向量（也可以说是n个神经元），隐层有m个神经元，因为是全连接，
一共有n*m个权重，故权重矩阵W大小为(n,m),每一列对应隐层的每个神经元的连接权重。

b是偏置，隐藏层有m个神经元，故b为m维向量。

rng即随机数生成器，numpy.random.RandomState，用于初始化W。

input训练模型为所用到的所有输入，并不是MLP的输入层，MLP的输入层的神经元个数时n_in，
而这里的参数input大小是（n_example, n_in），其中n_example是一个batch的大小，


activation:激活函数,这里定义为函数tanh
"""
        
        
 
"""
代码要兼容GPU，则W、b必须使用 dtype=theano.config.floatX,并且定义为theano.shared
另外，W的初始化有个规则：如果使用tanh函数，则在-sqrt(6./(n_in+n_hidden))到sqrt(6./(n_in+n_hidden))之间均匀
抽取数值来初始化W，若时sigmoid函数，则以上再乘4倍。
"""
#如果W未初始化，则根据上述方法初始化。
#加入这个判断的原因是：有时候我们可以用训练好的参数来初始化W，见我的上一篇文章。
		self.input = input  
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)
 
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
 
#用上面定义的W、b来初始化类HiddenLayer的W、b
        self.W = W
        self.b = b
 
#隐含层的输出
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
 
#隐含层的参数
        self.params = [self.W, self.b]



"""
定义分类层，Softmax回归
在deeplearning tutorial中，直接将LogisticRegression视为Softmax，
而我们所认识的二类别的逻辑回归就是当n_out=2时的LogisticRegression
"""
#参数说明：
#input，大小就是(n_example,n_in)，其中n_example是一个batch的大小，
#因为我们训练时用的是Minibatch SGD，因此input这样定义
#n_in,即上一层(隐含层)的输出
#n_out,输出的类别数 
class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
 
#W大小是n_in行n_out列，b为n_out维向量。即：每个输出对应W的一列以及b的一个元素。  
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
 
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
 
#input是(n_example,n_in)，W是（n_in,n_out）,点乘得到(n_example,n_out)，加上偏置b，
#再作为T.nnet.softmax的输入，得到p_y_given_x
#故p_y_given_x每一行代表每一个样本被估计为各类别的概率    
#PS：b是n_out维向量，与(n_example,n_out)矩阵相加，内部其实是先复制n_example个b，
#然后(n_example,n_out)矩阵的每一行都加b
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
 
#argmax返回最大值下标，因为本例数据集是MNIST，下标刚好就是类别。axis=1表示按行操作。
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
 
#params，LogisticRegression的参数     
        self.params = [self.W, self.b]        


#3层的MLP
class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )
 
#将隐含层hiddenLayer的输出作为分类层logRegressionLayer的输入，这样就把它们连接了
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
 
 
#以上已经定义好MLP的基本结构，下面是MLP模型的其他参数或者函数
 
#规则化项：常见的L1、L2_sqr
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )
 
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )
 
 
#损失函数Nll（也叫代价函数）
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
 
#误差      
        self.errors = self.logRegressionLayer.errors
 
#MLP的参数
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        # end-snippet-3        






# Calculate sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Network size
N_input = 4
N_hidden = 3
N_output = 2

np.random.seed(42)
# Make some fake data
X = np.random.randn(4)

weights_input_to_hidden = np.random.normal(0, scale=0.1, size=(N_input, N_hidden))
weights_hidden_to_output = np.random.normal(0, scale=0.1, size=(N_hidden, N_output))


# Make a forward pass through the network
hidden_layer_in = np.dot(X, weights_input_to_hidden)
hidden_layer_out = sigmoid(hidden_layer_in)

print('Hidden-layer Output:')
print(hidden_layer_out)

output_layer_in = np.dot(hidden_layer_out, weights_hidden_to_output)
output_layer_out = sigmoid(output_layer_in)

print('Output-layer Output:')
print(output_layer_out)
        