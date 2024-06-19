import  numpy as np

#前向传播函数################################################
"""
一个隐藏层的前向传播
x:包含输入数据的numpy数组形状为(N,d_1,...d_k)
w:形状为(D,W)
b：偏置，形状为(M,)
偏置向量 b 的维度为 (W,) 表示它是一个一维的数组，其中 W 是隐藏层中神经元的数量。
这意味着偏置向量 b 包含了与隐藏层中每个神经元相关联的偏置值。
具体来说，如果你有一个具有 W 个神经元的隐藏层，那么偏置向量 b 包含了 W 个实数值，每个实数值表示了
一个神经元的偏置。在前向传播中，这些偏置值将被添加到隐藏层的加权输入上，以计算每个神经元的激活值。
"""
def affine_forward(x,w,b):
    out=None
    N=x.shape[0]   #此时为4
    x_row=x.reshape(N,-1) # 重塑为(N,D) 4x2
    out=np.dot(x_row,w)+b   # H=x*w+b  (4x2) * (2*50) +(1*50)
    cache=(x,w,b)#缓存，留在反向传播时使用
    return out,cache

#反向传播函数
"""
仿射变换反向传播的最重要的3个目的，分别是：①更新参数w的值②计算流向下一个节点的数值
③更新参数b的值。“更新”的时候需要“旧”值，也就是缓存值
# 反向传播函数
# - x：包含输入数据的numpy数组，形状为（N，d_1，...，d_k）
# - w：形状（D，M）的一系列权重
# - b：偏置，形状为（M，）
"""
def affine_backward(dout,cache):
    #dout是上游传递到下游的的梯度为(N,M),N是批量大小，M是输出的特征数
    x,w,b=cache
    dx,dw,db=None,None,None
    #计算输入数据x的梯度dx,使用以下公式，因为反向传播的链式法则要求将上游梯度和权重w相乘，然后传递到前一层
    dx=np.dot(dout,w.T)             #(N,D)
    #将梯度dx的形状重新整形为与输入数据x具有相同形状的数组，这是因为在反向传播的梯度应该具有
    #与输入数据相同的形状，以便与输入数据进行逐元素相乘，x.shape返回输入数据x的形状,然后使用np.reshape
    #将dx调整为相同的形状
    dx=np.reshape(dx,x.shape)       #(N,d_1,...d_k)
    #这一行代码将输入数据x重新整形为一个二维数组，行数不变，列数变为-1，-1的意思是根据数据的总大小自动计算，
    # 这是为了将输入数据x从形状(N,d_1,...d_k)转化形状为(N,D)，N是批量大小，D是输入数据的总特征数
    #这个操作是为了后面计算权重梯度dw时方便，因为dw的形状是(D,M)，D是输入特征数，M是输出特征数
    x_row=x.reshape(x.shape[0],-1)  #(N,D)
    #关于计算权重w的梯度
    dw=np.dot(x_row.T,dout)         #(D,M)
    #计算偏置b的梯度，使用以下公式，因为偏置在所有样本上的梯度是相同的，
    # 所以需要对dout沿着批量维度(就是有多少对数据)求和
    db=np.sum(dout,axis=0,keepdims=True)#(1,M)
    return dx,dw,db

#参数初始化
X=np.array([[2,1],
           [-1,1],
           [-1,-1],
           [1,-1]])#用于训练的坐标，对应于I,II,III,IIII象限
t=np.array([0,1,2,3])#标签，对应于I,II,III,IIII象限
np.random.seed(1)
#对于训练数据以及训练模型已经确定的网络来说，为了得到更好的训练效果需要调节的参数就是上述的隐藏层维度、
# 正则化强度和梯度下降的学习率，以及下一节中的训练循环次数。
input_dim=X.shape[1]#输入参数的维度，此处为2表示每个坐标有两个数表示
num_classes=t.shape[0]#输出参数的维度，此处是4，对应四个象限
hidden_dim=50#隐藏层维度，为可调参数
reg=0.001#正则化强度,正则化惩罚系数，为可调参数
epsilon=0.001#梯度下降的学习率，为可调参数

W1=np.random.randn(input_dim,hidden_dim)#(2,50)
W2=np.random.randn(hidden_dim,num_classes)#(50,4)
b1=np.zeros((1,hidden_dim))#(1,50)
b2=np.zeros((1,num_classes))#(1,4)

#训练与迭代
for j in range(10000):#设置训练的循环次数为10000
    #前向传播
    H,fc_cache=affine_forward(X,W1,b1)#第一层前向传播
    H=np.maximum(0,H)#激活
    relu_cache=H#缓存第一层激活的结果
    Y,cachey=affine_forward(H,W2,b2)#第二层前向传播
    #softmax层计算
    probs=np.exp(Y-np.max(Y,axis=1,keepdims=True))#Y矩阵中每个值减掉改行最大值后再取对数。
    probs/=np.sum(probs,axis=1,keepdims=True)#softmax算法实现#以行为单位求出各个数值对应的比例。也就是最终实现了Softmax层的输出。
    #计算loss值
    N=Y.shape[0]#值为4
    print(probs[np.arange(N),t])#打印各个数据的正确标签对应的神经网络的输出
    loss=-np.sum(np.log(probs[np.arange(N),t]))/N#计算loss#交叉熵损失的求法是求对数的负数。
    #N中先求了N维数据中的交叉熵损失，然后对这N个交叉熵损失求平均值，作为最终loss值。
    print(loss)#打印loss
    #反向传播
    dx=probs.copy()#以softmax输出结果作为反向输出
    dx[np.arange(N),t]-=1#实现y-t,y就是softmax层的输出
    dx/=N#到这里是反向传播到softmax前
    dh1,dW2,db2=affine_backward(dx,cachey)#反向传播到第二层
    dh1[relu_cache<=0]=0#反向传播到激活层
    dX,dW1,db1=affine_backward(dh1,fc_cache)#反向传播到第一层
    #参数更新,前两行是引入正则化惩罚项更新dW，后四行是引入学习率更新W和b
    dW2+=reg*W2
    dW1+=reg*W1
    W2+=-epsilon*dW2
    b2+=-epsilon*db2
    W1+=-epsilon*dW1

#验证
"""
本例是一个很简单的神经网络的例子，我们只用了一组数据用来训练，其训练结果应该是比较勉强的。
之所以最终效果还行，只是我们选择验证的例子比较合适。要想得到比较完美的模型，需要有大量的、
分散的训练数据，比如第一象限不仅要有[1,1]这种数据，还要有[1000,1]，[1,1000]这种，这里就不再详述了。
"""
"""
给出了一组数据test，对已经训练好的网络进行验证。其实验证的方法和训练时的正向传播的过程基本一致，
即第一层网络线性计算→激活→第二层网络线性计算→Softmax→得到分类结果。
"""
test=np.array([[2,2],[-2,2],[-2,-2],[2,-2]])
H,fc_cache=affine_forward(test,W1,b1)#仿射
H=np.maximum(0,H)#激活
relu_cache=H
Y,cachey=affine_forward(H,W2,b2)#仿射
#softmax
probs=np.exp(Y - np.max(Y,axis=1,keepdims=True))
probs/=np.sum(probs,axis=1,keepdims=True) # softmax
print(probs)
for k in range(4):
    print(test[k,:],"所在的象限为",np.argmax(probs[k,:])+1)
