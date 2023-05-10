import torch

def make_tensor():
    # 封装了基本的张量构造，常规操作的例子
    x = torch.arange(12)
    print("x:",x)
    print("x的形状:",x.shape)
    print("x的元素总数:",x.numel())

    # reshape的-1表示自动计算
    # x_reshaped 元素为3的轴成为轴0，元素4的轴称为轴1，以此类推
    # 轴1的list有4个内容为数值的元素，轴0的list有三个内容为list的元素
    # 轴的规律：数字最大的轴即为shape的最右一个参数，这个轴对应的是一个元素为值的list
    x_reshaped = x.reshape(-1,4)
    print("x_reshaped:",x_reshaped)
    x_reshaped[0:2,0:2] = 12
    print("let x_reshaped[0:2,0:2] = 12:",x_reshaped)

    z = torch.zeros((2,3,4))
    print("z:",z)

    # randn方法的每个元素都从均值0、标准差为1的标准高斯分布（正态分布）中随机采样
    r = torch.randn(3,4)
    print("r:",r)

def operate():
    # 封装了基本的张量运算的例子
    x = torch.tensor([1.0,2,4,8])
    y = torch.tensor([2.0,2,2,2])

    print("x + y:", x + y)
    print("x - y:", x - y)
    print("x * y:", x * y)
    print("x / y:", x / y)
    # **表示求幂
    print("x ** y:", x ** y)
    print("exp(x)：",torch.exp(x))
    print("x == y:",x == y)
    print("x.sum():",x.sum())
    print("torch.dot(x,y):",torch.dot(x,y))

def concatenate():
    # 封装了张量连结的例子
    # cat的规律:沿着哪个dim进行cat，新tensor的哪个dim就会增加
    x = torch.arange(12,dtype = torch.float32).reshape(3,4)
    y = torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
    print("x cat y,dim = 0:",torch.cat((x,y),dim = 0))
    print(torch.cat((x,y),dim = 0).shape)
    print("x cat y,dim = 1:",torch.cat((x,y),dim = 1))
    print(torch.cat((x,y),dim = 1).shape)
    print()

def broadcast():
    # 封装了张量广播的例子
    # 广播机制：通过复制元素来扩展tensor,使两个tensor的shape相同,再对生成的数组执行按元素操作
    a = torch.arange(3).reshape((3,1))
    b = torch.arange(2).reshape((1,2))
    print("broadcast:")
    print("a:",a)
    print("b:",b)
    print("a + b：",a+b)
    print()

def memory():
    # 探索了tensor对id的处理
    x = torch.tensor([1.0,2,4,8])
    y = torch.tensor([2,2,2,2])
    print("Memory remains unchanged?")
    before = id(y)
    y = y + x
    print("y = y + x",id(y) == before)

    before = id(y)
    y += x
    print("y += x",id(y) == before)

    before = id(y)
    y[:] = x + y
    print("y[:] = x + y",id(y) == before)

    print()

def tensor_norm():
    # 范数的简单例子
    u = torch.tensor([3.0,-4.0])
    print("L2norm,torch.norm(u):",torch.norm(u))
    print("L1norm,torch.abs(u).sum():",torch.abs(u).sum())
    # 矩阵的2-范数
    print("torch.norm(torch.ones((4,9))):",torch.norm(torch.ones((4,9))))

if __name__ == "__main__":
    if torch.cuda.is_available():
        print('GPU is available!\n')
    else:
        print('GPU is not available!\n')

    make_tensor()
    operate()
    concatenate()
    broadcast()
    memory()
    tensor_norm()


