import torch

def example():
    x = torch.arange(4.0,requires_grad = True)
    print("x:",x)
    print("x.grad:",x.grad)
    print()

    # dot表示向量点积
    y = 2 * torch.dot(x,x)
    print("y:",y)
    y.backward()
    print("y = 2 * torch.dot(x,x),after y.backward():\nx.grad:",x.grad)
    print()

    x.grad.zero_()
    y = x.sum()
    y.backward()
    print("y = x.sum(),after y.backward():\nx.grad:",x.grad)
    print()

    # 非标量的反向传播
    # 这里的y.sum()不应该理解为一个数值，而是表达式y.sum()= x1^2+x2^2+x3^2+x4^2
    # 故x.grad = tensor([2 * x1,2 * x2,2 * x3,2 * x4])
    x.grad.zero_()
    y = x * x
    print("y:",y)
    y.sum().backward()
    print("y = x * x,after y.sum().backward():\nx.grad:",x.grad)
    print()

    # 分离计算
    # 利用detach(),返回u,丢弃计算图中计算y的任何信息,梯度不会从u流向x
    # 因此,z = u * x时,u被视为常数
    x.grad.zero_()
    y = x * x
    u = y.detach()
    z = u * x
    z.sum().backward()
    print("z:",z)
    print("x.grad == u? ",x.grad == u)
    x.grad.zero_()
    y.sum().backward()
    print("x.grad == 2 * x? ",x.grad == 2 * x)

def f(a):
    b = a * 2
    # norm()方法返回平方范数,即平方和开根号
    # 注意到，尽管有python控制流存在，但是c关于a是分段线性的，故一定有f(a) == c == k * a
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

def python_flow():
    # 验证在python控制流下,自动微分的适用性
    a = torch.randn(size = (),requires_grad =True)
    print("a:",a)
    c = f(a)
    print("c:",c)
    c.backward()
    print("a.grad == c / a? ",a.grad == c / a)

if __name__ == "__main__":
    example()
    python_flow()