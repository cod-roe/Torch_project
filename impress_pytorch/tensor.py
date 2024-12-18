#%%
import torch
# %%
torch.tensor([0,1,2,3])
# %%
torch.Tensor([0,1,2,3])
# %%
torch.Tensor(2,3)
# %%
torch.Tensor(3)
# %%
torch.tensor(range(10))
# %%
torch.arange(10)
# %%
torch.tensor([[0,1,2],[3,4,5]])
# %%
torch.arange(6).reshape(2,3)
# %%
a = torch.arange(6).reshape(2,3)
# %%
a
# %%
a.reshape(3,2)
# %%
a = torch.arange(5)
# %%
a
# %%
a + 2
# %%
a -2
# %%
2 * a
# %%
a / 2
# %%
a = torch.arange(6).reshape(2,3)
# %%
a
# %%
b = a + 1
# %%
b
# %%
a + b
# %%
a - b
# %%
a * b
# %%
a / b
# %%
a0 = torch.tensor([1.,2.,3.,4.])
a1 = torch.tensor([5.,6.,7.,8.])

# %%
torch.dot(a0,a1)
# %%
32 + 21 + 12 + 5
# %%
torch.matmul(a0,a1)
# %%
a0 = torch.tensor([1,2,3,4])
# %%
a0
# %%
a1 = torch.arange(8).reshape((2,4))
# %%
a1
# %%
torch.mv(a1,a0)
# %%
torch.matmul(a1,a0)
# %%
a0 = torch.arange(8).reshape(2,4)
# %%
a0
# %%
a1 = torch.arange(8).reshape(4,2)
# %%
a1
# %%
torch.mm(a0,a1)
# %%
torch.matmul(a0,a1)
# %%
a0 = torch.arange(24).reshape(-1,2,4)
a1 = torch.arange(24).reshape(-1,4,2)

# %%
torch.bmm(a0,a1)
# %%
torch.matmul(a0,a1)
# %%
a0
# %%
a = torch.tensor([1.,2.,3.])
# %%
torch.sin(a)
# %%
torch.log(a)
# %%
a = torch.tensor([1,2,3])
torch.sin(a)
# %%
a0 = torch.tensor([1,2,3])
# %%
a0.dtype
# %%
a0.type()
# %%
a1 = torch.tensor([1.,2.,3.,])

# %%
a1.dtype
# %%
a1.type()
# %%
a0 = torch.tensor([1,2,3])
# %%
a0.type()
# %%
a0 = torch.tensor([1,2,3],dtype=torch.float)
# %%
a0.type()
# %%
a1 = a0.type(torch.LongTensor)
# %%
a1
# %%
a1.type()
# %%
a1.dtype
# %%
a0
# %%
a0.dtype
# %%
a0 = torch.tensor([1,2,3])
# %%
a0.dtype
# %%
b0 = a0.numpy()
# %%
b0.dtype
# %%
a1 = torch.from_numpy(b0)
# %%
a1.dtype
# %%
a = torch.tensor([1.],requires_grad=True)

# %%
a.dtype
# %%
b = a.detach().numpy()
# %%
b.dtype

# %%
a = torch.zeros(6).reshape(2,3)
b = torch.ones(6).reshape(2,3)

# %%
torch.cat([a,b])
# %%
a
# %%
b
# %%
torch.cat([a,b],dim=1)
# %%
a = torch.zeros(6).reshape(2,3)
b = torch.ones(6).reshape(2,3)

# %%
c = b + 1
# %%
d = torch.stack([a,b,c])
# %%
d.shape
# %%
a = torch.arange(6).reshape(2,3)
# %%
a
# %%
d = a.unsqueeze(0)
# %%
d.shape
# %%
a = torch.arange(12).reshape(2,2,-1)
# %%
a
# %%
a.shape
# %%
a.permute(2,0,1)
# %%
