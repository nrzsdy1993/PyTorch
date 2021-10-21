import numpy as np
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

print(np.__version__)
print(torch.__version__)

"""- Create an array of ones"""

print(np.ones(5))
print(torch.ones(5))

"""- Create an array of zeros"""

print(np.zeros(5))
print(torch.zeros(5))

"""- Create a random array"""

print(np.random.rand(5))
print(torch.rand(5))

"""- Create an array from given values"""

print(np.array(5))
print(torch.tensor(5))

"""- Get an array shape"""

temp = np.zeros((2, 5))
print(temp.shape)

temp = torch.zeros((2, 5))
print(temp.shape)

"""## Tensor Data Type

- Define a tensor with a default data type
"""

x = torch.ones(2, 2)
print(x)
print(x.dtype)

"""- Specify the data type when defining a tensor"""

x = torch.ones(2, 2, dtype = torch.int8)
print(x)
print(x.dtype)

"""## Changing the tensor's data type"""

x = torch.ones(1, dtype=torch.uint8)
print(x.dtype)

x = x.type(torch.float)
print(x.dtype)

"""## Converting tensors into NumPy arrays"""

x = torch.rand(2, 2)
print(x)
print(x.dtype)

"""- Convert the tensor into a NumPy array"""

y = x.numpy()
print(y)
print(y.dtype)

"""## Arithmetic Operation
- 텐서를 활용하여 사칙 연산을 수행합니다.

### Scalar 
- 값 하나의 상수값을 의미한다.
"""

import torch
scalar_1 = torch.tensor([1.])
scalar_2 = torch.tensor([5.])

add = scalar_1 + scalar_2
torch_add = torch.add(scalar_1, scalar_2)
print(add)
print(torch_add)

sub = scalar_1 - scalar_2
torch_sub = torch.sub(scalar_1, scalar_2)
print(sub)
print(torch_sub)

mul = scalar_1 * scalar_2
torch_mul = torch.mul(scalar_1, scalar_2)
print(mul)
print(torch_mul)

div = scalar_1 / scalar_2
torch_div = torch.div(scalar_1, scalar_2)
print(div)
print(torch_div)

"""### Vector
- 하나의 값을 표현할 때 2개 이상의 수치로 표현한 것
- 사칙 연산의 문법은 Scalar와 동일하게 적용하기 때문에 생략합니다. 
- torch.dot에 대해 설명하면 다음과 같습니다. 
  + 공식: (1 * 4) + (2 * 5) + (3 * 6) = 4 + 10 + 18 = 32
"""

vector_1 = torch.tensor([1., 2., 3.])
vector_2 = torch.tensor([4., 5., 6.])
torch.dot(vector_1, vector_2)

"""### Matrix
- 2개 이상의 벡터 값을 통합해 구성된 값
- 선형 대수의 기본 단위임. 
- 사칙연산도 기존 Scalar와 동일하기 때문에 생략한다.
- 행렬 곱 연산은 torch.matmul을 활용할 수 있다. 

"""

matrix_1 = torch.tensor([[1., 2.], [3., 4.]])
matrix_2 = torch.tensor([[5., 6.], [7., 8.]])

print(torch.matmul(matrix_1, matrix_2))
print("-- 행렬 곱 연산 --")
print(1 * 5 + 2 * 7)
print(1 * 6 + 2 * 8)
print(3 * 5 + 4 * 7)
print(3 * 6 + 4 * 8)

"""### Tensor
- 행렬이 2차원의 배열이라 한다면, 텐서는 2차원 이상의 배열

"""

tensor_1 = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
tensor_2 = torch.tensor([[[9., 10.], [11., 12.]], [[13., 14.], [15., 16.]]])

torch.matmul(tensor_1, tensor_2)

"""## Converting NumPy arrays into tensors

"""

x = np.zeros((2, 2), dtype=np.float32)
print(x)
print(x.dtype)

"""- Convert the numpy array into PyTorch tensor """

y = torch.from_numpy(x)
print(y)
print(y.dtype)

"""## Moving between devices"""

x = torch.tensor([1.5, 2])
print(x)
print(x.device)

# """## CUDA Device
#
# """
#
# if torch.cuda.is_available():
#   device = torch.device("cuda:0")
#
# x = x.to(device)
# print(x)
# print(x.device)
#
# device = torch.device('cpu')
# x = x.to(device)
# print(x)
# print(x.device)
#
# device = torch.device("cuda:0")
# x = torch.ones(2, 2, device=device)
# print(x)