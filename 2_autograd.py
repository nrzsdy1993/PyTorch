import numpy as np
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if torch.cuda.is_available():
  DEVICE = torch.device('cuda')
else:
  DEVICE = torch.device('cpu')

"""### 기본 값 및 데이터 샘플링
- BATCH_SIZE: 파라미터 업데이트 시, 계산되는 데이터의 개수 (== INPUT)
- INPUT_SIZE: INPUT_SIZE는 INPUT의 크기이자 입력층의 노드 수를 의미. 

```
(BATCH_SIZE, INPUT_SIZE)
```

- HIDDEN_SIZE: INPUT을 다수의 파라미터를 이용해 계산한 결과에 한 번 더 계산되는 파라미터 수. (64, 1000) ~ 행렬 곱 ~ (1000, 100) 크기의 행렬과 행렬 곱 계산

- OUTPUT_SIZE: 최종적으로 출력되는 값의 벡터의 크기 의미. 

"""

BATCH_SIZE = 64
INPUT_SIZE = 1000
HIDDEN_SIZE = 100
OUTPUT_SIZE = 10

x  = torch.randn(BATCH_SIZE, INPUT_SIZE, device = DEVICE, dtype = torch.float, requires_grad = False)
y  = torch.randn(BATCH_SIZE, OUTPUT_SIZE, device = DEVICE, dtype = torch.float, requires_grad = False)
w1 = torch.randn(INPUT_SIZE, HIDDEN_SIZE, device=DEVICE, dtype=torch.float, requires_grad=True)
w2 = torch.randn(HIDDEN_SIZE, OUTPUT_SIZE, device=DEVICE, dtype=torch.float, requires_grad=True)

"""- randn은 평균이 0, 표준편차가 1인 정규분포에서 샘플링한 값 
- 업데이트할 파라미터 값 설정 (Back Propagation) 위해서는 requires_grad = True 이용

### 학습 구현
- 딥러닝 모델을 구현하는 코드를 작성합니다.
"""

learning_rate = 1e-6 
for t in range(1, 501):
  y_pred = x.mm(w1).clamp(min = 0).mm(w2)
  loss = (y_pred - y).pow(2).sum()
  if t % 100 == 0:
    print("Iteration: ", t, "\t", "Loss: ", loss.item())
  loss.backward() # back propagation 적용 시, 새로 계산함. 

  with torch.no_grad():
    w1 -= learning_rate * w1.grad
    w2 -= learning_rate * w2.grad

    w1.grad.zero_() # Gradient 값으로 0으로 설정. 
    w2.grad.zero_()

"""- clamp 함수는 비선형 함수를 적용한다. 
- 최솟값이 0이며, 0보다 큰 값은 자기 자신을 갖게 되는 메서드. 
- 참고자료: https://pytorch.org/docs/stable/generated/torch.clamp.html
- pow() method는 지수를 취하는 기본 메서. pow(2) 제곱차를 의미함.
"""