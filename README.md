# Depthwise Separable Convolution_Pytorch
Implementation of Depthwise Separable Convolution

Depthwise Separable Convolution was first introduced in [Xception: Deep Learning with Depthwise Separable Convolutions
](https://arxiv.org/pdf/1610.02357.pdf)

## Installation
```
git clone https://github.com/seungjunlee96/DepthwiseSeparableConvolution_Pytorch.git
cd DepthwiseSeparableConvolution_Pytorch/
python3 setup.py install --user
```
## Usage

```python
from DepthwiseSeparableConvolution import depthwise_separable_conv

## In your Network
depthwise_separable_conv(nin, nout, kernel_size = 3, padding = 1, bias=False)

```



## Explanation on Depthwise Separable Convolution
### 1.Depthwise Convolution
![depthwise](./images/depthwise.png)



```python
class depthwise_conv(nn.Module): 
  def __init__(self, nin, kernels_per_layer): 
    super(depthwise_separable_conv, self).__init__() 
    self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin) 
  
  def forward(self, x): 
    out = self.depthwise(x) 
    return out
 ```

### 2.Pointwise Convolution
![pointwise](./images/pointwise.png)

```python
class pointwise_conv(nn.Module):
  def __init__(self, nin, nout): 
    super(depthwise_separable_conv, self).__init__() 
    self.pointwise = nn.Conv2d(nin, nout, kernel_size=1) 
    
  def forward(self, x): 
    out = self.pointwise(x) 
    return out
 ```
### 3.Depthwise Separable Convoltion
![DepthwiseSeparable](./images/DepthwiseSeparable.jpeg)

 ```python
class depthwise_separable_conv(nn.Module):
  def __init__(self, nin, kernels_per_layer, nout): 
    super(depthwise_separable_conv, self).__init__() 
    self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin) 
    self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1) 
   
  def forward(self, x): 
    out = self.depthwise(x) 
    out = self.pointwise(out) 
    return out
 ```
 
 
## To Do
For example usage, please refer to `./example/Xception.py`.

## references

* [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357.pdf)
* https://discuss.pytorch.org/t/depthwise-and-separable-convolutions-in-pytorch/7315
* [hoya012/pytorch-Xception](https://github.com/hoya012/pytorch-Xception)
* https://wingnim.tistory.com/104
