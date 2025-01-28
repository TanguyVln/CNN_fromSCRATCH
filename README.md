# CNN from scratch

I wanted to have a better understanding of how a CNN works, so I decided to do this project from scratch using NumPy. 

## Convolutional Layer
The convolutional layer is responsible for detecting features in the input image or feature map. It uses trainable filters to scan the input image and detect patterns (like edges, textures...). Each filter is used sliding toward the input image, displaying a feature map.  


It takes as argument:  
* The number of filters to determine how many feature maps we'll use (here 16)  
* The size of the filters (here 3x3 matrix)  
* I decided to not use padding and use a stride of 1 (we move the filter one pixel everytime)  



![image](https://github.com/user-attachments/assets/b3858551-3182-445c-91fc-1609bc191b05)
https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks

### Forward pass
The convolution is done by multiplying each filter with every possible 3x3 zone of the input image and then summing the results. This gives us a new 3d output where the last dimension represent the number of filters.

```python
# start by creating the output volume (H-2) x (W-2) x (num_filters)
output = np.zeros((H - 2, W - 2, self.num_filters))

# slide each filter toward the input
for i in range(H - 2):
    for j in range(W - 2):
        for k in range(self.num_filters):
            # dot product between the filter and the input zone
            output[i, j, k] = np.sum(input_data[i:i+3, j:j+3] * self.filters[k])
```


### Backward pass
In the backward pass, we calcultate the gradient of the loss with respect to each filter. That allows us to uodate their value in the good way using gradient descent.

```python
# initialize the gradient array 
d_filters = np.zeros(self.filters.shape)

# loop in each zone of the original input 
for i in range(self.last_input.shape[0] - 2):
    for j in range(self.last_input.shape[1] - 2):
        for k in range(self.num_filters):
            # compute the gradient with respect to this filter
            d_filters[k] += d_out[i, j, k] * self.last_input[i:i+3, j:j+3]

# update the filters by gradient descent
self.filters -= learn_rate * d_filters

```
