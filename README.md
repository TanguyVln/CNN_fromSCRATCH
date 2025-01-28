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
In the backward pass, we calculate the gradient of the loss with respect to each filter. That allows us to update their values in the correct way using gradient descent.

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


## Max Pool Layer 
This layer is used to reduce the spatial dimension of the feature maps obtained from to the convolutional layer.

Reducing the height and width of the feature maps allow us to decrease the computational cost (fewer parameters) and to control overfitting by reducing the spatial size.

Here we use a 2x2 Max Pool, that means that each feature map is divided into 2x2 windows and the maximum value of each window is used as the single value of the output.



![image](https://github.com/user-attachments/assets/996dcd0f-4ae7-4126-9ce3-815eeb978e82)  
https://matlab1.com/max-pooling-in-convolutional-neural-network/


### Forward Pass
In the forward pass, we scan the input in 2x2 windows and select the maximum value of each window.

```python
# set the output (half of the size in height and width)
output = np.zeros((h // 2, w // 2, num_filters))  

# max pooling for each filter
for i in range(num_filters):
    out_y = 0
    for j in range(0, h, 2): # steps of 2
        out_x = 0
        for k in range(0, w, 2): # steps of 2
            # maximum of the region we're in (2x2)
            output[out_y, out_x, i] = np.max(input_data[j:j+2, k:k+2, i])
            out_x += 1
        out_y += 1
```

### Backward Pass
During the backpropagation, we need to pass the gradient of the pooled output to the original input (the position of the maximum value we used for each 2x2 window).

```python
# set the gradient of the input
grad_input = np.zeros(self.last_input.shape)

# same as forward
for i in range(num_filters):
    out_y = 0
    for j in range(0, h, 2):
        out_x = 0
        for k in range(0, w, 2):
            # find the max in all window
            window = self.last_input[j:j+2, k:k+2, i]
            max_val = np.max(window)
                                
            # the gradient goes to the position of the maximum value
            grad_input[j:j+2, k:k+2, i] = (window == max_val) * grad_output[out_y, out_x, i]
            
            out_x += 1
        out_y += 1
```

## Fully Connected Layer


![image](https://github.com/user-attachments/assets/73aff3ba-8a22-432d-a5f6-86e751ad687f)
https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks

