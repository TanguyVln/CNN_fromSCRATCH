import numpy as np

class Conv_Layer:

    def __init__(self, num_filters):
        # initiallization of the conv layer with the numbers of kernels as parameters 
        self.num_filters = num_filters

        # we use 3x3 filters
        self.filters = np.random.randn(num_filters, 3, 3) #shape (16, 3, 3) (assuming 16 filters)



    def forward(self, input_data):

        self.last_input = input_data # shape (28, 28)

        in_height, in_width = input_data.shape # 28, 28

        output = np.zeros((in_height - 2, in_width - 2, self.num_filters)) # shape (26, 26, 16)

        for i in range(in_height - 2):
            for j in range(in_width - 2):
                for k in range(self.num_filters):
                    output[i, j, k] = np.sum(input_data[i:i+3, j:j+3] * self.filters[k])

        return output

        
    
    def backward(self, d_out, learn_rate):
        d_filters = np.zeros(self.filters.shape) # shape (16, 3, 3)

        for i in range(self.last_input.shape[0] - 2):
            for j in range(self.last_input.shape[1] - 2):
                for k in range(self.num_filters):
                    d_filters[k] += d_out[i, j, k] * self.last_input[i:i+3, j:j+3] # shape (3, 3)

        self.filters -= learn_rate * d_filters # update the filters with every iteration

        return None
    

class Max_Pool_Layer:

    def forward(self, input_data):
        self.last_input = input_data # shape (26, 26, 16)

        h_prev, w_prev, num_filters = input_data.shape # 26, 26, 16

        output = np.zeros((h_prev // 2, w_prev // 2, num_filters)) # shape (13, 13, 16)

        for i in range(num_filters):
            out_y = 0
            for j in range(0, h_prev, 2):
                out_x = 0
                for k in range(0, w_prev, 2):
                    output[out_y, out_x, i] = np.max(input_data[j:j+2, k:k+2, i]) # shape (2, 2)
                    out_x += 1
                out_y += 1
        
        return output 

    def backward(self, grad_output):
        grad_input = np.zeros(self.last_input.shape) # shape (26, 26, 16)

        h_prev, w_prev, num_filters = self.last_input.shape # 26, 26, 16

        for i in range(num_filters):
            out_y = 0
            for j in range(0, h_prev, 2):
                out_x = 0
                for k in range(0, w_prev, 2):
                    window = self.last_input[j:j+2, k:k+2, i]
                    max_val = np.max(window)
                    grad_input[j:j+2, k:k+2, i] = (window == max_val) * grad_output[out_y, out_x, i]
                    out_x += 1
                out_y += 1

        return grad_input

class FC_layer:
    def __init__(self, len, num_neurons):
        self.weights = np.random.randn(len, num_neurons) / len # shape (2704, 10)
        self.biases = np.zeros(num_neurons) # shape (10,)
    
    def forward(self, input_data):
        self.last_input_shape = input_data.shape # shape (13, 13, 16)

        flattened = input_data.flatten() # shape (13, 13, 16) to (2704,)
        self.last_flattened_input = flattened  

        logits = np.dot(flattened, self.weights) + self.biases # (2704,) @ (2704, 10) = (10,)
        self.last_logits = logits  # shape (10,)

        exp_values = np.exp(logits) # shape (10,)
        self.exp_values = exp_values # shape (10,)

        return exp_values / np.sum(exp_values, axis=0)
    
    def backward(self, d_out, learn_rate):
        z = self.last_logits                # shape (10,)
        x = self.last_flattened_input       # shape (2704,)

       
        probs = np.exp(z) / np.sum(np.exp(z)) # shape (10,)
        dL_dz  = np.zeros_like(z) # shape (10,)

        for i in range(len(z)):
            for j in range(len(z)):
                if i == j:
                    dL_dz[i] += d_out[j] * (probs[i] * (1 - probs[j]))
                else:
                    dL_dz[i] -= d_out[j] * (probs[i] * probs[j])


        dL_dw = np.outer(x, dL_dz) # (2704,) outer (10,) = (2704, 10)
        dL_db = dL_dz # shape (10,)


        dL_dx = self.weights @ dL_dz # (2704, 10) @ (10,) = (2704,)

        self.weights -= learn_rate * dL_dw
        self.biases  -= learn_rate * dL_db

        return dL_dx.reshape(self.last_input_shape) # reshape to (13, 13, 16)
            
        