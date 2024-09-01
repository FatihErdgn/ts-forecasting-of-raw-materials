'BASIC FORWARD PROPOGATION MODEL'
import numpy as np

input_data = np.array([2,3])

weights = {
            'node_0':np.array([1,2]),
            'node_1':np.array([1,-1]),
            'output':np.array([3,4])
            }

node_0_val_in = (input_data * weights['node_0']).sum()
node_0_val_out = np.tanh(node_0_val_in) #activation function

node_1_val_in = (input_data * weights['node_1']).sum()
node_1_val_out = np.tanh(node_1_val_in)

hidden_layer_outputs = np.array([node_0_val_out,node_1_val_out])

output = (hidden_layer_outputs * weights['output']).sum()
print(output)

def relu(val):
    '''val > 0 then val else 0'''
    # Calculate the value for the output of the relu function: output
    output = max(0, val)
    
    # Return the value just calculated
    return(output)

'''SLOPE VE LEARNING RATE ILE WEIGHTS UZERINDE GUNCELLEME YAPILIR VE MODEL IYILESTIRILEREK TARGETE YAKLASTIRILIR (GRADIENT DESCENT)'''
target = 3
# Set the learning rate: learning_rate
learning_rate = 0.01

# Calculate the predictions: preds
preds = (weights * input_data).sum()

# Calculate the error: error
error = preds - target

# Calculate the slope: slope
slope = 2 * input_data * error

# Update the weights: weights_updated
weights_updated = weights - learning_rate * slope

# Get updated predictions: preds_updated
preds_updated = (input_data * weights_updated).sum()

# Calculate updated error: error_updated
error_updated = preds_updated - target

# Print the original error
print(error)

# Print the updated error
print(error_updated)