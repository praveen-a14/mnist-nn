import numpy as np
import gzip

def load_mnist(type='train'):
    labels_path = f'mnist-{type}-labels.gz'
    images_path = f'mnist-{type}-images.gz'
    
    with gzip.open(labels_path, 'rb') as lbpath:
        # read in labels, offset=8 bc first 8 bytes are metadata
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    
    with gzip.open(images_path, 'rb') as imgpath:
        # read in images, each image is 28x28 pixels so reshape into 2d array with size 784
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        # put pixel values on 0-1 scale
        images = images.astype(np.float32) / 255.0  
    
    return images, labels

# relu function and derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# function to convert raw matrix multiplicaion values to a valid probability dist (sums to 1)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# loss function (log loss)
def log_loss(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / m

# creates a 10x10 identity matrix, one row per number (0-9). i.e., 3 is [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        self.relu = relu
        self.relu_derivative = relu_derivative
        
        # initialize weights to random values, biases to zeros
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i+1]) * 0.01)
            self.biases.append(np.zeros((1, layers[i+1])))
    
    def forward(self, X):
        activations = [X]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(activations[-1], w) + b
            zs.append(z)
            # if the next layer is the output layer, softmax to get probabilities
            if w.shape[1] == 10: 
                activations.append(softmax(z))
            # if not, then relu and continue to feed forward
            else:
                activations.append(self.relu(z))
        return activations, zs
    
    def backward(self, activations, zs, y_true):
        m = y_true.shape[0]
        d_weights = []
        d_biases = []
        
        # Compute output layer error
        delta = activations[-1] - y_true
        
        for i in reversed(range(len(self.weights))):
            # compute weight gradient, transpose to fix dimension mismatches
            dW = np.dot(activations[i].T, delta) / m
            # compute bias gradient
            db = np.sum(delta, axis=0, keepdims=True) / m
            # insert gradients at front of list to maintain order
            d_weights.insert(0, dW)
            d_biases.insert(0, db)
            
            if i > 0:
                # compute delta for previous layer using the chain rule (backpropagation)
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(zs[i-1])
        
        return d_weights, d_biases
    
    # adjust weights and biases based on deltas obtained during backpropagation
    def update_weights(self, d_weights, d_biases):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * d_weights[i]
            self.biases[i] -= self.learning_rate * d_biases[i]
    
    def train(self, X, y, epochs, batch_size):
        y_encoded = one_hot_encode(y)
        for epoch in range(epochs):
            shuffled_indices = np.random.permutation(len(X))
            X, y_encoded = X[shuffled_indices], y_encoded[shuffled_indices]
            
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y_encoded[i:i+batch_size]
                
                activations, zs = self.forward(X_batch)
                d_weights, d_biases = self.backward(activations, zs, y_batch)
                self.update_weights(d_weights, d_biases)
            
            loss = log_loss(y_encoded, self.forward(X)[0][-1])
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}')
    
    def predict(self, X):
        activations, _ = self.forward(X)
        # gives the index (also value) of the highest probability digit
        return np.argmax(activations[-1], axis=1)
    
    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

# load mnist test and train sets
train_images, train_labels = load_mnist('train')
test_images, test_labels = load_mnist('test')

# create and train model
nn = NeuralNetwork(layers=[784, 128, 64, 10], learning_rate=0.01)
nn.train(train_images, train_labels, epochs=250, batch_size=64)

# evaluate
print(f'Test Accuracy: {nn.accuracy(test_images, test_labels):.4f}')
