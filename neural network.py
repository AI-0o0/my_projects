#neural network
# A neuron takes inputs, does some math with them, and produces one output.
# ex: a neuron that takes two inputs
# inputs: x1, x2
# weights: w1, w2
# bias: b
# output: y = w1*x1 + w2*x2 + b
import numpy as np
import math
def sigmoid(x):
    """computes the sigmoid activation function wich normalize the resulte to be bettwen 0 and 1"""
    return 1 / (1 + np.exp(-x))

#to calculate the errore for building the NN
def MSE_value(y_true, y_pred):
    return ((y_true - y_pred)**2).mean()

# y_true = np.array([1,0,0,1])
# y_pred = np.array([0,0,0,0])
# print(MSE_value(y_true, y_pred))

def derivative_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1-fx)

class Neuron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, x):
        total = np.dot(self.weights, x) + self.bias
        return sigmoid(total)

# weights = np.array([0, 1])
# bias = 4
# n = Neuron(weights, bias)
# print(n.feedforward(np.array([2, 2])))

class NeuralNetwork:

    def __init__(self):
        # Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        #biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self , data, all_y_trues , learn_rate = 0.1 , epochs = 1700):
        self.learn_rate = learn_rate
        self.epochs = epochs

        for epoch in range(epochs):
            for x , y_true in zip(data , all_y_trues):
                sum_h1 =  self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                d_ypred_d_w5 = h1 * derivative_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * derivative_sigmoid(sum_o1)
                d_ypred_d_b3 = derivative_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * derivative_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * derivative_sigmoid(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * derivative_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * derivative_sigmoid(sum_h1)
                d_h1_d_b1 = derivative_sigmoid(sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * derivative_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * derivative_sigmoid(sum_h2)
                d_h2_d_b2 = derivative_sigmoid(sum_h2)

                 # --- Update weights and biases
                # Neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

                  # --- Calculate total loss at the end of each epoch
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = MSE_value(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))
def What_is_the_Gender_of(gusse):
    if (round(gusse) == 1) :
        return(f"I'm {gusse * 100}% sure that you are a Female")
    else:
        return(f"I'm {(1- gusse) * 100}% sure you are a male")

data = np.array([
    [-69.4, -5.9],
    [-47.4, -3.9],
    [-36.4, -1.9],
    [-25.4, -2.9],
    [-14.4, -0.9],
    [-3.4, 0.1],
    [7.6, 2.1],
    [18.6, 3.1],
    [29.6, 4.1],
    [40.6, 6.1],
    [33, 3],
    [-9, -5]
])

avg_weight = 152
avg_height = 67

kg_to_pound = 2.20462
cm_to_inchs = 0.393701 

# 1 = Female, 0 = Male
all_y_trues = np.array([
    1, 1, 1, 1, 1,   
    0, 0, 0, 0, 0 , 0 , 0
])

network = NeuralNetwork()
network.train(data, all_y_trues , 0.1, 1000000)

weight = (float(input("pleas enter you weighth in Kg: ")) * kg_to_pound) - avg_weight
height = (float(input("pleas enter you heighth in cm: ")) * cm_to_inchs) - avg_height

user = np.array([weight, height])

print(What_is_the_Gender_of(network.feedforward(user)))
# emily = np.array([128-152, 63-67])
# frank = np.array([25.6,  -1.9]) 

# print("Emily:", What_is_the_Gender_of(network.feedforward(emily)))
# print("Frank:", What_is_the_Gender_of(network.feedforward(frank)))

# print("Emily: %.3f" % network.feedforward(emily))
# print("Frank: %.3f" % network.feedforward(frank))

# this weights are 98% accaurate 
# W1: 0.37628523319812246, W2: 1.1852598965957664, W3: 0.3374869997443513, W4: 0.1393183523371986, W5: -6.365756521920572, W6: -2.8601363996979208 , 
# B1: 3.144094787329415, B2: 2.239148579767488, B3: 4.13255016719388
# print(f"W1: {network.w1}, W2: {network.w2}, W3: {network.w3}, W4: {network.w4}, W5: {network.w5}, W6: {network.w6} , B1: {network.b1}, B2: {network.b2}, B3: {network.b3}" )