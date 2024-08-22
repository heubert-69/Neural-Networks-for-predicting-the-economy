#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "../lib/Eigen/Dense"
#include <vector>
#include <iostream>

// Simple activation function (ReLU)
Eigen::VectorXf relu(const Eigen::VectorXf &x);

// Derivative of ReLU
Eigen::VectorXf relu_derivative(const Eigen::VectorXf &x);

// Simple neural network class using Eigen
class NeuralNetwork {
public:
    NeuralNetwork(int input_size, int hidden_size, int output_size);
    Eigen::VectorXf forward(const Eigen::VectorXf &input);
    void backward(const Eigen::VectorXf &input, const Eigen::VectorXf &target, float learning_rate);
    void train(const std::vector<Eigen::VectorXf> &inputs, const std::vector<float> &targets, int epochs, float learning_rate);

private:
    Eigen::MatrixXf W1, W2;
    Eigen::VectorXf b1, b2;
    Eigen::VectorXf hidden, output;
};

#endif // NEURAL_NETWORK_HPP
