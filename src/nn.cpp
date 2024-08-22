#include "../include/NN_class.hpp"

Eigen::VectorXf relu(const Eigen::VectorXf &x) {
    return x.cwiseMax(0.0f);
}

Eigen::VectorXf relu_derivative(const Eigen::VectorXf &x) {
    return (x.array() > 0.0f).cast<float>();
}

// Constructor to initialize weights and biases
NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, int output_size) {
    W1 = Eigen::MatrixXf::Random(hidden_size, input_size);  // Weights for the first layer
    b1 = Eigen::VectorXf::Random(hidden_size);              // Bias for the first layer
    W2 = Eigen::MatrixXf::Random(output_size, hidden_size); // Weights for the second layer
    b2 = Eigen::VectorXf::Random(output_size);              // Bias for the second layer
}

// Forward pass
Eigen::VectorXf NeuralNetwork::forward(const Eigen::VectorXf &input) {
    hidden = relu(W1 * input + b1);
    output = W2 * hidden + b2;
    return output;
}

// Backward pass and gradient update
void NeuralNetwork::backward(const Eigen::VectorXf &input, const Eigen::VectorXf &target, float learning_rate) {
    // Compute loss derivative
    Eigen::VectorXf error = output - target;
    Eigen::VectorXf d_output = error;  // Derivative of MSE Loss with respect to output

    // Backpropagation
    Eigen::MatrixXf d_W2 = d_output * hidden.transpose();
    Eigen::VectorXf d_b2 = d_output;

    Eigen::VectorXf d_hidden = W2.transpose() * d_output;
    Eigen::VectorXf d_relu = d_hidden.cwiseProduct(relu_derivative(hidden));

    Eigen::MatrixXf d_W1 = d_relu * input.transpose();
    Eigen::VectorXf d_b1 = d_relu;

    // Update weights and biases
    W2 -= learning_rate * d_W2;
    b2 -= learning_rate * d_b2;
    W1 -= learning_rate * d_W1;
    b1 -= learning_rate * d_b1;
}

// Training function
void NeuralNetwork::train(const std::vector<Eigen::VectorXf> &inputs, const std::vector<float> &targets, int epochs, float learning_rate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;

        for (size_t i = 0; i < inputs.size(); ++i) {
            // Forward pass
            Eigen::VectorXf target(1);
            target(0) = targets[i];
            forward(inputs[i]);

            // Compute loss (Mean Squared Error)
            float loss = 0.5f * (output - target).squaredNorm();
            total_loss += loss;

            // Backward pass and update weights
            backward(inputs[i], target, learning_rate);
        }

        std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << total_loss / inputs.size() << std::endl;
    }
}
