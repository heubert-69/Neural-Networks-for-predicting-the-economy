#include "NN_class.hpp"
#include "CsvReader.hpp"
#include <iostream>

int main() {
    // Define the size of the dataset
    int rows = 1000; // Adjust to your dataset's row count
    int input_features = 9;
    int output_features = 1;

    // Load the dataset from the CSV file
    std::string csv_file = "path/to/your/dataset.csv";
    Eigen::MatrixXd data = CSVReader::readCSV(csv_file, rows, input_features + output_features);

    // Split the data into inputs and targets
    Eigen::MatrixXd inputs = data.leftCols(input_features);
    Eigen::VectorXd targets = data.rightCols(output_features);

    // Convert inputs and targets to vector of Eigen vectors
    std::vector<Eigen::VectorXf> input_data;
    std::vector<float> target_data;

    for (int i = 0; i < rows; ++i) {
        input_data.push_back(inputs.row(i).cast<float>());
        target_data.push_back(targets(i, 0));
    }

    // Initialize the neural network
    NeuralNetwork nn(input_features, 5, output_features);

    // Training parameters
    int epochs = 1000;
    float learning_rate = 0.01f;

    // Train the network
    nn.train(input_data, target_data, epochs, learning_rate);

    // Test the trained network with a new input
    Eigen::VectorXf test_input(9);
    test_input << 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 1.0f;  // Example numerical vector
    Eigen::VectorXf output = nn.forward(test_input);
    std::cout << "Predicted Output: " << output << std::endl;

    return 0;
}
