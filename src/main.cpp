#include "../include/NN_class.hpp"
#include "../include/CsvReader.hpp"
#include <iostream>
#include <string>



int main() {
    // Define the size of the dataset
    int train_rows = 17232; // Adjust to your dataset's row count
    int test_rows = 8391;
    int input_features = 9;
    int output_features = 1;

    // Load the dataset from the CSV file
    std::string train_csv_file = "../data/processed_data/training_data.csv";
    Eigen::MatrixXd train_data = CSVReader::readCSV(train_csv_file, train_rows, input_features + output_features);

    // Split the data into inputs and targets
    Eigen::MatrixXd train_input = train_data.leftCols(input_features);
    Eigen::VectorXd train_target = train_data.rightCols(output_features);

    // Convert inputs and targets to vector of Eigen vectors
    std::vector<Eigen::VectorXf> train_input_data;
    std::vector<float> train_target_data;

    for (int i = 0; i < train_rows; ++i) {
        train_input_data.push_back(train_input.row(i).cast<float>());
        train_target_data.push_back(train_target(i));
    }
    
    
    std::string test_csv_file = "../data/processed_data/testing_data.csv";
    Eigen::MatrixXd test_data = CSVReader::readCSV(test_csv_file, test_rows, input_features + output_features);
    
    Eigen::MatrixXd test_inputs = test_data.leftCols(input_features);
    Eigen::VectorXd test_targets = test_data.rightCols(output_features);
    
    
     std::vector<Eigen::VectorXf> test_input_data;
     std::vector<float> test_target_data;
 
     for (int i = 0; i < test_rows; ++i) {
         test_input_data.push_back(test_inputs.row(i).cast<float>());
         train_target_data.push_back(test_targets(i));
     }
    

    // Initialize the neural network
    NeuralNetwork nn(input_features, 5, output_features);

    // Training parameters
    int epochs = 1000;
    float learning_rate = 0.01f;

    // Train the network
    nn.train(train_input_data, train_target_data, epochs, learning_rate);
    
    //evaluate model on test data
    float total_error = 0.0f;
    
    
    for (int i = 0; i < test_rows; ++i) {
        Eigen::VectorXf prediction = nn.forward(test_input_data[i]);
        float loss  = 0.5f * (prediction(0) - test_target_data[i]) * (prediction(0) - test_target_data[i]);
        
        std::cout << "Test Sample " << i + 1 << " - Actual: " << test_target_data[i] << ", Predicted: " << prediction(0) << std::endl;
    }
    std::cout << "Average Test Loss: " << total_error / test_rows << std::endl;
    return 0;
}
