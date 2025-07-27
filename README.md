📊 Neural Networks for Predicting the Economy
---
This project is a lightweight feedforward neural network implemented in C++, designed to train on structured tabular data (e.g., CSV files). It uses Eigen for linear algebra and includes custom CSV parsing and a neural network class.
---
📌 Overview
---
The program performs the following steps:

Reads training and testing data from CSV files.

Splits data into input features and targets.

Converts data into Eigen-compatible vector format.

Trains a feedforward neural network on the training set.

Evaluates the model on the test set and prints loss per sample.
---
📂 Project Structure
---
```makefile
project-root/
├── src/
│   └── main.cpp               # The main file (this script)
├── include/
│   ├── NN_class.hpp           # Neural network class definition
│   └── CsvReader.hpp          # CSV reading utility
├── data/
│   └── processed_data/
│       ├── training_data.csv  # Training dataset
│       └── testing_data.csv   # Testing dataset
```
---
⚙️ Build Requirements
---
C++17 or later

Eigen 3 (header-only)

A C++ compiler like g++ or clang++

🏗️ Compilation Example
```bash
g++ -std=c++17 -I ./include src/main.cpp -o nn_trainer
```
🚀 Usage
```bash
./nn_trainer
```
This will:

- Load training_data.csv and testing_data.csv

- Train a simple feedforward neural network

- Output predictions vs actual values for test samples

- Print average test loss

🧠 Model Architecture
Inputs: 9 features

Hidden Layer: 5 neurons (ReLU)

Outputs: 1 target (regression)

⚙️ Training Configuration
Epochs: 1000

Learning Rate: 0.01

📝 Code Walkthrough
1. Load Data
```cpp
Eigen::MatrixXd train_data = CSVReader::readCSV(train_csv_file, train_rows, input_features + output_features);
```
Splits into:

train_input → Feature matrix

train_target → Target values

Converted to vectors for neural network input.

2. Model Initialization
```cpp
NeuralNetwork nn(input_features, 5, output_features);
```
Creates a simple neural net with 1 hidden layer.

3. Training
```cpp
nn.train(train_input_data, train_target_data, epochs, learning_rate);
```
Performs forward and backward passes to minimize mean squared error.

4. Evaluation
```cpp
Eigen::VectorXf prediction = nn.forward(test_input_data[i]);
```
Outputs predictions and compares with actual target values.

⚠️ Notes
There’s a bug in the code: test_target_data is not filled correctly. It uses train_target_data.push_back(...) instead of test_target_data.push_back(...). Fix:

```cpp
test_target_data.push_back(test_targets(i));
```
Add error accumulation to calculate average test loss:

```cpp
total_error += loss;
```
📈 Example Output
```yaml
Test Sample 1 - Actual: 135.0, Predicted: 133.57
Test Sample 2 - Actual: 142.0, Predicted: 140.32
...
Average Test Loss: 4.23
```

📚 Dependencies
Eigen for matrix math

STL (vector, iostream, fstream, etc.)

🧪 Future Improvements
Add activation functions and layer customization

Save and load model weights

Add support for classification tasks

👨‍💻 Author
Made with 💻 and ☕ by Heubert-69

