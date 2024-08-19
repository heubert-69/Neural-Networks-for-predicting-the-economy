#ifndef CSVREADER_HPP
#define CSVREADER_HPP

#include "../lib/Eigen/Dense"
#include <fstream>
#include <sstream>
#include <vector>

class CSVReader {
public:
    static Eigen::MatrixXd readCSV(const std::string &file, int rows, int cols) {
        Eigen::MatrixXd data(rows, cols);
        std::ifstream infile(file);

        if (!infile.is_open()) {
            throw std::runtime_error("Could not open file");
        }

        std::string line;
        int row = 0;
        while (std::getline(infile, line) && row < rows) {
            std::stringstream ss(line);
            std::string value;
            int col = 0;
            while (std::getline(ss, value, ',') && col < cols) {
                data(row, col) = std::stod(value);
                col++;
            }
            row++;
        }

        infile.close();
        return data;
    }
};

#endif // CSVREADER_HPP
