#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

void load_csv(const string &file_path, vector<vector<double>> &X, vector<int> &y) {
    auto load_start = std::chrono::high_resolution_clock::now();

    ifstream file(file_path);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << file_path << endl;
        exit(1);
    }

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<double> row;

        getline(ss, value, ',');
        y.push_back(stoi(value));

        while (getline(ss, value, ',')) {
            row.push_back(stod(value));
        }

        X.push_back(row);
    }

    file.close();
    cout << "Loaded " << X.size() << " samples from " << file_path << "\n";

    auto load_end = std::chrono::high_resolution_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count();
    cout << "Loading " << file_path << " took " << load_duration << " ms\n";
}

class LogisticRegression {
private:
    double learning_rate;
    int epochs;
    MatrixXd weights;
    vector<double> train_losses;
    vector<double> test_losses;

    MatrixXd sigmoid(const MatrixXd& z) {
        return 1.0 / (1.0 + (-z.array()).exp());
    }

    MatrixXd softmax(const MatrixXd& z) {
        MatrixXd exp_z = (z.rowwise() - z.colwise().maxCoeff()).array().exp();
        return exp_z.array().rowwise() / exp_z.rowwise().sum().array();
    }

    double categorical_crossentropy(const MatrixXd& y, const MatrixXd& y_pred, double epsilon = 1e-15) {
        MatrixXd clipped = y_pred.cwiseMax(epsilon).cwiseMin(1 - epsilon);
        return -((y.array() * clipped.array().log()).rowwise().sum()).mean();
    }

    MatrixXd gradient_descent(const MatrixXd& X, const MatrixXd& y, const MatrixXd& weights, double lr) {
        MatrixXd y_pred = softmax(X * weights);
        MatrixXd error = y_pred - y;
        MatrixXd gradient = X.transpose() * error / y.rows();
        return weights - lr * gradient;
    }

public:
    LogisticRegression(double lr, int ep) : learning_rate(lr), epochs(ep) {}

    void fit(const MatrixXd& X_train, const MatrixXd& y_train, const MatrixXd& X_test, const MatrixXd& y_test) {
        int num_features = X_train.cols();
        int num_classes = y_train.cols();
        weights = MatrixXd::Random(num_features, num_classes);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            weights = gradient_descent(X_train, y_train, weights, learning_rate);

            MatrixXd train_pred = softmax(X_train * weights);
            double train_loss = categorical_crossentropy(y_train, train_pred);
            train_losses.push_back(train_loss);

            double test_loss = 0;
            if (X_test.rows() > 0 && y_test.rows() > 0) {
                MatrixXd test_pred = softmax(X_test * weights);
                test_loss = categorical_crossentropy(y_test, test_pred);
                test_losses.push_back(test_loss);
            }
            cout << "Epoch " << epoch + 1 << "/" << epochs
                 << ": Train Loss = " << train_loss
                 << ", Test Loss = " << test_loss << endl;
        }
    }
};

int main() {
    vector<vector<double>> X_train, X_test;
    vector<int> y_train, y_test;

    load_csv("train.csv", X_train, y_train);
    load_csv("test.csv", X_test, y_test);

    MatrixXd X_train_eigen = Map<MatrixXd>(X_train[0].data(), X_train.size(), X_train[0].size());
    MatrixXd y_train_eigen = MatrixXd::Zero(y_train.size(), 10);
    for (size_t i = 0; i < y_train.size(); ++i) {
        y_train_eigen(i, y_train[i]) = 1;
    }

    MatrixXd X_test_eigen = Map<MatrixXd>(X_test[0].data(), X_test.size(), X_test[0].size());
    MatrixXd y_test_eigen = MatrixXd::Zero(y_test.size(), 10);
    for (size_t i = 0; i < y_test.size(); ++i) {
        y_test_eigen(i, y_test[i]) = 1;
    }

    LogisticRegression lr(0.01, 10);
    lr.fit(X_train_eigen, y_train_eigen, X_test_eigen, y_test_eigen);
    return 0;
}