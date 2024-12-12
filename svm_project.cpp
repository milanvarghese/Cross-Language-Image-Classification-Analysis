#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cmath>
#include <chrono>
using namespace std;

// SVM Class Definition
class SVM {
private:
    vector<vector<double>> weights; // Weights for each class
    vector<double> biases;          // Bias for each class
    double learning_rate;           // Learning rate
    double lambda_param;            // Regularization parameter
    int n_epochs;                   // Number of epochs
    int n_classes;                  // Number of classes

public: 
    // Class constructor
    SVM(double learning_rate, double lambda_param, int n_epochs, int n_classes)
        : learning_rate(learning_rate), lambda_param(lambda_param), n_epochs(n_epochs), n_classes(n_classes) {
        weights.resize(n_classes, vector<double>(0)); // Resize weights and biases for num_classes
        biases.resize(n_classes, 0.0);
    }

    // Training fit method
    double fit(const vector<vector<double>> &X, const vector<int> &y, int n_features,
               const vector<vector<double>> &X_test, const vector<int> &y_test) {
        for (int i = 0; i < n_classes; ++i) {
            weights[i].resize(n_features, 0.0); // Resize weights to match n_features
        }

        double best_accuracy = 0.0;

        // Start timer for training
        auto train_start = std::chrono::high_resolution_clock::now();

        for (int epoch = 0; epoch < n_epochs; ++epoch) {
            for (int c = 0; c < n_classes; ++c) {
                vector<double> &w = weights[c];
                double &b = biases[c];

                for (size_t i = 0; i < X.size(); ++i) {
                    int y_binary = (y[i] == c) ? 1 : -1; // Convert labels to binary
                    double linear_output = inner_product(w.begin(), w.end(), X[i].begin(), -b);

                    // Update weights and biases
                    if (y_binary * linear_output < 1) { // Check margin
                        for (int j = 0; j < n_features; ++j) {
                            w[j] -= learning_rate * (2 * lambda_param * w[j] - y_binary * X[i][j]);
                        }
                        b -= learning_rate * (-y_binary);
                    } else {
                        for (int j = 0; j < n_features; ++j) {
                            w[j] -= learning_rate * (2 * lambda_param * w[j]);
                        }
                    }
                }
            }

            // Compute accuracy
            int correct = 0;
            for (size_t i = 0; i < X_test.size(); ++i) {
                int pred = predict(X_test[i]);
                if (pred == y_test[i]) {
                    ++correct;
                }
            }
            double accuracy = static_cast<double>(correct) / X_test.size() * 100.0;
            best_accuracy = max(best_accuracy, accuracy);
            cout << "Epoch " << epoch + 1 << "/" << n_epochs << " - Accuracy: " << fixed << setprecision(2) << accuracy << "%\n";
        }

        auto train_end = std::chrono::high_resolution_clock::now();
        auto train_duration = std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start).count();
        cout << "Training took " << train_duration << " ms\n";

        return best_accuracy;
    }


    // Predict method
    int predict(const vector<double> &x) const {
        double max_score = -INFINITY;
        int best_class = -1;

        for (int c = 0; c < n_classes; ++c) {
            double linear_output = inner_product(weights[c].begin(), weights[c].end(), x.begin(), -biases[c]);

            // Return class with maximum score and return prediction
            if (linear_output > max_score) {
                max_score = linear_output;
                best_class = c;
            }
        }

        return best_class;
    }
};

// Function to load CIFAR-10 CSV files
void load_csv(const string &file_path, vector<vector<double>> &X, vector<int> &y) {
    // Start timer for loading CSV
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

        // First value is the label
        getline(ss, value, ',');
        y.push_back(stoi(value));

        // Remaining values are the features
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

int main() {
    int n_features = 3072; // Flattened 32x32x3 images
    int n_classes = 10;    // CIFAR-10 has 10 classes

    vector<vector<double>> X_train, X_test;
    vector<int> y_train, y_test;

    cout << "Loading CIFAR-10 CSV files...\n";
    load_csv("train.csv", X_train, y_train);
    load_csv("test.csv", X_test, y_test);

    cout << "Training SVM...\n";
    SVM svm(1e-4, 0.1, 5, n_classes);
    double best_accuracy = svm.fit(X_train, y_train, n_features, X_test, y_test);

    cout << "Best Accuracy: " << fixed << setprecision(2) << best_accuracy << "%\n";

    return 0;
}
