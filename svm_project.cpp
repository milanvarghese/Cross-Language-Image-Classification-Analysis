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

// Function to load CIFAR-10 CSV files
void load_csv(const string &file_path, vector<vector<double>> &X, vector<int> &y) {
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

        row.push_back(1.0); // Bias feature
        X.push_back(row);
    }

    file.close();
    cout << "Loaded " << X.size() << " samples from " << file_path << "\n";
}

// Evaluation class for metrics
class Evaluation {
public:
    // Compute the confusion matrix
    static vector<vector<int>> confusion_matrix(const vector<int>& y_true, const vector<int>& y_pred, int num_classes) {
        // Initialize confusion matrix with zeros
        vector<vector<int>> cm(num_classes, vector<int>(num_classes, 0));
        for (size_t i = 0; i < y_true.size(); ++i) {
            int true_label = y_true[i];
            int pred_label = y_pred[i];
            if (true_label >= 0 && true_label < num_classes && pred_label >= 0 && pred_label < num_classes) {
                cm[true_label][pred_label]++;
            }
        }
        return cm;
    }

    // Compute accuracy
    static double accuracy(const vector<int>& y_true, const vector<int>& y_pred) {
        int correct = 0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            if (y_true[i] == y_pred[i]) {
                ++correct;
            }
        }
        return static_cast<double>(correct) / y_true.size();
    }

    // Compute precision
    static double precision(const vector<vector<int>>& cm) {
        int num_classes = cm.size();
        double precision_sum = 0.0;
        for (int i = 0; i < num_classes; ++i) {
            int tp = cm[i][i];
            int fp = 0;
            for (int j = 0; j < num_classes; ++j) {
                if (j != i) {
                    fp += cm[j][i];
                }
            }
            if (tp + fp > 0) {
                precision_sum += static_cast<double>(tp) / (tp + fp);
            }
        }
        return precision_sum / num_classes;
    }

    // Compute recall
    static double recall(const vector<vector<int>>& cm) {
        int num_classes = cm.size();
        double recall_sum = 0.0;
        for (int i = 0; i < num_classes; ++i) {
            int tp = cm[i][i];
            int fn = 0;
            for (int j = 0; j < num_classes; ++j) {
                if (j != i) {
                    fn += cm[i][j];
                }
            }
            if (tp + fn > 0) {
                recall_sum += static_cast<double>(tp) / (tp + fn);
            }
        }
        return recall_sum / num_classes;
    }

    // Compute F1-score
    static double f1_score(double precision, double recall) {
        if (precision + recall == 0.0) return 0.0;
        return 2 * (precision * recall) / (precision + recall);
    }
};

// Normalize features
void normalize_features(vector<vector<double>>& X) {
    for (size_t i = 0; i < X.size(); ++i) {
        for (size_t j = 0; j < X[i].size(); ++j) {
            X[i][j] /= 255.0;
        }
    }
}

// SVM Class Definition
class SVM {
private:
    vector<vector<double>> weights; // Weights for each class
    double learning_rate;           // Learning rate
    double lambda_param;            // Regularization parameter
    int n_epochs;                   // Number of epochs
    int n_classes;                  // Number of classes

public: 
    // Class constructor
    SVM(double learning_rate, double lambda_param, int n_epochs, int n_classes)
        : learning_rate(learning_rate), lambda_param(lambda_param), n_epochs(n_epochs), n_classes(n_classes) {
        weights.resize(n_classes, vector<double>(0)); // Resize weights
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

                for (size_t i = 0; i < X.size(); ++i) {
                    int y_binary = (y[i] == c) ? 1 : -1; // Convert labels to binary
                    double linear_output = inner_product(w.begin(), w.end(), X[i].begin(), 0.0);

                    // Update weights
                    if (y_binary * linear_output < 1) { // Check margin
                        for (int j = 0; j < n_features; ++j) {
                            w[j] -= learning_rate * (2 * lambda_param * w[j] - y_binary * X[i][j]);
                        }
                    } else {
                        for (int j = 0; j < n_features; ++j) {
                            w[j] -= learning_rate * (2 * lambda_param * w[j]);
                        }
                    }
                }
            }

            // Compute accuracy
            int correct = 0;
            vector<int> y_pred;
            y_pred.reserve(X_test.size());
            for (size_t i = 0; i < X_test.size(); ++i) {
                int pred = predict(X_test[i]);
                y_pred.push_back(pred);
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
            double linear_output = inner_product(weights[c].begin(), weights[c].end(), x.begin(), 0.0);

            // Return class with maximum score and return prediction
            if (linear_output > max_score) {
                max_score = linear_output;
                best_class = c;
            }
        }

        return best_class;
    }
};

int main() {
    int n_features = 3073; // Flattened 32x32x3 images
    int n_classes = 10;    // CIFAR-10 has 10 classes

    vector<vector<double>> X_train, X_test;
    vector<int> y_train, y_test;

    cout << "Loading CIFAR-10 CSV files...\n";
    load_csv("train.csv", X_train, y_train);
    load_csv("test.csv", X_test, y_test);

    cout << "Training SVM...\n";
    SVM svm(1e-5, 0.1, 100, n_classes);
    double best_accuracy = svm.fit(X_train, y_train, n_features, X_test, y_test);

    cout << "Best Accuracy: " << fixed << setprecision(2) << best_accuracy << "%\n";

    // Generate predictions on the test set
    vector<int> y_pred;
    y_pred.reserve(X_test.size());
    for (const auto& sample : X_test) {
        y_pred.push_back(svm.predict(sample));
    }

    // Compute confusion matrix
    vector<vector<int>> cm = Evaluation::confusion_matrix(y_test, y_pred, n_classes);

    // Compute other metrics
    double accuracy = Evaluation::accuracy(y_test, y_pred) * 100.0;
    double precision = Evaluation::precision(cm) * 100.0;
    double recall = Evaluation::recall(cm) * 100.0;
    double f1 = Evaluation::f1_score(precision, recall);

    // Display metrics
    cout << fixed << setprecision(2);
    cout << "Evaluation Metrics:\n";
    cout << "Accuracy: " << accuracy << "%\n";
    cout << "Precision: " << precision << "\n";
    cout << "Recall: " << recall << "\n";
    cout << "F1 Score: " << f1 << "\n";

    return 0;
}