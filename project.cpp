#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <Eigen/Dense>
#include <iomanip>
#include <numeric>

using namespace std;
using namespace Eigen;

void normalize_data(vector<vector<double>>& X) {
    for (size_t i = 0; i < X.size(); ++i) {
        for (size_t j = 0; j < X[i].size() - 1; ++j) { // Exclude last feature (bias)
            X[i][j] /= 255.0;
        }
        // Ensure the bias term remains 1.0
        X[i].back() = 1.0;
    }
}

MatrixXd vectorXiToMatrixXd(const VectorXi& vec, int num_classes) {
    MatrixXd mat = MatrixXd::Zero(vec.size(), num_classes);
    for (int i = 0; i < vec.size(); ++i) {
        if (vec(i) >= 0 && vec(i) < num_classes) {
            mat(i, vec(i)) = 1.0;
        } else {
            cerr << "Error: Invalid class index.\n";
            exit(1);
        }
    }
    return mat;
}

MatrixXd vectorToMatrix(const vector<vector<double>>& vec) {
    int rows = vec.size();
    int cols = vec[0].size();
    MatrixXd mat(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            mat(i, j) = vec[i][j];
    return mat;
}

VectorXi vectorToVectorXi(const vector<int>& vec) {
    VectorXi v(vec.size());
    for (size_t i = 0; i < vec.size(); ++i)
        v(i) = vec[i];
    return v;
}

// Function to load CIFAR-10 CSV files with one-hot encoded labels
void load_csv(const string &file_path, vector<vector<double>> &X, vector<int> &y) {
    ifstream file(file_path);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << file_path << endl;
        exit(1);
    }

    string line;
    int line_number = 0;
    while (getline(file, line)) {
        line_number++;
        stringstream ss(line);
        string value;
        vector<double> row;
        vector<double> label_one_hot;

        // Read the label (first value in the row)
        if (!getline(ss, value, ',')) {
            cerr << "Error: Missing label in line " << line_number << endl;
            exit(1);
        }
        int label = stoi(value);
        y.push_back(label);

        // Read the remaining values as features
        while (getline(ss, value, ',')) {
            row.push_back(stod(value));
        }

        // Append bias term to the feature vector
        row.push_back(1.0);
        X.push_back(row);
    }

    file.close();
    cout << "Loaded " << X.size() << " samples from " << file_path << "\n";
}

// Evaluation class for logistic regression metrics
class Evaluation {
public:
    static MatrixXi confusion_matrix(const VectorXi& y_true, const VectorXi& y_pred, int num_classes) {
        MatrixXi cm = MatrixXi::Zero(num_classes, num_classes);
        for (int i = 0; i < y_true.size(); ++i) {
            cm(y_true[i], y_pred[i])++;
        }
        return cm;
    }

    static double accuracy(const VectorXi& y_true, const VectorXi& y_pred) {
        int correct = (y_true.array() == y_pred.array()).count();
        return static_cast<double>(correct) / y_true.size();
    }

    static double precision(const MatrixXi& cm) {
        double precision_sum = 0;
        for (int i = 0; i < cm.rows(); ++i) {
            int tp = cm(i, i);
            int fp = cm.col(i).sum() - tp;
            precision_sum += (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0;
        }
        return precision_sum / cm.rows();
    }

    static double recall(const MatrixXi& cm) {
        double recall_sum = 0;
        for (int i = 0; i < cm.rows(); ++i) {
            int tp = cm(i, i);
            int fn = cm.row(i).sum() - tp;
            recall_sum += (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0;
        }
        return recall_sum / cm.rows();
    }

    static double f1_score(double precision, double recall) {
        return (precision + recall > 0) ? 2 * (precision * recall) / (precision + recall) : 0;
    }
};

class LogisticRegression {
private:
    double learning_rate;
    int epochs;
    MatrixXd weights;

public:
    vector<double> train_losses;
    vector<double> test_losses;

    LogisticRegression(double lr, int e) : learning_rate(lr), epochs(e) {}

    MatrixXd softmax(const MatrixXd& z) {
        MatrixXd exp_z = (z.rowwise() - z.colwise().maxCoeff()).array().exp();
        MatrixXd sum_exp = exp_z.rowwise().sum().replicate(1, z.cols());
        return exp_z.array() / sum_exp.array();
    }

    VectorXi predict(const MatrixXd& X) {
        MatrixXd y_pred = softmax(X * weights);
        VectorXi predictions(X.rows());
        for (int i = 0; i < y_pred.rows(); ++i) {
            y_pred.row(i).maxCoeff(&predictions(i));
        }
        return predictions;
    }

    void fit(const MatrixXd& X_train, const MatrixXd& y_train, const MatrixXd& X_test = MatrixXd(), const MatrixXd& y_test = MatrixXd()) {
        int num_features = X_train.cols();
        int num_classes = y_train.cols();

        random_device rd;
        mt19937 gen(rd());
        normal_distribution<> d(0, 1);

        weights = MatrixXd::NullaryExpr(num_features, num_classes, [&]() { return d(gen) * 0.01; });

        // Start timer for training
        auto train_start = std::chrono::high_resolution_clock::now();

        for (int epoch = 0; epoch < epochs; ++epoch) {
            MatrixXd y_pred_train = softmax(X_train * weights);
            MatrixXd error = y_pred_train - y_train;
            MatrixXd gradient = (X_train.transpose() * error) / y_train.rows();
            weights -= learning_rate * gradient;

            // Calculate and print accuracy for the current epoch
            VectorXi train_predictions = predict(X_train);
            VectorXi true_labels = VectorXi::Zero(y_train.rows());
            for (int i = 0; i < y_train.rows(); ++i) {
                y_train.row(i).maxCoeff(&true_labels(i));
            }

            double accuracy = Evaluation::accuracy(true_labels, train_predictions);
            cout << "Epoch " << epoch + 1 << "/" << epochs << " - Training Accuracy: " << accuracy * 100 << "%\n";
        }

        auto train_end = std::chrono::high_resolution_clock::now();
        auto train_duration = std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start).count();
        cout << "Training took " << train_duration << " ms\n";
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

            // Return class with maximum score
            if (linear_output > max_score) {
                max_score = linear_output;
                best_class = c;
            }
        }

        return best_class;
    }
};

// Evaluation class for svm metrics (NO EIGEN)
class EvaluationSVM {
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

int main() {
    int n_features = 3073; // Flattened 32x32x3 images
    int n_classes = 10;    // CIFAR-10 has 10 classes

    // Load CIFAR-10 data
    vector<vector<double>> X_train_vec, X_test_vec;
    vector<int> y_train_vec, y_test_vec;

    load_csv("train.csv", X_train_vec, y_train_vec);
    load_csv("test.csv", X_test_vec, y_test_vec);

    // Normalize data
    normalize_data(X_train_vec);
    normalize_data(X_test_vec);

    // Convert data to Eigen types
    MatrixXd X_train = vectorToMatrix(X_train_vec);
    MatrixXd X_test = vectorToMatrix(X_test_vec);
    VectorXi y_train = vectorToVectorXi(y_train_vec);
    VectorXi y_test = vectorToVectorXi(y_test_vec);

    // ---- Logistic Regression ----
    LogisticRegression lr(0.01, 500);

    // Convert labels to MatrixXd (one-hot encoding)
    MatrixXd y_train_onehot = vectorXiToMatrixXd(y_train, y_train.maxCoeff() + 1);
    MatrixXd y_test_onehot = vectorXiToMatrixXd(y_test, y_test.maxCoeff() + 1);

    lr.fit(X_train, y_train_onehot, X_test, y_test_onehot);

    VectorXi train_pred_lr = lr.predict(X_train);
    VectorXi test_pred_lr = lr.predict(X_test);

    // Evaluation for Logistic Regression
    MatrixXi train_cm_lr = Evaluation::confusion_matrix(y_train, train_pred_lr, y_train.maxCoeff() + 1);
    MatrixXi test_cm_lr = Evaluation::confusion_matrix(y_test, test_pred_lr, y_test.maxCoeff() + 1);

    double train_accuracy_lr = Evaluation::accuracy(y_train, train_pred_lr);
    double test_accuracy_lr = Evaluation::accuracy(y_test, test_pred_lr);

    double train_precision_lr = Evaluation::precision(train_cm_lr);
    double test_precision_lr = Evaluation::precision(test_cm_lr);

    double train_recall_lr = Evaluation::recall(train_cm_lr);
    double test_recall_lr = Evaluation::recall(test_cm_lr);

    double train_f1_lr = Evaluation::f1_score(train_precision_lr, train_recall_lr);
    double test_f1_lr = Evaluation::f1_score(test_precision_lr, test_recall_lr);

    cout << "Logistic Regression Evaluation Results:\n";
    cout << "Training Metrics:\n";
    cout << "Confusion Matrix:\n" << train_cm_lr << "\n";
    cout << "Accuracy: " << train_accuracy_lr << "\n";
    cout << "Precision: " << train_precision_lr << "\n";
    cout << "Recall: " << train_recall_lr << "\n";
    cout << "F1 Score: " << train_f1_lr << "\n";
    cout << "Validation Metrics:\n";
    cout << "Confusion Matrix:\n" << test_cm_lr << "\n";
    cout << "Accuracy: " << test_accuracy_lr << "\n";
    cout << "Precision: " << test_precision_lr << "\n";
    cout << "Recall: " << test_recall_lr << "\n";
    cout << "F1 Score: " << test_f1_lr << "\n";
    
    // ---- SVM ----
    cout << "Training SVM...\n";
    SVM svm(1e-5, 0.1, 20, n_classes);
    double best_accuracy = svm.fit(X_train_vec, y_train_vec, n_features, X_test_vec, y_test_vec);

    cout << "Best Accuracy: " << fixed << setprecision(2) << best_accuracy << "%\n";

    // Generate predictions on the training set
    vector<int> y_pred_train_svm;
    y_pred_train_svm.reserve(X_train_vec.size());
    for (const auto& sample : X_train_vec) {
        y_pred_train_svm.push_back(svm.predict(sample));
    }

    // Generate predictions on the test set
    vector<int> y_pred_test_svm;
    y_pred_test_svm.reserve(X_test_vec.size());
    for (const auto& sample : X_test_vec) {
        y_pred_test_svm.push_back(svm.predict(sample));
    }

    // Compute confusion matrix for the training set
    vector<vector<int>> cm_svm_train = EvaluationSVM::confusion_matrix(y_train_vec, y_pred_train_svm, n_classes);

    // Compute confusion matrix for the test set
    vector<vector<int>> cm_svm_test = EvaluationSVM::confusion_matrix(y_test_vec, y_pred_test_svm, n_classes);

    // Compute training metrics
    double accuracy_svm_train = EvaluationSVM::accuracy(y_train_vec, y_pred_train_svm) * 100.0;
    double precision_svm_train = EvaluationSVM::precision(cm_svm_train) * 100.0;
    double recall_svm_train = EvaluationSVM::recall(cm_svm_train) * 100.0;
    double f1_svm_train = EvaluationSVM::f1_score(precision_svm_train, recall_svm_train);

    // Compute testing metrics
    double accuracy_svm_test = EvaluationSVM::accuracy(y_test_vec, y_pred_test_svm) * 100.0;
    double precision_svm_test = EvaluationSVM::precision(cm_svm_test) * 100.0;
    double recall_svm_test = EvaluationSVM::recall(cm_svm_test) * 100.0;
    double f1_svm_test = EvaluationSVM::f1_score(precision_svm_test, recall_svm_test);


   // Display training metrics
    cout << fixed << setprecision(2);
    cout << "Training Evaluation Metrics:\n";
    cout << "Training Confusion Matrix:\n";
    for (const auto& row : cm_svm_train) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << "\n";
    }
    cout << "Accuracy: " << accuracy_svm_train << "%\n";
    cout << "Precision: " << precision_svm_train << "%\n";
    cout << "Recall: " << recall_svm_train << "%\n";
    cout << "F1 Score: " << f1_svm_train << "%\n";

    // Display testing metrics
    cout << "\nTesting Evaluation Metrics:\n";
    cout << "Testing Confusion Matrix:\n";
    for (const auto& row : cm_svm_test) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << "\n";
    }
    cout << "Accuracy: " << accuracy_svm_test << "%\n";
    cout << "Precision: " << precision_svm_test << "%\n";
    cout << "Recall: " << recall_svm_test << "%\n";
    cout << "F1 Score: " << f1_svm_test << "%\n";
    cout << "Evaluation completed.\n";

    return 0;
}
