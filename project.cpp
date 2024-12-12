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

using namespace std;
using namespace Eigen;

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

        if (!getline(ss, value, ',')) {
            cerr << "Error: Invalid line format in file " << file_path << endl;
            continue;
        }

        try {
            y.push_back(stoi(value));
        } catch (const invalid_argument& e) {
            cerr << "Error: Invalid label value in file " << file_path << endl;
            continue;
        }

        while (getline(ss, value, ',')) {
            try {
                row.push_back(stod(value));
            } catch (const invalid_argument& e) {
                cerr << "Error: Invalid feature value in file " << file_path << endl;
                row.clear();
                break;
            }
        }

        if (!row.empty()) {
            X.push_back(row);
        }
    }

    file.close();
    cout << "Loaded " << X.size() << " samples from " << file_path << "\n";
}

// Evaluation class for metrics
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

        weights = MatrixXd::NullaryExpr(num_features, num_classes, [&]() { return d(gen); });

        for (int epoch = 0; epoch < epochs; ++epoch) {
            MatrixXd y_pred_train = softmax(X_train * weights);
            MatrixXd error = y_pred_train - y_train;
            MatrixXd gradient = (X_train.transpose() * error) / y_train.rows();
            weights -= learning_rate * gradient;

            cout << "Epoch " << epoch + 1 << "/" << epochs << endl;
        }
    }
};

// SVM Class Definition
class SVM {
private:
    MatrixXd weights;
    double learning_rate;           
    double lambda_param;            
    int n_epochs;                   
    int n_classes;                  

public: 
    // Class constructor
    SVM(double lr, double lambda, int epochs, int classes)
        : learning_rate(lr), lambda_param(lambda), n_epochs(epochs), n_classes(classes) {}

    double fit(const MatrixXd& X, const VectorXi& y, const MatrixXd& X_test, const VectorXi& y_test) {
        int n_features = X.cols();
        weights = MatrixXd::Zero(n_classes, n_features);

        double best_accuracy = 0.0;

        // Start training
        auto train_start = chrono::high_resolution_clock::now();

        for (int epoch = 0; epoch < n_epochs; ++epoch) {
            for (int c = 0; c < n_classes; ++c) {
                for (int i = 0; i < X.rows(); ++i) {
                    int y_binary = (y(i) == c) ? 1 : -1; // Convert labels to binary
                    double linear_output = X.row(i).dot(weights.row(c)); // Linear output

                    // Update weights
                    if (y_binary * linear_output < 1) { // Check margin
                        weights.row(c) += learning_rate * (y_binary * X.row(i) - 2 * lambda_param * weights.row(c));
                    } else {
                        weights.row(c) -= learning_rate * (2 * lambda_param * weights.row(c));
                    }
                }
            }

            // Evaluate on test set
            VectorXi predictions = predict(X_test);
            int correct = (predictions.array() == y_test.array()).count();
            double accuracy = static_cast<double>(correct) / X_test.rows() * 100.0;
            best_accuracy = max(best_accuracy, accuracy);

            cout << "Epoch " << epoch + 1 << "/" << n_epochs << " - Accuracy: " << fixed << setprecision(2) << accuracy << "%\n";
        }

        auto train_end = chrono::high_resolution_clock::now();
        auto train_duration = chrono::duration_cast<chrono::milliseconds>(train_end - train_start).count();
        cout << "Training took " << train_duration << " ms\n";

        return best_accuracy;
    }

    int predict_sample(const RowVectorXd &sample) const {
        VectorXd scores = weights * sample.transpose();
        int predicted_class;
        scores.maxCoeff(&predicted_class);
        return predicted_class;
    }

    // Batch Predict Function
    VectorXi predict(const MatrixXd &X) const {
        VectorXi predictions(X.rows());
        for (int i = 0; i < X.rows(); ++i) {
            predictions(i) = predict_sample(X.row(i));
        }
        return predictions;
    }
};

int main() {
    // Load CIFAR-10 data
    vector<vector<double>> X_train_vec, X_test_vec;
    vector<int> y_train_vec, y_test_vec;

    load_csv("train.csv", X_train_vec, y_train_vec);
    load_csv("test.csv", X_test_vec, y_test_vec);

    // Convert data to Eigen matrices
    MatrixXd X_train(X_train_vec.size(), X_train_vec[0].size() + 1); // Add bias
    MatrixXd X_test(X_test_vec.size(), X_test_vec[0].size() + 1);

    // Add bias column to X_train and X_test
    for (size_t i = 0; i < X_train_vec.size(); ++i) {
        X_train(i, 0) = 1.0; // Bias term
        X_train.block(i, 1, 1, X_train_vec[i].size()) = Map<RowVectorXd>(X_train_vec[i].data(), X_train_vec[i].size());
    }

    for (size_t i = 0; i < X_test_vec.size(); ++i) {
        X_test(i, 0) = 1.0; // Bias term
        X_test.block(i, 1, 1, X_test_vec[i].size()) = Map<RowVectorXd>(X_test_vec[i].data(), X_test_vec[i].size());
    }

    VectorXi y_train = Map<VectorXi>(y_train_vec.data(), y_train_vec.size());
    VectorXi y_test = Map<VectorXi>(y_test_vec.data(), y_test_vec.size());

    // Train Logistic Regression
    LogisticRegression lr(0.01, 1000);
    lr.fit(X_train, y_train.cast<double>(), X_test, y_test.cast<double>());

    VectorXi train_pred_lr = lr.predict(X_train);
    VectorXi test_pred_lr = lr.predict(X_test);

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
    cout << "Accuracy: " << train_accuracy_lr << " Precision: " << train_precision_lr
         << " Recall: " << train_recall_lr << " F1 Score: " << train_f1_lr << "\n";
    cout << "Validation Metrics:\n";
    cout << "Accuracy: " << test_accuracy_lr << " Precision: " << test_precision_lr
         << " Recall: " << test_recall_lr << " F1 Score: " << test_f1_lr << "\n";

    // Train SVM
    SVM svm(1e-6, 0.01, 100, y_train.maxCoeff() + 1);
    svm.fit(X_train, y_train, X_test, y_test);

    VectorXi train_pred_svm = svm.predict(X_train);
    VectorXi test_pred_svm = svm.predict(X_test);

    MatrixXi train_cm_svm = Evaluation::confusion_matrix(y_train, train_pred_svm, y_train.maxCoeff() + 1);
    MatrixXi test_cm_svm = Evaluation::confusion_matrix(y_test, test_pred_svm, y_test.maxCoeff() + 1);

    double train_accuracy_svm = Evaluation::accuracy(y_train, train_pred_svm);
    double test_accuracy_svm = Evaluation::accuracy(y_test, test_pred_svm);

    double train_precision_svm = Evaluation::precision(train_cm_svm);
    double test_precision_svm = Evaluation::precision(test_cm_svm);

    double train_recall_svm = Evaluation::recall(train_cm_svm);
    double test_recall_svm = Evaluation::recall(test_cm_svm);

    double train_f1_svm = Evaluation::f1_score(train_precision_svm, train_recall_svm);
    double test_f1_svm = Evaluation::f1_score(test_precision_svm, test_recall_svm);

    cout << "SVM Evaluation Results:\n";
    cout << "Training Metrics:\n";
    cout << "Accuracy: " << train_accuracy_svm << " Precision: " << train_precision_svm
         << " Recall: " << train_recall_svm << " F1 Score: " << train_f1_svm << "\n";
    cout << "Validation Metrics:\n";
    cout << "Accuracy: " << test_accuracy_svm << " Precision: " << test_precision_svm
         << " Recall: " << test_recall_svm << " F1 Score: " << test_f1_svm << "\n";

    cout << "Evaluation completed.\n";

    return 0;
}