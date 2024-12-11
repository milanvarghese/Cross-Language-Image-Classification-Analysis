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

    auto load_end = std::chrono::high_resolution_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count();
    cout << "Loading " << file_path << " took " << load_duration << " ms\n";
}

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

int main() {
    vector<vector<double>> X_train, X_test;
    vector<int> y_train, y_test;

    load_csv("train.csv", X_train, y_train);
    load_csv("test.csv", X_test, y_test);

    MatrixXd X_train_mat(X_train.size(), X_train[0].size());
    MatrixXd X_test_mat(X_test.size(), X_test[0].size());

    for (size_t i = 0; i < X_train.size(); ++i) {
        X_train_mat.row(i) = Map<RowVectorXd>(X_train[i].data(), X_train[i].size());
    }

    for (size_t i = 0; i < X_test.size(); ++i) {
        X_test_mat.row(i) = Map<RowVectorXd>(X_test[i].data(), X_test[i].size());
    }

    VectorXi y_train_vec = Map<VectorXi>(y_train.data(), y_train.size());
    VectorXi y_test_vec = Map<VectorXi>(y_test.data(), y_test.size());

    LogisticRegression lr(0.01, 1000);
    lr.fit(X_train_mat, y_train_vec.cast<double>(), X_test_mat, y_test_vec.cast<double>());

    VectorXi train_pred = lr.predict(X_train_mat);
    VectorXi test_pred = lr.predict(X_test_mat);

    MatrixXi train_cm = Evaluation::confusion_matrix(y_train_vec, train_pred, y_train_vec.maxCoeff() + 1);
    MatrixXi test_cm = Evaluation::confusion_matrix(y_test_vec, test_pred, y_test_vec.maxCoeff() + 1);

    double train_accuracy = Evaluation::accuracy(y_train_vec, train_pred);
    double test_accuracy = Evaluation::accuracy(y_test_vec, test_pred);

    double train_precision = Evaluation::precision(train_cm);
    double test_precision = Evaluation::precision(test_cm);

    double train_recall = Evaluation::recall(train_cm);
    double test_recall = Evaluation::recall(test_cm);

    double train_f1 = Evaluation::f1_score(train_precision, train_recall);
    double test_f1 = Evaluation::f1_score(test_precision, test_recall);

    cout << "Training Metrics:\n";
    cout << "Accuracy: " << train_accuracy << " Precision: " << train_precision << " Recall: " << train_recall << " F1 Score: " << train_f1 << "\n";

    cout << "Validation Metrics:\n";
    cout << "Accuracy: " << test_accuracy << " Precision: " << test_precision << " Recall: " << test_recall << " F1 Score: " << test_f1 << "\n";

    return 0;
}
