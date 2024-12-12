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

void normalize_data(vector<vector<double>>& X) {
    for (auto& row : X)
        for (auto& val : row)
            val /= 255.0;
}


MatrixXd vectorXiToMatrixXd(const VectorXi& vec, int num_classes) {
    MatrixXd mat = MatrixXd::Zero(vec.size(), num_classes);
    for (int i = 0; i < vec.size(); ++i) {
        mat(i, vec(i)) = 1.0;
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
    vector<vector<double>> weights;  // Class-specific weights
    vector<double> biases;           // Class-specific biases
    double learning_rate;            // Learning rate
    double lambda_param;             // Regularization parameter
    int n_epochs;                    // Number of epochs
    int n_classes;                   // Number of classes

public:
    // Constructor
    SVM(double learning_rate, double lambda_param, int n_epochs, int n_classes)
        : learning_rate(learning_rate), lambda_param(lambda_param), n_epochs(n_epochs), n_classes(n_classes) {
        weights.resize(n_classes, vector<double>(0));
        biases.resize(n_classes, 0.0);
    }

    // Public Getters
    int getNumClasses() const { return n_classes; }
    const vector<vector<double>>& getWeights() const { return weights; }
    const vector<double>& getBiases() const { return biases; }

    // Training the SVM
    double fit(const vector<vector<double>>& X, const vector<int>& y, int n_features, const vector<vector<double>>& X_test, const vector<int>& y_test) {
        for (int i = 0; i < n_classes; ++i) {
            weights[i].resize(n_features, 0.0);  // Initialize weights to zero
        }

        double best_accuracy = 0.0;

        for (int epoch = 0; epoch < n_epochs; ++epoch) {
            for (size_t i = 0; i < X.size(); ++i) {
                int y_true = y[i];

                for (int c = 0; c < n_classes; ++c) {
                    double linear_output = inner_product(weights[c].begin(), weights[c].end(), X[i].begin(), -biases[c]);
                    int y_binary = (y_true == c) ? 1 : -1;

                    // Hinge Loss Update
                    if (y_binary * linear_output < 1) {
                        for (int j = 0; j < n_features; ++j) {
                            weights[c][j] += learning_rate * (y_binary * X[i][j] - 2 * lambda_param * weights[c][j]);
                        }
                        biases[c] += learning_rate * y_binary;
                    } else {
                        for (int j = 0; j < n_features; ++j) {
                            weights[c][j] -= learning_rate * 2 * lambda_param * weights[c][j];
                        }
                    }
                }
            }

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

        return best_accuracy;
    }

    // Predict Function
    int predict(const vector<double>& x) const {
        double max_score = -INFINITY;
        int best_class = -1;

        for (int c = 0; c < n_classes; ++c) {
            double linear_output = inner_product(weights[c].begin(), weights[c].end(), x.begin(), -biases[c]);
            if (linear_output > max_score) {
                max_score = linear_output;
                best_class = c;
            }
        }

        return best_class;
    }
};




// class EnsembleModel {
// private:
//     vector<pair<SVM*, double>> models;

// public:
//     void addModel(SVM* model, double accuracy) {
//         models.push_back(make_pair(model, accuracy));
//     }

//     int predict(const vector<double> &x) const {
//         vector<double> class_scores(models[0].first->n_classes, 0.0);
//         for (const auto& model_pair : models) {
//             const SVM* model = model_pair.first;
//             double weight = model_pair.second;
//             vector<double> probs(model->n_classes);
//             for (int c = 0; c < model->n_classes; ++c) {
//                 probs[c] = inner_product(model->weights[c].begin(), model->weights[c].end(), x.begin(), -model->biases[c]);
//             }
//             for (int c = 0; c < model->n_classes; ++c) {
//                 class_scores[c] += weight * probs[c];
//             }
//         }
//         return max_element(class_scores.begin(), class_scores.end()) - class_scores.begin();
//     }
// };

class EnsembleModel {
private:
    vector<pair<SVM*, double>> models;  // List of models and their accuracies

public:
    void addModel(SVM* model, double accuracy) {
        models.push_back(make_pair(model, accuracy));
    }

    int predict(const vector<double>& x) const {
        if (models.empty()) {
            cerr << "Error: No models in the ensemble.\n";
            exit(1);
        }

        vector<double> class_scores(models[0].first->getNumClasses(), 0.0);

        for (const auto& model_pair : models) {
            const SVM* model = model_pair.first;
            double weight = model_pair.second;
            const vector<vector<double>>& model_weights = model->getWeights();
            const vector<double>& model_biases = model->getBiases();

            for (int c = 0; c < model->getNumClasses(); ++c) {
                double linear_output = inner_product(model_weights[c].begin(), model_weights[c].end(), x.begin(), -model_biases[c]);
                class_scores[c] += weight * linear_output;
            }
        }
        return max_element(class_scores.begin(), class_scores.end()) - class_scores.begin();
    }
};


int main() {
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

    /*
    // ---- Logistic Regression ----
    LogisticRegression lr(0.1, 10);

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
    */

    // ---- SVM ----
    SVM svm(1.0, 0.1, 15, y_train.maxCoeff() + 1);
    svm.fit(X_train_vec, y_train_vec, X_train_vec[0].size(), X_test_vec, y_test_vec);

    vector<int> train_pred_svm(y_train_vec.size());
    vector<int> test_pred_svm(y_test_vec.size());

    for (size_t i = 0; i < y_train_vec.size(); ++i) {
        train_pred_svm[i] = svm.predict(X_train_vec[i]);
    }
    for (size_t i = 0; i < y_test_vec.size(); ++i) {
        test_pred_svm[i] = svm.predict(X_test_vec[i]);
    }

    MatrixXi train_cm_svm = Evaluation::confusion_matrix(vectorToVectorXi(y_train_vec), vectorToVectorXi(train_pred_svm), y_train.maxCoeff() + 1);
    MatrixXi test_cm_svm = Evaluation::confusion_matrix(vectorToVectorXi(y_test_vec), vectorToVectorXi(test_pred_svm), y_test.maxCoeff() + 1);

    double train_accuracy_svm = Evaluation::accuracy(vectorToVectorXi(y_train_vec), vectorToVectorXi(train_pred_svm));
    double test_accuracy_svm = Evaluation::accuracy(vectorToVectorXi(y_test_vec), vectorToVectorXi(test_pred_svm));

    double train_precision_svm = Evaluation::precision(train_cm_svm);
    double test_precision_svm = Evaluation::precision(test_cm_svm);

    double train_recall_svm = Evaluation::recall(train_cm_svm);
    double test_recall_svm = Evaluation::recall(test_cm_svm);

    double train_f1_svm = Evaluation::f1_score(train_precision_svm, train_recall_svm);
    double test_f1_svm = Evaluation::f1_score(test_precision_svm, test_recall_svm);

    cout << "SVM Evaluation Results:\n";
    cout << "Training Metrics:\n";
    cout << "Confusion Matrix:\n" << train_cm_svm << "\n";
    cout << "Accuracy: " << train_accuracy_svm << "\n";
    cout << "Precision: " << train_precision_svm << "\n";
    cout << "Recall: " << train_recall_svm << "\n";
    cout << "F1 Score: " << train_f1_svm << "\n";
    cout << "Validation Metrics:\n";
    cout << "Confusion Matrix:\n" << test_cm_svm << "\n";
    cout << "Accuracy: " << test_accuracy_svm << "\n";
    cout << "Precision: " << test_precision_svm << "\n";
    cout << "Recall: " << test_recall_svm << "\n";
    cout << "F1 Score: " << test_f1_svm << "\n";

    /*
    // ---- Ensemble Model ----
    EnsembleModel ensemble;
    ensemble.addModel(&svm, test_accuracy_svm);

    vector<int> test_pred_ensemble(y_test_vec.size());
    for (size_t i = 0; i < y_test_vec.size(); ++i) {
        test_pred_ensemble[i] = ensemble.predict(X_test_vec[i]);
    }

    MatrixXi test_cm_ensemble = Evaluation::confusion_matrix(vectorToVectorXi(y_test_vec), vectorToVectorXi(test_pred_ensemble), y_test.maxCoeff() + 1);
    double test_accuracy_ensemble = Evaluation::accuracy(vectorToVectorXi(y_test_vec), vectorToVectorXi(test_pred_ensemble));
    double test_precision_ensemble = Evaluation::precision(test_cm_ensemble);
    double test_recall_ensemble = Evaluation::recall(test_cm_ensemble);
    double test_f1_ensemble = Evaluation::f1_score(test_precision_ensemble, test_recall_ensemble);

    cout << "\nEnsemble Model Evaluation Results:\n";
    cout << "Validation Metrics:\n";
    cout << "Confusion Matrix:\n" << test_cm_ensemble << "\n";
    cout << "Accuracy: " << test_accuracy_ensemble << "\n";
    cout << "Precision: " << test_precision_ensemble << "\n";
    cout << "Recall: " << test_recall_ensemble << "\n";
    cout << "F1 Score: " << test_f1_ensemble << "\n";
    */

    cout << "Evaluation completed.\n";

    return 0;
}
