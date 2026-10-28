#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

using namespace std;

double relu(double x) { return (x > 0) ? x : 0; }
double reluDerivative(double x) { return (x > 0) ? 1 : 0; }
double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double sigmoidDerivative(double x) { return sigmoid(x) * (1 - sigmoid(x)); }

double calculateMSE(const vector<double>& predictions, const vector<double>& targets) {
    if (predictions.size() != targets.size() || predictions.empty())
        throw invalid_argument("Predictions and targets must be non-empty and of the same size.");
    double sum = 0;
    for (size_t i = 0; i < predictions.size(); i++) sum += pow(predictions[i] - targets[i], 2);
    return sum / predictions.size();
}

class NeuralNetwork {
private:
    mutable vector<double> cacheHidden, cacheOutput;
    vector<double> cacheOutGradients, cacheHiddenGradients;

public:
    vector<vector<double>> weightsInputHidden, weightsHiddenOutput;
    vector<double> biasHidden, biasOutput;

    NeuralNetwork(int inputLayerSize, int hiddenLayerSize, int outputLayerSize) {
        if (inputLayerSize <= 0 || hiddenLayerSize <= 0 || outputLayerSize <= 0)
            throw invalid_argument("Layer sizes must be greater than zero.");
        weightsInputHidden.resize(inputLayerSize, vector<double>(hiddenLayerSize));
        weightsHiddenOutput.resize(hiddenLayerSize, vector<double>(outputLayerSize));
        biasHidden.resize(hiddenLayerSize, 0.1);
        biasOutput.resize(outputLayerSize, 0.1);
        cacheHidden.resize(hiddenLayerSize, 0.0);
        cacheOutput.resize(outputLayerSize, 0.0);
        cacheOutGradients.resize(outputLayerSize, 0.0);
        cacheHiddenGradients.resize(hiddenLayerSize, 0.0);
        initWeights();
    }

    void initWeights() {
        for (auto& row : weightsInputHidden)
            for (double& w : row) w = (double)rand() / RAND_MAX - 0.5;
        for (auto& row : weightsHiddenOutput)
            for (double& w : row) w = (double)rand() / RAND_MAX - 0.5;
    }

    const vector<double>& forward(const vector<double>& input, vector<double>& hiddenLayer) const {
        if (input.size() != weightsInputHidden.size())
            throw invalid_argument("Input size does not match input layer size.");
        hiddenLayer.assign(weightsInputHidden[0].size(), 0.0);
        for (size_t i = 0; i < weightsInputHidden.size(); ++i) {
            double current_input = input[i];
            for (size_t j = 0; j < weightsInputHidden[0].size(); ++j)
                hiddenLayer[j] += current_input * weightsInputHidden[i][j];
        }
        for (size_t j = 0; j < hiddenLayer.size(); ++j)
            hiddenLayer[j] = relu(hiddenLayer[j] + biasHidden[j]);
        cacheOutput.assign(weightsHiddenOutput[0].size(), 0.0);
        for (size_t j = 0; j < weightsHiddenOutput.size(); ++j) {
            double current_hidden = hiddenLayer[j];
            for (size_t k = 0; k < weightsHiddenOutput[0].size(); ++k)
                cacheOutput[k] += current_hidden * weightsHiddenOutput[j][k];
        }
        for (size_t j = 0; j < cacheOutput.size(); ++j)
            cacheOutput[j] = sigmoid(cacheOutput[j] + biasOutput[j]);
        return cacheOutput;
    }

    vector<double> predict(const vector<double>& input) const { return forward(input, cacheHidden); }

    void train(const vector<vector<double>>& inputs, const vector<vector<double>>& targets, double lr, int epochs) {
        if (inputs.empty() || targets.empty() || inputs.size() != targets.size())
            throw invalid_argument("Training inputs and targets cannot be empty and must match batch size.");
        if (lr <= 0 || epochs <= 0)
            throw invalid_argument("Learning rate and epochs must be greater than zero.");
        size_t expectedInputSize = weightsInputHidden.size();
        size_t expectedOutputSize = weightsHiddenOutput[0].size();
        for (size_t i = 0; i < inputs.size(); i++) {
            if (inputs[i].size() != expectedInputSize || targets[i].size() != expectedOutputSize)
                throw invalid_argument("Input/target dimensions do not match network configuration.");
        }

        for (int e = 0; e < epochs; e++) {
            double totalLoss = 0;
            for (size_t i = 0; i < inputs.size(); i++) {
                const vector<double>& outps = forward(inputs[i], cacheHidden);
                totalLoss += calculateMSE(outps, targets[i]);
                for (size_t j = 0; j < outps.size(); j++) {
                    double mseDerivative = 2.0 * (outps[j] - targets[i][j]) / outps.size();
                    double sigmoidPrime = outps[j] * (1.0 - outps[j]);
                    cacheOutGradients[j] = mseDerivative * sigmoidPrime;
                }
                for (size_t j = 0; j < cacheHidden.size(); j++) {
                    cacheHiddenGradients[j] = 0;
                    for (size_t k = 0; k < outps.size(); k++)
                        cacheHiddenGradients[j] += cacheOutGradients[k] * weightsHiddenOutput[j][k];
                    cacheHiddenGradients[j] *= reluDerivative(cacheHidden[j]);
                }
                for (size_t j = 0; j < weightsHiddenOutput.size(); j++)
                    for (size_t k = 0; k < weightsHiddenOutput[0].size(); k++)
                        weightsHiddenOutput[j][k] -= lr * cacheOutGradients[k] * cacheHidden[j];
                for (size_t j = 0; j < biasOutput.size(); j++)
                    biasOutput[j] -= lr * cacheOutGradients[j];
                for (size_t j = 0; j < weightsInputHidden.size(); j++)
                    for (size_t k = 0; k < weightsInputHidden[0].size(); k++)
                        weightsInputHidden[j][k] -= lr * cacheHiddenGradients[k] * inputs[i][j];
                for (size_t j = 0; j < biasHidden.size(); j++)
                    biasHidden[j] -= lr * cacheHiddenGradients[j];
            }
            if (e % 1000 == 0) cout << "Epoch: " << e << " Loss: " << totalLoss / inputs.size() << endl;
        }
    }
};

int main() {
    try {
        NeuralNetwork nn(2, 3, 1);
        vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        vector<vector<double>> targets = {{0}, {1}, {1}, {0}};
        nn.train(inputs, targets, 0.05, 10000000); //lol
        cout << "\n Training Done \n" << endl;
        cout << "Predictions After Training: " << endl;
        for (const auto& input : inputs) {
            vector<double> prediction = nn.predict(input);
            cout << "Input: [" << input[0] << " , " << input[1] << "] -> Prediction: " << prediction[0] << endl;
        }
    } catch (const exception& ex) {
        cerr << "Error: " << ex.what() << endl;
        return 1;
    }
    return 0;
}