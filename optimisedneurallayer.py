#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <vector>
#include <memory>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <cmath>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                                   std::to_string(__LINE__) + " - " + cudaGetErrorString(err)); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error(std::string("cuBLAS error at ") + __FILE__ + ":" + \
                                   std::to_string(__LINE__)); \
        } \
    } while(0)

enum class ActivationType { ReLU, Sigmoid, Tanh };

class NeuralLayer {
private:
    float *d_weights, *d_biases, *d_input, *d_output, *d_gradients;
    float *d_weight_grads, *d_bias_grads;
    // Adam optimizer states
    float *d_weight_m, *d_weight_v, *d_bias_m, *d_bias_v;
    
    int input_size, output_size, batch_size;
    size_t weights_size, biases_size, batch_input_size, batch_output_size;
    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    ActivationType activation;
    float learning_rate, beta1, beta2, epsilon;  // Adam parameters
    static const int THREADS_PER_BLOCK = 256;
    int timestep;  // For Adam optimization

    void allocateDeviceMemory() {
        weights_size = input_size * output_size * sizeof(float);
        biases_size = output_size * sizeof(float);
        batch_input_size = input_size * batch_size * sizeof(float);
        batch_output_size = output_size * batch_size * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_weights, weights_size));
        CUDA_CHECK(cudaMalloc(&d_biases, biases_size));
        CUDA_CHECK(cudaMalloc(&d_input, batch_input_size));
        CUDA_CHECK(cudaMalloc(&d_output, batch_output_size));
        CUDA_CHECK(cudaMalloc(&d_gradients, batch_output_size));
        CUDA_CHECK(cudaMalloc(&d_weight_grads, weights_size));
        CUDA_CHECK(cudaMalloc(&d_bias_grads, biases_size));
        CUDA_CHECK(cudaMalloc(&d_weight_m, weights_size));
        CUDA_CHECK(cudaMalloc(&d_weight_v, weights_size));
        CUDA_CHECK(cudaMalloc(&d_bias_m, biases_size));
        CUDA_CHECK(cudaMalloc(&d_bias_v, biases_size));
        
        CUDA_CHECK(cudaMemset(d_weight_m, 0, weights_size));
        CUDA_CHECK(cudaMemset(d_weight_v, 0, weights_size));
        CUDA_CHECK(cudaMemset(d_bias_m, 0, biases_size));
        CUDA_CHECK(cudaMemset(d_bias_v, 0, biases_size));
    }

    __global__ static void applyActivation(float* data, int size, ActivationType type) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            switch(type) {
                case ActivationType::ReLU: data[idx] = fmaxf(0.0f, data[idx]); break;
                case ActivationType::Sigmoid: data[idx] = 1.0f / (1.0f + expf(-data[idx])); break;
                case ActivationType::Tanh: data[idx] = tanhf(data[idx]); break;
            }
        }
    }

    __global__ static void computeGradient(float* gradient, float* output, int size, ActivationType type) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            float val = output[idx];
            switch(type) {
                case ActivationType::ReLU: gradient[idx] *= (val > 0.0f) ? 1.0f : 0.0f; break;
                case ActivationType::Sigmoid: gradient[idx] *= val * (1.0f - val); break;
                case ActivationType::Tanh: gradient[idx] *= (1.0f - val * val); break;
            }
        }
    }

    // Adam update kernel
    __global__ static void adamUpdate(float* param, float* grad, float* m, float* v, int size,
                                    float lr, float beta1, float beta2, float epsilon, int t) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            float g = grad[idx];
            float m_t = beta1 * m[idx] + (1.0f - beta1) * g;
            float v_t = beta2 * v[idx] + (1.0f - beta2) * g * g;
            m[idx] = m_t;
            v[idx] = v_t;
            
            float m_hat = m_t / (1.0f - powf(beta1, t));
            float v_hat = v_t / (1.0f - powf(beta2, t));
            param[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
        }
    }

public:
    NeuralLayer(int in_size, int out_size, int batch, ActivationType act,
                float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f,
                cudaStream_t s = 0)
        : input_size(in_size), output_size(out_size), batch_size(batch),
          activation(act), learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps),
          stream(s), d_weights(nullptr), timestep(0) {
        
        CUDA_CHECK(cublasCreate(&cublas_handle));
        if (stream) CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
        allocateDeviceMemory();
        initializeWeights();
    }

    ~NeuralLayer() {
        cudaFree(d_weights); cudaFree(d_biases); cudaFree(d_input);
        cudaFree(d_output); cudaFree(d_gradients); cudaFree(d_weight_grads);
        cudaFree(d_bias_grads); cudaFree(d_weight_m); cudaFree(d_weight_v);
        cudaFree(d_bias_m); cudaFree(d_bias_v);
        cublasDestroy(cublas_handle);
    }

    void initializeWeights() {
        curandGenerator_t rand_gen;
        CUDA_CHECK(curandCreateGenerator(&rand_gen, CURAND_RNG_PSEUDO_DEFAULT));
        CUDA_CHECK(curandSetPseudoRandomGeneratorSeed(rand_gen, time(NULL)));
        
        float limit = sqrtf(6.0f / (input_size + output_size));
        CUDA_CHECK(curandGenerateUniform(rand_gen, d_weights, input_size * output_size));
        float alpha = 2.0f * limit, beta = -limit;
        CUBLAS_CHECK(cublasSscal(cublas_handle, input_size * output_size, &alpha, d_weights, 1));
        CUBLAS_CHECK(cublasSaxpy(cublas_handle, input_size * output_size, &beta, d_weights, 1, d_weights, 1));
        CUDA_CHECK(cudaMemset(d_biases, 0, biases_size));
        CUDA_CHECK(curandDestroyGenerator(rand_gen));
    }

    void forward(const float* input) {
        CUDA_CHECK(cudaMemcpyAsync(d_input, input, batch_input_size,
                                 cudaMemcpyHostToDevice, stream));
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               output_size, batch_size, input_size,
                               &alpha, d_weights, output_size,
                               d_input, input_size, &beta,
                               d_output, output_size));
        int blocks = (batch_size * output_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        applyActivation<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(d_output, batch_size * output_size, activation);
    }

    void backward(const float* next_grad, const float* prev_input) {
        timestep++;
        int blocks = (batch_size * output_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        CUDA_CHECK(cudaMemcpyAsync(d_gradients, next_grad, batch_output_size,
                                 cudaMemcpyHostToDevice, stream));
        computeGradient<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(d_gradients, d_output,
                                                                batch_size * output_size, activation);
        
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                               output_size, input_size, batch_size,
                               &alpha, d_gradients, output_size,
                               d_input, input_size, &beta,
                               d_weight_grads, output_size));
        CUBLAS_CHECK(cublasSgemv(cublas_handle, CUBLAS_OP_N,
                               output_size, batch_size, &alpha,
                               d_gradients, output_size,
                               d_output, 0, &beta,
                               d_bias_grads, 1));
        
        // Adam updates
        blocks = (input_size * output_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        adamUpdate<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(d_weights, d_weight_grads, d_weight_m, d_weight_v,
                                                            input_size * output_size, learning_rate, beta1, beta2, epsilon, timestep);
        blocks = (output_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        adamUpdate<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(d_biases, d_bias_grads, d_bias_m, d_bias_v,
                                                            output_size, learning_rate, beta1, beta2, epsilon, timestep);
    }

    float* getOutput() { return d_output; }
    float* getGradients() { return d_gradients; }
    int getOutputSize() const { return output_size; }
    cudaStream_t getStream() const { return stream; }
    void synchronize() { CUDA_CHECK(cudaStreamSynchronize(stream)); }
};

class NeuralNetwork {
private:
    std::vector<std::unique_ptr<NeuralLayer>> layers;
    std::vector<cudaStream_t> streams;
    int batch_size;
    float *d_input, *d_output, *d_loss_grad;
    size_t input_size, output_size;
    cublasHandle_t cublas_handle;

    // Cross-entropy loss computation
    __global__ static void computeCrossEntropyGradient(float* gradient, float* output, float* target, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            gradient[idx] = output[idx] - target[idx];  // Derivative of cross-entropy with softmax
        }
    }

public:
    NeuralNetwork(int batch, const std::vector<int>& layer_sizes,
                  const std::vector<ActivationType>& activations, float lr = 0.001f,
                  float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
        : batch_size(batch) {
        
        if (layer_sizes.size() < 2 || activations.size() != layer_sizes.size() - 1) {
            throw std::runtime_error("Invalid network configuration");
        }

        input_size = layer_sizes[0] * batch * sizeof(float);
        output_size = layer_sizes.back() * batch * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_input, input_size));
        CUDA_CHECK(cudaMalloc(&d_output, output_size));
        CUDA_CHECK(cudaMalloc(&d_loss_grad, output_size));
        
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        
        streams.resize(activations.size());
        for (auto& stream : streams) {
            CUDA_CHECK(cudaStreamCreate(&stream));
        }

        for (size_t i = 0; i < activations.size(); i++) {
            layers.push_back(std::make_unique<NeuralLayer>(
                layer_sizes[i], layer_sizes[i + 1], batch, activations[i],
                lr, beta1, beta2, epsilon, streams[i]
            ));
        }
    }

    ~NeuralNetwork() {
        cudaFree(d_input); cudaFree(d_output); cudaFree(d_loss_grad);
        for (auto& stream : streams) { cudaStreamDestroy(stream); }
        cublasDestroy(cublas_handle);
    }

    void forward(const float* h_input) {
        CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, input_size,
                                 cudaMemcpyHostToDevice, streams[0]));
        layers[0]->forward(d_input);
        for (size_t i = 1; i < layers.size(); i++) {
            layers[i]->forward(layers[i-1]->getOutput());
        }
    }

    float computeLoss(const float* h_target) {
        CUDA_CHECK(cudaMemcpyAsync(d_output, h_target, output_size,
                                 cudaMemcpyHostToDevice, streams.back()));
        
        float* d_network_output = layers.back()->getOutput();
        float loss = 0.0f;
        int size = batch_size * layers.back()->getOutputSize();
        
        // Compute cross-entropy loss
        float alpha = -1.0f / batch_size;
        CUBLAS_CHECK(cublasSaxpy(cublas_handle, size, &alpha, d_output, 1, d_network_output, 1));
        CUDA_CHECK(cudaMemcpy(&loss, d_network_output, sizeof(float), cudaMemcpyDeviceToHost));
        
        // Reset d_network_output for gradient computation
        CUDA_CHECK(cudaMemcpyAsync(d_network_output, layers.back()->getOutput(), output_size,
                                 cudaMemcpyDeviceToDevice, streams.back()));
        return loss;
    }

    void backward(const float* h_target) {
        CUDA_CHECK(cudaMemcpyAsync(d_output, h_target, output_size,
                                 cudaMemcpyHostToDevice, streams.back()));
        
        int size = batch_size * layers.back()->getOutputSize();
        int blocks = (size + 256 - 1) / 256;
        computeCrossEntropyGradient<<<blocks, 256, 0, streams.back()>>>(d_loss_grad, layers.back()->getOutput(), d_output, size);
        
        layers.back()->backward(d_loss_grad, layers[layers.size()-2]->getOutput());
        for (int i = layers.size() - 2; i >= 0; i--) {
            layers[i]->backward(layers[i+1]->getGradients(),
                              i > 0 ? layers[i-1]->getOutput() : d_input);
        }
    }

    void train(const float* h_input, const float* h_target) {
        forward(h_input);
        float loss = computeLoss(h_target);
        backward(h_target);
        for (auto& layer : layers) { layer->synchronize(); }
        return loss;
    }

    void getOutput(float* h_output) {
        CUDA_CHECK(cudaMemcpy(h_output, layers.back()->getOutput(),
                            output_size, cudaMemcpyDeviceToHost));
    }
};

int main() {
    try {
        const int BATCH_SIZE = 32;
        std::vector<int> layer_sizes = {784, 256, 128, 10};  // MNIST-like
        std::vector<ActivationType> activations = {
            ActivationType::ReLU,
            ActivationType::ReLU,
            ActivationType::Sigmoid  // Softmax-like behavior with cross-entropy
        };

        NeuralNetwork nn(BATCH_SIZE, layer_sizes, activations);
        
        std::vector<float> h_input(layer_sizes[0] * BATCH_SIZE);
        std::vector<float> h_target(layer_sizes.back() * BATCH_SIZE);
        std::vector<float> h_output(layer_sizes.back() * BATCH_SIZE);

        // Dummy data (one-hot encoded targets for classification)
        for (auto& val : h_input) val = static_cast<float>(rand()) / RAND_MAX;
        for (size_t i = 0; i < BATCH_SIZE; i++) {
            int label = rand() % 10;
            for (int j = 0; j < 10; j++) {
                h_target[i * 10 + j] = (j == label) ? 1.0f : 0.0f;
            }
        }

        // Training loop
        for (int epoch = 0; epoch < 50; epoch++) {
            float loss = nn.train(h_input.data(), h_target.data());
            if (epoch % 10 == 0) {
                nn.getOutput(h_output.data());
                printf("Epoch %d: Loss = %f\n", epoch, loss);
            }
        }
    }
    catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}