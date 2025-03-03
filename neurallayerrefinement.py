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

enum class ActivationType { ReLU, Sigmoid, Tanh, Softmax };

class NeuralLayer {
private:
    float *d_weights, *d_biases, *d_input, *d_output, *d_gradients;
    float *d_weight_grads, *d_bias_grads;
    float *d_weight_m, *d_weight_v, *d_bias_m, *d_bias_v;
    int input_size, output_size, batch_size, max_batch_size;
    size_t weights_size, biases_size, batch_input_size, batch_output_size;
    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    ActivationType activation;
    float learning_rate, beta1, beta2, epsilon;
    int THREADS_PER_BLOCK;  // Per-instance configuration
    int timestep;

    void allocateDeviceMemory(int max_batch) {
        weights_size = input_size * output_size * sizeof(float);
        biases_size = output_size * sizeof(float);
        batch_input_size = input_size * max_batch * sizeof(float);
        batch_output_size = output_size * max_batch * sizeof(float);

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

    void freeDeviceMemory() {
        cudaFree(d_weights); cudaFree(d_biases); cudaFree(d_input);
        cudaFree(d_output); cudaFree(d_gradients); cudaFree(d_weight_grads);
        cudaFree(d_bias_grads); cudaFree(d_weight_m); cudaFree(d_weight_v);
        cudaFree(d_bias_m); cudaFree(d_bias_v);
    }

    __global__ static void applyActivation(float* data, int batch_size, int output_size, ActivationType type) {
        int batch_idx = blockIdx.x;
        int out_idx = threadIdx.x;
        if (batch_idx < batch_size && out_idx < output_size) {
            int idx = batch_idx * output_size + out_idx;
            switch(type) {
                case ActivationType::ReLU: data[idx] = fmaxf(0.0f, data[idx]); break;
                case ActivationType::Sigmoid: data[idx] = 1.0f / (1.0f + expf(-data[idx])); break;
                case ActivationType::Tanh: data[idx] = tanhf(data[idx]); break;
                case ActivationType::Softmax: {
                    extern __shared__ float sh_mem[];
                    float* sh_data = sh_mem;
                    float max_val = -INFINITY;
                    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
                        sh_data[i] = data[batch_idx * output_size + i];
                        max_val = fmaxf(max_val, sh_data[i]);
                    }
                    __syncthreads();
                    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
                        if (threadIdx.x < stride) {
                            max_val = fmaxf(max_val, sh_data[threadIdx.x + stride]);
                        }
                        __syncthreads();
                    }
                    float sum = 0.0f;
                    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
                        sh_data[i] = expf(sh_data[i] - max_val);
                        sum += sh_data[i];
                    }
                    __syncthreads();
                    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
                        if (threadIdx.x < stride) {
                            sum += sh_data[threadIdx.x + stride];
                        }
                        __syncthreads();
                    }
                    data[idx] = sh_data[out_idx] / sum;
                } break;
            }
        }
    }

    __global__ static void computeGradient(float* gradient, float* output, int batch_size, int output_size, ActivationType type) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < batch_size * output_size) {
            float val = output[idx];
            gradient[idx] *= (type == ActivationType::ReLU) ? (val > 0.0f ? 1.0f : 0.0f) :
                             (type == ActivationType::Sigmoid) ? (val * (1.0f - val) + 1e-8f) :
                             (type == ActivationType::Tanh) ? (1.0f - val * val + 1e-8f) : 1.0f; // Softmax handled by loss
        }
    }

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
    NeuralLayer(int in_size, int out_size, int batch, int max_batch, ActivationType act,
                float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f,
                cudaStream_t s = 0)
        : input_size(in_size), output_size(out_size), batch_size(batch), max_batch_size(max_batch),
          activation(act), learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps),
          stream(s), timestep(0) {
        
        CUDA_CHECK(cublasCreate(&cublas_handle));
        if (stream) CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
        THREADS_PER_BLOCK = getOptimalThreadsPerBlock();
        allocateDeviceMemory(max_batch);
        initializeWeights();
    }

    ~NeuralLayer() { freeDeviceMemory(); cublasDestroy(cublas_handle); }

    void resizeBatch(int new_batch_size) {
        if (new_batch_size > max_batch_size) {
            freeDeviceMemory();
            allocateDeviceMemory(new_batch_size);
            max_batch_size = new_batch_size;
        }
        batch_size = new_batch_size;
    }

    static int getOptimalThreadsPerBlock() {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        return prop.maxThreadsPerBlock > 512 ? 512 : prop.maxThreadsPerBlock;
    }

    void initializeWeights() {
        curandGenerator_t rand_gen;
        CUDA_CHECK(curandCreateGenerator(&rand_gen, CURAND_RNG_PSEUDO_DEFAULT));
        CUDA_CHECK(curandSetPseudoRandomGeneratorSeed(rand_gen, time(NULL)));
        
        float limit = (activation == ActivationType::ReLU) ? sqrtf(6.0f / (input_size + output_size)) : sqrtf(1.0f / input_size);
        CUDA_CHECK(curandGenerateUniform(rand_gen, d_weights, input_size * output_size));
        float alpha = 2.0f * limit, beta = -limit;
        CUBLAS_CHECK(cublasSscal(cublas_handle, input_size * output_size, &alpha, d_weights, 1));
        CUBLAS_CHECK(cublasSaxpy(cublas_handle, input_size * output_size, &beta, d_weights, 1, d_weights, 1));
        CUDA_CHECK(cudaMemset(d_biases, 0, biases_size));
        CUDA_CHECK(curandDestroyGenerator(rand_gen));
    }

    void forward(const float* input) {
        CUDA_CHECK(cudaMemcpyAsync(d_input, input, input_size * batch_size * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 output_size, batch_size, input_size,
                                 &alpha, d_weights, output_size,
                                 d_input, input_size, &beta,
                                 d_output, output_size));
        applyActivation<<<batch_size, THREADS_PER_BLOCK, output_size * sizeof(float), stream>>>(d_output, batch_size, output_size, activation);
    }

    void backward(const float* next_grad, const float* prev_input) {
        timestep++;
        CUDA_CHECK(cudaMemcpyAsync(d_gradients, next_grad, output_size * batch_size * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
        if (activation != ActivationType::Softmax) {
            int blocks = (batch_size * output_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            computeGradient<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(d_gradients, d_output, batch_size, output_size, activation);
        }
        
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                 output_size, input_size, batch_size,
                                 &alpha, d_gradients, output_size,
                                 prev_input, input_size, &beta,
                                 d_weight_grads, output_size));
        CUBLAS_CHECK(cublasSgemv(cublas_handle, CUBLAS_OP_N,
                                 output_size, batch_size, &alpha,
                                 d_gradients, output_size,
                                 d_output, 0, &beta,
                                 d_bias_grads, 1));
        
        int weight_blocks = (input_size * output_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        int bias_blocks = (output_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        adamUpdate<<<weight_blocks, THREADS_PER_BLOCK, 0, stream>>>(d_weights, d_weight_grads, d_weight_m, d_weight_v,
                                                                    input_size * output_size, learning_rate, beta1, beta2, epsilon, timestep);
        adamUpdate<<<bias_blocks, THREADS_PER_BLOCK, 0, stream>>>(d_biases, d_bias_grads, d_bias_m, d_bias_v,
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
    int batch_size, max_batch_size;
    float *d_input, *d_output, *d_loss_grad, *d_temp_loss;
    size_t input_size, output_size;
    cublasHandle_t cublas_handle;

    __global__ static void computeCrossEntropyGradient(float* gradient, float* output, float* target, int batch_size, int output_size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < batch_size * output_size) {
            gradient[idx] = output[idx] - target[idx];
        }
    }

    __global__ static void computeCrossEntropyLoss(float* loss, float* output, float* target, int batch_size, int output_size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < batch_size * output_size) {
            loss[idx] = -target[idx] * logf(fmaxf(output[idx], 1e-8f));
        }
    }

public:
    NeuralNetwork(int batch, int max_batch, const std::vector<int>& layer_sizes,
                  const std::vector<ActivationType>& activations, float lr = 0.001f,
                  float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
        : batch_size(batch), max_batch_size(max_batch) {
        
        if (layer_sizes.size() < 2 || activations.size() != layer_sizes.size() - 1) {
            throw std::runtime_error("Invalid network configuration");
        }

        input_size = layer_sizes[0] * max_batch * sizeof(float);
        output_size = layer_sizes.back() * max_batch * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_input, input_size));
        CUDA_CHECK(cudaMalloc(&d_output, output_size));
        CUDA_CHECK(cudaMalloc(&d_loss_grad, output_size));
        CUDA_CHECK(cudaMalloc(&d_temp_loss, output_size));
        
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        
        streams.resize(activations.size());
        for (auto& stream : streams) {
            CUDA_CHECK(cudaStreamCreate(&stream));
        }

        for (size_t i = 0; i < activations.size(); i++) {
            layers.push_back(std::make_unique<NeuralLayer>(
                layer_sizes[i], layer_sizes[i + 1], batch, max_batch, activations[i],
                lr, beta1, beta2, epsilon, streams[i]
            ));
        }
    }

    ~NeuralNetwork() {
        cudaFree(d_input); cudaFree(d_output); cudaFree(d_loss_grad); cudaFree(d_temp_loss);
        for (auto& stream : streams) { cudaStreamDestroy(stream); }
        cublasDestroy(cublas_handle);
    }

    void resizeBatch(int new_batch_size) {
        if (new_batch_size > max_batch_size) {
            throw std::runtime_error("Batch size exceeds max capacity");
        }
        batch_size = new_batch_size;
        for (auto& layer : layers) {
            layer->resizeBatch(new_batch_size);
        }
    }

    void forward(const float* h_input) {
        CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, layer_sizes[0] * batch_size * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[0]));
        layers[0]->forward(d_input);
        for (size_t i = 1; i < layers.size(); i++) {
            layers[i]->forward(layers[i-1]->getOutput());
        }
    }

    float computeLoss(const float* h_target) {
        CUDA_CHECK(cudaMemcpyAsync(d_output, h_target, output_size * batch_size * sizeof(float),
                                   cudaMemcpyHostToDevice, streams.back()));
        
        int size = batch_size * layers.back()->getOutputSize();
        int blocks = (size + 256 - 1) / 256;
        computeCrossEntropyLoss<<<blocks, 256, 0, streams.back()>>>(d_temp_loss, layers.back()->getOutput(), d_output, batch_size, layers.back()->getOutputSize());
        
        float loss = 0.0f;
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSasum(cublas_handle, size, d_temp_loss, 1, &loss));
        return loss / batch_size;
    }

    void backward(const float* h_target) {
        CUDA_CHECK(cudaMemcpyAsync(d_output, h_target, output_size * batch_size * sizeof(float),
                                   cudaMemcpyHostToDevice, streams.back()));
        
        int size = batch_size * layers.back()->getOutputSize();
        int blocks = (size + 256 - 1) / 256;
        computeCrossEntropyGradient<<<blocks, 256, 0, streams.back()>>>(d_loss_grad, layers.back()->getOutput(), d_output, batch_size, layers.back()->getOutputSize());
        
        layers.back()->backward(d_loss_grad, layers[layers.size()-2]->getOutput());
        for (int i = layers.size() - 2; i >= 0; i--) {
            layers[i]->backward(layers[i+1]->getGradients(),
                                i > 0 ? layers[i-1]->getOutput() : d_input);
        }
    }

    float train(const float* h_input, const float* h_target) {
        forward(h_input);
        float loss = computeLoss(h_target);
        backward(h_target);
        return loss;
    }

    void getOutput(float* h_output) {
        CUDA_CHECK(cudaMemcpy(h_output, layers.back()->getOutput(),
                              layers.back()->getOutputSize() * batch_size * sizeof(float), cudaMemcpyDeviceToHost));
    }
};

int main() {
    try {
        const int BATCH_SIZE = 64;
        const int MAX_BATCH_SIZE = 128;
        std::vector<int> layer_sizes = {784, 512, 256, 128, 64, 10};  // Deeper MNIST-like network
        std::vector<ActivationType> activations = {
            ActivationType::ReLU,
            ActivationType::ReLU,
            ActivationType::ReLU,
            ActivationType::ReLU,
            ActivationType::Softmax
        };

        NeuralNetwork nn(BATCH_SIZE, MAX_BATCH_SIZE, layer_sizes, activations);
        
        float *h_input, *h_target;
        CUDA_CHECK(cudaHostAlloc(&h_input, layer_sizes[0] * BATCH_SIZE * sizeof(float), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_target, layer_sizes.back() * BATCH_SIZE * sizeof(float), cudaHostAllocDefault));
        std::vector<float> h_output(layer_sizes.back() * BATCH_SIZE);

        // Dummy data
        for (int i = 0; i < layer_sizes[0] * BATCH_SIZE; i++) {
            h_input[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        std::vector<int> labels(BATCH_SIZE);
        for (size_t i = 0; i < BATCH_SIZE; i++) {
            int label = rand() % 10;
            labels[i] = label;
            for (int j = 0; j < 10; j++) {
                h_target[i * 10 + j] = (j == label) ? 1.0f : 0.0f;
            }
        }

        // Training loop
        for (int epoch = 0; epoch < 100; epoch++) {
            float loss = nn.train(h_input, h_target);
            nn.getOutput(h_output.data());
            
            int correct = 0;
            for (int i = 0; i < BATCH_SIZE; i++) {
                float max_val = -INFINITY;
                int pred = 0;
                for (int j = 0; j < 10; j++) {
                    if (h_output[i * 10 + j] > max_val) {
                        max_val = h_output[i * 10 + j];
                        pred = j;
                    }
                }
                if (pred == labels[i]) correct++;
            }
            float accuracy = static_cast<float>(correct) / BATCH_SIZE;

            if (epoch % 10 == 0) {
                printf("Epoch %d: Loss = %f, Accuracy = %.2f%%\n", epoch, loss, accuracy * 100);
            }
        }

        // Test resize
        nn.resizeBatch(32);
        CUDA_CHECK(cudaFreeHost(h_input));
        CUDA_CHECK(cudaFreeHost(h_target));
        CUDA_CHECK(cudaHostAlloc(&h_input, layer_sizes[0] * 32 * sizeof(float), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_target, layer_sizes.back() * 32 * sizeof(float), cudaHostAllocDefault));
        h_output.resize(layer_sizes.back() * 32);
        printf("Resized batch to 32, running one epoch...\n");
        float loss = nn.train(h_input, h_target);
        printf("Loss after resize: %f\n", loss);
        
        CUDA_CHECK(cudaFreeHost(h_input));
        CUDA_CHECK(cudaFreeHost(h_target));
    }
    catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}