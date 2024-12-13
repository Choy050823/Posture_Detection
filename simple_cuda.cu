#include <iostream>

__global__ void hello_cuda() {
    printf("Hello from CUDA kernel!\n");
}

int main() {  
    // Launch the kernel
    hello_cuda<<<1, 10>>>();
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();

    std::cout << "Hello from CPU!" << std::endl;
    return 0;
}
