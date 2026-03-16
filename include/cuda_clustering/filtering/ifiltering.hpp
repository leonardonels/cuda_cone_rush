#pragma once 

class IFilter
{
    protected:
        
    public:
        virtual void filterPoints(float* input, unsigned int inputSize, float** output, unsigned int* outputSize, cudaStream_t stream) = 0;
        virtual ~IFilter() = default;
};