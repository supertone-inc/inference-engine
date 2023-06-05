#include "inference_engine/TfLiteInferenceEngine.hpp"

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <vector>

namespace inference_engine
{
class Shape
{
public:
    template <typename U>
    Shape(const std::vector<U> &values)
        : Shape(values.data(), values.size())
    {
    }

    template <typename U>
    Shape(const U *data, size_t size)
    {
        for (auto i = 0; i < size; i++)
        {
            this->values.push_back(data[i] > 0 ? data[i] : 0);
        }
        element_count = count_elements(values);
    }

    Shape &operator=(const std::vector<size_t> &values)
    {
        this->values = values;
        element_count = count_elements(values);
        return *this;
    }

    operator const std::vector<size_t> &() const
    {
        return values;
    }

    template <typename U>
    operator std::vector<U>() const
    {
        return {values.begin(), values.end()};
    }

    const size_t *data() const
    {
        return values.data();
    }

    size_t size() const
    {
        return values.size();
    }

    size_t get_element_count() const
    {
        return element_count;
    }

private:
    std::vector<size_t> values;
    size_t element_count;

    template <typename U>
    static size_t count_elements(const std::vector<U> &shape)
    {
        if (shape.empty())
        {
            return 0;
        }

        auto element_count = 1;

        for (auto v : shape)
        {
            if (v <= 0)
            {
                return 0;
            }

            element_count *= v;
        }

        return element_count;
    }
};

class TfLiteInferenceEngine::Impl
{
public:
    Impl(const void *model_data, size_t model_data_size_bytes)
    {
        model = tflite::FlatBufferModel::BuildFromBuffer(
            static_cast<const char *>(model_data),
            model_data_size_bytes
        );

        if (!model)
        {
            throw std::runtime_error("failed to load model");
        }

        tflite::ops::builtin::BuiltinOpResolver op_resolver;
        tflite::InterpreterBuilder builder(*model, op_resolver);

        if (builder.SetNumThreads(1) != kTfLiteOk)
        {
            throw std::runtime_error("failed to set the number of CPU threads");
        }

        if (builder(&interpreter) != kTfLiteOk)
        {
            throw std::runtime_error("failed to build the interpreter");
        }

        if (interpreter->AllocateTensors() != kTfLiteOk)
        {
            throw std::runtime_error("failed to allocate tensor buffers");
        }

        input_count = interpreter->inputs().size();
        for (auto i = 0; i < input_count; i++)
        {
            auto tensor = interpreter->input_tensor(i);
            auto dims = tensor->dims;
            input_shapes.emplace_back(dims->data, dims->size);
            input_data.emplace_back(nullptr);
        }

        output_count = interpreter->outputs().size();
        for (auto i = 0; i < output_count; i++)
        {
            auto tensor = interpreter->output_tensor(i);
            auto dims = tensor->dims;
            output_shapes.emplace_back(dims->data, dims->size);
            output_data.emplace_back(nullptr);
        }
    }

    size_t get_input_count() const
    {
        return input_count;
    }

    size_t get_output_count() const
    {
        return output_count;
    }

    const std::vector<size_t> &get_input_shape(size_t index) const
    {
        return input_shapes[index];
    }

    const std::vector<size_t> &get_output_shape(size_t index) const
    {
        return output_shapes[index];
    }

    void set_input_shape(size_t index, const std::vector<size_t> &shape)
    {
        if (interpreter->ResizeInputTensor(interpreter->inputs()[index], {shape.begin(), shape.end()}) != kTfLiteOk)
        {
            throw std::runtime_error("failed to resize input tensor");
        }

        if (interpreter->AllocateTensors() != kTfLiteOk)
        {
            throw std::runtime_error("failed to allocate tensor buffers for resized input tensor");
        }

        {
            auto tensor = interpreter->input_tensor(index);
            auto dims = tensor->dims;
            input_shapes.emplace(input_shapes.begin() + index, dims->data, dims->size);
            input_data[index] = nullptr;
        }

        for (auto i = 0; i < output_count; i++)
        {
            auto tensor = interpreter->output_tensor(i);
            auto dims = tensor->dims;
            output_shapes.emplace(output_shapes.begin() + i, dims->data, dims->size);
            output_data[i] = nullptr;
        }
    }

    void set_output_shape(size_t index, const std::vector<size_t> &shape)
    {
        throw std::runtime_error("reshape output tensor is not supported");
    }

    float *get_input_data(size_t index)
    {
        if (input_data[index])
        {
            return input_data[index];
        }

        return interpreter->typed_input_tensor<float>(index);
    }

    const float *get_output_data(size_t index) const
    {
        if (output_data[index])
        {
            return output_data[index];
        }

        return interpreter->typed_output_tensor<float>(index);
    }

    void set_input_data(size_t index, const float *data)
    {
        input_data[index] = const_cast<float *>(data);

        if (data)
        {
            std::copy_n(
                data,
                interpreter->input_tensor(index)->bytes / sizeof(float),
                interpreter->typed_input_tensor<float>(index)
            );
        }
    }

    void set_output_data(size_t index, float *data)
    {
        output_data[index] = data;
    }

    void run()
    {
        if (interpreter->Invoke() != kTfLiteOk)
        {
            throw std::runtime_error("failed to invoke the interpreter");
        }

        for (auto i = 0; i < output_count; i++)
        {
            if (output_data[i])
            {
                std::copy_n(
                    interpreter->typed_output_tensor<float>(i),
                    interpreter->output_tensor(i)->bytes / sizeof(float),
                    output_data[i]
                );
            }
        }
    }

private:
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;

    size_t input_count;
    size_t output_count;

    std::vector<Shape> input_shapes;
    std::vector<Shape> output_shapes;

    std::vector<float *> input_data;
    std::vector<float *> output_data;
};

TfLiteInferenceEngine::TfLiteInferenceEngine(const void *model_data, size_t model_data_size_bytes)
    : impl(new Impl(model_data, model_data_size_bytes))
{
}

size_t TfLiteInferenceEngine::get_input_count() const
{
    return impl->get_input_count();
}

size_t TfLiteInferenceEngine::get_output_count() const
{
    return impl->get_output_count();
}

const std::vector<size_t> &TfLiteInferenceEngine::get_input_shape(size_t index) const
{
    return impl->get_input_shape(index);
}

const std::vector<size_t> &TfLiteInferenceEngine::get_output_shape(size_t index) const
{
    return impl->get_output_shape(index);
}

void TfLiteInferenceEngine::set_input_shape(size_t index, const std::vector<size_t> &shape)
{
    impl->set_input_shape(index, shape);
}

void TfLiteInferenceEngine::set_output_shape(size_t index, const std::vector<size_t> &shape)
{
    impl->set_output_shape(index, shape);
}

float *TfLiteInferenceEngine::get_input_data(size_t index)
{
    return impl->get_input_data(index);
}

const float *TfLiteInferenceEngine::get_output_data(size_t index) const
{
    return impl->get_output_data(index);
}

void TfLiteInferenceEngine::set_input_data(size_t index, const float *data)
{
    impl->set_input_data(index, data);
}

void TfLiteInferenceEngine::set_output_data(size_t index, float *data)
{
    impl->set_output_data(index, data);
}

void TfLiteInferenceEngine::run()
{
    impl->run();
}
} // namespace inference_engine
