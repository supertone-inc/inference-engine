#include "inference-engine/TfliteInferenceEngine.hpp"

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

class TfliteInferenceEngine::Impl
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

        auto inputs = interpreter->inputs();
        input_count = inputs.size();
        for (auto tensor_index : inputs)
        {
            auto tensor = interpreter->tensor(tensor_index);
            auto dims = tensor->dims;
            input_shapes.emplace_back(dims->data, dims->size);
        }

        auto outputs = interpreter->outputs();
        output_count = outputs.size();
        for (auto tensor_index : outputs)
        {
            auto tensor = interpreter->tensor(tensor_index);
            auto dims = tensor->dims;
            output_shapes.emplace_back(dims->data, dims->size);
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
        input_shapes[index] = shape;

        auto tensor_index = interpreter->inputs()[index];

        if (interpreter->ResizeInputTensor(tensor_index, input_shapes[index]) != kTfLiteOk)
        {
            throw std::runtime_error("failed to resize input tensor");
        }

        if (interpreter->AllocateTensors() != kTfLiteOk)
        {
            throw std::runtime_error("failed to allocate tensor buffers");
        }

        auto outputs = interpreter->outputs();
        for (auto i = 0; i < output_count; i++)
        {
            auto tensor_index = outputs[i];
            auto tensor = interpreter->tensor(tensor_index);
            auto dims = tensor->dims;
            output_shapes.emplace(output_shapes.begin() + 1, dims->data, dims->size);
        }
    }

    void set_output_shape(size_t index, const std::vector<size_t> &shape)
    {
        throw std::runtime_error("not supported");
    }

    void set_input_data(size_t index, const float *data)
    {
        interpreter->input_tensor(index)->data.f = const_cast<float *>(data);
    }

    void set_output_data(size_t index, float *data)
    {
        interpreter->output_tensor(index)->data.f = data;
    }

    void run()
    {
        if (interpreter->Invoke() != kTfLiteOk)
        {
            throw std::runtime_error("failed to invoke the interpreter");
        }
    }

private:
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;

    size_t input_count;
    size_t output_count;

    std::vector<Shape> input_shapes;
    std::vector<Shape> output_shapes;
};

TfliteInferenceEngine::TfliteInferenceEngine(const void *model_data, size_t model_data_size_bytes)
    : impl(new Impl(model_data, model_data_size_bytes))
{
}

size_t TfliteInferenceEngine::get_input_count() const
{
    return impl->get_input_count();
}

size_t TfliteInferenceEngine::get_output_count() const
{
    return impl->get_output_count();
}

const std::vector<size_t> &TfliteInferenceEngine::get_input_shape(size_t index) const
{
    return impl->get_input_shape(index);
}

const std::vector<size_t> &TfliteInferenceEngine::get_output_shape(size_t index) const
{
    return impl->get_output_shape(index);
}

void TfliteInferenceEngine::set_input_shape(size_t index, const std::vector<size_t> &shape)
{
    impl->set_input_shape(index, shape);
}

void TfliteInferenceEngine::set_output_shape(size_t index, const std::vector<size_t> &shape)
{
    impl->set_output_shape(index, shape);
}

void TfliteInferenceEngine::set_input_data(size_t index, const float *data)
{
    impl->set_input_data(index, data);
}

void TfliteInferenceEngine::set_output_data(size_t index, float *data)
{
    impl->set_output_data(index, data);
}

void TfliteInferenceEngine::run()
{
    impl->run();
}
} // namespace inference_engine
