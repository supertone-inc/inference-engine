#include "inference_engine/OrtInferenceEngine.hpp"

#include <onnxruntime_cxx_api.h>
#include <vector>

namespace inference_engine
{
class Shape
{
public:
    template <typename U>
    Shape(const std::vector<U> &values)
    {
        for (auto v : values)
        {
            this->values.push_back(v > 0 ? v : 0);
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

class OrtInferenceEngine::Impl
{
public:
    Impl(const void *model_data, size_t model_data_size_bytes)
        : env()
        , session(
              env,
              model_data,
              model_data_size_bytes,
              static_cast<Ort::SessionOptions &>(
                  Ort::SessionOptions().SetIntraOpNumThreads(1)
              )
          )
        , io_binding(session)
        , allocator()
        , memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU))
        , run_options(nullptr)
        , input_count(session.GetInputCount())
        , output_count(session.GetOutputCount())
    {
        for (auto i = 0; i < input_count; i++)
        {
            input_names.push_back(session.GetInputNameAllocated(i, allocator));
            input_shapes.push_back(session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
            input_values.push_back(Ort::Value::CreateTensor<float>(
                allocator,
                reinterpret_cast<const int64_t *>(input_shapes[i].data()),
                input_shapes[i].size()
            ));
            io_binding.BindInput(input_names[i].get(), input_values[i]);
        }

        for (auto i = 0; i < output_count; i++)
        {
            output_names.push_back(session.GetOutputNameAllocated(i, allocator));
            output_shapes.push_back(session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
            output_values.push_back(Ort::Value::CreateTensor<float>(
                allocator,
                reinterpret_cast<const int64_t *>(output_shapes[i].data()),
                output_shapes[i].size()
            ));
            io_binding.BindOutput(output_names[i].get(), output_values[i]);
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
        input_values[index] = Ort::Value::CreateTensor<float>(
            allocator,
            reinterpret_cast<const int64_t *>(input_shapes[index].data()),
            input_shapes[index].size()
        );
        io_binding.BindInput(input_names[index].get(), input_values[index]);
    }

    void set_output_shape(size_t index, const std::vector<size_t> &shape)
    {
        output_shapes[index] = shape;
        output_values[index] = Ort::Value::CreateTensor<float>(
            allocator,
            reinterpret_cast<const int64_t *>(output_shapes[index].data()),
            output_shapes[index].size()
        );
        io_binding.BindOutput(output_names[index].get(), output_values[index]);
    }

    float *get_input_data(size_t index)
    {
        return input_values[index].GetTensorMutableData<float>();
    }

    const float *get_output_data(size_t index) const
    {
        return output_values[index].GetTensorData<float>();
    }

    void set_input_data(size_t index, const float *data)
    {
        input_values[index] = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float *>(data),
            input_shapes[index].get_element_count(),
            reinterpret_cast<const int64_t *>(input_shapes[index].data()),
            input_shapes[index].size()
        );
        io_binding.BindInput(input_names[index].get(), input_values[index]);
    }

    void set_output_data(size_t index, float *data)
    {
        output_values[index] = Ort::Value::CreateTensor<float>(
            memory_info,
            data,
            output_shapes[index].get_element_count(),
            reinterpret_cast<const int64_t *>(output_shapes[index].data()),
            output_shapes[index].size()
        );
        io_binding.BindOutput(output_names[index].get(), output_values[index]);
    }

    void run()
    {
        session.Run(run_options, io_binding);
    }

private:
    Ort::Env env;
    Ort::Session session;
    Ort::IoBinding io_binding;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info;
    Ort::RunOptions run_options;

    const size_t input_count;
    const size_t output_count;

    std::vector<Ort::AllocatedStringPtr> input_names;
    std::vector<Ort::AllocatedStringPtr> output_names;

    std::vector<Shape> input_shapes;
    std::vector<Shape> output_shapes;

    std::vector<Ort::Value> input_values;
    std::vector<Ort::Value> output_values;
};

OrtInferenceEngine::OrtInferenceEngine(const void *model_data, size_t model_data_size_bytes)
    : impl(new Impl(model_data, model_data_size_bytes))
{
}

size_t OrtInferenceEngine::get_input_count() const
{
    return impl->get_input_count();
}

size_t OrtInferenceEngine::get_output_count() const
{
    return impl->get_output_count();
}

const std::vector<size_t> &OrtInferenceEngine::get_input_shape(size_t index) const
{
    return impl->get_input_shape(index);
}

const std::vector<size_t> &OrtInferenceEngine::get_output_shape(size_t index) const
{
    return impl->get_output_shape(index);
}

void OrtInferenceEngine::set_input_shape(size_t index, const std::vector<size_t> &shape)
{
    impl->set_input_shape(index, shape);
}

void OrtInferenceEngine::set_output_shape(size_t index, const std::vector<size_t> &shape)
{
    impl->set_output_shape(index, shape);
}

float *OrtInferenceEngine::get_input_data(size_t index)
{
    return impl->get_input_data(index);
}

const float *OrtInferenceEngine::get_output_data(size_t index) const
{
    return impl->get_output_data(index);
}

void OrtInferenceEngine::set_input_data(size_t index, const float *data)
{
    impl->set_input_data(index, data);
}

void OrtInferenceEngine::set_output_data(size_t index, float *data)
{
    impl->set_output_data(index, data);
}

void OrtInferenceEngine::run()
{
    impl->run();
}
} // namespace inference_engine
