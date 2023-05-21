#include "inference-engine/OrtInferenceEngine.hpp"

#include <onnxruntime_cxx_api.h>
#include <vector>

namespace inference_engine
{
class OrtInferenceEngine::Impl
{
public:
    Impl(const std::byte *model_data, size_t model_data_size)
        : session(env, model_data, model_data_size, Ort::SessionOptions().SetIntraOpNumThreads(1))
        , memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU))
        , run_options(Ort::RunOptions{nullptr})
        , input_count(session.GetInputCount())
        , output_count(session.GetOutputCount())
    {
        for (auto i = 0; i < input_count; i++)
        {
            input_names.push_back(session.GetInputName(i, allocator));

            auto shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            input_shapes.push_back(shape);
            input_element_counts.push_back(get_element_count(shape));
        }

        for (auto i = 0; i < output_count; i++)
        {
            output_names.push_back(session.GetOutputName(i, allocator));

            auto shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            output_shapes.push_back(shape);
            output_element_counts.push_back(get_element_count(shape));
        }
    }

    const std::vector<std::vector<int64_t>> &get_input_shapes() const
    {
        return input_shapes;
    }

    void set_input_shapes(const std::vector<std::vector<int64_t>> &shapes)
    {
        input_shapes = shapes;

        input_element_counts.clear();
        for (auto &shape : input_shapes)
        {
            input_element_counts.push_back(get_element_count(shape));
        }

        for (auto i = 0; i < input_values.size(); i++)
        {
            input_values[i] = Ort::Value::CreateTensor<float>(
                memory_info,
                input_values[i].GetTensorMutableData<float>(),
                input_element_counts[i],
                input_shapes[i].data(),
                input_shapes[i].size()
            );
        }
    }

    void set_input_data(const float *const *data)
    {
        if (input_values.size() == input_count)
        {
            for (auto i = 0; i < input_count; i++)
            {
                auto tensor_data = input_values[i].GetTensorMutableData<float>();
                auto tensor_data_ptr = &tensor_data;
                *tensor_data_ptr = const_cast<float *>(data[i]);
            }

            return;
        }

        input_values.clear();
        for (auto i = 0; i < input_count; i++)
        {
            input_values.push_back(Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float *>(data[i]),
                input_element_counts[i],
                input_shapes[i].data(),
                input_shapes[i].size()
            ));
        }
    }

    const std::vector<std::vector<int64_t>> &get_output_shapes() const
    {
        return output_shapes;
    }

    void set_output_shapes(const std::vector<std::vector<int64_t>> &shapes)
    {
        output_shapes = shapes;

        output_element_counts.clear();
        for (auto &shape : output_shapes)
        {
            output_element_counts.push_back(get_element_count(shape));
        }

        for (auto i = 0; i < output_values.size(); i++)
        {
            output_values[i] = Ort::Value::CreateTensor<float>(
                memory_info,
                output_values[i].GetTensorMutableData<float>(),
                output_element_counts[i],
                output_shapes[i].data(),
                output_shapes[i].size()
            );
        }
    }

    void set_output_data(float **data)
    {
        if (output_values.size() == output_count)
        {
            for (auto i = 0; i < output_count; i++)
            {
                auto tensor_data = output_values[i].GetTensorMutableData<float>();
                auto tensor_data_ptr = &tensor_data;
                *tensor_data_ptr = data[i];
            }

            return;
        }

        output_values.clear();
        for (auto i = 0; i < output_count; i++)
        {
            output_values.push_back(Ort::Value::CreateTensor<float>(
                memory_info,
                data[i],
                output_element_counts[i],
                output_shapes[i].data(),
                output_shapes[i].size()
            ));
        }
    }

    void run()
    {
        session.Run(
            run_options,
            input_names.data(),
            input_values.data(),
            input_values.size(),
            output_names.data(),
            output_values.data(),
            output_values.size()
        );
    }

private:
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    const Ort::MemoryInfo memory_info;
    const Ort::RunOptions run_options;

    const size_t input_count;
    std::vector<const char *> input_names;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<int64_t> input_element_counts;
    std::vector<Ort::Value> input_values;

    const size_t output_count;
    std::vector<const char *> output_names;
    std::vector<std::vector<int64_t>> output_shapes;
    std::vector<int64_t> output_element_counts;
    std::vector<Ort::Value> output_values;

    static int64_t get_element_count(const std::vector<int64_t> &shape)
    {
        if (shape.empty())
        {
            return 0;
        }

        auto element_count = 1;

        for (auto v : shape)
        {
            if (v < 0)
            {
                return -1;
            }

            element_count *= v;
        }

        return element_count;
    }
};

OrtInferenceEngine::OrtInferenceEngine(const std::byte *model_data, size_t model_data_size)
    : impl(new Impl(model_data, model_data_size))
{
}

const std::vector<std::vector<int64_t>> &OrtInferenceEngine::get_input_shapes() const
{
    return impl->get_input_shapes();
}

void OrtInferenceEngine::set_input_shapes(const std::vector<std::vector<int64_t>> &shapes)
{
    impl->set_input_shapes(shapes);
}

void OrtInferenceEngine::set_input_data(const float *const *data)
{
    impl->set_input_data(data);
}

const std::vector<std::vector<int64_t>> &OrtInferenceEngine::get_output_shapes() const
{
    return impl->get_output_shapes();
}

void OrtInferenceEngine::set_output_shapes(const std::vector<std::vector<int64_t>> &shapes)
{
    impl->set_output_shapes(shapes);
}

void OrtInferenceEngine::set_output_data(float **data)
{
    impl->set_output_data(data);
}

void OrtInferenceEngine::run()
{
    impl->run();
}
} // namespace inference_engine
