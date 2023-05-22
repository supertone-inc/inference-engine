#include "OrtInferenceEngine.hpp"

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>

const std::filesystem::path PROJECT_DIR = std::filesystem::path(__FILE__).parent_path().parent_path();

std::vector<std::byte> read_file(const std::filesystem::path &file_path)
{
    auto file_size = std::filesystem::file_size(file_path);
    std::vector<std::byte> file(file_size);
    std::ifstream ifs(file_path, std::ios::binary);
    ifs.read(reinterpret_cast<char *>(file.data()), file.size());

    return std::move(file);
}

template <typename T>
T unwrap(Result<T> &&result)
{
    if (result.code == ResultCode::Ok)
    {
        return std::move(result.value);
    }

    throw std::runtime_error(result.error.get_message());
}

TEST_CASE("OrtInferenceEngine with invalid model data")
{
    auto result = OrtInferenceEngine::create(nullptr, 0);
    REQUIRE(result.code == ResultCode::Error);
    REQUIRE(std::string(result.error.get_message()) == "No graph was found in the protobuf.");
}

TEST_CASE("OrtInferenceEngine with dynamic-shape model")
{
    auto model = read_file(PROJECT_DIR / "../ort-cpp/test-models/mat_mul_dynamic_shape.onnx");
    auto engine = unwrap(OrtInferenceEngine::create(model.data(), model.size()));
    REQUIRE(engine.get_input_count() == 2);
    REQUIRE(engine.get_output_count() == 1);

    {
        std::vector<std::vector<size_t>> input_shapes;
        for (auto i = 0; i < engine.get_input_count(); ++i)
        {
            auto shape = engine.get_input_shape(i);
            input_shapes.push_back({shape.data, shape.data + shape.size});
        }
        REQUIRE(input_shapes == std::vector<std::vector<size_t>>{{0, 0}, {0, 0}});
    }

    {
        std::vector<std::vector<size_t>> output_shapes;
        for (auto i = 0; i < engine.get_output_count(); ++i)
        {
            auto shape = engine.get_output_shape(i);
            output_shapes.push_back({shape.data, shape.data + shape.size});
        }
        REQUIRE(output_shapes == std::vector<std::vector<size_t>>{{0, 0}});
    }

    {
        std::vector<std::vector<size_t>> input_shapes{{2, 1}, {1, 2}};
        for (auto i = 0; i < engine.get_input_count(); ++i)
        {
            unwrap(engine.set_input_shape(i, input_shapes[i].data(), input_shapes[i].size()));
        }
    }

    {
        std::vector<std::vector<size_t>> input_shapes;
        for (auto i = 0; i < engine.get_input_count(); ++i)
        {
            auto shape = engine.get_input_shape(i);
            input_shapes.push_back({shape.data, shape.data + shape.size});
        }
        REQUIRE(input_shapes == std::vector<std::vector<size_t>>{{2, 1}, {1, 2}});
    }

    {
        std::vector<std::vector<size_t>> output_shapes{{2, 2}};
        for (auto i = 0; i < engine.get_output_count(); ++i)
        {
            unwrap(engine.set_output_shape(i, output_shapes[i].data(), output_shapes[i].size()));
        }
    }

    {
        std::vector<std::vector<size_t>> output_shapes;
        for (auto i = 0; i < engine.get_output_count(); ++i)
        {
            auto shape = engine.get_output_shape(i);
            output_shapes.push_back({shape.data, shape.data + shape.size});
        }
        REQUIRE(output_shapes == std::vector<std::vector<size_t>>{{2, 2}});
    }

    std::vector<std::vector<float>> inputs{{1, 2}, {3, 4}};
    for (auto i = 0; i < engine.get_input_count(); ++i)
    {
        unwrap(engine.set_input_data(i, inputs[i].data()));
    }

    std::vector<std::vector<float>> outputs{{0, 0, 0, 0}};
    for (auto i = 0; i < engine.get_output_count(); ++i)
    {
        unwrap(engine.set_output_data(i, outputs[i].data()));
    }

    unwrap(engine.run());

    REQUIRE(outputs == std::vector<std::vector<float>>{{{3, 4, 6, 8}}});
}
