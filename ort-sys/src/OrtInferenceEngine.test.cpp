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

std::vector<std::vector<size_t>> get_input_shapes(const OrtInferenceEngine &engine)
{
    std::vector<std::vector<size_t>> shapes;

    for (auto i = 0; i < engine.get_input_count(); ++i)
    {
        auto shape = engine.get_input_shape(i);
        shapes.push_back({shape.data, shape.data + shape.size});
    }

    return std::move(shapes);
}

void set_input_shapes(OrtInferenceEngine &engine, const std::vector<std::vector<size_t>> &shapes)
{
    for (auto i = 0; i < engine.get_input_count(); ++i)
    {
        unwrap(engine.set_input_shape(i, shapes[i].data(), shapes[i].size()));
    }
}

void set_input_data(OrtInferenceEngine &engine, const std::vector<std::vector<float>> &data)
{
    for (auto i = 0; i < engine.get_input_count(); ++i)
    {
        unwrap(engine.set_input_data(i, data[i].data()));
    }
}

std::vector<std::vector<size_t>> get_output_shapes(const OrtInferenceEngine &engine)
{
    std::vector<std::vector<size_t>> shapes;

    for (auto i = 0; i < engine.get_output_count(); ++i)
    {
        auto shape = engine.get_output_shape(i);
        shapes.push_back({shape.data, shape.data + shape.size});
    }

    return std::move(shapes);
}

void set_output_shapes(OrtInferenceEngine &engine, const std::vector<std::vector<size_t>> &shapes)
{
    for (auto i = 0; i < engine.get_output_count(); ++i)
    {
        unwrap(engine.set_output_shape(i, shapes[i].data(), shapes[i].size()));
    }
}

void set_output_data(OrtInferenceEngine &engine, std::vector<std::vector<float>> &data)
{
    for (auto i = 0; i < engine.get_output_count(); ++i)
    {
        unwrap(engine.set_output_data(i, data[i].data()));
    }
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
    REQUIRE(get_input_shapes(engine) == std::vector<std::vector<size_t>>{{0, 0}, {0, 0}});
    REQUIRE(get_output_shapes(engine) == std::vector<std::vector<size_t>>{{0, 0}});

    set_input_shapes(engine, {{2, 1}, {1, 2}});
    REQUIRE(get_input_shapes(engine) == std::vector<std::vector<size_t>>{{2, 1}, {1, 2}});

    set_output_shapes(engine, {{2, 2}});
    REQUIRE(get_output_shapes(engine) == std::vector<std::vector<size_t>>{{2, 2}});

    std::vector<std::vector<float>> inputs{{1, 2}, {3, 4}};
    set_input_data(engine, inputs);

    std::vector<std::vector<float>> outputs{{0, 0, 0, 0}};
    set_output_data(engine, outputs);

    unwrap(engine.run());
    REQUIRE(outputs == std::vector<std::vector<float>>{{{3, 4, 6, 8}}});
}
