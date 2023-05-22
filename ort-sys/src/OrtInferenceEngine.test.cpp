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

TEST_CASE("OrtInferenceEngine with invalid model data")
{
    auto result = OrtInferenceEngine::create(nullptr, 0);
    REQUIRE(result.code == ResultCode::Error);
    REQUIRE(std::string(result.error.get_message()) == "No graph was found in the protobuf.");
}

TEST_CASE("OrtInferenceEngine with fixed-shape model")
{
    auto model = read_file(PROJECT_DIR / "../ort-cpp/test-models/mat_mul.onnx");
    auto result = OrtInferenceEngine::create(model.data(), model.size());
    REQUIRE(result.code == ResultCode::Ok);

    auto engine = std::move(result.value);
    REQUIRE(engine.get_input_count() == 2);
    REQUIRE(engine.get_output_count() == 1);

    std::vector<std::vector<size_t>> input_shapes;
    for (auto i = 0; i < engine.get_input_count(); ++i)
    {
        auto shape = engine.get_input_shape(i);
        input_shapes.push_back({shape.data, shape.data + shape.size});
    }
    REQUIRE(input_shapes == std::vector<std::vector<size_t>>{{2, 2}, {2, 2}});

    std::vector<std::vector<size_t>> output_shapes;
    for (auto i = 0; i < engine.get_output_count(); ++i)
    {
        auto shape = engine.get_output_shape(i);
        output_shapes.push_back({shape.data, shape.data + shape.size});
    }
    REQUIRE(output_shapes == std::vector<std::vector<size_t>>{{2, 2}});
}
