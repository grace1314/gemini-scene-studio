#pragma once
#include <filesystem>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace studio {
namespace fs = std::filesystem;
using json = nlohmann::json;
json read_json(const fs::path &path);
void write_json(const fs::path &path, const json &value);
std::string read_text(const fs::path &path);
std::string sha256(const fs::path &path);
std::string id();
double now();
bool contained(const fs::path &child, const fs::path &parent);
fs::path scene_path(const fs::path &root, const std::string &scene);
std::vector<json> runs(const fs::path &scene);
json active_runs(const fs::path &scene);
json validate_config(const std::string &stage, json config);
void verify_inputs(const fs::path &scene);
extern const std::map<std::string, std::vector<std::string>> dependencies;
} // namespace studio
