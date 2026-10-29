#include "studio/core.hpp"
#include <cassert>
#include <iostream>
using namespace studio;
int main() {
    assert(contained("/tmp/studio/a", "/tmp/studio"));
    assert(!contained("/tmp/studio-other/a", "/tmp/studio"));
    assert(!contained("/tmp/studio/../private", "/tmp/studio"));
    assert(validate_config("depth", json::object())["width"] == 1280);
    assert(validate_config("poses", {{"backend", "colmap_rgb"}})["backend"] == "colmap_rgb");
    bool rejected = false;
    try {
        validate_config("depth", {{"min_depth", 2}, {"max_depth", 1}});
    } catch (...) {
        rejected = true;
    }
    assert(rejected);
    rejected = false;
    try {
        validate_config("train", {{"iterations", 1.5}});
    } catch (...) {
        rejected = true;
    }
    assert(rejected);
    rejected = false;
    try {
        validate_config("tsdf", {{"voxel_size", 0.01}, {"sdf_trunc", 0.001}});
    } catch (...) {
        rejected = true;
    }
    assert(rejected);
    auto temp = fs::temp_directory_path() / ("studio-test-" + id());
    write_json(temp / "manifest.json", {{"revision", 1}});
    auto run = [&](const char *i, const char *stage, double t, json deps) {
        write_json(temp / "runs" / i / "status.json", {{"id", i},
                                                       {"stage", stage},
                                                       {"revision", 1},
                                                       {"state", "succeeded"},
                                                       {"created_at", t},
                                                       {"dependencies", deps}});
    };
    run("a", "depth", 1, json::object());
    run("b", "poses", 2, {{"depth", "a"}});
    run("c", "cloud", 3, {{"depth", "a"}, {"poses", "b"}});
    assert(active_runs(temp).contains("cloud"));
    run("d", "depth", 4, json::object());
    assert(!active_runs(temp).contains("poses"));
    assert(!active_runs(temp).contains("cloud"));
    // Remove only this newly created, uniquely identified test directory.
    fs::remove_all(temp);
    std::cout << "core invariants passed\n";
}
