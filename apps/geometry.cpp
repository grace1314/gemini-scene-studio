#include "studio/reconstruction.hpp"
#include <iostream>
int main(int argc, char **argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("Usage: studio_geometry RUN_DIR");
        const auto out = studio::fs::canonical(argv[1]);
        const auto scene = out.parent_path().parent_path();
        const auto request = studio::read_json(out / "request.json");
        studio::write_json(out / "result.json", studio::execute_geometry(scene, out, request));
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR " << e.what() << std::endl;
        return 1;
    }
}
