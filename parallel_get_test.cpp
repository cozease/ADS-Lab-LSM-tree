#include <iostream>
#include <fstream>
#include <chrono>

#include "kvstore.h"

const int TEST_CASES = 1000;

std::vector<std::string> load_text(const std::string &filename) {
    std::ifstream file(filename);
    std::vector<std::string> texts;
    std::string line;
    for (int i = 0; i < TEST_CASES; ++i) {
        std::getline(file, line);
        texts.push_back(line);
    }
    return texts;
}

int main() {
    KVStore store("data/");

    // 加载答案
    std::vector<std::string> ans = load_text("data/cleaned_text_100k.txt");

    // 测试普通的 get 操作
    std::cout << "--- Normal Get Test ---" << std::endl;
    bool flag = true;
    long long duration = 0;
    for (int i = 0; i < TEST_CASES; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        std::string result = store.get(i);
        auto end = std::chrono::high_resolution_clock::now();
        duration += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        if (result != ans[i]) {
            std::cout << "Normal test failed at key " << i << "!" << std::endl;
            flag = false;
        }
    }
    if (flag) {
        std::cout << "Normal get test passed." << std::endl;
        std::cout << "Total normal get time: " << duration << "us" << std::endl;
        std::cout << "Average normal get time: " << (double)duration / TEST_CASES << "us" << std::endl;
    }
    
    std::cout << std::endl;

    // 测试并行的 get 操作
    std::cout << "--- Parallel Get Test ---" << std::endl;
    flag = true;
    duration = 0;
    for (int i = 0; i < TEST_CASES; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        std::string result = store.parallel_get(i);
        auto end = std::chrono::high_resolution_clock::now();
        duration += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        if (result != ans[i]) {
            std::cout << "Parallel test failed at key " << i << "!" << std::endl;
            // std::cout << "Expected: " << ans[i] << ", Got: " << result << std::endl;
            flag = false;
        }
    }
    if (flag) {
        std::cout << "Parallel get test passed." << std::endl;
        std::cout << "Total parallel get time: " << duration << "us" << std::endl;
        std::cout << "Average parallel get time: " << (double)duration / TEST_CASES << "us" << std::endl;
    }

    return 0;
}