#include <iostream>
#include <fstream>
#include <sstream>

#include "kvstore.h"

std::vector<std::vector<float>> load_embedding(const std::string &filename) {
    std::ifstream file(filename);
    std::vector<std::vector<float>> vecs;

    std::string line;
    while (std::getline(file, line)) {
        std::vector<float> values;
        std::istringstream ss(line);
        float value;
        char c;

        ss.ignore(1);
        while (ss >> value >> c)
            values.push_back(value);
        
        vecs.push_back(values);
    }

    return vecs;
}

std::vector<std::string> load_text(const std::string &filename) {
    std::ifstream file(filename);
    std::vector<std::string> texts;

    std::string line;
    while (std::getline(file, line))
        texts.push_back(line);
    
    return texts;
}

int main() {
    KVStore store("data/");
    store.reset();

    std::vector<std::string> texts = load_text("data/cleaned_text_100k.txt");
    std::vector<std::vector<float>> vecs = load_embedding("data/embedding_100k.txt");
    
    int total = 100000;
    for (int i = 0; i < total; i++) {
        store.put_with_embedding(i, texts[i], vecs[i]);
        if (i % 10000 == 0) std::cout << "Inserted " << i << " items." << std::endl;
    }
    
    store.save_hnsw_index_to_disk("hnsw_data/");

    return 0;
}
