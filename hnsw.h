#ifndef HNSW_H
#define HNSW_H

#include <vector>

class HNSWNode {
public:
    int level;  // 当前节点的层级
    uint64_t key;
    std::string val;
    std::vector<float> vec;
    
    // 每层中的邻居节点和与该邻居节点的距离（注意：这里的距离是余弦相似度，余弦相似度越大，距离越小）
    std::vector<std::vector<std::pair<float, HNSWNode *>>> neighbors;

    HNSWNode(int level, uint64_t key, const std::string &val, const std::vector<float> &vec)
        : level(level), key(key), val(val), vec(vec) {
        neighbors.resize(level + 1);
    }

    bool operator<(const HNSWNode &other) const {
        return true;
    }
    bool operator>(const HNSWNode &other) const {
        return true;
    }
};

class HNSW {
private:
    int M;
    int M_max;
    int efConstruction;
    int m_L;

    HNSWNode *entry_point;

public:
    HNSW() {}
    HNSW(int M, int M_max, int efConstruction, int m_L)
        : M(M), M_max(M_max), efConstruction(efConstruction), m_L(m_L) {
        entry_point = nullptr;
    }

    int rand_level();

    void insert(uint64_t key, const std::string &val, const std::vector<float> &vec);
    std::vector<std::pair<uint64_t, std::string>> search(std::string query, int k);
};

#endif // HNSW_H
