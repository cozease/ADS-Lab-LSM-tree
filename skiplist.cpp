#include "skiplist.h"

#include "utils.h"

double skiplist::my_rand() {
    return (double)rand() / RAND_MAX;
}

int skiplist::randLevel() {
    int level = 1;
    while (my_rand() < p && level < MAX_LEVEL)
        level++;
    return level;
}

void skiplist::insert(uint64_t key, const std::string &str) {
    int level = randLevel();
    if (level > curMaxL)
        curMaxL = level;
    std::vector<slnode *> update(level);
    slnode *p = head;
    for (int i = level - 1; i >= 0; --i) {
        while (p->nxt[i]->key < key) p = p->nxt[i];
        update[i] = p;
    }
    if (p->nxt[0]->key == key) {
        bytes += str.length() - p->nxt[0]->val.length();
        p->nxt[0]->val = str;
        if (str == "~DELETE~")
            p->nxt[0]->vec = std::vector<float>(dim, std::numeric_limits<float>::max());
        else
            p->nxt[0]->vec = embedding_single(str);
        return;
    }
    slnode *newNode = new slnode(key, str, NORMAL);
    for (int i = 0; i < level; ++i) {
        newNode->nxt[i] = update[i]->nxt[i];
        update[i]->nxt[i] = newNode;
    }
    bytes += 12 + str.length();
}

std::string skiplist::search(uint64_t key) {
    slnode *p = head;
    for (int i = curMaxL - 1; i >= 0; --i) {
        while (p->nxt[i]->key < key) p = p->nxt[i];
        if (p->nxt[i]->key == key) return p->nxt[i]->val;
    }
    return "";
}

std::vector<float> skiplist::search_vec(uint64_t key) {
    slnode *p = head;
    for (int i = curMaxL - 1; i >= 0; --i) {
        while (p->nxt[i]->key < key) p = p->nxt[i];
        if (p->nxt[i]->key == key) return p->nxt[i]->vec;
    }
    return {};
}

bool skiplist::del(uint64_t key, uint32_t len) {

}

void skiplist::scan(uint64_t key1, uint64_t key2, std::vector<std::pair<uint64_t, std::string>> &list) {
    if (key1 >= key2) return;

    slnode *start = head, *end = head;
    for (int i = curMaxL - 1; i >= 0; --i) {
        while (start->nxt[i]->key < key1) start = start->nxt[i];
        while (end->nxt[i]->key < key2) end = end->nxt[i];
    }
    if (end->nxt[0]->key == key2) end = end->nxt[0];
    for (slnode *p = start; p != end; p = p->nxt[0]) {
        list.push_back(std::make_pair(p->nxt[0]->key, p->nxt[0]->val));
    }
}

struct memPair {
    uint64_t key;
    std::string val;
    float sim;
    memPair(uint64_t key, const std::string &val, float sim) {
        this->key = key;
        this->val = val;
        this->sim = sim;
    }
    bool operator>(const memPair &other) const {
        return sim > other.sim;
    }
};

std::vector<std::pair<float, std::pair<uint64_t, std::string>>> skiplist::search_knn(
    std::vector<float> query_vec,
    int k,
    const std::vector<std::pair<uint64_t, std::vector<float>>> &deleted_nodes
) {
    std::priority_queue<memPair, std::vector<memPair>, std::greater<memPair>> heap;
    slnode *p = head->nxt[0];
    while (p != tail) {
        // skip deleted nodes
        if (std::find_if(deleted_nodes.begin(), deleted_nodes.end(),
            [&p](const std::pair<uint64_t, std::vector<float>> &node) {
                bool flag = true;
                const float epsilon = 1e-6f;
                for (size_t i = 0; i < dim; ++i) {
                    if (fabs(node.second[i] - p->vec[i]) > epsilon) {
                        flag = false;
                        break;
                    }
                }
                return flag && node.first == p->key;
            }) != deleted_nodes.end()) {
            p = p->nxt[0];
            continue;
        }

        float sim = common_embd_similarity_cos(query_vec.data(), p->vec.data(), query_vec.size());
        if (heap.size() < k) {
            heap.push(memPair(p->key, p->val, sim));
        } else {
            if (sim > heap.top().sim) {
                heap.pop();
                heap.push(memPair(p->key, p->val, sim));
            }
        }
        p = p->nxt[0];
    }
    std::vector<std::pair<float, std::pair<uint64_t, std::string>>> res;
    while (!heap.empty()) {
        memPair cur = heap.top();
        heap.pop();
        res.push_back(std::make_pair(cur.sim, std::make_pair(cur.key, cur.val)));
    }
    return res;
}

void skiplist::hnsw_insert_all(HNSW &hnsw) {
    slnode *p = head->nxt[0];
    while (p != tail) {
        hnsw.insert(p->key, p->vec);
        p = p->nxt[0];
    }
}

slnode *skiplist::lowerBound(uint64_t key) {

}

void skiplist::reset() {
    slnode *p = head->nxt[0];
    while (p != tail) {
        slnode *q = p;
        p = p->nxt[0];
        delete q;
    }
    bytes = 0;
    curMaxL = 1;
    for (int i = 0; i < MAX_LEVEL; ++i) {
        head->nxt[i] = tail;
    }
}

uint32_t skiplist::getBytes() {
    return bytes;
}

void skiplist::putEmbeddingFile() {
    // 如果文件不存在，则新建并写入文件头
    std::string path = "./embedding_data/";
    std::string filename = path + "embedding_data.bin";
    if (!utils::dirExists(path)) {
        utils::mkdir(path.data());
        FILE *file = fopen(filename.c_str(), "wb");
        fseek(file, 0, SEEK_SET);
        fwrite(&dim, 8, 1, file);
        fflush(file);
        fclose(file);
    }

    // 写入当前 skip list 中的所有 embedding
    FILE *file = fopen(filename.c_str(), "ab+");
    fseek(file, 8, SEEK_SET);
    slnode *p = head->nxt[0];
    while (p != tail) {
        fwrite(&p->key, 8, 1, file);
        fwrite(p->vec.data(), 4, dim, file);
        p = p->nxt[0];
    }
}
