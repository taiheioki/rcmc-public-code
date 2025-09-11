#ifndef RCMC_NETWORK_HPP
#define RCMC_NETWORK_HPP

#include <algorithm>
#include <cassert>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

#include "numdef.hpp"

namespace rcmc
{

template<class R>
struct UndirectedNetwork
{
    struct Edge
    {
        int i, j;
        R value;

        bool operator<(const Edge& other) const
        {
            return std::tie(i, j) < std::tie(other.i, other.j);
        }
    };

    std::vector<R> vertices;
    std::set<Edge> edges;

    explicit UndirectedNetwork(const int n) : vertices(n) {}

    int num_vertices() const
    {
        return vertices.size();
    }

    int num_edges() const
    {
        return edges.size();
    }

    void add_edge(int i, int j, const R value)
    {
        assert(0 <= i && i < num_vertices());
        assert(0 <= j && j < num_vertices());
        if(i > j) {
            std::swap(i, j);
        }

        edges.insert({i, j, value});
    }
};

}  // namespace rcmc

#endif
