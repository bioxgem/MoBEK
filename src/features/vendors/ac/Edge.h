#ifndef EDGE_H
#define EDGE_H

#include <iostream>
using namespace std;

struct Edge {
    int	first;
    int second;
    int	id;

    Edge()                      : first(0), second(0){}
    Edge(const Edge& e)         : first(e.first), second(e.second), id(e.id) {}
    Edge(int f, int s, int i)   : first(f), second(s), id(i) {}

    bool operator==(const Edge& comp) const {
        return  (this->first==comp.first and this->second==comp.second) or
                (this->first==comp.second and this->second==comp.first);
    }

    bool operator!=(const Edge& comp) const {
        return  (this->first!=comp.first and this->second!=comp.second) or
                (this->first!=comp.second and this->second!=comp.first);
    }

    bool hasVertex(const int& vertex) {
        return this->first==vertex or this->second==vertex;
    }

    int opposite(const int q) {
        if( q==first ) return second;
        else if( q==second ) return first;
        else {
            cerr << "Vertex "<< q+1 << " has no opposite vertex" << endl;
            return -1;
        }
    }
};

ostream& operator<<(ostream& os, const Edge& edge)
{
    os << "(" << edge.first+1 << " " << edge.second+1 << ")";
    return os;
}

int Opposite( const int v, const Edge& e )
{
    if( v==e.first )     return e.second;
    else if( v==e.second) return e.first;
    else {
        cerr << "Vertex "<< v+1 << " has no opposite vertex" << endl;
        exit(EXIT_FAILURE);
    }
}

#endif
