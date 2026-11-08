#ifndef COMPOUND_H
#define COMPOUND_H

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <string>
#include "Edge.h"
using namespace std;

class Compound {
    vector<string>        m_atoms;  // must read into this variable
    vector<vector<int> >  m_bonds;  // must read into this variable
    vector<Edge>	      m_edges;
    vector<Edge>          m_adj_edges;
    vector<int>           m_adj_nodes;

public:
    bool    read_mol(const string& file);
    bool    read_mol2(const string& file) { return false;}
    bool    read_pdb(const string& file);

    size_t	BondNumber()    { return m_edges.size(); }
    size_t	AtomNumber()    { return m_atoms.size(); }
    size_t	RingNumber()	{ return m_edges.size() - m_atoms.size() + 1; }

    void setAtomTable(const vector<string>& a)  { m_atoms = a; }
    void setEdgeTable(const vector<Edge>& e)    { m_edges = e; }
    void setBondTable(const vector<vector<int> >& b) { m_bonds=b; }
    vector<string>	        getAtomTable() { return m_atoms; }
    vector<vector<int> >    getBondTable() { return m_bonds; }
    vector<Edge>	        getEdgeTable() { return m_edges; }

    vector<int>& 	incidentVertices( const int v ) {
        m_adj_nodes.clear();
        for(size_t i=0;i<m_bonds[v].size();++i)
            if(m_bonds[v][i] == 1)
                m_adj_nodes.push_back( i );
        return m_adj_nodes;
    }
    vector<Edge>&	incidentEdges( const int v ) {
        m_adj_edges.clear();
        for(size_t i=0;i<m_edges.size();++i)
            if( m_edges[i].hasVertex(v) )
                m_adj_edges.push_back( m_edges[i] );
        return m_adj_edges;
    }

    ~Compound() {
        m_atoms.clear();
        m_bonds.clear();
        m_edges.clear();
        m_adj_edges.clear();
        m_adj_nodes.clear();
    }
};


bool Compound::read_mol(const string& file)
{
    ifstream fin( file.c_str() );
    if( ! fin ) {
        //cerr << "Error: " <<  file << " doesn't exist" << endl;
        return false;
    }
    bool header=false;
    string buffer;
    size_t atom_number=0, bond_number=0;
    while( getline(fin,buffer) )
    {
        if( buffer.size() >=6 )
        {
            atom_number = atoi( buffer.substr(0,3).c_str() );
            bond_number = atoi( buffer.substr(3,3).c_str() );
            if( atom_number>1 and bond_number>0 ) {
                header=true;
                break;
            }
        }
    }

    //cout << "Header=" << atom_number << ", " << bond_number << endl;

    if( not header ) {
        cerr << "Error: In "<< file <<" file, no header record exist!" << endl;
        return false;
    }

    m_atoms.clear();
    for(size_t i=0;i<atom_number;++i)		// save atom info
    {
        getline(fin,buffer);
        istringstream sin( buffer.substr(31) );
        string atom_type;
        sin >> atom_type;
        m_atoms.push_back( atom_type );
    }

    m_bonds.clear();
    vector<int> temp( atom_number,999999 );// temp size = atom_number, default value = 999999
    for(size_t i=0;i<atom_number;++i)
        m_bonds.push_back( temp );

    //cout <<"#atom:" << atom_number << endl;
    //cout <<"#bond:" << bond_number << endl;
    for(size_t i=0;i<bond_number;++i)
    {
        getline(fin,buffer);
        if( buffer.empty() )
            break;
        int gg = atoi(buffer.substr(0,3).c_str());
        int yy = atoi(buffer.substr(3,3).c_str());
        m_bonds[gg-1][yy-1] = 1;
        m_bonds[yy-1][gg-1] = 1;
        m_edges.push_back( Edge(gg-1, yy-1, i) );
    }
    fin.close();
    return true;
}

bool Compound::read_pdb(const string& file)  {

    ifstream fin( file.c_str() );
    string buffer;
    map< string,int > number_to_vector;
    map< int,map< int,int > > edge_check;
    vector<int> temp;
    int edge_count = 0;
    size_t atom_number = 0;
    if( ! fin ) {
        return false;
    }
    m_atoms.clear();
    m_bonds.clear();
    while( getline(fin,buffer) )
	{
        if( buffer.substr( 0,6 ) != "HETATM" && buffer.substr( 0,6 ) != "CONECT" && buffer.substr( 0,4 ) != "ATOM")
		    continue;
		if( buffer.substr( 0,6 ) == "HETATM" || buffer.substr( 0,4 ) == "ATOM")
		{
		    istringstream sin( buffer );
      	    string number;
      	    string atom_type;
	        sin >> number;
  	        sin >> number;
  	        sin >> atom_type;
    	    m_atoms.push_back( atom_type );
    	    number_to_vector[number] = m_atoms.size()-1;
    	    atom_number++;
        }
        if( buffer.substr( 0,6 ) == "CONECT" )
        {
            if( atom_number != 0 )
            {
                vector<int>temp( atom_number,999999 );// temp size = atom_number, default value = 999999
                for(size_t i=0;i<atom_number;++i)
                    m_bonds.push_back( temp );
                atom_number = 0;
            }
            istringstream sin( buffer );
            string first;
            string connect;
            sin >> first;
            sin >> first;
            while( sin >> connect )
            {
                m_bonds[ number_to_vector[first] ][ number_to_vector[connect] ] = 1;
                if( edge_check[ number_to_vector[first] ][ number_to_vector[connect] ] != 1 &&
                    edge_check[ number_to_vector[connect] ][ number_to_vector[first] ] != 1 )
                {
                    edge_check[ number_to_vector[connect] ][ number_to_vector[first] ] = 1;
                    edge_check[ number_to_vector[first] ][ number_to_vector[connect] ] = 1;
                    m_edges.push_back( Edge(number_to_vector[first], number_to_vector[connect], edge_count) );
                    edge_count++;
                }
            }
        }
    }
	fin.close();
    return true;
}



#endif
