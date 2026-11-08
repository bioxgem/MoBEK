#ifndef COMPOSITION_H
#define COMPOSITION_H

#include <dirent.h>
#include "String.h"
#include "Compound.h"


const string sep("\t");

void DFS(Compound& g,
         const int v,
         vector<bool>& visitV,
         vector<bool>& visitE,
         int source,
         bool &cyc)
{
	if( cyc ) return;
	vector<Edge> e = ( g.incidentEdges(v) );
	for(size_t i=0;i<e.size();++i) {
	    if(! visitE[ e[i].id ] ) {
			visitE[ e[i].id ] = true;
			int w = Opposite( v,e[i] );
			if( ! visitV[ w ] )	{
				visitV[ w ] = true;
				DFS(g,w,visitV,visitE,source,cyc);
			}
			else if(w == source) {
				cyc = true; return;
			}
		}
	}
}

void getCycleMarker(Compound& g, vector<bool>& ret)
{
	for(size_t i=0;i<g.AtomNumber();i++) {
	    vector<bool> visitE( g.BondNumber(), false );
        vector<bool> visitV( g.AtomNumber(), false );
		visitV[i] = true;
		bool cyc = false;
		DFS( g,i,visitV,visitE,i,cyc );
		if( cyc ) ret.push_back(true);
		else	  ret.push_back(false);
	}
}

string getCNOPSX_title()
{
    ostringstream oss;
	oss << "#Compound"<< sep
        << "C.ring" << sep
        << "C.other"<< sep
        << "N.ring" << sep
        << "N.other"<< sep
        << "O.ring" << sep
        << "O.other"<< sep
        << "P"      << sep
        << "S"      << sep
        << "X"      << sep
        << "#.of.Ring"<< endl;
    return oss.str();
}

string C_composition(Compound& molecular)
{
	float
         C_ring=0, C_other=0,
		 N_ring=0, N_other=0,
		 O_ring=0, O_other=0,
         P=0, S=0, X=0,
         Total=0.0000001;

	vector<string>	atom( molecular.getAtomTable() );
	vector<bool>    on_ring;

    getCycleMarker(molecular, on_ring);

	for(size_t i=0;i<atom.size();++i) {
        if(      atom[i] == "C" and   on_ring[i] )  C_ring++;
        else if( atom[i] == "C" and ! on_ring[i] )  C_other++;
        else if( atom[i] == "N" and	  on_ring[i] )  N_ring++;
        else if( atom[i] == "N" and ! on_ring[i] )  N_other++;
        else if( atom[i] == "O" and	  on_ring[i] )  O_ring++;
        else if( atom[i] == "O" and ! on_ring[i])   O_other++;
        else if( atom[i] == "P" )                   P++;
        else if( atom[i] == "S" )                   S++;
        else if( atom[i] == "Cl" or
                 atom[i] == "Br" or
                 atom[i] == "I" or
                 atom[i] == "At" )                  X++;

        if( atom[i] != "H" )                        Total++;
	}

    ostringstream oss;
    oss << C_ring   << sep
        << C_other  << sep
        << N_ring   << sep
        << N_other  << sep
        << O_ring   << sep
        << O_other  << sep
        << P        << sep
        << S        << sep
        << X        << sep
        << molecular.RingNumber()<<endl;
    return oss.str();
}

string get_composition(const string& file)
{
    ostringstream oss;
	string id( STRING::RemovePath(file) );
	Compound molecule;
    bool ret;
    if( FORMAT::isPDB( file ) ) {
        ret = molecule.read_pdb( file.c_str() );
        cout << "format is pdb" << endl;
    }
    else
        ret = molecule.read_mol( file.c_str() );
    if( ret ) {
    	if( molecule.BondNumber() >= molecule.AtomNumber()-1 )
    	{
	    	oss << id << sep;
		    oss << C_composition( molecule );
    	}
 	}
	else {
	    cerr << file << " is an abnormal mol file!! Please check the integraty of file" << endl;
        cerr.flush();
	}
    return oss.str();
}

void ACgenByDir(const string& dir_name, const string& outputdir=".")
{
	DIR *dir;
	struct dirent *dp;
	if ((dir = opendir( dir_name.c_str() )) == NULL) {
		cerr << "Cannot open dir: " << dir << endl;
		exit(1);
	}
	int file_number=0;
	while ((dp = readdir(dir)) != NULL) {
		file_number++;
	}
	closedir(dir);
	cerr << file_number-2 << " files in target directory" << endl;
    cerr.flush();

	dir = opendir( dir_name.c_str() );
	int counter = 1;
	while ((dp = readdir(dir)) != NULL) {
		counter++;
		cerr.precision(5);
		cerr << "Percentage: "<< ((double)counter/(double)file_number)*100 << "%\r";
        cerr.flush();

		string file_name(dp->d_name);
		if( file_name[0] == '.' ) continue;

		if( file_name.size() > 5 and file_name.substr(file_name.size()-4)==".mol" )
		{
			Compound mol;
            mol.read_mol( dir_name+"/"+file_name );

			if( mol.BondNumber() >= mol.AtomNumber()-1	) {
				cout << file_name << sep;
                cout.flush();
				cout << C_composition( mol );
			}
			else {
				cerr << "Error: " << dp->d_name << " is an abnormal mol file!! Please check the integraty of file\n";
                cerr.flush();
			}
		}
	} // end of readdir
	closedir(dir);
}


void ACgen(const string& input_file, string& output_file)
{
    if( output_file.size() == 0 )
        output_file = "ac-"+STRING::getNoPathPrefix(input_file)+".txt" ;

    cout << "Generating composition descriptor to " << output_file << endl;

    ofstream fout( output_file.c_str() );
    fout << getCNOPSX_title();

    ifstream fin( input_file.c_str() );
    if( not fin.good() ) {
        cerr << "Cannot open file " << input_file << endl;
        exit(EXIT_FAILURE);
    }

    string molecule;
    while( getline(fin,molecule) )
    {
        if(molecule.size()>1 && molecule[0] != '#' )
        {
            fout << get_composition( molecule );
        }
    }
    cout << "done!" << endl;
}

void ACgenSingle(const string& s, const string& output_file)
{
    cout << getCNOPSX_title();
    cout << get_composition( s );
}

bool isLST(const string& file)
{
    ifstream fin(file.c_str());
    if( not fin.good() )
        return false;

    int  error_count = 0;
    string buffer;
    while( getline(fin,buffer) ) {
        if( buffer.size()>0 && buffer[0] != '#' ) {
            ifstream fin2( buffer.c_str() );
            if( not fin2.good() ) {
                if(++error_count>5) {
                    return false;
                }
            }
            fin2.close();
        }
    }
    fin.close();
    return true;

}

#endif
