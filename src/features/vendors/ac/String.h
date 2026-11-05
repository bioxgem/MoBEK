#ifndef LIB_STRING_H
#define LIB_STRING_H

#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>

namespace STRING {
    using namespace std;

    string getPrefix(const string& s)
    {
        int pos = s.find_last_of('.');
        if( pos > 0 )
            return s.substr(0,pos);
        else return s;
    }

    string getSuffix(const string& s)
    {
        size_t pos = s.find_last_of('.');
        if( pos == string::npos )
            return s;
        return s.substr(pos);
    }

    string RemovePath(const string& s)
    {
        std::string::size_type pos = s.find('/');
        if( pos == std::string::npos )
        {   // no '/' contained
            std::string::size_type pos2 = s.find('\\');
            if( pos2==std::string::npos ) {
                // no '\\' contained
                return s;
                //return s.substr(s.find_last_of('/')+1);
            }
            return s.substr(s.find_last_of('\\')+1);
        }
        return s.substr(s.find_last_of('/')+1);

        //string::size_type pos = s.find('\\');
        //if( pos == string::npos )
        //    return s.substr(s.find_last_of(Slash)+1);
        //return s.substr(s.find_last_of('\\')+1);
    }

    string getNoPathPrefix(const string& s)
    {
        return getPrefix(RemovePath(s));
    }

    template<typename T>  string Num2Str(const T &number)
    {
        ostringstream oss;
        oss << number;
        return oss.str();
    }

    template<typename T>  string itos(const T &number)
    {
        ostringstream oss;
        oss << setfill('0') << number;
        return oss.str();
    }
}

namespace FORMAT {
    using namespace std;
    bool isPDB(const string& file)
    {
        ifstream fin(file.c_str());
        if( not fin.good() )
        {
            return false;
            //cerr << "Cannot open pdb file: " << file << endl;
            //fin.clear();
            //string traditional_WCav="../WCavPDB/" + file;
            //fin.open( traditional_WCav.c_str() );
            //if( not fin.good() )
            //{
            //    fin.clear();
            //    string traditional_Ligand="../Ligand/" + file;
            //    fin.open( traditional_Ligand.c_str() );
            //    if( not fin.good() )
            //        return false;
            //}
        }

        string buffer;
        int num_HETATM=0;
        int num_CONNECT=0;
        //int num_SINGLE=0;
        int num_ATOM=0;
        while(getline(fin,buffer))
        {
            if(buffer.size()>=54 && num_HETATM<3)
            {
                if(buffer.substr(0,6)=="HETATM")
                {
                    bool have_atom=false;
                    string atom=buffer.substr(12,4);
                    for(size_t i=0;i<atom.size();i++)
                    {
                        if( isalpha( atom[i] ) )
                        {
                            have_atom = true;
                            //break;
                        }
                    }

                    double x = atof( buffer.substr(30,8).c_str() );
                    double y = atof( buffer.substr(38,8).c_str() );
                    double z = atof( buffer.substr(46,8).c_str() );
                    if( //have_atom and
                        x != 0.0 and
                        y != 0.0 and
                        z != 0.0    ) ++num_HETATM;
                }
                if(buffer.substr(0,4)=="ATOM")
                {
                    if( atof( buffer.substr(30,8).c_str() )!= 0.0 and
                        atof( buffer.substr(38,8).c_str() )!= 0.0 and
                        atof( buffer.substr(46,8).c_str() )!= 0.0    ) ++num_ATOM;
                }
            }
            getline(fin,buffer);
            if( buffer.size()>16 ) {
                if(buffer.substr(0,6)=="CONECT")
                {
                    if( atoi( buffer.substr(6,5).c_str() )!= 0 and
                        atoi( buffer.substr(11,5).c_str() )!= 0 ) ++num_CONNECT;
                    if( num_CONNECT > 1 ) break;
                }
                if( (num_HETATM>2 && num_CONNECT>1 ) || num_ATOM>2 ) break;
            }

            //if( buffer.size()>=16 ) {
            //    if(buffer.substr(0,6)=="SINGLE")
            //    {
            //        if( atoi( buffer.substr(6,5).c_str() )!= 0.0 and
            //            atoi( buffer.substr(11,5).c_str() )!= 0.0 ) ++num_SINGLE;
            //        if( num_SINGLE > 1 ) break;
            //    }
            //}

        }
        fin.close();
        if( (num_HETATM>1 or num_CONNECT>=1 ) or num_ATOM>2 )
            return true;
        else return false;
    }
}



#endif
