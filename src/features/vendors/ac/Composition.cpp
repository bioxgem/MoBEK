#include "Composition.h"
#include <getopt.h>

void usage( const string& baseName )
{
    cerr << "Usage : " << baseName << " [OPTIONS] FILE" << endl;
    cerr << "  OPTIONS"<< endl;
    cerr << "   -o output_file  - specify output file name as output_file" << endl;
    exit(0);
}

int main(int argc, char* argv[])
{
    bool is_dir = false;
    string input_file;
    string output_file;

    int opt;
    while( (opt=getopt( argc, argv, "do:")) != -1 ) {
        switch(opt) {
            case 'o':   output_file=optarg; break;
            case 'd':   is_dir = true;      break;
            default: usage(argv[0]);
        }
    }
    if( optind < argc ) {
        input_file = argv[optind];
    }
    else
        usage(argv[0]);

    if( isLST(input_file) ) {
        cout << "list" << endl;
        ACgen( input_file, output_file );
    }
    else {
        ACgenSingle( input_file, output_file );
    }
}

