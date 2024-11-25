#ifndef __PHY_3DVARRAY__
#define __PHY_3DVARRAY__

#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include "GenArray.h"

using std::vector;
using std::endl;
using std::cout;
using std::stringstream;
using std::string;
using std::ifstream;
using std::ofstream;

namespace CBA {

class array3DV
{
public:
    void readConfig(ifstream &in);
    bool setParas(string s, string t);
    void addSelector();
    void removeSelector();
    bool debug;
    void build(string);
private:
    GenArray gen;
    int n, m, h;
    bool hasSelector, hasCapacity;
    vector<string> Vwl;
    vector<string> Vbl;
    vector<int> useSourceLine;
    vector<int> useBitline;
    vector<int> useWordline;

};


}


#endif