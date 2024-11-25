#ifndef __PHY_3DHARRAY__
#define __PHY_3DHARRAY__

#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include "GenArray.h"
#include <cassert>

using std::vector;
using std::endl;
using std::cout;
using std::stringstream;
using std::string;
using std::ifstream;
using std::ofstream;

namespace CBA {

class array3DH: public phyArray
{
public:
    bool setParas(string s, string t);
    void addSelector();
    void removeSelector();
    void setArraySize(int k, int n, int m)
    {
        this->n = n, this->m = m, this->h = k;
        arr = vector<vector<vector<int>>>(h, vector<vector<int>>(n, vector<int>(m)));
        useLeftWordline = useRightWordline = vector<vector<int>>((h+1)/2, vector<int>(n));
        useUpBitline = useDownBitline = vector<vector<int>>(h/2+1, vector<int>(m));

        VleftWordline = VrightWordline = vector<vector<string>>((h+1)/2, vector<string>(n));
        VupBitline = VdownBitline = vector<vector<string>>(h/2+1, vector<string>(m));

        RwlLine =vector<vector<vector<double>>>((h+1)/2, vector<vector<double>>(n, vector<double>(m+1)));
        RblLine =vector<vector<vector<double>>>(h/2+1, vector<vector<double>>(m, vector<double>(n+1)));

    }   

private:
    int n, m, h; // number of wordline, bitline, high
    vector<vector<vector<int>>> arr;
    bool hasSelector, hasCapacity;

    vector<vector<string>> VleftWordline;
    vector<vector<string>> VrightWordline;
    vector<vector<string>> VupBitline;
    vector<vector<string>> VdownBitline;

    vector<vector<int>> useLeftWordline;
    vector<vector<int>> useRightWordline;
    vector<vector<int>> useUpBitline;
    vector<vector<int>> useDownBitline;
    vector<vector<vector<double>>> RblLine, RwlLine;
    bool debug;
    void build(string);
    int getUp(int ht, int y);
    int getLeft(int ht, int x);
    int getDown(int ht, int y);
    int getRight(int ht, int x);


};


}


#endif
