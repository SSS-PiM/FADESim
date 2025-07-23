#ifndef __PHY_2DARRAY__
#define __PHY_2DARRAY_

#include "abstract_array.h"
#include <vector>
#include <string>
#include <map>
#include "GenArray.h"

using std::vector;
using std::string;

namespace cba {

class Array2D: public AbstractArray
{
public:
    void setSize(int n, int m)
    {
        this->n = n, this->m = m;
        arr = vector<vector<int>>(n, vector<int>(m));
        RULine = vector<vector<double>>(n, vector<double>(m+1));
        RDLine = vector<vector<double>>(m, vector<double>(n+1));
        v[0] = v[2] = vector<double>(m);
        v[3] = v[1] = vector<double>(n);

        v_str[0] = v_str[2] = vector<string>(m);
        v_str[3] = v_str[1] = vector<string>(n);

        bv[2] = bv[0] = vector<int>(m);
        bv[3] = bv[1] = vector<int>(n);
        for (int i=0; i<4; ++i)
            r_load[i] = -1;       
    }

    std::pair<int, int> size()
    {
        return std::make_pair(n, m);
    }

    bool hasSelector = false, hasCapacity = false;
    string capacitance[4];//Cw2g, Cb2g, Cw2w, Cb2b 
    vector<double> v[4];//up, left, down, right
    vector<string> v_str[4];
    vector<int> bv[4];  // value = 0 -> not used, 1-> used, 2-> special usage of not constant voltage
    double r_load[4];
    vector<vector<int>> arr;
    vector<vector<double>> RULine, RDLine;
    bool fast_mode;
    vector<double> cellR_states, inputV_states;
    int cellR_num;
    int inputV_num;
    string dc_ac_tran_type = "dc";
    bool useRtypeCell = true;
    string table_name;
    GenArray gen;
};          

}
#endif