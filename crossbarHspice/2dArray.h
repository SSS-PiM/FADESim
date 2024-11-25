#ifndef __PHY_2DARRAY__
#define __PHY_2DARRAY__

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

typedef vector<double> vec;
typedef vector<vec> Mat;

class array2D: public phyArray
{
public:
    array2D(): fast_mode(false),  table_name("") {}
    bool setParas(string s, string t);

    void setArraySize(int n, int m) 
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
    void addSelector();
    void removeSelector();
    void buildSpice(string);
    void nodeBasedGS(int, double, int, bool, double, vec *);
    void nodeBasedGeneralGS(int, double, bool, double);
    void fastsolve(int, bool, bool, double);
    void IR_aihwkit(double ir_drop_beta, double vin_max);
    void IR_sciChina(double Rs);
    void IR_neurosim();
    void IR_free();
    void IR_PBIA(int times, bool, double);
    void IR_FCM(int mode, string file);

private:
    int n, m;
    //double line_resistance;
    bool hasSelector, hasCapacity;
    string capacitance[4];//Cw2g, Cb2g, Cw2w, Cb2b 
    vector<double> v[4];//up, left, down, right
    vector<string> v_str[4];
    vector<int> bv[4];
    double r_load[4];
    vector<vector<int>> arr;
    vector<vector<double>> RULine, RDLine;
    bool fast_mode;
    vector<double> cellR_states, inputV_states;
    int cellR_num;
    int inputV_num;
    string dc_ac_tran_type;
    bool useRtypeCell;
    string table_name;
};

}

#endif
