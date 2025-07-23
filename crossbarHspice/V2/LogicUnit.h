#ifndef __LOGICUNIT__
#define __LOGICUNIT__ 

#include <vector>
#include <iostream>
#include <string>
#include <cstdio>
#include <map>
#include <set>

using std::set;
using std::map;
using std::pair;
using std::string;
using std::multimap;
using std::to_string;
using std::ostream;
using std::vector;
using std::endl;
using std::make_pair;

namespace cba {

const int startLinkNo = 1;

enum struct UnitType: uint8_t
{
    none,
    linearR,
    selector,
    ReRAM,
    transistor,
    voltage,
    voltage_pp,
    current,
    senseV,
    senseI,
    capacity,
};

class LogicUnit
{
public:
    LogicUnit(UnitType t = UnitType::none, int No = -1): \
        type(t), unitNo(No) {}

    LogicUnit(UnitType t, const vector<int> &lk, const vector<string> &p = {}, int No = -1): \
        type(t), unitNo(No) 
    {
        link = lk;
        paras = p;
    }

    void setType(UnitType t);
    UnitType getType()
    {
        return type;
    }

    vector<string> getParas()
    {
        return paras;
    }

    void setLink(const vector<int> &lk);
    void setParas(const vector<string> &p);
    void setNo(int No);

    void clearLink();
    void clearParas();
    void clear();

    void addLink(int num);
    void addPara(const string &v);
    //bool operator ==(const LogicUnit &lu) const;
    void print(ostream &o, vector<string> *addition = nullptr, bool *dc_print = nullptr);
    friend void changeUnit();
    friend class GenArray;
    void checkPrintI(int x, int y, const string &str);

private:
    static int printCnt[20];
    UnitType type;
    int unitNo;
    static multimap<pair<int, int>, string> unitNameForPrintI; 
    static set<pair<int, int>> needPrintI;
    vector<int> link;
    vector<string> paras;
};

string toStr(double x, int y=6);

string toStr(int x);


}
#endif
