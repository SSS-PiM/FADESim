#include "LogicUnit.h"

using namespace CBA;


string CBA::toStr(double x, int y)
{
    char tmp[20];
    string s = "%." + to_string(y)+"f";
    snprintf(tmp, 18, s.c_str(), x);
    return string(tmp);
}

string CBA::toStr(int x)
{
    return to_string(x);
}

void LogicUnit::setType(UnitType t)
{
    type = t;
}

void LogicUnit::addLink(int num)
{
    link.push_back(num);
}

void LogicUnit::setNo(int No)
{
    unitNo = No;
}

void LogicUnit::addPara(const string &v)
{
    paras.push_back(v);
}

void LogicUnit::setLink(const vector<int> &lk)
{
    link = lk;
}

void LogicUnit::setParas(const vector<string> &p)
{
    paras = p;
}

// bool LogicUnit::operator ==(const LogicUnit &lu) const
// {

// }

void LogicUnit::clearLink()
{
    link.clear();
}

void LogicUnit::clearParas()
{
    paras.clear();
}

int LogicUnit::printCnt[20]={0};
multimap<pair<int, int>, string> LogicUnit::unitNameForPrintI;
set<pair<int, int>> LogicUnit::needPrintI;

void LogicUnit::clear()
{
    clearLink();
    clearParas();
}

void LogicUnit::checkPrintI(int x, int y, const string &str)
{
    if (x>y)
        std::swap(x, y);
    if (LogicUnit::needPrintI.find(make_pair(x, y))!=LogicUnit::needPrintI.end())
    {
        LogicUnit::unitNameForPrintI.insert(make_pair(make_pair(x, y), str));
    }
}

void LogicUnit::print(ostream &os, vector<string> *addition, bool *dc_print)
{
    string tmp;
    string print = ".print";
    switch (type)
    {
        case UnitType::linearR: 
            ++printCnt[0];
            os << "R"+to_string(printCnt[0]) + " " + \
                to_string(link[0]) + " " + to_string(link[1]) + " " + \
                paras[0] << endl;
            checkPrintI(link[0], link[1], "R"+to_string(printCnt[0]));
            break;

        case UnitType::selector:
            ++printCnt[1];
            os << "Xsel"+to_string(printCnt[1]) + " " + \
                to_string(link[0]) + " " + to_string(link[1]) + " " + \
                "selector" << endl;
            checkPrintI(link[0], link[1], "Xsel"+to_string(printCnt[1]));
            break;

        case UnitType::ReRAM:
            ++printCnt[2];
            os << "Xreram"+to_string(printCnt[2]) + " " + \
                to_string(link[0]) + " " + to_string(link[1]) + " " + \
                " reram_mod state = " + to_string(link[2]) << endl;
            checkPrintI(link[0], link[1], "Xreram"+to_string(printCnt[2]));
            break;

        case UnitType::voltage:
            ++printCnt[3];
            os << "V"+to_string(printCnt[3]) + " " + \
                to_string(link[0]) + " 0 " + \
                paras[0]  << endl;
            checkPrintI(link[0], 0, "V"+to_string(printCnt[3]));
            if (dc_print!=nullptr && addition!=nullptr && *dc_print == true)
            {
                *dc_print = false;
                addition->push_back(".dc V"+to_string(printCnt[3]) + " " + paras[0] + " " + paras[0] + " 0.1");
            }
            break;
        
        case UnitType::capacity:
            ++printCnt[4];
            os << "C"+to_string(printCnt[4]) + " " + \
                to_string(link[0]) + " " + to_string(link[1]) + " " + \
                paras[0] << endl;
            break;

        case UnitType::senseV:
            if (paras.size()==1)
                print += " " + paras[0] + " ";
            if (link.size()==1)
                os << print + " V(" + to_string(link[0]) + ")" << endl; 
            else if (link.size()>1)
                os << print + " V(" + to_string(link[0]) + ")" << endl;
            else
            {
                os << link.size() << "???" << endl;
            }
            break; 

        case UnitType::senseI:
            if (paras.size()==1)
                print += " " + paras[0] + " ";
            if (link.size()==1)
                os << print + " I(" + to_string(link[0]) + ")" << endl; 
            else if (link.size()>1)
            {
                int x = link[0], y = link[1];
                if (x>y) std::swap(x, y);
                auto ret = unitNameForPrintI.equal_range(make_pair(x, y));
                for (auto it = ret.first; it!=ret.second; ++it)
                    os << print + " I(" + it->second + ")" << endl;
            }
            else
            {
                os << link.size() << "I???" << endl;
            }
            break; 

        case UnitType::voltage_pp:
            ++printCnt[3];
            tmp = "";
            for (auto &i : paras)
                tmp += i + " ";
            os << "V"+to_string(printCnt[3]) + " " + \
                to_string(link[0]) + " 0 " + \
                tmp  << endl;
            checkPrintI(link[0], 0, "V"+to_string(printCnt[3]));
            break; 

    }
}


