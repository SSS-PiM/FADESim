#ifndef __ARRAY__
#define __ARRAY__

#include "LogicUnit.h"
#include <fstream>

using std::ifstream;

namespace cba {

class GenArray 
{
public:
    void push(const LogicUnit &lu, int num=-1);
    void pop_back();
    void push2top(const string &s);
    void push2bot(const string &s);
    void push2bot_top(const string &s);
    void push2bot(const LogicUnit &lu, int num=-1);
    void print2Hspice(ostream &, bool = false);
    GenArray(): mode(0) {}
    int size() { return arch.size()+arch_bot.size(); }
    int mode; //0-> static mode, 1-> dynamic mode
    // void setHspiceConfig();
    vector<LogicUnit> *getArch() {return &arch;}
private:
    vector<LogicUnit> arch, arch_bot;
    vector<string> top_str, bottom_str;
};

}

#endif
