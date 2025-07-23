#include "GenArray.h"


namespace cba {
void GenArray::push(const LogicUnit &lu, int num)
{
    if (num==-1) ;
       // lu.setNo(arch.size()+1);
    arch.push_back(lu);
    if (lu.type == UnitType::senseI && lu.link.size()>1)
    {
        if (lu.link[0]<lu.link[1])
            LogicUnit::needPrintI.insert(make_pair(lu.link[0], lu.link[1]));
        else
            LogicUnit::needPrintI.insert(make_pair(lu.link[1], lu.link[0]));
    }
}

void GenArray::push2top(const string &s)
{
    top_str.push_back(s);    
}

void GenArray::push2bot(const string &s)
{
    bottom_str.push_back(s);
}

void GenArray::push2bot_top(const string &s)
{
    bottom_str.insert(bottom_str.begin(), s);
}

void GenArray::push2bot(const LogicUnit &lu, int num)
{
    if (num==-1) ;
    arch_bot.push_back(lu);
    if (lu.type == UnitType::senseI && lu.link.size()>1)
    {
        if (lu.link[0]<lu.link[1])
            LogicUnit::needPrintI.insert(make_pair(lu.link[0], lu.link[1]));
        else
            LogicUnit::needPrintI.insert(make_pair(lu.link[1], lu.link[0]));
    }
}

void GenArray::pop_back()
{
    arch.pop_back();
}

void GenArray::print2Hspice(ostream &o, bool dc_print)
{
    vector<string> addition; // dc mode, we need to print ".dc V0 st ed step", use addition to store this command

    for (auto &i : top_str)
        o << i << endl;

    for (auto &i : arch)
        i.print(o, &addition, &dc_print);

    for (auto &i : arch_bot)
        i.print(o, &addition, &dc_print);
    
    for (auto &i : addition)
        o << i << endl;
    for (auto &i : bottom_str)
        o << i << endl;
}



}


