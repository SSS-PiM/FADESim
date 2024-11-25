#include "3dVArray.h"

using namespace CBA;

void array3DV::addSelector()
{
    hasSelector = true;
}

void array3DV::removeSelector()
{
    hasSelector = false;
}


void array3DV::readConfig(ifstream &in)
{
    string cmd, paras;
    while (in >> cmd)
    {
        getline(in, paras);
        if (cmd=="%" || cmd=="//") continue;
        setParas(cmd, paras);
    }
}

bool array3DV::setParas(string s, string t)
{
    return true;
}