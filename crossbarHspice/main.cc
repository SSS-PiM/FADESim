#include <iostream>
#include "2dArray.h"
#include "3dHArray.h"
#include <memory>

using namespace std;
using namespace CBA;

int main(int argc, char *argv[])
{
    string str;
    shared_ptr<phyArray> arr;
    if (argc==1)
    {
        arr=make_shared<array2D>();
        str = "config";
    }
    else if (argc>3)
    {
        cout << "argc number wrong" << endl;
        return 0;
    }
    else if (argc==3)
    {
        string tmp = argv[2];
        if (tmp == "-debug")
        {
            arr=make_shared<array2D>();
            arr->debug=true;
            cout << "in debug mode" << endl;
        }
        else if (tmp == "-3DH" || tmp == "-3dh")
            arr=make_shared<array3DH>();
        else 
            arr=make_shared<array2D>();

        str = argv[1];
    }
    else if (argc==2)
        str = argv[1], arr=make_shared<array2D>();
    ifstream in(str);
    arr->readConfig(in);
    return 0;
}
