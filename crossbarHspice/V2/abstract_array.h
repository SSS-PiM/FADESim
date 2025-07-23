#ifndef __ABSTRACT_ARRAY__
#define __ABSTRACT_ARRAY__

#include <vector>
#include <string>
#include <iostream>

namespace cba
{
/*
* Currently, only 2D and 3D arrays are supported.
* because there are no 4D or higher-dimensional arrays.
*/
class AbstractArray
{
public:
    void pushLog(const std::string &x)
    {
        history.push_back(x);
    }
    
    void printLog(std::ostream &os)
    {
        os << std::endl << "--------Operation Log---------" << std::endl;
        for (auto &i : history)
            os << i << std::endl;
        os << "--------Log end--------" << std::endl;
    }

    int n = 0, m = 0, h = 0;  // (n, m) for 2D, (n, m, h) for 3D
    
    // store history of operations
    std::vector<std::string> history;
};

}
#endif