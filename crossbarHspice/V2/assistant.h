#ifndef __ASSISTANT__
#define __ASSISTANT__

#include <exception>
#include <string>
#include <random>
#include <sys/time.h>
#include <numeric>
#include <algorithm>
#include <cctype>

namespace cba {

struct CommandException: public std::exception
{
    const char *what() const throw()
    {
        return msg.c_str();
    }
    
    std::string msg;
    CommandException(const std::string &str): msg(str) {}
};

std::string string2lower(const std::string &in_str)
{
    std::string str(in_str);
    for (char &c : str)
        c = ::tolower(c);
    return std::move(str);
}

template <typename T>
std::vector<T> generate_sequence(const std::vector<double> &p,
                            const std::vector<T> &items, int length)
{
    if (p.size() != items.size())
        throw CommandException("p size should equal to item size");
    double sum = std::accumulate(p.begin(), p.end(), 0.0);
    if (std::abs(sum-1.0)>1e-6)
        throw CommandException("Error: p sum is not 1");

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(p.begin(), p.end());

    std::vector<T> seq;
    seq.reserve(length);

    for (int i=0; i<length; ++i)
    {
        int index = dist(gen);
        seq.push_back(items[index]);
    }
    return std::move(seq);
}

bool checkYes(std::string str)
{
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    if (str=="yes" || str=="1" || str=="true" || str=="y" || str=="t")
        return true;
    return false;
}

struct time_record
{
    struct timeval t1, t2;
    std::ostream &os;

    time_record(std::ostream &os): os(os)
    {
        gettimeofday(&t1, NULL);
    }

    ~time_record()
    {
        gettimeofday(&t2, NULL);
        os << "Duration time = " << (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0 << " s" << std::endl;
    }
};


}
#endif