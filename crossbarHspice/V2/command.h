#ifndef __ARRAY_COMMAND__
#define __ARRAY_COMMAND__

#include "abstract_array.h"
#include "assistant.h"
#include <vector>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <iterator>
#include <cstring>
#include <functional>
#include <memory>


namespace cba {

using std::string;
using std::vector;

class CommandFactory;
class Command
{
public:

    virtual void run(AbstractArray &) = 0;
    virtual string getCmdName() const { return "default_name"; };
    virtual ~Command() = default;
    std::ostringstream os;
    CommandFactory *cmd_factory;
};

class CommandFactory
{
public:
    
    template<typename T, typename... Args>
    void registerCommand(const string &cmd_name)
    {
        auto func = [this](const vector<string> &args) 
            -> std::unique_ptr<Command>
        {
            auto t = parseArgs<Args...>(args);
            return std::apply([](auto&&... params)
            {
                return std::unique_ptr<T>(new T(std::forward<decltype(params)>(params)...));
            },
            t);
        };
        
        // use int in the pair<creator, int> to store the size of Args.
        // if args is vector, size set to -1 means infinite
        int params_cnt = sizeof...(Args);
        if constexpr (sizeof...(Args)>0)
        {
            if constexpr (is_vector<std::tuple_element_t<0, std::tuple<Args...>>>())
                params_cnt = -1;
        }

        cmd_creator.insert({cmd_name, std::make_pair(func, params_cnt)}); 
    }
    
    std::unique_ptr<Command> createCommand(const std::string &cmd_str)
    {
        std::istringstream iss(cmd_str);
        std::istream_iterator<string> isit(iss);               
        std::istream_iterator<string> ed;
        
        string cmd_name;
        
        // empty line, use empty command
        if (isit==ed)
            cmd_name = "empty"; 
        else
            cmd_name = *isit++;
        
        auto range = cmd_creator.equal_range(cmd_name);
        if (range.first == range.second)
            throw CommandException("Unknown command: " + cmd_name);
        
        auto params = vector<string>(isit, ed);
        CreatorFunc ret = findMatchCreator(params.size(), range.first, range.second);
        auto ptr = ret(params);
        ptr->cmd_factory = this;
        return ptr;
    }
    
private:
    using CreatorFunc = std::function<std::unique_ptr<Command>(const vector<string> &)>;
    using MpIt = std::unordered_multimap<string, std::pair<CreatorFunc, int>>::iterator;
    std::unordered_multimap<string, std::pair<CreatorFunc, int>> cmd_creator;
    
    CreatorFunc findMatchCreator(int n, MpIt first, MpIt second)
    {
        // find the creator that has same params as n
        for (auto it=first; it!=second; ++it)
        {
            auto [func, params_cnt] = it->second;
            if (params_cnt == n)
                return func;
        }
        
        if (n==0)
            throw CommandException("Failed to parse argument with 0 args.");

        // if can't find the creator that has same params as n
        // then we find if the first params is vector<string> to store all args
        for (auto it=first; it!=second; ++it)
        {
            auto [func, params_cnt] = it->second;
            // -1 means vector, so use vector to get all args
            if (params_cnt == -1)
                return func;
        }

        throw CommandException("Failed to parse argument due to the number of"
            " args not match.");
    }
    
    template<typename... Args>
    std::tuple<Args...> parseArgs(const vector<string> &args) 
    {
        std::tuple<Args...> result;
        if constexpr (sizeof...(Args)>=1)
        {
            if constexpr (is_vector<std::tuple_element_t<0, std::tuple<Args...>>>())
            {
                if (sizeof...(Args) != 1) 
                    throw CommandException("Only one argument is allowed when"
                        " the first argument is a vector. You should use the"
                        " vector to store all args.");
                std::get<0>(result) = args;
            }
            else 
                parseArgsHelper(result, args, std::index_sequence_for<Args...>{});
        }
        return result;
    }
    
    template<typename T>
    struct is_vector : std::false_type {};
    
    template<typename T, typename Alloc>
    struct is_vector<std::vector<T, Alloc>> : std::true_type {};
    
    
    // Function to parse a single argument
    template<typename T>
    T parseArg(const std::string &arg) 
    {
        T value;
        std::istringstream iss(arg);
        iss >> value;
        if (iss.fail() || !iss.eof()) 
            throw CommandException("Failed to parse argument: " + arg);
        return value;
    }

    template<typename Tuple, std::size_t... Is>
    void parseArgsHelper(Tuple &t, const std::vector<std::string> &args, 
        std::index_sequence<Is...>) 
    {
        ((std::get<Is>(t) = (args.size()<=Is? throw CommandException("Command args provided not enough"):
            parseArg<std::tuple_element_t<Is, Tuple>>(args[Is]))), ...);
    }    
    
};


}
#endif