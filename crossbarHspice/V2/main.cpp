#include <iostream>
#include "concrete_command.h"
#include <fstream>

using namespace std;

void init_cmd();
void init_params(int, char *[]);
void read_file_and_execute();
void print_log_to_default_file();

string input_file;
cba::CommandFactory cmd_factory;
unique_ptr<cba::AbstractArray> arr;
int line = 0;

int main(int argc, char *argv[])
{
    try 
    {
        init_cmd();
        init_params(argc, argv);
        read_file_and_execute();
        print_log_to_default_file();
        cout << "--------Run end && Succeed---------" << endl;
    }
    catch (const exception &e)
    {
        cout << "Error(s) occured in line " << line << ": " << e.what() << endl;
        arr->printLog(cout);
    }
    return 0;
}

void init_params(int argc, char *argv[])
{
    //default input file is config
    if (argc==1)
        input_file = "config";
    else if (argc>=2)
        input_file = argv[1];
    

    // now only support 2d array, 3d remains to be done
    arr = make_unique<cba::Array2D>();
}

void read_file_and_execute()
{
    string cmd_str;
    ifstream fin(input_file);
    while (getline(fin, cmd_str))
    {
        ++line;
        auto cmd = cmd_factory.createCommand(cmd_str);
        cmd->run(*arr);
        // cout << cmd->getCmdName() << endl;
    }
}

void print_log_to_default_file()
{
    std::cout << "Log print to default.log" << std::endl;
    ofstream os("default.log");
    arr->printLog(os);
}

void init_cmd()
{
    // addtestcommand are only for unit test, not used.
    cmd_factory.registerCommand<cba::AddTestCommand, int, int>("add");
    cmd_factory.registerCommand<cba::AddTestCommand, int>("add");
    cmd_factory.registerCommand<cba::AddTestCommand, vector<string>>("add");

    // empty command; do not delete it!!
    cmd_factory.registerCommand<cba::EmptyCommand>("empty");
    
    // cmd name is %, //, or # means that it's a comment line, so do nothing
    cmd_factory.registerCommand<cba::EmptyCommand, vector<string>>("%");
    cmd_factory.registerCommand<cba::EmptyCommand, vector<string>>("//");
    cmd_factory.registerCommand<cba::EmptyCommand, vector<string>>("#");
    
    // register our command
    cmd_factory.registerCommand<cba::SetArraySize, int, int>("arraySize");
    cmd_factory.registerCommand<cba::SetLineR, double>("line_resistance");
    cmd_factory.registerCommand<cba::SetLineR, string, int, int, double>("setLineR");
    cmd_factory.registerCommand<cba::SetRload, string, double>("rload");
    
    // when use vector<string>, it means decode the args by the command itself
    // see command function run to check the decode of args.
    cmd_factory.registerCommand<cba::SetUseLine, vector<string>>("setUseLine");
    cmd_factory.registerCommand<cba::SetNotUseLine, vector<string>>("setNotUseLine");
    cmd_factory.registerCommand<cba::SetUseLinePlus, vector<string>>("setUseLine++");
    cmd_factory.registerCommand<cba::SetLineV, vector<string>>("setLineV");
    cmd_factory.registerCommand<cba::SetLineVPlus, vector<string>>("setLineV++");
    cmd_factory.registerCommand<cba::SetCellR, int, int, int>("setCellR");
    cmd_factory.registerCommand<cba::CellRStates, vector<string>>("cellRStates");
    cmd_factory.registerCommand<cba::InputVStates, vector<string>>("inputVStates");
    cmd_factory.registerCommand<cba::StringAdd, vector<string>>("topString");
    cmd_factory.registerCommand<cba::StringAddBot, vector<string>>("bottomString");
    cmd_factory.registerCommand<cba::BuildSpice, string>("build");
    cmd_factory.registerCommand<cba::Capacity, vector<string>>("capacity");
    cmd_factory.registerCommand<cba::Capacity, vector<string>>("capacitance");
    cmd_factory.registerCommand<cba::SenseCellV, vector<string>>("senseCellV");
    cmd_factory.registerCommand<cba::SenseBitlineI, vector<string>>("senseBitlineI");
    cmd_factory.registerCommand<cba::SenseWordlineI, vector<string>>("senseWordlineI");
    cmd_factory.registerCommand<cba::SimpleWriteForward, vector<string>>("simpleWriteOne");
    cmd_factory.registerCommand<cba::SimpleWriteReverse, vector<string>>("simpleWriteZero");
    cmd_factory.registerCommand<cba::RandSetInputV, vector<string>>("randSetInputV");
    cmd_factory.registerCommand<cba::UseRtypeReRAMCell, string>("useRtypeReRAMCell");
    cmd_factory.registerCommand<cba::RandSetR, vector<string>>("randSetR");
    cmd_factory.registerCommand<cba::DCMode>("dc");
    cmd_factory.registerCommand<cba::ACMode>("ac");
    cmd_factory.registerCommand<cba::SimpleFFRead, vector<string>>("simpleReadOut");
    cmd_factory.registerCommand<cba::NodebasedGSMethod, vector<string>>("nodebasedGSMethod");
    
}

