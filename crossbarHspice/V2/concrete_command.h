#ifndef __CONCRETE_COMMAND__
#define __CONCRETE_COMMAND__

#include "command.h"
#include <numeric>
#include "array_2d.h"
#include "LogicUnit.h"
#include <cmath>
#include <iomanip>


namespace cba
{

typedef vector<double> vec;
typedef vector<vec> Mat;

using std::endl;
struct EmptyCommand: public Command
{
    void run(AbstractArray &) {}
    string getCmdName() const
    {
        return "empty";
    }
    EmptyCommand() {}
    EmptyCommand(const vector<string>&) {}
};

struct AddTestCommand: public Command
{
    void run(AbstractArray &) 
    {
        std::cout << x+y << std::endl;
    }

    AddTestCommand(int x, int y): x(x), y(y) {}
    
    AddTestCommand(const vector<string>&) 
    { std::cout << "in string constrcut" << std::endl; }
    
    AddTestCommand(int x): x(x), y(0) {}

    string getCmdName() const
    {
        return "add";
    }
    int x, y;
};

struct SetArraySize: public Command
{
    void run(AbstractArray &arr)
    {
        auto ptr = (reinterpret_cast<Array2D*>(&arr));
        ptr->setSize(n, m);
        os << "Set array size to ( " << n << ", " << m << " )";
        ptr->pushLog(os.str());
    }

    SetArraySize(int n, int m): n(n), m(m) {}
    string getCmdName() const
    {
        return "arraySize";
    }
    int n, m;
};

struct SetLineR: public Command
{
    void run(AbstractArray &arr)
    {
        auto arr_ptr = (reinterpret_cast<Array2D*>(&arr));
        
        if (dir == "")
        {
            auto [n, m] = arr_ptr->size();
            if (n==0 || m==0)
                throw CommandException("Please set arraySize before"
                    " set line resistance"); 
                
            for (auto &i : arr_ptr->RULine)
                for (auto &j : i)
                    j = r;

            for (auto &i : arr_ptr->RDLine)
                for (auto &j : i)
                    j = r;
            
            os << "Set all line resistance to " << r;
        }
        else
        {
            auto [n, m] = arr_ptr->size();

            if (x>=n || y>=m || r<0)
                throw CommandException("Arg Direction not matched:"
                    "setLineR (up|down) x:int<n y:int<m r:double>0");
            if (dir == "up")
            {
                arr_ptr->RULine[x][y] = r;
                os << "RULine (" << x << ", " << y << ") to " << r << endl; 
            }
            else if (dir == "down")
            {
                arr_ptr->RDLine[x][y] = r;
                os << "RDLine (" << x << ", " << y << ") to " << r << endl; 
            }
            else
                throw CommandException("Arg Direction not matched:"
                    "setLineR up|down x:int y:int r:double");
        }
        arr_ptr->pushLog(os.str());
    }

    SetLineR(double r): dir(), r(r) {}
    SetLineR(const string &dir, int x, int y, double r):
        dir(dir), x(x), y(y), r(r) {}

    string getCmdName() 
    {
        return "line_resistance";
    }
    string dir;
    int x, y;
    double r;
};

// load resistance is the most left or right line resistance of the wordline
// or the most up or down line resistance of the bitline
struct SetRload: public Command
{
    void run(AbstractArray &arr_2d)
    {
        auto arr = (reinterpret_cast<Array2D*>(&arr_2d));
        if (mode == "all")
        {
            for (auto &i : arr->RULine)
                i[0] = i[i.size()-1] = r;
            for (auto &i : arr->RDLine)
                i[0] = i[i.size()-1] = r;
            for (int i=0; i<4; ++i)
                arr->r_load[i] = r;
            os << "set all r_load (wordline left&right) & (bitline up&down) to "
                << r << " omega" << endl;
        }
        else if (mode == "left" || mode == "right")
        {
            for (auto &i : arr->RULine)
                i[mode=="left"? 0 : (i.size()-1)] = r;
            arr->r_load[mode=="left"? 1 : 3] = r; 
        }   
        else if (mode == "down" || mode == "up")
        {
            for (auto &i : arr->RDLine)
                i[mode=="up"? 0 : (i.size()-1)] = r;
            arr->r_load[mode=="up"? 0 : 2] = r; 
        }
        else
            throw CommandException("Arg str not matched");
        os << "set " + mode + " line r_load to " << r << " omega" << endl; 
        arr->pushLog(os.str());
    }
    
    SetRload(const string &m, double r): mode(m), r(r) {}

    string getCmdName()
    {
        return "rload";
    }

    string mode;
    double r;
};

int getDir(const string &dir)
{
    int d = -1;
    if (dir == "up")
        d = 0;
    else if (dir == "left")
        d = 1;
    else if (dir == "down")
        d = 2;
    else if (dir == "right")
        d = 3;
    return d;
}

#define CHECK_DIR_NUM(d) \
    if ((d)==-1) \
        throw CommandException("Arg str not matched" \
            " Direction should be up|left|down|right.");

struct SetUseLine: public Command
{
    void run(AbstractArray &arr_2d)
    {
        auto arr = (reinterpret_cast<Array2D*>(&arr_2d));
        
        int d = -1;
        for (int i=0; i<args.size(); ++i)
        {
            if (i==0)
            {
                d=getDir(args[0]);
                CHECK_DIR_NUM(d);
            }
            else
            {
                int x = stoi(args[i]);
                if (x==-1)
                {
                    for (auto &i : arr->bv[d])
                        i = use_line;
                    if (use_line)
                        os << "set use Direction " << d << (string(" all line") 
                            + (use_line>1? " plus" : ""));
                    else
                        os << "set not use Direction " << d << " all line";
                }   
                else
                {
                    arr->bv[d].at(x) = use_line;
                    if (use_line)
                        os << "set use Direction " << d << "line No. " << x 
                            << (use_line>1? " plus" : "");
                    else
                        os << "set not use Direction " << d << "line No. " << x;

                }
            }
        }
        arr->pushLog(os.str());
    }

    SetUseLine(vector<string> &args): args(std::move(args)), use_line(1) {}

    vector<string> args;
    
    int use_line;
};

struct SetNotUseLine: public SetUseLine
{
    SetNotUseLine(vector<string> &args): SetUseLine(args)
    {
        use_line = 0;
    }
};

struct SetUseLinePlus: public SetUseLine
{
    SetUseLinePlus(vector<string> &args): SetUseLine(args)
    {
        use_line = 2;
    }
};

struct SetLineV: public Command
{
    void run(AbstractArray &arr_2d)
    {
        auto arr = (reinterpret_cast<Array2D*>(&arr_2d));
        
        int d = -1;
        
        if (plus_version == 0)
        {
            int x;
            double y;
            
            if (args.size()!=3)
                throw CommandException("Args size not matched!");
            
                d = getDir(args[0]), x = stoi(args[1]), y = stod(args[2]);
            
            if (x==-1)
            {
                for (auto &i : arr->v[d])
                    i = y;
                os << "set direction " << d << " all line to voltage " 
                    << y << " [d=(0, 1, 2, 3)->(up, left, down, right)]";
            }   
            else
            {
                arr->v[d].at(x) = y;
                os << "set direction " << d << " line No." << x 
                    << " to voltage " << y << 
                    " [d=(0, 1, 2, 3)->(up, left, down, right)]";
            }
        }
        else
        {
            int x;
            d = getDir(args[0]), x = stoi(args[1]);
            string y;
            for (int i=2; i<args.size(); ++i)
                y += args[i] + " ";
            if (x==-1)
            {
                for (auto &i : arr->v_str[d])
                    i = y;
                os << "set direction " << d << " all line to voltage " 
                    << y << " [d=(0, 1, 2, 3)->(up, left, down, right)]";
            }   
            else
            {
                arr->v_str[d].at(x) = y;
                os << "set direction " << d << " line No." << x << 
                    " to voltage " << y << 
                    " [d=(0, 1, 2, 3)->(up, left, down, right)]";
            }
        }
        arr->pushLog(os.str());
    }

    SetLineV(vector<string> &args): args(std::move(args)), plus_version(0) {}

    vector<string> args;
    int plus_version;
};

struct SetLineVPlus: public SetLineV
{
    SetLineVPlus(vector<string> &args): SetLineV(args)
    {
        plus_version = 1;
    }
};

struct SetCellR: public Command
{
    SetCellR(int x, int y, int z): x(x), y(y), z(z) {};
    
    void run(AbstractArray &arr_2d)
    {
        auto arr = (reinterpret_cast<Array2D*>(&arr_2d));
        
        os << "setCellR " << x << ' ' << y << ' ' << z;
        if (x==-1)
        {
            for (auto &i : arr->arr)
            {
                if (y==-1)
                {
                    for (auto &j : i)
                        j = z;
                }
                else 
                    i.at(y) = z;
            }
        }
        else
        {
            if (y==-1)
            {
                for (auto &j : arr->arr.at(x))
                    j = z;
            }
            else
                arr->arr.at(x).at(y) = z; 
        }

        arr->pushLog(os.str());
    }

    int x, y, z;
};

struct CellRStates: public Command
{
    CellRStates(vector<string> &args): args(std::move(args)) {}
    
    void run(AbstractArray &arr_2d)
    {
        auto arr = (reinterpret_cast<Array2D*>(&arr_2d));
        if (!arr->useRtypeCell)
            os << "warning: set cell R states, note that, this is only avaliable "
                "when cell R is linear. But now useRtypecell is false!" << endl;
        
        try
        {
            arr->cellR_num = stoi(args.at(0));
            arr->cellR_states = vector<double>(arr->cellR_num);
            for (int i=0; i<arr->cellR_num; ++i)
                arr->cellR_states.at(i) = stod(args.at(i+1));
        }
        catch (std::exception &e)
        {
            throw CommandException("cell r set fail!");
        }
        arr->pushLog(os.str());
    }

    vector<string> args;
};

struct InputVStates: public Command
{
    InputVStates(vector<string> &args): args(std::move(args)) {}
    
    void run(AbstractArray &arr_2d)
    {
        auto arr = (reinterpret_cast<Array2D*>(&arr_2d));
        os << "set cell input v states, note that, this is useful when calling randSetInputV" << endl;
        
        try
        {
            arr->inputV_num = stoi(args.at(0));
            arr->inputV_states = vector<double>(arr->inputV_num);
            for (int i=0; i<arr->inputV_num; ++i)
                arr->inputV_states.at(i) = stod(args.at(i+1));
        }
        catch (std::exception &e)
        {
            throw CommandException("input set fail!");
        }
        arr->pushLog(os.str());
    }

    vector<string> args;
};


struct StringAdd: public Command
{
    StringAdd(vector<string> &args): args(std::move(args)), dir(Direction::up) {}
    
    void run(AbstractArray &arr_2d)
    {
        auto arr = (reinterpret_cast<Array2D*>(&arr_2d));
        string str = std::accumulate(args.begin(), args.end(), 
                string(), [](const string &a, const string &b) 
                {
                    return a.empty()? b : a+" "+b;
                });

        if (dir==Direction::up)
            arr->gen.push2top(str);
        else
            arr->gen.push2bot(str);

    }
    
    enum class Direction{up, down} dir;
    vector<string> args;
};

struct StringAddBot: public StringAdd
{
    StringAddBot(vector<string> &args): StringAdd(args) 
    {
        dir = Direction::down;
    }
};

struct BuildSpice: public Command
{
    BuildSpice(string &file): file(std::move(file)) {}
    
    void run(AbstractArray &arr_2d)
    {
        auto array = (reinterpret_cast<Array2D*>(&arr_2d));
        int n = array->n, m = array->m;
        bool hasSelector = array->hasSelector;
        int otherPoint = hasSelector? 3*n*m : 2*n*m;
        auto &gen = array->gen;
        auto &arr = array->arr;
        auto &cellR_states = array->cellR_states;
        auto &useRtypeCell = array->useRtypeCell;
        auto &RULine = array->RULine;
        auto &RDLine = array->RDLine;
        auto &capacitance = array->capacitance;
        auto &hasCapacity = array->hasCapacity;
        auto &bv = array->bv;
        auto &v = array->v;
        auto &v_str = array->v_str;
        auto &dc_ac_tran_type = array->dc_ac_tran_type;

        //build ReRAM cells and selectors
        for (int i=0; i<n; ++i)
        {
            for (int j=0; j<m; ++j)
            {
                int now = i*m+j+1;
                if (hasSelector)
                {
                    gen.push(LogicUnit(UnitType::selector, {now, now+m*n*2}, {}));
                    if (!useRtypeCell)
                        gen.push(LogicUnit(UnitType::ReRAM, {now+n*m*2, now+m*n, arr[i][j]}, {}));
                    else
                        gen.push(LogicUnit(UnitType::linearR, {now+n*m*2, now+n*m}, {toStr(cellR_states[arr[i][j]])} ));
                }
                else
                {
                    if (!useRtypeCell)
                        gen.push(LogicUnit(UnitType::ReRAM, {now, now+n*m, arr[i][j]}, {}));
                    else
                        gen.push(LogicUnit(UnitType::linearR, {now, now+n*m}, {toStr(cellR_states[arr[i][j]])} ));
                }
            }
        }

        //build worlines
        for (int i=0; i<n; ++i)
        {
            for (int j=0; j<=m; ++j)
            {
                int pa, pb;
                if (j!=m)
                    pb = i*m+j+1;
                else
                    pb = otherPoint+m*2+n+i+1;
                if (j!=0)
                    pa = i*m+j;
                else
                    pa = otherPoint+m+i+1;

                gen.push(LogicUnit(UnitType::linearR, {pa, pb}, { toStr(RULine[i][j])} ));
            }
        }

        //build bitlines
        for (int j=0; j<m; ++j)
        {
            for (int i=0; i<=n; ++i)
            {
                int pa, pb;
                if (i!=n)
                    pb = i*m+j+1+m*n;
                else
                    pb = otherPoint+m+n+j+1;
                if (i!=0)
                    pa = i*m+j+1+m*n-m;
                else
                    pa = otherPoint+j+1;

                gen.push(LogicUnit(UnitType::linearR, {pa, pb}, { toStr(RDLine[j][i]) } ));
            }
        }

        //build voltages to the worline  
        for (int i=0; i<n; ++i)
        {
            if (bv[1][i]==1)
                gen.push(LogicUnit(UnitType::voltage, {otherPoint+m+i+1}, { toStr(v[1][i]) } ));
            else if (bv[1][i]==2)
                gen.push(LogicUnit(UnitType::voltage, {otherPoint+m+i+1}, { v_str[1][i] } ));
            

            if (bv[3][i]==1)
                gen.push(LogicUnit(UnitType::voltage, {otherPoint+n+2*m+i+1}, { toStr(v[3][i]) }));
            else if (bv[3][i]==2)
                gen.push(LogicUnit(UnitType::voltage, {otherPoint+n+2*m+i+1}, { v_str[3][i] } ));
            
        }

        //build voltages to the bitline  
        for (int j=0; j<m; ++j)
        {
            if (bv[0][j]==1)
                gen.push(LogicUnit(UnitType::voltage, {otherPoint+j+1}, { toStr(v[0][j]) }));
            else if (bv[0][j]==2)
                gen.push(LogicUnit(UnitType::voltage, {otherPoint+j+1}, { v_str[0][j] }));

            if (bv[2][j]==1)
                gen.push(LogicUnit(UnitType::voltage, {otherPoint+m+n+j+1}, { toStr(v[2][j]) }));
            else if (bv[2][j]==2)
                gen.push(LogicUnit(UnitType::voltage, {otherPoint+m+n+j+1}, { v_str[2][j] }));
        }

        //build capacity
        if (hasCapacity)
        {
            for (int i=0; i<n; ++i)
            {
                for (int j=0; j<m; ++j)
                {
                    int now = i*m +j +1;
                    gen.push(LogicUnit(UnitType::capacity, {now, 0}, { capacitance[0] }));
                    if (j!=m-1)
                        gen.push(LogicUnit(UnitType::capacity, {now, now+1}, { capacitance[2] }));
                }
            }

            for (int j=0; j<m; ++j)
            {
                for (int i=0; i<n; ++i)
                {
                    int now = m*n+i*m+j+1;
        
                    gen.push(LogicUnit(UnitType::capacity, {now, 0}, {capacitance[1]}));
                    if (i!=n-1)
                        gen.push(LogicUnit(UnitType::capacity, {now, now+m}, {capacitance[3]}));
                }
            }
        }


        os << "build over; start print to spice" << endl; 
        std::ofstream out(file);
        gen.print2Hspice(out, dc_ac_tran_type=="dc");
        os << "print2 spice over" << endl;
        array->pushLog(os.str());
    }
    
    string file;
};

struct Capacity: public Command
{
    Capacity(vector<string> &args): args(std::move(args)) {}
    
    void run(AbstractArray &arr_2d)
    {
        auto arr = (reinterpret_cast<Array2D*>(&arr_2d));
        
        if (args.size()<1)
        {
            throw CommandException("Capacity set fail");
        }
        try 
        {
            int x = stoi(args[0]);
            if (x != 0)
                arr->hasCapacity = true;
            else
                arr->hasCapacity = false;
                    
            for (int i=0; i<4; ++i)
                arr->capacitance[i] = stod(args.at(i+1));
        }
        catch (std::exception &e)
        {
            throw CommandException("Capacity set fail");
        }
    }

    vector<string> args;
};

struct SenseCellV: public Command
{
    SenseCellV(vector<string> &args): args(std::move(args)) 
    {
        if (this->args.size()%2!=0)
            throw CommandException("sense cell (x, y) not paired");
        for (int i=0; i<this->args.size(); ++i)
        {
            int value = stoi(this->args[i]);
            if (i&1)
                y.push_back(value);
            else
                x.push_back(value);
        }
    }

    void run(AbstractArray &arr_2d)
    {
        auto arr = (reinterpret_cast<Array2D*>(&arr_2d));
        int n = arr->n, m = arr->m;
        for (int i=0; i<x.size(); ++i)
        {
            int nx = x[i], ny = y[i];
            if (nx>=arr->n || ny>=arr->m)
                throw CommandException("x >=n or y>=m");
            vector<int> prt_x, prt_y;
            if (nx==-1)
            {
                if (ny==-1)
                {
                    for (int j=0; j<n; ++j)
                    {
                        for (int k=0; k<m; ++k)
                        {
                            prt_x.push_back(j);
                            prt_y.push_back(k);
                        }
                    }
                }
                else
                {
                    for (int j=0; j<n; ++j)
                    {
                        prt_x.push_back(j);
                        prt_y.push_back(ny);
                    }
                }
            }
            else
            {
                if (ny==-1)
                {
                    for (int k=0; k<m; ++k)
                    {
                        prt_x.push_back(nx);
                        prt_y.push_back(k);
                    }
                }
                else
                {
                    prt_x.push_back(nx);
                    prt_y.push_back(ny);
                }
            }
            for (int id=0; id<prt_x.size(); ++id)
            {
                arr->gen.push2bot(LogicUnit(UnitType::senseV, {prt_x[id]*m+prt_y[id]+1}, {arr->dc_ac_tran_type}));
                arr->gen.push2bot(LogicUnit(UnitType::senseV, {prt_x[id]*m+prt_y[id]+1+n*m}, {arr->dc_ac_tran_type}));
            }
        }
        
    }

    vector<string> args;
    vector<int> x, y;
};

struct SenseBitlineI: public Command
{
    SenseBitlineI(vector<string> &args): args(std::move(args)) {}

    void run(AbstractArray &arr_2d)
    {
        auto arr = (reinterpret_cast<Array2D*>(&arr_2d));
        
        if (args.size()<1)
            throw CommandException("sensebitlineI args number not mathed");
        
        string &dir = args[0];
        vector<int> y;
        int n = arr->n, m = arr->m;
        for (int i=1; i<args.size(); ++i)
        {
            int p = stoi(args[i]);
            if (p>=arr->m || p<-1)
                throw CommandException("sensebitlineI bitline No. wrong");
            
            if (p>=0)
                y.push_back(p);
            else
            {
                for (int j=0; j<arr->m; ++j)
                    y.push_back(j);
                break;
            }
        }
        int otherPoint = arr->hasSelector? 3*n*m : 2*n*m;
        for (int ny : y)
        {
            if (dir=="down")
                arr->gen.push2bot(LogicUnit(UnitType::senseI, {otherPoint+m+n+ny+1, 0}, {arr->dc_ac_tran_type}));
            else if (dir=="up") 
                arr->gen.push2bot(LogicUnit(UnitType::senseI, {otherPoint+ny+1, 0}, {arr->dc_ac_tran_type}));
            else 
                throw CommandException("Direction should be up or down");
        }
    }

    vector<string> args;
};

struct SenseWordlineI: public Command
{
    SenseWordlineI(vector<string> &args): args(std::move(args)) {}

    void run(AbstractArray &arr_2d)
    {
        auto arr = (reinterpret_cast<Array2D*>(&arr_2d));
        
        if (args.size()<1)
            throw CommandException("sensewordlineI args number not mathed");
        
        string &dir = args[0];
        vector<int> x;
        int n = arr->n, m = arr->m;
        for (int i=1; i<args.size(); ++i)
        {
            int p = stoi(args[i]);
            if (p>=arr->n || p<-1)
                throw CommandException("sensewordlineI wordline No. wrong");
            
            if (p>=0)
                x.push_back(p);
            else
            {
                for (int j=0; j<arr->n; ++j)
                    x.push_back(j);
                break;
            }
        }
        int otherPoint = arr->hasSelector? 3*n*m : 2*n*m;
        for (int nx : x)
        {
            if (dir=="left")
                arr->gen.push2bot(LogicUnit(UnitType::senseI, {otherPoint+m+nx+1, 0}, {arr->dc_ac_tran_type}));
            else if (dir=="right")
                arr->gen.push2bot(LogicUnit(UnitType::senseI, {otherPoint+m*2+n+nx+1, 0}, {arr->dc_ac_tran_type}));
            else
                throw CommandException("Direction should be left or right");
        }
    }

    vector<string> args;
};

/*
* half-voltage write method
* Allow to write multiple cells on the same wordline
*/
struct SimpleWriteForward: public Command
{
    SimpleWriteForward(vector<string> &args): args(std::move(args)), dir(Direction::forward) {}
    
    void run(AbstractArray &arr_2d)
    {
        auto arr = (reinterpret_cast<Array2D*>(&arr_2d));
        double writeV;

        if (args.size()<3 || args.size()%2==0)
            throw CommandException("simplewrite args number not matched");
        
        writeV = stod(args[0]);
        
        vector<int> x, y;

        for (int i=1; i<args.size(); ++i)
        {
            int p = stoi(args[i]);
            if (i&1)
                x.push_back(p);
            else
                y.push_back(p);
        }
        
        for (int i=x.size()-1; i>=1; --i)
        {
            if (x[i]!=x[i-1])
                throw CommandException("Simple write, wordline is not same");
        }
        
        cmd_factory->createCommand("setUseLine left -1")->run(arr_2d);
        cmd_factory->createCommand("setLineV left -1 " + std::to_string(writeV/2))->run(arr_2d);
        
        if (dir==Direction::forward)
            cmd_factory->createCommand("setLineV left " + std::to_string(x[0])+ " "+ std::to_string(writeV))->run(arr_2d);
        else
            cmd_factory->createCommand("setLineV left " + std::to_string(x[0])+ " "+ std::to_string(0))->run(arr_2d);

        
        cmd_factory->createCommand("setUseLine down -1")->run(arr_2d);
        cmd_factory->createCommand("setLineV down -1 " + std::to_string(writeV/2))->run(arr_2d);

        for (auto &i : y)
        {
            if (dir==Direction::forward)
                cmd_factory->createCommand("setLineV down " + std::to_string(i) + " " + std::to_string(0))->run(arr_2d);
            else
                cmd_factory->createCommand("setLineV down " + std::to_string(i) + " " + std::to_string(writeV))->run(arr_2d);
        }
        
        for (int i=0; i<x.size(); ++i)
        {
            int nx = x[i], ny = y[i];
            
            arr->gen.push2bot(LogicUnit(UnitType::senseV, {nx*arr->m+ny+1}, {arr->dc_ac_tran_type}));
            arr->gen.push2bot(LogicUnit(UnitType::senseV, {nx*arr->m+ny+1+arr->n*arr->m}, {arr->dc_ac_tran_type}));
        }
    }

    vector<string> args;
    
    enum class Direction {forward, reverse} dir;
};

struct SimpleWriteReverse: public SimpleWriteForward
{
    SimpleWriteReverse(vector<string> &args): SimpleWriteForward(args)
    {
        dir = Direction::reverse;
    }
};

struct RandSetInputV: public Command
{
    RandSetInputV(vector<string> &args): args(std::move(args)) {}
    
    void run(AbstractArray &arr_2d)
    {
        auto arr = (reinterpret_cast<Array2D*>(&arr_2d));
        int d;
        
        if (arr->inputV_states.size() == 0)
            throw CommandException("randSetInputV should be used after inputVstates");
        if (arr->inputV_states.size()+1 != args.size())
            throw CommandException("Randsetinputv args number wrong, "
                "should be direction(up|left|down|right) + a series "
                "of probility for each input state");
        d = getDir(args[0]);
        CHECK_DIR_NUM(d);
        vector<double> p;
        p.reserve(arr->inputV_num);
        for (int i=0; i<arr->inputV_num; ++i)
        {
            double x;
            x = stod(args[i+1]);
            p.push_back(x);
        } 
        int len = (d==0 || d==2)? arr->m : arr->n;
        auto v_in = generate_sequence<double>(p, arr->inputV_states, len);
        bool warn = false;
        for (int i=0; i<len; ++i)
        {
            arr->v[d].at(i) = v_in[i];
            if (arr->bv[d].at(i) == 0 && !warn)
            {
                warn = true;
                os << "warning: randsetinputV but the input line is not used. maybe should call setUseLine first";
            }
        }
        arr->pushLog(os.str());
    }
    vector<string> args;
};

/*
* if set to true, HSPICE will directly print resistance to represent cell
* rather than use verilog-a mod.
*/
struct UseRtypeReRAMCell: public Command
{
    UseRtypeReRAMCell(string &x): str(std::move(x)) {}

    void run(AbstractArray &arr_2d)
    {
        auto arr = (reinterpret_cast<Array2D*>(&arr_2d));
        if (cba::checkYes(str))
            arr->useRtypeCell = true;
        else
            arr->useRtypeCell = false;
    }

    std::string str;
};

struct RandSetR: public Command
{
    RandSetR(vector<string> &args): args(std::move(args)) {}
    
    void run(AbstractArray &arr_2d)
    {
        auto arr = (reinterpret_cast<Array2D*>(&arr_2d));
        
        if (arr->cellR_states.size() == 0)
            throw CommandException("randSetR should be used after cellRstates");
        if (arr->cellR_states.size() != args.size())
            throw CommandException("RandsetR args number wrong, "
                "should be a series "
                "of probility for each input state");

        vector<double> p;
        p.reserve(arr->cellR_num);
        for (int i=0; i<arr->cellR_num; ++i)
        {
            double x;
            x = stod(args[i]);
            p.push_back(x);
        } 
        int len = arr->m * arr->n;
        vector<int> states(arr->cellR_num);
        int cnt = 0;
        std::generate(states.begin(),
                    states.end(), [&cnt]() {return cnt++;});
        auto r = generate_sequence<int>(p, states, len);
        bool warn = false;
        for (int i=0; i<arr->n; ++i)
            for (int j=0; j<arr->m; ++j)
            {
                arr->arr[i][j] = r[i*arr->m + j];
            }
    }
    vector<string> args;
};


struct UseSelector: public Command
{
    UseSelector(string &x): str(std::move(x)) {}

    void run(AbstractArray &arr_2d)
    {
        auto arr = (reinterpret_cast<Array2D*>(&arr_2d));
        if (cba::checkYes(str))
            arr->hasSelector = true;
        else
            arr->hasSelector = false;
    }

    std::string str;
};

struct DCMode: public Command
{

    void run(AbstractArray &arr_2d)
    {
        auto arr = (reinterpret_cast<Array2D*>(&arr_2d));
        arr->dc_ac_tran_type = "dc";
    }

};

struct ACMode: public Command
{

    void run(AbstractArray &arr_2d)
    {
        auto arr = (reinterpret_cast<Array2D*>(&arr_2d));
        arr->dc_ac_tran_type = "ac";
    }

};

struct SimpleFFRead: public Command
{
    SimpleFFRead(vector<string> &args): args(std::move(args)) {}
    void run(AbstractArray &arr_2d)
    {
        auto arr = (reinterpret_cast<Array2D*>(&arr_2d));
        
        double readV, Rsense;
        bool isvoltageRead;
        int st;

        if (args.size()<4)
            throw CommandException("simpleFFread args number not matched");
        
        if (args[0]=="volread")
        {
            Rsense = std::stod(args[1]);
            isvoltageRead = true;
            st = 2;
        }
        else if (args[0]=="Iread")
            isvoltageRead = false, st=1;
        else
            throw CommandException("the first args should be volread or Iread, "
                "to indicate it's a voltage read mode or current read mode");
        
        readV= stod(args[st++]);
        
        vector<int> x, y;

        for (int i=st; i<args.size(); ++i)
        {
            int p = stoi(args[i]);
            if ((i-st)&1)
                y.push_back(p);
            else
                x.push_back(p);
        }
        if (x.size()!=y.size())
            throw CommandException("simpleffread, args x.size!=y.size");
        for (int i=x.size()-1; i>=1; --i)
        {
            if (x[i]!=x[i-1])
                throw CommandException("Simpleffread, wordline is not same");
        }
       
        cmd_factory->createCommand("setUseLine left "+std::to_string(x[0]))->run(arr_2d);
        cmd_factory->createCommand("setLineV left "+std::to_string(x[0])+" "+std::to_string(readV))->run(arr_2d);
        int otherPoint = arr->hasSelector? 3*arr->n*arr->m : 2*arr->n*arr->m;
        for (auto &i : y)
        {
            cmd_factory->createCommand("setUseLine down "+std::to_string(i))->run(arr_2d);
            cmd_factory->createCommand("setLineV down "+std::to_string(i)+" "+std::to_string(0))->run(arr_2d);
            if (isvoltageRead) 
            {
                arr->RDLine[i][arr->n] = Rsense;
                arr->gen.push2bot(LogicUnit(UnitType::senseV, {(arr->n-1)*arr->m+i+1+arr->n*arr->m}, {arr->dc_ac_tran_type}));
            }
            else
            {
                arr->gen.push2bot(LogicUnit(UnitType::senseI, {otherPoint+arr->m+arr->n+i+1, 0}, {arr->dc_ac_tran_type}));
            }
        }
    }
    
    vector<string> args;
};

struct NodebasedGSMethod: public Command
{
    NodebasedGSMethod(vector<string> &args): args(std::move(args)) {}

    void run(AbstractArray &arr_2d)
    {
        auto arr = (reinterpret_cast<Array2D*>(&arr_2d));
        
        int iter_time = -1;
        double w;
        int mode = 0;
        bool enable_break = false;
        double threshould_break = 0;

        if (args.size()<3)
            throw CommandException("args number not matched (<3)");

        iter_time = stoi(args[0]);
        w = stod(args[1]);
        mode = stoi(args[2]);
        
        if (args.size()>3)
        {
            if (args.size()!=5)
                throw CommandException("args number not matched (!=5)");
            
            enable_break = checkYes(args[3]);
            
            if (enable_break)
                threshould_break = stod(args[4]);
        }
        
        nodeBaseGS(arr, iter_time, w, mode, enable_break, threshould_break, nullptr);
        arr->pushLog(os.str());
    }
    
    void nodeBaseGS(Array2D *, int iter_number = -1, double w = 1.0, int mode = 0, bool enable_break = false, double threshould_break = 0, vec *ret_out = nullptr);
    
    vector<string> args;
};



void NodebasedGSMethod::nodeBaseGS(Array2D *array, int iter_number, double w, int mode, bool enable_break, double threshould_break, vec *ret_out)
{

    auto &RDLine = array->RDLine;
    auto &n = array->n, &m = array->m;
    auto &v = array->v;
    // currently, we only support one kind of wire resistance
    double g_wire = 1.0/RDLine[0][0];

    vector<double> vin(n), vout(m);

    //n is row size, m is col size
    Mat g(n, vector<double>(m)), R = g;
    vec out(m);
    vector<vector<vector<double>>> vup(2, vector<vector<double>>(n, vector<double>(m)));
    vector<vector<vector<double>>> vdown(2, vector<vector<double>>(n, vector<double>(m)));
   
    //set up
    for (int i=0; i<n; ++i)
        vin[i] = v[1][i];
    
    for (int j=0; j<m; ++j)
        vout[j] = v[2][j];
    
    auto &cellR_states = array->cellR_states;
    auto &arr = array->arr;
    for (int i=0; i<n; ++i)
    {
        for (int j=0; j<m; ++j)
        {
            g[i][j] = 1.0/cellR_states[arr[i][j]];
            R[i][j] = 1.0/g[i][j];
        }
    }
   
    for (int i=0; i<n; ++i)
    {
        for (int j=0; j<m; ++j)
        {
            vup[1][i][j] = vup[0][i][j] = vin[i];
            vdown[1][i][j] = vdown[0][i][j] = vout[j];
        }
    }   
   
    auto pre = [](int x) -> int
    {
        return 1-(x&1);
    };
   
    auto now = [](int x) -> int
    {
        return x&1;
    };
   
    // use sci_china way to init the array, then use GS method
    // we found it useless, do delete the code, but remains the mode bit.
    if ((mode &1) == 1)
    {
        throw CommandException("mode bit 0 is not supported now");
    }
   
    int record = -1;
    double g2_wire = g_wire*2;
    if (!(m>=2 && n>=2)) //assert(m>=2&&n>=2); 为了避免一些特判，我这边假定阵列是2x2以上。原因是当一个维度大小为1时，需要进行边界的特别处理
    {
        throw CommandException("m and n should greater than 2");
    }
    
    // calculate the best omega via Theoretical calculation
    // now it's useless. Use fast omega method calculate via trisection method is better.
    if ((mode&4)!=0)
    {
        // time_record t(os);
        // w = best_omega_for_GSmethod(vin, g, g_wire, n, m);
        // os << "best w is " << w << endl;
        throw CommandException("mode bit 2 is not supported now");
    }
    
    auto &RULine = array->RDLine;
    if ((mode &2) == 0)
    {
        time_record t(os);
        for (int iter = 0; iter<iter_number; ++iter)
        {
            int x = now(iter);
            int y = 1-x;
            for (int i=0; i<n; ++i)
            {
                for (int j=0; j<m; ++j)
                {
                    double sumg;// sumi=0;
                    double l=(j==0? vin[i] : vup[x][i][j-1]), r, o = vdown[y][i][j];
                    if (j==m-1)
                    {
                        sumg = 1.0/RULine[i][j]+g[i][j];
                        vup[x][i][j] = w*(l/RULine[i][j]+g[i][j]*o)/sumg + (1-w)*vup[y][i][j];
                    }
                    else 
                    {
                        r = vup[y][i][j+1];
                        sumg = 1.0/RULine[i][j] + 1.0/RULine[i][j+1] +g[i][j];
                        vup[x][i][j] = w*(l/RULine[i][j]+r/RULine[i][j+1]+g[i][j]*o)/sumg + (1-w)*vup[y][i][j];
                    }
   
                    r = (i==n-1? vout[j] : vdown[y][i+1][j]);
                    o = vup[x][i][j];
                    if (i==0)
                    {
                        sumg = 1.0/RDLine[j][1]+g[i][j];
                        vdown[x][i][j] = w*(r/RDLine[j][1]+g[i][j]*o)/sumg + (1-w)*vdown[y][i][j];
                    }
                    else
                    {
                        l = vdown[x][i-1][j];
                        sumg = 1.0/RDLine[j][i] + 1.0/RDLine[j][i+1]+g[i][j];
                        vdown[x][i][j] = w*(l/RDLine[j][i] + r/RDLine[j][i+1]+g[i][j]*o)/sumg + (1-w)*vdown[y][i][j];
                    }
                }
            }
   
            if (enable_break)
            {
                double mx = 0;
                for (int i=0; i<n; ++i)
                {
                    for (int j=0; j<m; ++j)
                    {
                        mx = std::max(mx, std::abs(vdown[x][i][j]-vdown[y][i][j]));
                        mx = std::max(mx, std::abs(vup[x][i][j]-vup[y][i][j]));
                    }
                }
                if (mx<threshould_break)
                {
                    record = iter;
                    break;
                }
            }
        }
    }
   
    os << "GS method mem usage  = " << 
        (sizeof(int)*5 + sizeof(double)*3 + 
        m*n*sizeof(double)*5 + m*sizeof(double)*2 + n*sizeof(double)) 
        << " bytes (Compuated by simply multiplying the number"
        " of variables by the variable space)" << std::endl;
   
    int final_i = pre(iter_number);
    if (record != -1)
    {
        os << "threshould break, iter times =  " << record << endl;
        final_i = now(record);
    }
    else
        os << "iter times = " << iter_number << endl;
        
    {
        // Obtain the sum of current (I) errors for each node
        auto get_errI_sum = [&]()->double
        {
            double errI_sum = 0;
            for (int i=0; i<n; ++i)
            {
                // errI_sum += fabs( (vin[i]-vup[final_i][i][0]+(m>1? vup[final_i][i][1]-vup[final_i][i][0] : 0))*g_wire+(vdown[final_i][i][0]-vup[final_i][i][0])*g[i][0]);
                for (int j=1; j<m-1; ++j)
                {
                    errI_sum += fabs( (vup[final_i][i][j-1]-vup[final_i][i][j]+vup[final_i][i][j+1]-vup[final_i][i][j])*g_wire + (vdown[final_i][i][j]-vup[final_i][i][j])*g[i][j]  );
                }
                if (m>1)
                    errI_sum += fabs( (vup[final_i][i][m-2]-vup[final_i][i][m-1])*g_wire + (vdown[final_i][i][m-1]-vup[final_i][i][m-1])*g[i][m-1]  );
            }
            //answer vdown[final_i] check
            for (int j=0; j<m; ++j)
            {
                errI_sum += fabs( (vup[final_i][0][j]-vdown[final_i][0][j])*g[0][j] + (n>1? vdown[final_i][1][j]-vdown[final_i][0][j] : 0)*g_wire ); 
            }
            for (int i=1; i<n-1; ++i)
            {
                for (int j=0; j<m; ++j)
                {
                    errI_sum += fabs( (vup[final_i][i][j]-vdown[final_i][i][j])*g[i][j]+ (vdown[final_i][i-1][j]-vdown[final_i][i][j]+vdown[final_i][i+1][j]-vdown[final_i][i][j])*g_wire );
                }
            }
            return errI_sum;
        };
        
        // output answer to fastmode_nodebaseGS.out
        std::ofstream of("fastmode_nodebasedGS.out");
        double errIsum = get_errI_sum();
        of << "error I sum = " << errIsum << endl;
        os << "error I sum = " << errIsum << endl;
        of << "vup = " << endl;
   
        of << std::scientific << std::setprecision(12);
        for (int i=0; i<n; ++i)
        {
            for (int j=0; j<m; ++j)
                of << std::setw(20) << vup[final_i][i][j];
            of << endl;
        }
        of << endl << "vdown = " << endl; 
        for (int i=0; i<n; ++i)
        {
            for (int j=0; j<m; ++j)
                of << std::setw(20) << vdown[final_i][i][j];
            of << endl;
        }
        of << endl << "iout = " << endl;
   
        for (int i=0; i<n; ++i)
        {
            for (int j=0; j<m; ++j)
                out[j] += (vup[final_i][i][j]-vdown[final_i][i][j])*g[i][j];
        }
        for (int j=0; j<m; ++j)
        {
            of << std::setw(20) << out[j];
        }
        of << endl;
   
        if (ret_out != nullptr)
        {
            for (int j = 0; j < m; ++j)
                (*ret_out)[j] = out[j];
        }
    }
}

}

#endif