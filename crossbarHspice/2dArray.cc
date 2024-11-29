#include "2dArray.h"
#include "ir_drop_fast_solver.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cctype>

using std::accumulate;
using std::fabs;
using std::min;
using std::cerr;
using namespace CBA;

bool checkYes(string str)
{
    std::transform(str.begin(), str.end(), str.begin(), tolower);
    if (str=="yes" || str=="1" || str=="true" || str=="y" || str=="t")
        return true;
    return false;
}

bool array2D::setParas(string s, string t)
{
    stringstream ss;
    ss << t << endl;
    if (debug) 
        cout << s << ' ' << t << endl;
    auto getDir=[](const string &dir)->int
    {
        int d = 0;
        if (dir == "up")
            d = 0;
        else if (dir == "left")
            d = 1;
        else if (dir == "down")
            d = 2;
        else if (dir == "right")
            d = 3;
        return d;
    };

    if (s == "arraySize")
    {
        int n, m;
        ss >> n >> m;
        setArraySize(n, m);
        cout << "set array size to wordline = " << n << " bitline = " << m << endl;
    }
    else if (s == "nodebasedGSMethod") // remains to check the correctness
    {
        string str;
        int iter_time = -1;
        double w = 1;
        int cnt = 0;
        int mode = 0;
        bool enable_break = false;
        double threshould_break = 0;
        while (ss >> str)
        {
            if (cnt == 0)
                iter_time = stoi(str);
            else if (cnt==1)
                w = stof(str);
            else if (cnt == 2)
            {
                mode = stoi(str);
            }
            else
            {
                if (checkYes(str))
                {
                    enable_break = true;
                    ss >> threshould_break;
                }
                break;
            }
            ++cnt;
        }
        cout << "evalutate ir drop using GS-method, w = " << w << " , print to file fastmode_nodebasedGS.out" << endl;
        nodeBasedGS(iter_time, w, mode, enable_break, threshould_break, nullptr);

        auto check_status = [&]()->bool  // return it is Singular type. if true , use general GS
        {
            for (auto &i : bv[0])
                if (i==1) 
                    return true;
            for (auto &i : bv[1])
                if (i==0)
                    return true;
            for (auto &i : bv[2])
                if (i==0)
                    return true;
            for (auto &i : bv[3])
                if (i==1)
                    return true;
            
            return false;
        };
        if (!check_status())
        {
            cout << "evalutate ir drop using GS-method (special version for vmm only), w = " << w << " , print to file fastmode_nodebasedGS.out" << endl;
            nodeBasedGS(iter_time, w, mode, enable_break, threshould_break, nullptr);
        }
        else 
        {
            cout << "evalutate ir drop using general GS-method, w = " << w << " , print to file fastmode_nodebasedGeneralGS.out" << endl;
            nodeBasedGeneralGS(iter_time, w, enable_break, threshould_break);
        }
        cout << endl << endl;
    }
    else if (s == "inputVstates")
    {
        cout << "set cell input v states, note that, this is useful when calling randSetInputV" << endl;
        ss >> inputV_num;
        inputV_states = vector<double>(inputV_num);
        for (int i=0; i<inputV_num; ++i)
            ss >> inputV_states[i];
    }
    else if (s == "randSetInputV")
    {
        string dir;
        double p_sum = 0;
        vector<double> p;

        ss >> dir;
        int d = getDir(dir);

        cout << "randSetInputV should be used after inputVstates, rand p = ";
        for (int i=0; i<inputV_num; ++i)
        {
            double x;
            ss >> x;
            p_sum += x;
            p.push_back(x);
            cout << x << " ";
            if (i>1)
                p[i] += p[i-1];
        }
        cout << endl;
        if (std::fabs(p_sum-1)>1e-6)
        {
            cout << "Error: p sum is not 1" << endl;
            throw 0;
        }
        std::srand(time(NULL));

        int len = (dir=="up" || dir=="down")? m : n;
        for (int i=0; i<len; ++i)
        {
            int data;
            int rd = rand()%10000;
            bool fd = false;
            for (int k=0; k<inputV_num; ++k)
            {
                if (p[k]*10000>rd)
                {
                    fd = true;
                    data = k;
                    break;
                }
            }
            if (!fd)
            {
                data=inputV_num-1;
            }
            // setParas("setLineV", dir+" "+toStr(i)+" "+toStr(inputV_states[data], 8));
            d = getDir(dir);
            v[d].at(i) = inputV_states[data];
            // cout << "set dir " << d << " line No." << i << " to voltage " << inputV_states[data] << " [d=(0, 1, 2, 3)->(up, left, down, right)]" << endl;
        }
        cout << "randSetInputV "+dir+" finish" << endl;
    }
    else if (s == "randSetR")
    {
        vector<double> p;
        double p_sum = 0;
        cout << "randSetR should be used after cellRstates, rand p = ";
        for (int i=0; i<cellR_num; ++i)
        {
            double x;
            ss >> x;
            p_sum += x;
            p.push_back(x);
            cout << x << " ";
            if (i>1)
                p[i] += p[i-1];
        }
        cout << endl;
        if (std::fabs(p_sum-1)>1e-6)
        {
            cout << "Error: p sum is not 1" << endl;
            throw 0;
        }
        std::srand(time(NULL));
        for (int i=0; i<n; ++i)
        {
            for (int j=0; j<m; ++j)
            {
                int rd = rand()%10000;
                bool fd = false;
                for (int k=0; k<cellR_num; ++k)
                {
                    if (p[k]*10000>rd)
                    {
                        fd = true;
                        arr[i][j] = k;
                        break;
                    }
                }
                if (!fd)
                {
                    arr[i][j]=cellR_num-1;
                }
            }
        }
        cout << "randSetR finish" << endl;
    }
    else if (s == "cellRstates")
    {
        cout << "set cell R states, note that, this is only avaliable when cell R is linear" << endl;
        ss >> cellR_num;
        cellR_states = vector<double>(cellR_num);
        for (int i=0; i<cellR_num; ++i)
            ss >> cellR_states[i];
    }
    else if (s == "useRtypeReRAMCell")
    {
        string temp;
        ss >> temp;
        if (checkYes(temp))
            useRtypeCell = true;
        else
            useRtypeCell = false;
    }
    else if (s == "fastmode")
    {
        string temp;
        ss >> temp;
        if (checkYes(temp))
        {
            this->fast_mode = true;
            ss >> cellR_num;
            cellR_states = vector<double>(cellR_num);
            for (int i=0; i<cellR_num; ++i)
                ss >> cellR_states[i];
        }
        else
            this->fast_mode = false;
        cout << "fast mode " << (this->fast_mode? "on" : "off") << endl;
    }
    else if (s == "selector")
    {
        string str;
        ss >> str;
        if (str == "yes" || str == "y")
        {
            addSelector();
            cout << "selector put into crossbar" << endl;
        }
        else
        {
            removeSelector();
            cout << "selector remove" << endl;
        }
        
    }
    else if (s == "line_resistance")
    {
        double r;
        ss >> r;
        for (auto &i : RULine)
          for (auto &j : i)
              j = r;

        for (auto &i : RDLine)
          for (auto &j : i)
              j = r;
        cout << "set line_resistance to " << r << endl;
    }
    else if (s == "rload")
    {
        string str;
        double r;
        ss >> str >> r;
        if (str == "all")
        {
            for (auto &i : RULine)
                i[0] = i[i.size()-1] = r;
            for (auto &i : RDLine)
                i[0] = i[i.size()-1] = r;
            for (int i=0; i<4; ++i)
                r_load[i] = r;
            cout << "set all line resistance load to r_load " << r << " omega" << endl;
        }
        else if (str == "left" || str == "right")
        {
            for (auto &i : RULine)
                i[str=="left"? 0 : (i.size()-1)] = r;
            r_load[str=="left"? 1 : 3] = r; 
        }   
        else if (str == "down" || str == "up")
        {
            for (auto &i : RDLine)
                i[str=="up"? 0 : (i.size()-1)] = r;
            r_load[str=="up"? 0 : 2] = r; 
        }
        cout << "set " + str + " line resistance load to r_load " << r << " omega" << endl; 
    }
    else if (s == "build")
    {
        string str;
        ss >> str;
        buildSpice(str);
        cout << "build the array to file " << str << endl;
    }
    else if (s == "fastsolve")
    {
        int iter_times, get_nodal_voltage_from_table = 0, enable_break = 0;
        double break_th = 1e-9;
        ss >> iter_times;
        ss >> get_nodal_voltage_from_table; 
        ss >> enable_break >> break_th;
        fastsolve(iter_times, get_nodal_voltage_from_table, enable_break, break_th);
        cout << "fast solve, iter_times = " + to_string(iter_times) + ", print to file fastmode.out" << endl;
    }
    else if (s == "iras_table")
    {
        string temp;
        ss >> temp;
        this->table_name = temp;
    }
    else if (s == "ir_aihwkit")
    {
        double ir_drop_beta = 0.3;
        double vin_max = 1.0;
        ss >> ir_drop_beta >> vin_max;

        IR_aihwkit(ir_drop_beta, vin_max);
        cout << "evaluate ir drop using the method in aihwkit, ir_drop beta = "+ to_string(ir_drop_beta)+ ", print to file fastmode_aihwkit.out" << endl;
        cout << endl << endl;
    }
    else if (s == "ir_scichina")
    {
        double Rs = 2.93;
        ss >> Rs;
        IR_sciChina(Rs);
        cout << "evaluate ir drop using sci china method, print to file fastmode_scichina.out" << endl;
        cout << endl << endl;
    }
    else if (s == "ir_neurosim")
    {
        IR_neurosim();
        cout << "evaluate ir drop using neurosim method, print to file fastmode_neurosim.out" << endl;
        cout << endl << endl;
    }
    else if (s == "ir_free")
    {
        IR_free();
        cout << "evaluate ir drop using neurosim method, print to file fastmode_neurosim.out" << endl;
        cout << endl << endl;
    }
    else if (s == "ir_pbia")
    {
        int iter_times;
        ss >> iter_times; 
        bool enable_break = false;
        string str;
        double threshould_break=0;
        while (ss >> str)
        {
            if (checkYes(str))
            {
                enable_break = true;
                ss >> threshould_break;
                break;
            }
        }
        IR_PBIA(iter_times, enable_break, threshould_break);
        cout << "evalutate ir drop using Physics-Based Iterative Algorithm, print to file fastmode_pbia.out" << endl;
        cout << endl << endl;
    }
    else if (s == "setLineR")
    {
        string dir;
        int x, y;
        double z;
        ss >> dir >> x >> y >> z;
        if (dir == "up")
        {
            RULine.at(x).at(y) = z;
            cout << "RDLine (" << x << ", " << y << ") to " << z << endl; 
        }
        else
        {
            RDLine.at(x).at(y) = z;
            cout << "RDLine (" << x << ", " << y << ") to " << z << endl; 
        }
    }
    else if (s == "capacitance" || s=="capacity")
    {
        int x;
        ss >> x;
        if (x != 0)
            hasCapacity = true;
        else
            hasCapacity = false;
        if (!hasCapacity)
        {
            throw 0;
            // return false;
        }
        ss >> capacitance[0] >> capacitance[1] >> capacitance[2] >> capacitance[3];
    }
    else if (s == "setUseLine")
    {
        string dir;
        int x, d;
        ss >> dir;
        d = getDir(dir);
       
        while (ss >> x)
        {
            if (x==-1)
            {
                for (auto &i : bv[d])
                    i = 1;
                cout << "set use dir " << d << " all line" << endl;
            }   
            else
            {
                bv[d].at(x) = 1;
                cout << "set use dir " << d << "line No. " << x << endl;
            }
        }
    }
    else if (s == "setNotUseLine")
    {
        string dir;
        int x, d;
        ss >> dir;
        d = getDir(dir);
     
        while (ss >> x)
        {
            if (x==-1)
            {
                for (auto &i : bv[d])
                    i = 0;
                cout << "set not use dir " << d << " all line [d=(0, 1, 2, 3)->(up, left, down, right)]" << endl;
            }   
            else
            {
                bv[d].at(x) = 0;
                cout << "set not use dir " << d << "line No. " << x << endl;
            }
        }
    }
    else if (s == "setLineV")
    {
        string dir;
        int d, x;
        double y;
        ss >> dir >> x >> y;
        d = getDir(dir);
     
        if (x==-1)
        {
            for (auto &i : v[d])
                i = y;
            cout << "set dir " << d << " all line to voltage " << y << " [d=(0, 1, 2, 3)->(up, left, down, right)]" << endl;
        }   
        else
        {
            v[d].at(x) = y;
            cout << "set dir " << d << " line No." << x << " to voltage " << y << " [d=(0, 1, 2, 3)->(up, left, down, right)]" << endl;
        }
    }
    else if ( s == "setUseLine++" )
    {
        string dir;
        int x, d;
        ss >> dir >> x;
        d = getDir(dir);
     
        if (x==-1)
        {
            for (auto &i : bv[d])
                i = 2;
            cout << "set use(++) dir " << d << " all line [d=(0, 1, 2, 3)->(up, left, down, right)]" << endl;
        }   
        else
        {
            bv[d].at(x) = 2;
            cout << "set use(++) dir " << d << "line No. " << x << endl;
        }
    }
    else if (s == "dc" || s=="ac")
    {
        dc_ac_tran_type = s;
    }
    else if (s == "setLineV++")
    {
        string dir, type;
        int d, x;
        string y;
        ss >> dir >> x;
        getline(ss, y);

        d = getDir(dir);
     
        if (x==-1)
        {
            for (auto &i : v_str[d])
                i = y;
            cout << "set (++)dir " << d << " all line to voltage " << y << " [d=(0, 1, 2, 3)->(up, left, down, right)]" << endl;
        }   
        else
        {
            v_str[d].at(x) = y;
            cout << "set (++)dir " << d << " line No." << x << " to voltage " << y << " [d=(0, 1, 2, 3)->(up, left, down, right)]" << endl;
        }
    }
    else if (s == "setCellR")
    {
        int x, y, z;
        ss >> x >> y >> z;
        cout << "setCellR " << x << ' ' << y << ' ' << z << endl;
        if (x==-1)
        {
            for (auto &i : arr)
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
                for (auto &j : arr.at(x))
                    j = z;
            }
            else
                arr.at(x).at(y) = z;
        }
    }
    else if (s == "simpleWriteOne")
    {
        vector<int> x;
        vector<int> y;
        int p;
        bool w=true;
        double Vset;
        ss >> Vset;
        while (ss >> p)
        {
            if (w)
                x.push_back(p);
            else
                y.push_back(p);
            w=!w;
        }
        if (x.size()!=y.size())
        {
            cerr << "simpleWriteOne error" << endl;
            throw 0;
        }
        for (auto i=x.size()-1; i>=1; --i)
            if (x[i]!=x[i-1])
            {
                printf("simpleWriteOne wordline is not same!\n");
                return false;
            }

        setParas("setUseLine", "left -1");
        setParas("setLineV", "left -1 " + std::to_string(Vset/2));
        setParas("setLineV", "left "+std::to_string(x[0])+" "+std::to_string(Vset));

        setParas("setUseLine", "down -1");
        setParas("setLineV", "down -1 "+std::to_string(Vset/2));
        for (auto &i : y)
        {
            setParas("setLineV", "down "+std::to_string(i)+" "+std::to_string(0));
        }
        int len=x.size();
        for (int i=0; i<len; ++i)
        {
            int nx = x[i], ny = y[i];

            gen.push2bot(LogicUnit(UnitType::senseV, {nx*m+ny+1}, {dc_ac_tran_type}));
            gen.push2bot(LogicUnit(UnitType::senseV, {nx*m+ny+1+n*m}, {dc_ac_tran_type}));
        }
    }
    else if (s == "topString")
    {
        gen.push2top(t);
    }
    else if (s == "bottomString")
    {
        gen.push2bot(t);
    }
    else if (s == "mode")
    {
        string str;
        ss >> str;
        if (str == "static" || str == "0")
            gen.mode = 0;
        else
            gen.mode = 1;
    }
    else if (s == "simpleWriteZero")
    {
        vector<int> x;
        vector<int> y;
        int p;
        bool w=true;
        double Vrst;
        ss >> Vrst;
        while (ss >> p)
        {
            if (w)
                x.push_back(p);
            else
                y.push_back(p);
            w=!w;
        }
        if (x.size()!=y.size())
        {
            cerr << "simpleWriteZero error" << endl;
            throw 0;
        }
        for (auto i=x.size()-1; i>=1; --i)
            if (x[i]!=x[i-1])
            {
                printf("simpleWriteZero wordline is not same!\n");
                return false;
            }

        setParas("setUseLine", "left -1");
        setParas("setLineV", "left -1 "+std::to_string(Vrst/2));
        setParas("setLineV", "left "+std::to_string(x[0])+" "+std::to_string(0));

        setParas("setUseLine", "down -1");
        setParas("setLineV", "down -1 "+std::to_string(Vrst/2));
        for (auto &i : y)
        {
            setParas("setLineV", "down "+std::to_string(i)+" "+std::to_string(Vrst));
        }
        int len=x.size();
        for (int i=0; i<len; ++i)
        {
            int nx = x[i], ny = y[i];
            gen.push2bot(LogicUnit(UnitType::senseV, {nx*m+ny+1}, {dc_ac_tran_type}));
            gen.push2bot(LogicUnit(UnitType::senseV, {nx*m+ny+1+n*m}, {dc_ac_tran_type}));
        }
    }
    else if (s == "simpleReadOut") 
    {
        vector<int> x;
        vector<int> y;
        int p;
        bool w=true;
        double Vread;
        double Rsense;
        bool isvoltageRead;
        std::string str;
        ss >> str;
        if (str=="volread")
        {
            isvoltageRead = true;
            ss >> Rsense;
        }
        else 
            isvoltageRead = false;
        ss >> Vread;
        while (ss >> p)
        {
            if (w)
                x.push_back(p);
            else
                y.push_back(p);
            w=!w;
        }
        if (x.size()!=y.size())
        {
            cerr << "simpleReadOut error" << endl;
            throw 0;
        }
        for (auto i=x.size()-1; i>=1; --i)
            if (x[i]!=x[i-1])
            {
                printf("readout wordline is not same!\n");
                return false;
            }

        setParas("setUseLine", "left "+std::to_string(x[0]));
        setParas("setLineV", "left "+std::to_string(x[0])+" "+std::to_string(Vread));
        int otherPoint = hasSelector? 3*n*m : 2*n*m;
        for (auto &i : y)
        {
            setParas("setUseLine", "down "+std::to_string(i));
            setParas("setLineV", "down "+std::to_string(i)+" "+std::to_string(0));
            if (isvoltageRead) 
            {
                RDLine[i][n] = Rsense;
                gen.push2bot(LogicUnit(UnitType::senseV, {(n-1)*m+i+1+n*m}, {dc_ac_tran_type}));
            }
            else
            {
                gen.push2bot(LogicUnit(UnitType::senseI, {otherPoint+m+n+i+1, 0}, {dc_ac_tran_type}));
            }
        }
    }
    else if (s == "senseCellV")
    {
        vector<int> x, y;
        int p;
        bool w = true;
        while (ss >> p)
        {
            if (w)
                x.push_back(p);
            else
                y.push_back(p);
            w=!w;
        }
        if (x.size()!=y.size()) throw 1;
        for (auto &i : x)
          if (i>=n)
              throw 1;
        for (auto &j : y)
          if (j>=m)
              throw 1;
        
        for (int i=0; i<x.size(); ++i)
        {
            int nx = x[i], ny = y[i];
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
                gen.push2bot(LogicUnit(UnitType::senseV, {prt_x[id]*m+prt_y[id]+1}, {dc_ac_tran_type}));
                gen.push2bot(LogicUnit(UnitType::senseV, {prt_x[id]*m+prt_y[id]+1+n*m}, {dc_ac_tran_type}));
            }
        }
    }
    else if (s == "senseBitlineI")
    {
        vector<int> y;
        int p;
        string dir;
        ss >> dir;
        while (ss >> p)
        {
            if (p>=m)
                throw 1;
            if (p>=0) 
                y.push_back(p);
            else if (p==-1)
            {
                for (int j=0; j<m; ++j)
                    y.push_back(j);
                break;
            } 
        }
        int otherPoint = hasSelector? 3*n*m : 2*n*m;
        for (int ny : y)
        {
            if (debug) cout << "senseI " << ny << endl;
            if (dir=="down")
                gen.push2bot(LogicUnit(UnitType::senseI, {otherPoint+m+n+ny+1, 0}, {dc_ac_tran_type}));
            else 
                gen.push2bot(LogicUnit(UnitType::senseI, {otherPoint+ny+1, 0}, {dc_ac_tran_type}));
        }
    }
    else if (s == "senseWordlineI")
    {
        vector<int> x;
        int p;
        string dir;
        ss >> dir;
        while (ss >> p)
        {
            if (p>=n)
                throw 1;
            x.push_back(p);
        }
        int otherPoint = hasSelector? 3*n*m : 2*n*m;
        for (int nx : x)
        {
            if (dir=="left")
                gen.push2bot(LogicUnit(UnitType::senseI, {otherPoint+m+nx+1, 0}, {dc_ac_tran_type}));
            else 
                gen.push2bot(LogicUnit(UnitType::senseI, {otherPoint+m*2+n+nx+1, 0}, {dc_ac_tran_type}));
        }
    }
    else if (s == "FCM")
    {
        string file, mode;
        int mode_int = 0;
        ss >> mode >> file;
        if (mode == "") 
            mode = "test";
        
        mode_int = mode=="test"? 0 : 1;
        if (file == "")
        {
            if (mode_int == 0)
                file = "IR_FCM.out";
            else
                file = "IR_FCM.conf";
        }
        if (mode_int == 0)
        {
            cout << "Test according to IR_FCM.conf, and test output to file IR_FCM.out" << endl;
        }
        else
        {
            cout << "Generate G', i.e., G_non_ideal corresponding to origin G with line resistance effect, to file " << file << endl;
        }
        IR_FCM(mode_int, file);
    }
    else
    {
        cout << "wrong commnd" << endl;
        //throw 0;
    }
    return true;
}

//only support vmm
// fcm model, see RxNN: A framework for evaluating deep neural networks on resistive crossbars, TCAD, 2020
void array2D::IR_FCM(int mode, string file)
{
    if (mode == 0) // test mode, use file as input, to generate Iout
    {
        ifstream fin(file);
        int file_n, file_m, file_cellR_num;
        Mat file_arr(n, vec(m));
        fin >> file_n >> file_m;
        fin >> file_cellR_num;
        if (!(file_n == n && file_m == m && file_cellR_num == cellR_num))
        {
            cout << "fcm config file has wrong parameters!" << endl;
            return;
        }
        for (int i = 0; i < cellR_num; ++i)
        {
            double temp_R;
            fin >> temp_R;
            if (fabs(temp_R - cellR_states[i]) > 1e-5)
            {
                cout << "fcm config file wrong, has wrong r states" << endl;
                return;
            }
        }
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                int temp_states;
                fin >> temp_states;
                if (arr[i][j] != temp_states)
                {
                    cout << "fcm config file wrong, has wrong cell states" << endl;
                    return;
                }
            }
        }

        vec vin(n);
        for (int i = 0; i < n; ++i)
            vin[i] = v[1][i];

        Mat g(n, vec(m));
        vec Iout(m);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                fin >> g[i][j];
            }
        }

        { // start calculate
            time_record t;
            for (int j = 0; j < m; ++j)
            {
                for (int i = 0; i < n; ++i)
                    Iout[j] += g[i][j]*vin[i];
            }
        }

        cout << "use memory " << 1.0 * (m * n + n) * sizeof(double) / 1024 / 1024 << " MB" << endl;
        ofstream of("IR_FCM.out");
        of << std::scientific << std::setprecision(12);
        of << endl
           << "iout = " << endl;
        for (int j = 0; j < m; ++j)
        {
            of << std::setw(20) << Iout[j];
        }
        of << endl;
    }
    else
    {
        vec record_vin(n);
        Mat g_non_ideal(n, vec(m));
        vec Iout(m);
        for (int i = 0; i < n; ++i)
            record_vin[i] = v[1][i];
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
                v[1][j] = 0;
            v[1][i] = 1;
            nodeBasedGS(1000000, 1.97, 0, true, 1e-13, &g_non_ideal[i]);
        }

        for (int i = 0; i < n; ++i)
            v[1][i] = record_vin[i];

        { // start calculate
            time_record t;
            for (int j = 0; j < m; ++j)
            {
                for (int i = 0; i < n; ++i)
                    Iout[j] += g_non_ideal[i][j] * v[1][i];
            }
        }

        {
            ofstream of("IR_FCM.out");
            of << std::scientific << std::setprecision(12);
            of << endl
               << "iout = " << endl;
            for (int j = 0; j < m; ++j)
            {
                of << std::setw(20) << Iout[j] << " ";
            }
            of << endl;
        }
        {
            ofstream of(file);
            of << std::scientific << std::setprecision(12);
            of << n << " " << m << " " << cellR_num << endl;
            for (int i = 0; i < cellR_num; ++i)
                of << cellR_states[i] << " ";
            of << endl;
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < m; ++j)
                {
                    of << arr[i][j] << " ";
                }
                of << endl;
            }
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < m; ++j)
                {
                    of << g_non_ideal[i][j] << " ";
                }
                of << endl;
            }
        }
    }
}

//only support vmm
// remains to check the correctness
void array2D::IR_aihwkit(double ir_drop_beta = 0.3, double vin_max = 1.0)
{

    // currently, we only support one kind of wire resistance
    double g_wire = 1.0/RDLine[0][0];
    vector<vector<double>> g(n, vector<double>(m));
    vector<double> vin(n), out(m);
    for (int i=0; i<n; ++i)
        vin[i] = v[1][i];
    
    for (int j=0; j<m; ++j)
    {
        if (v[2][j]>0)
        {
            cout << "error: ir_drop evaluation in aihwkit, only support vmm, but now bitline is connected to some voltages" << endl;
            return;
        }
    }

    for (int i=0; i<n; ++i)
    {
        for (int j=0; j<m; ++j)
        {
            g[i][j] = 1.0/cellR_states[arr[i][j]];
        }
    }
    std::cout << "aihwkit fastmode ";
    {
        time_record t;
        ir_drop_fastsolve_cpu_singleMat_in_aihwkit(vin, out, arr, g_wire, 1.0/cellR_states.back(), ir_drop_beta, n, m);
    }

    std::cout << "ir aihwkit mem = " << (sizeof(int)*4 + sizeof(double)*6 + m*n*sizeof(double) + 3*m*sizeof(double) + 2*n*sizeof(double)) << " bytes" << std::endl;

    //aihwkit 产生的结果不是电流，而是转换成数值后的结果，为了与其他模型公平比较，需要重新转换为对应电流
    ofstream of("fastmode_aihwkit.out");
    of << std::scientific << std::setprecision(12);
    of << endl << "iout = " << endl;
    for (int j=0; j<m; ++j)
    {
        of << std::setw(20) << (out[j]/(vin_max*n*(cellR_states.size()-1)))*vin_max*n*(1.0/cellR_states.back()-1.0/cellR_states.front());
    }
    of << endl;
}

//only support vmm
// remains to check the correctness
void array2D::IR_sciChina(double Rs)
{
    // currently, we only support one kind of wire resistance
    double r_wire = RDLine[0][0];
    vector<vector<double>> R(n, vector<double>(m));
    vector<double> vin(n), Vout(m), Iout(m);
    for (int i=0; i<n; ++i)
        vin[i] = v[1][i];
    
    for (int j=0; j<m; ++j)
    {
        if (v[2][j]>0)
        {
            cout << "error: ir_drop evaluation in scichina, only support vmm, but now bitline is connected to some voltages" << endl;
            return;
        }
    }

    for (int i=0; i<n; ++i)
    {
        for (int j=0; j<m; ++j)
        {
            R[i][j] = cellR_states[arr[i][j]];
        }
    }
    
    std::cout << "sci china (em) fastmode ";
    {
        time_record t;
        ir_drop_fastsolve_cpu_singleMat_in_scichina(vin, Vout, Iout, R, r_wire, Rs, n, m);
    }

    std::cout << "ir scichina (em) mem = " << (sizeof(int)*4 + sizeof(double)*2 + 8*m*n*sizeof(double) + 3*m*sizeof(double) + n*sizeof(double)) << " bytes" << std::endl;

    ofstream of("fastmode_scichina.out");
    of << std::scientific << std::setprecision(6);

    of << endl << "iout = " << endl;
    for (int j=0; j<m; ++j)
    {
        of << std::setw(20) << Iout[j];
    }
    of << endl;
}

//only support vmm
// remains to check the correctness
void array2D::IR_neurosim()
{
 // currently, we only support one kind of wire resistance
    double r_wire = RDLine[0][0];
    Mat R(n, vector<double>(m)); // n is rowsize
    vector<double> vin(n), Iout(m); // m is colsize
    for (int i=0; i<n; ++i)
        vin[i] = v[1][i];
    
    for (int j=0; j<m; ++j)
    {
        if (v[2][j]>0)
        {
            cout << "error: ir_drop evaluation in neurosim, only support vmm, but now bitline is connected to some voltages" << endl;
            return;
        }
    }

    for (int i=0; i<n; ++i)
    {
        for (int j=0; j<m; ++j)
        {
            R[i][j] = cellR_states[arr[i][j]];
        }
    }
    
    std::cout << "neurosim fastmode ";
    {
        time_record t;
        ir_drop_fastsolve_cpu_singleMat_in_neurosim(vin, Iout, R, r_wire, n, m);
    }

    std::cout << "ir neurosim mem = " << (sizeof(int)*4 + sizeof(double)*2 + m*n*sizeof(double) + m*sizeof(double) + n*sizeof(double)) << " bytes" << std::endl;

    ofstream of("fastmode_neurosim.out");
    of << std::scientific << std::setprecision(12);

    of << endl << "iout = " << endl;
    for (int j=0; j<m; ++j)
    {
        of << std::setw(20) << Iout[j];
    }
    of << endl;
}

//only support vmm
// remains to check the correctness
void array2D::IR_free()
{
 // currently, we only support one kind of wire resistance
    double r_wire = RDLine[0][0];
    Mat R(n, vector<double>(m)); // n is rowsize
    vector<double> vin(n), Iout(m); // m is colsize
    for (int i=0; i<n; ++i)
        vin[i] = v[1][i];
    
    for (int j=0; j<m; ++j)
    {
        if (v[2][j]>0)
        {
            cout << "error: ir_drop evaluation in ir free, only support vmm, but now bitline is connected to some voltages" << endl;
            return;
        }
    }

    for (int i=0; i<n; ++i)
    {
        for (int j=0; j<m; ++j)
        {
            R[i][j] = cellR_states[arr[i][j]];
        }
    }
    
    std::cout << "ir free mode  ";
    {
        time_record t;
        for (int j=0; j<m; ++j)
        {
            Iout[j] = 0;
            for (int i=0; i<n; ++i)
            {
                Iout[j] += vin[i]/(R[i][j]);
            }
        }
    }

    std::cout << "ir free mem = " << (sizeof(int)*2 + m*n*sizeof(double) + m*sizeof(double) + n*sizeof(double)) << " bytes" << std::endl;

    ofstream of("ir_drop_free.out");
    of << std::scientific << std::setprecision(6);

    of << endl << "iout = " << endl;
    for (int j=0; j<m; ++j)
    {
        of << std::setw(14) << Iout[j];
    }
    of << endl;
}

void array2D::nodeBasedGeneralGS(int iter_number = -1, double w = 1.0, bool enable_break = false, double threshould_break = 0)
{
     // currently, we only support one kind of wire resistance
    double g_wire = 1.0/RDLine[0][0];


    //n is row size, m is col size
    Mat g(n, vector<double>(m)), R = g;
    vector<vector<vector<double>>> vup(2, vector<vector<double>>(n, vector<double>(m)));
    vector<vector<vector<double>>> vdown(2, vector<vector<double>>(n, vector<double>(m)));

    // any initials are ok, but we set wordline voltage assuming ir drop free from left wordline's input v as initials
    // and we set vdown all to zero for simplity.
    for (int i=0; i<n; ++i)
    {
        for (int j=0; j<m; ++j)
        {
            if (bv[1][i]==1)
                vup[1][i][j] = vup[0][i][j] = v[1][i];
            else
                vup[1][i][j] = vup[0][i][j] = 0;
            
            vdown[1][i][j] = vdown[0][i][j] = 0;
        }
    }

    for (int i=0; i<n; ++i)
    {
        for (int j=0; j<m; ++j)
        {
            g[i][j] = 1.0/cellR_states[arr[i][j]];
            R[i][j] = 1.0/g[i][j];
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


    int record = -1;
    {
        time_record tim;

        for (int iter = 0; iter<iter_number; ++iter)
        {
            int x = now(iter), y=1-x;
            for (int i=0; i<n; ++i)
            {
                for (int j=0; j<m; ++j)
                {
                    double sumg;
                    double l_side, r_side, other_side;
                    sumg = g[i][j] + ((j!=0 || (j==0 && bv[1][i]==1))? 1.0/RULine[i][j]: 0) + ((j!=m-1 || (j==m-1 && bv[3][i]==1)) ? 1.0/RULine[i][j+1] : 0);
                    l_side = (j==0? (bv[1][i]==0? 0 : v[1][i]) : vup[x][i][j-1])/RULine[i][j];
                    r_side = (j==m-1? (bv[3][i]==0? 0 : v[3][i]) : vup[y][i][j+1])/RULine[i][j+1];
                    other_side = vdown[y][i][j]*g[i][j];

                    vup[x][i][j] = w*(l_side+r_side+other_side)/sumg + (1-w)*vup[y][i][j];

                    
                    sumg = g[i][j] + ((i!=0 || (i==0 && bv[0][j]==1))? 1.0/RDLine[j][i]: 0) + ((i!=n-1 || (i==n-1 && bv[2][j]==1)) ? 1.0/RDLine[j][i+1] : 0);
                    l_side = (i==0? (bv[0][j]==0? 0 : v[0][j]) : vdown[x][i-1][j])/RDLine[j][i];
                    r_side = (i==n-1? (bv[2][j]==0? 0 : v[2][j]) : vdown[y][i+1][j])/RDLine[j][i+1];
                    other_side = vup[x][i][j]*g[i][j];

                    vdown[x][i][j] = w*(l_side+r_side+other_side)/sumg + (1-w)*vdown[y][i][j];


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
    int final_i = pre(iter_number);
    if (record!=-1)
    {
        cout << "threshould break, iter times =  " << record << endl;
        final_i = now(record);
    }
    {
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
        ofstream of("fastmode_nodebasedGeneralGS.out");
        double errIsum = get_errI_sum();
        of << "error I sum = " << errIsum << endl;
        cout << "error I sum = " << errIsum << endl;
        of << "vup = " << endl;

        of << std::scientific << std::setprecision(6);
        for (int i=0; i<n; ++i)
        {
            for (int j=0; j<m; ++j)
                of << std::setw(14) << vup[final_i][i][j];
            of << endl;
        }
        of << endl << "vdown = " << endl; 
        for (int i=0; i<n; ++i)
        {
            for (int j=0; j<m; ++j)
                of << std::setw(14) << vdown[final_i][i][j];
            of << endl;
        }
        of << endl << "iout = " << endl;

        vec out(m);
        for (int j=0; j<m; ++j)
            if (bv[2][j]==1)
                out[j] = (vdown[final_i][n-1][j]-v[2][j])/RDLine[j][n];
        for (int j=0; j<m; ++j)
        {
            of << std::setw(14) << out[j];
        }
        of << endl;
    }

}

//remain to check the correctness
void array2D::nodeBasedGS(int iter_number = -1, double w = 1.0, int mode = 0, bool enable_break = false, double threshould_break = 0, vec *ret_out = nullptr)
{
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

    // sci_china way
    if ((mode &1) == 1)
    {
        int M = n, N = m;
        vector<vector<double>> Reqv_up(M, vector<double>(N));
        vector<vector<double>> Reqv_down(M, vector<double>(N));
        vector<vector<double>> Reqv(M, vector<double>(N));
        vector<vector<double>> f(M, vector<double>(N));
        vector<vector<double>> RR(M, vector<double>(N));
        vector<vector<double>> RRR(M, vector<double>(N));
        vector<vector<double>> V_wl(M, vector<double>(N));
        vector<vector<double>> V_bl(M, vector<double>(N));

        vector<double> g_sum_for_bl(N);
        double r_wire = 1/g_wire;
        double Rs = 1/g_wire;

        for (int j=0; j<N; ++j)
        {
            for (int i=0; i<M; ++i)
                g_sum_for_bl[j]+=g[i][j];
        }
        for (int k=0; k<M; ++k)
        {
            for (int j=0; j<N; ++j)
            {
                RR[k][j] = R[k][j]+(M-k)*r_wire+(g_sum_for_bl[j]*R[k][j])*Rs;
            }
        }
        for (int k=0; k<M; ++k)
        {
            for (int j=N-1; j>=0; --j)
            {
                if (j==N-1)
                    RRR[k][N-1] = RR[k][N-1]+r_wire;
                else
                    RRR[k][j] = (RR[k][j]*RRR[k][j+1])/(RR[k][j]+RRR[k][j+1]) + r_wire;
            }
        }

        for (int k=0; k<M; ++k)
        {
            for (int j=0; j<N; ++j)
            {
                if (j==0)
                {
                    V_wl[k][j] = (RRR[k][j]-r_wire)/RRR[k][j]*vin[k]; 
                    // cout << V_wl[k][j] << ' ';
                    continue;
                }
                V_wl[k][j] = (RRR[k][j]-r_wire)/RRR[k][j]*V_wl[k][j-1];
                // cout << V_wl[k][j] << ' ';
            }
            // cout << endl;
        }


        auto parallel = [](double x, double y) -> double
        {
            return x*y/(x+y);
        };

        for (int i=0; i<M; ++i)
        {
            for (int j=0; j<N; ++j)
            {
                if (i==0)
                    Reqv_up[i][j] = r_wire + R[i][j];
                else
                    Reqv_up[i][j] = parallel(Reqv_up[i-1][j], R[i][j]) + r_wire;
            }
        }

        for (int i=M-1; i>=0; --i)
        {
            for (int j=0; j<N; ++j)
            {    
                if (i!=M-1)
                {
                    Reqv_down[i][j] = r_wire+ parallel(Reqv_down[i+1][j], R[i+1][j]);
                    f[i][j] = f[i+1][j]*parallel(Reqv_down[i+1][j], R[i+1][j])/(parallel(Reqv_down[i+1][j], R[i+1][j]) + r_wire);
                }
                else
                {
                    Reqv_down[i][j] = Rs;
                    f[i][j] = 1;
                }
            }
        }

        for (int i=0; i<M; ++i)
        {
            for (int j=0; j<N; ++j)
            {
                if (i==0)
                    Reqv[i][j] = Reqv_down[i][j];
                else
                    Reqv[i][j] = parallel(Reqv_up[i-1][j], Reqv_down[i][j]);
            }
        }

        for (int j=0; j<N; ++j)
        {
            for (int i=0; i<M; ++i)
            {
                V_bl[i][j] = V_wl[i][j]*(Reqv[i][j])/(Reqv[i][j]+R[i][j]);
                vup[1][i][j] = vup[0][i][j] = V_wl[i][j];
                vdown[1][i][j] = vdown[0][i][j] = V_bl[i][j];
            }
        }
    }

    int record = -1;
    double g2_wire = g_wire*2;
    std::cout << "nodebase fast mode ";
    if (!(m>=2 && n>=2)) //assert(m>=2&&n>=2); 为了避免一些特判，我这边假定阵列是2x2以上。原因是当一个维度大小为1时，需要进行边界的特别处理
    {
        std::cout << "m and n should greater than 2" << std::endl;
        return;
    }
    if ((mode&4)!=0)
    {
        time_record t;
        w = best_omega_for_GSmethod(vin, g, g_wire, n, m);
        cout << "best w is " << w << endl;
    }
    if ((mode &2) == 0)
    {
        time_record t;
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

    std::cout << "gs mem = " << (sizeof(int)*5 + sizeof(double)*3 + m*n*sizeof(double)*5 + m*sizeof(double)*2 + n*sizeof(double)) << " bytes" << std::endl;

    int final_i = pre(iter_number);
    if (record!=-1)
    {
        cout << "threshould break, iter times =  " << record << endl;
        final_i = now(record);
    }
    {
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
        ofstream of("fastmode_nodebasedGS.out");
        double errIsum = get_errI_sum();
        of << "error I sum = " << errIsum << endl;
        cout << "error I sum = " << errIsum << endl;
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

// if get_nodal_vaol is true, we will read config file to set initial nodal voltage for Vup and Vdown 
// times is the iterative numbers
void array2D::fastsolve(int times, bool get_nodal_vol = false, bool enable_break=false, double break_th = 1e-9)
{
    // currently, we only support one kind of wire resistance
    double g_wire = 1.0/RDLine[0][0];

    vector<double> vin(n), vout(m);

    //n is row size, m is col size
    vector<vector<double>> g(n, vector<double>(m));
    vector<double> out(m);
    vector<vector<double>> vup(n, vector<double>(m));
    vector<vector<double>> vdown(n, vector<double>(m));
    vector<vector<double>> iarr(n, vector<double>(m));

    //set up
    for (int i=0; i<n; ++i)
        vin[i] = v[1][i];
    
    for (int j=0; j<m; ++j)
        vout[j] = v[2][j];
    
    for (int i=0; i<n; ++i)
    {
        for (int j=0; j<m; ++j)
            g[i][j] = 1.0/cellR_states[arr[i][j]];
    }

    for (int i=0; i<n; ++i)
    {
        for (int j=0; j<m; ++j)
        {
            vup[i][j] = vin[i];
            vdown[i][j] = 0;
        }
    }
    
    auto get_from_file = [&](bool from_iras_table, string table_name="")->void
    {
        if (get_nodal_vol && !from_iras_table)
        {
            cout << "get nodal voltage from file" << endl;
            string name = table_name==""? "table"+to_string(n)+"x"+to_string(m)+".out" : table_name;
            ifstream in(name);
            string str;
            double sum = 0;
            for (int i=0; i<n; ++i)
                sum+=vin[i];
            while (in >> str)
            {
                if (str=="vup")
                {
                    in >> str;
                    for (int i=0; i<n; ++i)
                    {
                        for (int j=0; j<m; ++j)
                        {
                            in >> vup[i][j];
                            vup[i][j]*=vin[i];
                        }
                    }
                }
                if (str=="vdown")
                {
                    in >> str;
                    for (int i=0; i<n; ++i)
                    {
                        for (int j=0; j<m; ++j)
                        {
                            in >> vdown[i][j];
                            vdown[i][j] = vdown[i][j]*sum/n;
                        }
                    }
                }
            }
        }
        else if (get_nodal_vol)
        {
            table_name = table_name==""? "table.out" : table_name;
            ifstream in_table(table_name);
            vector<vector<vector<Mat>>> table_up;
            vector<vector<vector<Mat>>> table_down;
            int M, N;
            int bit_vwl, bit_g, bit_input;
            int vwl_number, g_number, input_states;

            // read table_up & table_down
            {
                ifstream in("gen_table_config");
                double r_wire;
                int fast_solve_iterative_times;
                int cellStates;
                vector<double> states;

                in >> M >> N >> r_wire;
                in >> bit_vwl >> bit_g >> bit_input;

                in >> fast_solve_iterative_times;
                in >> cellStates;
                for (int i=0; i<cellStates; ++i)
                {
                    double temp;
                    in >> temp;
                    states.push_back(temp);
                }
                srand(time(NULL));
                if (bit_input!=1 || cellStates!=2)
                {
                    cerr << "bit input can only support 1 now, but get " << bit_input << endl;
                    cerr << "cell states can only support 2 now, but get " << cellStates << endl;
                    throw 1;
                }
                vwl_number = 1 << bit_vwl;
                g_number = 1 << bit_g;
                input_states = 1 << bit_input;
            
                table_up = vector<vector<vector<Mat>>>(vwl_number, vector<vector<Mat>>(g_number, vector<Mat>(input_states, Mat(M, vec(N)))));
                table_down = vector<vector<vector<Mat>>>(vwl_number, vector<vector<Mat>>(g_number, vector<Mat>(input_states, Mat(M, vec(N)))));
                for (int i=0; i<vwl_number; ++i)
                    for (int j=0; j<g_number; ++j)
                        for (int k=0; k<input_states; ++k)
                            for (int w=0; w<M; ++w)
                                for (int y=0; y<N; ++y)
                                {
                                    in_table >> table_up[i][j][k][w][y] >> table_down[i][j][k][w][y];
                                }
            }

            double sum = accumulate(vin.begin(), vin.end(), 0);
            sum/=(M*(input_states-1));
            int vwl_weight = sum*vwl_number;
            if (vwl_weight==vwl_number)
                --vwl_weight;
            
            sum = 0;
            for (auto &i : arr)
            {
                sum += accumulate(i.begin(), i.end(), 0);
            }
            sum/=(M*N);
            int g_weight = (1-sum)*g_number;
            if (g_weight==g_number)
                --g_weight;
            
            for (int wl=0; wl<M; ++wl)
            {
                int k = vin[wl];
                for (int bl=0; bl<N; ++bl)
                {
                    vup[wl][bl] = table_up[vwl_weight][g_weight][k][wl][bl];
                    vdown[wl][bl] = table_down[vwl_weight][g_weight][k][wl][bl];
                }
            }
        }
    };


    auto get_errI_sum = [&]()->double
    {
        double errI_sum = 0;
        for (int i=0; i<n; ++i)
        {
            errI_sum += fabs( (vin[i]-vup[i][0]+(m>1? vup[i][1]-vup[i][0] : 0))*g_wire+(vdown[i][0]-vup[i][0])*g[i][0]);
            for (int j=1; j<m-1; ++j)
            {
                errI_sum += fabs( (vup[i][j-1]-vup[i][j]+vup[i][j+1]-vup[i][j])*g_wire + (vdown[i][j]-vup[i][j])*g[i][j]  );
            }
            if (m>1)
                errI_sum += fabs( (vup[i][m-2]-vup[i][m-1])*g_wire + (vdown[i][m-1]-vup[i][m-1])*g[i][m-1]  );
        }
        //answer vdown check
        for (int j=0; j<m; ++j)
        {
            errI_sum += fabs( (vup[0][j]-vdown[0][j])*g[0][j] + (n>1? vdown[1][j]-vdown[0][j] : 0)*g_wire ); 
        }
        for (int i=1; i<n-1; ++i)
        {
            for (int j=0; j<m; ++j)
            {
                errI_sum += fabs( (vup[i][j]-vdown[i][j])*g[i][j]+ (vdown[i-1][j]-vdown[i][j]+vdown[i+1][j]-vdown[i][j])*g_wire );
            }
        }
        if (n>1)
        { 
            for (int j=0; j<m; ++j)
                errI_sum += fabs( (0-vdown[n-1][j]+vdown[n-2][j]-vdown[n-1][j])*g_wire + (vup[n-1][j]-vdown[n-1][j])*g[n-1][j]  );
        }
        return errI_sum;
    };


    cout << "start fast solve" << endl << "ir2s fast mode ";
    int record;
    get_from_file(true, table_name);
    {
        time_record t;
        record=ir_drop_fastSolve_cpu_singleMat(vin, vout, g, g_wire, (r_load[1]<0? 1/g_wire : r_load[1]), out, vup, vdown, iarr, n, m, times, enable_break, break_th);
    }    

    std::cout << "ir2s fast iter times = " << record << std::endl;

    //answer vup check
    

    ofstream of("fastmode.out");
    double errIsum = get_errI_sum();
    of << "error I sum = " << errIsum << endl;
    cout << "error I sum = " << errIsum << endl;
    of << "vup = " << endl;

    of << std::scientific << std::setprecision(12);
    for (int i=0; i<n; ++i)
    {
        for (int j=0; j<m; ++j)
            of << std::setw(20) << vup[i][j];
        of << endl;
    }
    of << endl << "vdown = " << endl; 
    for (int i=0; i<n; ++i)
    {
        for (int j=0; j<m; ++j)
            of << std::setw(20) << vdown[i][j];
        of << endl;
    }
    of << endl << "iout = " << endl;
    for (int j=0; j<m; ++j)
        of << std::setw(20) << out[j];
    of << endl;
}

void array2D::buildSpice(string file)
{
    int otherPoint = hasSelector? 3*n*m : 2*n*m;

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

    // if (dc_ac_tran_type=="dc")
    // {
    //     auto arch = *gen.getArch();
    //     for (auto i : arch)
    //     {
    //         if (i.getType()==UnitType::voltage || i.getType()==UnitType::voltage_pp)
    //         {
    //             if (i.getType()==UnitType::voltage_pp)
    //             {
    //                 cout << "Error: dc should use voltage, not voltage_pp" << endl;
    //                 throw 0;
    //             }

    //             auto paras = i.getParas();
    //             gen.push2bot_top(".dc V1 "+paras[0]+" "+paras[0]+" 0.1");
    //             break;
    //         }
    //     }       
    // }

    cout << "build over; start print to spice" << endl; 
    ofstream out(file);
    gen.print2Hspice(out, dc_ac_tran_type=="dc");
    cout << "print2 spice over" << endl;
}

void array2D::addSelector()
{
    hasSelector = true;
}

void array2D::removeSelector()
{
    hasSelector = false;
}

void print_mat(const Mat &a)
{
    int rowSize = a.size(), colSize = a[0].size();
    for (int i=0; i<rowSize; ++i)
    {
        for (int j=0; j<colSize; ++j)
            cout << a[i][j] << ' ';
        cout << endl;
    }
    cout << endl;
}

void array2D::IR_PBIA(int iter_times, bool enable_break, double threshould_break)  // array size is [m n]
{
    int rowSize = n, colSize = m; 

    if (m!=n)
        throw("please see paper, Modeling and Mitigating the Interconnect Resistance Issue in Analog RRAM Matrix Computing Circuits. Due to some reasons, we need rowsize equals to colsize.");

    auto matrix_dot_matrix = [&](const Mat &a, const Mat &b, int m, int n) -> Mat
    {
        Mat c(m, vec(n));
        for (int i=0; i<m; ++i)
        {
            for (int j=0; j<n; ++j)
                c[i][j] = a[i][j]*b[i][j];
        }
        return c;
    };

    auto scale_matrix = [&](double k, const Mat &a, int m, int n) -> Mat
    {
        Mat c(m, vec(n));
        for (int i=0; i<m; ++i)
        {
            for (int j=0; j<n; ++j)
                c[i][j] = a[i][j]*k;
        }
        return c;
    };

    auto matrix_mul_matrix = [&](const Mat &a, const Mat &b, int m, int k, int n) -> Mat
    {
        Mat c(m, vec(n));
        for (int i=0; i<m; ++i)
        {
            for (int j=0; j<n; ++j)
            {
                for (int t=0; t<k; ++t)
                    c[i][j] += a[i][t]*b[t][j];
            }
        }
        return c;
    };

    auto matrix_sub_matrix = [&](const Mat &a, const Mat &b, int m, int n) -> Mat
    {
        Mat c(m, vec(n));
        for (int i=0;  i<m; ++i)
            for (int j=0; j<n; ++j)
                c[i][j] = a[i][j]-b[i][j];
        return c;
    };


    Mat D1 = Mat(rowSize, vec(rowSize)), D1_inv = D1, D2 = Mat(colSize, vec(colSize)), D2_inv = D2;

    for (int i=0; i<rowSize; ++i)
    {
        if (i!=rowSize-1)
            D1[i][i] = 2;
        else 
            D1[i][i] = 1;
        
        if (i!=0 && i!=rowSize-1)
            D1[i][i-1] = D1[i][i+1] = -1;
        else if (i==0)
            D1[i][i+1] = -1;
        else
            D1[i][i-1] = -1;
    }

    for (int i=0; i<rowSize; ++i)
    {
        for (int j=0; j<rowSize; ++j)
        {
            if (j<i)
                D1_inv[i][j] = j+1;
            else
                D1_inv[i][j] = i+1;
        }
    }

    for (int i=0; i<colSize; ++i)
    {
        if (i!=0)
            D2[i][i] = 2;
        else 
            D2[i][i] = 1;
        
        if (i!=0 && i!=colSize-1)
            D2[i][i-1] = D2[i][i+1] = -1;
        else if (i==0)
            D2[i][i+1] = -1;
        else
            D2[i][i-1] = -1;
    }

    for (int i=0; i<colSize; ++i)
    {
        for (int j=0; j<colSize; ++j)
        {
            if (j<i)
                D2_inv[n-i-1][n-j-1] = j+1;
            else
                D2_inv[n-i-1][n-j-1] = i+1;
        }
    }

    Mat g(rowSize, vec(colSize));

    for (int i=0; i<rowSize; ++i)
    {
        for (int j=0; j<colSize; ++j)
            g[i][j] = 1.0/cellR_states[arr[j][i]];
    }

    vector<Mat> Vdown(2, Mat(rowSize, vec(colSize)));
    double r_wire = RDLine[0][0]; //only support 1 kind of  r_wire now.
    Mat V_ideal(rowSize, vec(colSize));
    for (int j=0; j<colSize; ++j)
    {
        for (int i=0; i<rowSize; ++i)
            V_ideal[i][j] = v[1][j];
    }
    
    vec Iout(rowSize);
    int record = iter_times;
    std::cout << "pbia fastmode ";
    {
        time_record t;
        for (int iter = 0; iter<iter_times; ++iter)
        {
            int now = iter%2;
            int pre = 1-now;
            Mat temp = matrix_mul_matrix(D1_inv, Vdown[pre], rowSize, rowSize, colSize);

            temp = matrix_mul_matrix(temp, D2, rowSize, colSize, colSize);
            temp = matrix_sub_matrix(matrix_sub_matrix(V_ideal, temp, rowSize, colSize), Vdown[pre], rowSize, colSize);
            temp = matrix_mul_matrix(matrix_dot_matrix(g, temp, rowSize, colSize), D2_inv, rowSize, colSize, colSize);
            Vdown[now] = scale_matrix(r_wire, temp, rowSize, colSize);

            if (enable_break)
            {
                double mx = 0;
                for (int i=0; i<rowSize; ++i)
                {
                    for (int j=0; j<colSize; ++j)
                    {
                        mx = std::max(mx, std::abs(Vdown[now][i][j]-Vdown[pre][i][j]));
                    }
                }
                if (mx<threshould_break)
                {
                    record = iter;
                    break;
                }
            }

        }

        for (int i=0; i<rowSize; ++i)
        {
            Iout[i] = Vdown[1-iter_times%2][i][colSize-1]/r_wire;
        }
    }

    std::cout << "pbia iter times = " << record << std::endl;

    std::cout << "pbia mem = " << (sizeof(int)*8 + rowSize*rowSize*sizeof(double)*2 + colSize*colSize*sizeof(double)*2 + rowSize*colSize*sizeof(double)*4 + rowSize*sizeof(double) + sizeof(double)) << " bytes" << std::endl;

    ofstream of("fastmode_pbia.out");

    of << std::scientific << std::setprecision(12);
    of << endl << "iout = " << endl;
    for (int j=0; j<rowSize; ++j)
        of << std::setw(20) << Iout[j];
    of << endl;

}
