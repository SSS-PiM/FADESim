#include <iostream>
#include <vector>
#include <fstream>
#include <regex>
#include <cmath>
#include <iomanip>

using namespace std;


typedef vector<double> vec;
typedef vector<vec> Mat;
typedef pair<Mat, Mat> Mat_pair;

int bit_vwl=2, bit_g=2, bit_input=1;
int M=256, N=256;
double r_wire = 2.8;
int fast_solve_iterative_times = 10;
int cellStates = 2;
vec states;

void check(string fastmode_out, string hspice_out);
Mat_pair get_from_hspiceout_to_gen_nodal_voltage(string hspice_out_file, int only_one_line);
double get_scale(char);

/*
* 从gen_table_config读取以下各种参数
* M, N：crossbar row/col size
* r_wire: wire resistance
* IRAS table #bit for wl input voltage distribution (bit_vwl), 
* #bit for cell conductance distribution (bit_g)
* #bit_input is the input voltage puluse bit
*/
void read_config()
{
    ifstream in("gen_table_config");
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
        cout << "bit input can only support 1 now, but get " << bit_input << endl;
        cout << "cell states can only support 2 now, but get " << cellStates << endl;
        throw 1;
    }
}

int main(int argc, char *argv[])
{
    read_config();
    if (argc==1)
        check("fastmode.out", "hspice.out");//对比流出电流的相对误差
    else if (argc==2 && string(argv[1])=="cell")//将hspice的所有结点电压打印到hspice.out2
    {
        auto s = get_from_hspiceout_to_gen_nodal_voltage("hspice.out", -1);
        ofstream of("hspice.out2");
        of << "vup = " << endl;

        of << std::scientific << std::setprecision(6);
        of << std::setw(12);
        for (int i=0; i<M; ++i)
        {
            for (int j=0; j<N; ++j)
                of << std::setw(14) << s.first[i][j];
            of << endl;
        }
        of << endl << "vdown = " << endl; 
        for (int i=0; i<M; ++i)
        {
            for (int j=0; j<N; ++j)
                of << std::setw(14) << s.second[i][j];
            of << endl;
        }

        of << endl << "iout = " << endl;
        for (int j=0; j<N; ++j)
            of << std::setw(14) << (s.second[M-1][j]/r_wire);
        of << endl;
    }
    else if (argc>=2 && string(argv[1])=="cmp")//对比fastmode.out与hspice.out2结点电压的差别
    {
        auto read_data = [&](ifstream &in) -> Mat_pair
        {
            string temp;
            Mat up(M, vec(N)), down(M, vec(N));
            while (in >> temp)
            {
                if (temp == "vup")
                {
                    in >> temp;
                    for (int i=0; i<M; ++i)
                        for (int j=0; j<N; ++j)
                        {
                            in >> up[i][j];
                        } 
                }
                if (temp == "vdown")
                {
                    in >> temp;
                    for (int i=0; i<M; ++i)
                        for (int j=0; j<N; ++j)
                        {
                            in >> down[i][j];
                        } 
                }
            }
            return make_pair(up, down);
        };
        string name1 = "fastmode.out";
        string name2 = "hspice.out2";
        if (argc>=4)
        {
            name1 = string(argv[2]);
            name2 = string(argv[3]);
        }
        ifstream fast(name1), hsp(name2);
        auto fast_out = read_data(fast);
        auto hsp_out = read_data(hsp);
        
        double avg_relative_err_up = 0;
        double avg_relative_err_down = 0;
        double max_relative_err_up = 0;
        double max_relative_err_down = 0;

        double max_relative_cell_vol_err = 0;
        double avg_relative_cell_vol_err = 0;
        for (int i=0; i<M; ++i)
            for (int j=0; j<N; ++j)
            {
                double x = abs(fast_out.first[i][j]-hsp_out.first[i][j])/abs(hsp_out.first[i][j]);
                max_relative_err_up = max(max_relative_err_up, x);
                avg_relative_err_up += x;
                x = abs(fast_out.second[i][j]-hsp_out.second[i][j])/abs(hsp_out.second[i][j]);
                max_relative_err_down = max(x, max_relative_err_down);
                avg_relative_err_down += x;

                double our_cell_v = fast_out.first[i][j]-fast_out.second[i][j];
                double hsp_cell_v = hsp_out.first[i][j]-hsp_out.second[i][j];

                x = abs(our_cell_v-hsp_cell_v)/abs(hsp_cell_v);
                avg_relative_cell_vol_err += x;
                max_relative_cell_vol_err = max(max_relative_cell_vol_err, x);

            }

        avg_relative_err_down/=M*N;
        avg_relative_err_up/=M*N;
        avg_relative_cell_vol_err/=M*N;
        cout << std::scientific << std::setprecision(6);
        cout << std::setw(12);
        cout << "avg relative error in wordline voltage = " << avg_relative_err_up*100 << "%" << endl;
        cout << "avg relative error in bitline voltage = " << avg_relative_err_down*100 << "%" << endl;
        cout << "avg relative error in all voltage = " << (avg_relative_err_down+avg_relative_err_up)*50 << "%" << endl;
        cout << endl;
        cout << "max relative error in wordine voltage = " << max_relative_err_up*100 << "%" << endl;
        cout << "max relative error in bitline voltage = " << max_relative_err_down*100 << "%" << endl;
        cout << "max relative error in all voltage = " << max(max_relative_err_down, max_relative_err_up)*100 << "%" << endl;
        cout << endl;
        cout << "max relative cell voltage error = " << max_relative_cell_vol_err*100 << "%" << endl;
        cout << "avg relative cell voltage error = " << avg_relative_cell_vol_err*100 << "%" << endl;
    }
    return 0;
}

//对比fastmode.out与hspice输出，最后位线流出电流的区别。
void check(string fastmode_out, string hspice_out)
{
    vector<double> iout_fastmode, iout_hspice;
    ifstream fast(fastmode_out), hsp(hspice_out);
    string temp;
    while (fast >> temp)
    {
        if (temp == "iout")
        {
            fast >> temp;
            double x;
            while (fast >> x)
                iout_fastmode.push_back(x);
            break;
        }
    }
    stringstream ss;
    ss << hsp.rdbuf();
    temp = ss.str();
    regex s("time\\s+current\\s+v\\d{1,4}\\s+0\\.\\s+(\\d*(\\.){0,1}\\d*)([A-Za-z]*)");

    for (sregex_iterator it(temp.begin(), temp.end(), s), ed; it!=ed; ++it)
    {
        double x = 1;
        char k = toupper(it->str().back());
        x = get_scale(k); 

        stringstream s;
        s << it->str(1);
        double temp;
        s >> temp;
        iout_hspice.push_back(temp*x);
    }
    if (iout_fastmode.size()!=iout_hspice.size())
    {
        cout << "size is not equal" << endl;
    } 
    else
    {
        int len = iout_hspice.size();
        double sum = 0;
        double max_err = 0;
        for (int i=0; i<len; ++i)
        {
            double tmp = fabs((iout_hspice[i]-iout_fastmode[i])/iout_hspice[i]);
            max_err = max(max_err, tmp);
            sum += tmp;
        }
        cout << "avg relative deviation = " << sum*100/len << " %"<< endl;
        cout << "max relative deviation = " << max_err*100 << " %"<< endl;
    }
}

//生成
void gen_model_config_file(string config_name, vec input_v, Mat crx_states, double r_wire, string hspice_file_name, bool enable_fast_solve, bool enable_hspice, bool enable_fast_solve_table)
{
    ofstream out(config_name);

    out << "topString .title twoDArray" << endl;
    out << "topString .hdl 'reram_mod.va' reram_mod" << endl;
    out << "bottomString .tran 1 1" << endl;
    out << "bottomString .end" << endl;
    out << ("arraySize "+to_string(M)+" "+to_string(N)) << endl;
    out << "selector no" << endl;
    out << "line_resistance " << r_wire << endl;
    out << "setUseLine left -1" << endl;
    out << "setUseLine down -1" << endl;
    out << "setLineV down -1 0" << endl;
    out << "senseBitlineI down -1" << endl;
    //set left input voltage
    for (int i=0; i<M; ++i)
        out << "setLineV left " << i << " " << input_v[i] << endl;
    
    //set cell state
    for (int i=0; i<M; ++i)
    {
        for (int j=0; j<N; ++j)
        {
            out << "setCellR " << i << " " << j << " " << crx_states[i][j] << endl;
        }
    }

    if (enable_hspice)
        out << ("build "+hspice_file_name) << endl;
    
    if (enable_fast_solve)
    {
        out << ("fastmode yes "+to_string(cellStates)+" ");
        for (int i=0; i<cellStates; ++i)
            out << states[i] << " ";
        out << endl;
        out << "fastsolve " << fast_solve_iterative_times;
        if (enable_fast_solve_table)
            out << " 1" << endl;
        else
            out << endl;
    }
}

//从hspice.out读取所有结点的结点电压，当only_one_line=-1
//当only_one_line>=0，只设置该一行的结点电压
Mat_pair get_from_hspiceout_to_gen_nodal_voltage(string hspice_out_file, int only_one_line)
{
    ifstream hsp(hspice_out_file);
    stringstream ss;

    ss << hsp.rdbuf();
    string temp = ss.str();
    regex s("time\\s+voltage\\s+\\d+\\s+0\\.\\s+(\\d*(\\.){0,1}\\d*)([A-Za-z]*)");
    vec vup, vdown;
    bool up = true;

    for (sregex_iterator it(temp.begin(), temp.end(), s), ed; it!=ed; ++it)
    {
        double x;
        char k = toupper(it->str().back());
        stringstream s;
        double temp;

        x = get_scale(k);
        s << it->str(1);
        s >> temp;

        temp*=x;

        if (up)
            vup.push_back(temp);
        else 
            vdown.push_back(temp);
        up = !up;
    }

    int number = only_one_line>=0? N : M*N;
        
    if (vup.size()!=number || vdown.size()!=number)
    {
        cout << "size error in get_from_hspiceout_to_gen_nodal_voltage" << endl;
        cout << "number is " << number << " but get vup size " << vup.size() << " vdown size " << vdown.size() << endl;
        throw 1;
    }

    Mat ret_vup(M, vec(N)), ret_vdown(M, vec(N));
    int st=0, ed=M;

    if (only_one_line>=0)
        st=only_one_line, ed=only_one_line+1;

    for (int i=st, k=0; i<ed; ++i)
    {
        for (int j=0; j<N; ++j)
        {
            ret_vup[i][j] = vup[k];
            ret_vdown[i][j] = vdown[k];
            ++k;
        }
    }

    return make_pair(ret_vup, ret_vdown);
}

double get_scale(char k)
{
    double x;
    if (k=='F')
        x = 1e-15;
    else if (k=='M')
        x = 1e-3;
    else if (k=='P')
        x = 1e-12;
    else if (k=='K')
        x = 1e3;
    else if (k=='N')
        x = 1e-9;
    else if (k=='X')
        x = 1e6;
    else if (k=='U')
        x = 1e-6;
    else if (k=='G')
        x = 1e9;
    else
        x = 1;
    return x;
}