#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <regex>
#include <map>
#include <cstdlib>
#include <cstdio>
#include <ctime>

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

//table[vwl_weight][g_weight][input_states][wordline_number][bitline_number] = 
vector<vector<vector<Mat>>> table_up;
vector<vector<vector<Mat>>> table_down;

Mat_pair get_from_hspiceout_to_gen_nodal_voltage(string hspice_out_file = "hspice.out", int only_one_line = -1);
void get_params(int, char *[]);
double get_scale(char);
void print(string str);
void generate_model_config_file(vec input_v, Mat crx_states, double r_wire, int see_only_one_line = -1, string hspice_file_name = "out", bool enable_fast_solve = true, bool enable_hspice = false, bool enable_fast_solve_table = false);

void work();

Mat_pair read_data(ifstream &in) 
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
        cerr << "bit input can only support 1 now, but get " << bit_input << endl;
        cerr << "cell states can only support 2 now, but get " << cellStates << endl;
        throw 1;
    }
}

int main()//生成IRAS table，名为table.out
{
    read_config();
    //get_params(argc, argv);

    int vwl_number = 1 << bit_vwl;
    int g_number = 1 << bit_g;
    int input_states = 1 << bit_input;

    table_up = vector<vector<vector<Mat>>>(vwl_number, vector<vector<Mat>>(g_number, vector<Mat>(input_states, Mat(M, vec(N)))));
    table_down = vector<vector<vector<Mat>>>(vwl_number, vector<vector<Mat>>(g_number, vector<Mat>(input_states, Mat(M, vec(N)))));

    for (int i=0; i<vwl_number; ++i)
    {
        for (int j=0; j<g_number; ++j)
        {
            for (int k=0; k<input_states; ++k)
            {
                for (int w=0; w<M; ++w)
                {
                    vector<double> input_v(M);
                    Mat crx_state(M, vec(N));

                    int g_high_state_num = (1.0*j+0.5)/g_number*M;
                    for (int bl=0; bl<N; ++bl)
                    {
                        for (int wl=0; wl<M; ++wl)
                        {
                            crx_state[wl][bl] = (rand()%M<g_high_state_num)? 0 : 1;
                        }
                    }

                    Mat_pair nodal_v_1, nodal_v_2;
                    //each interval has its maximum & minimum case
                    //gen nodal voltage from (maximum and minimum)/2
                    {// generate maximum one 
                        int high_vwl_num = 1.0*(1+i)/vwl_number*M;
                        for (int wl=0; wl<M; ++wl)
                        {
                            input_v[wl] = (rand()%M<high_vwl_num)? 1 : 0;
                        }
                        input_v[w]=k;

                        generate_model_config_file(input_v, crx_state, r_wire, w);
                        if (system("./sim")<0) 
                        {
                            cout << "error" << endl;
                            return 0;
                        }
                        // if (system("hspice out > hspice.out")<0)
                        // {
                        //     cout << "error" << endl;
                        //     return 0;
                        // }
                        ifstream in("fastmode.out");
                        nodal_v_1 = read_data(in);
                    }


                    {// generate minimum one
                        int low_vwl_num = 1.0*(1+i)/vwl_number*M;
                        for (int wl=0; wl<M; ++wl)
                        {
                            input_v[wl] = (rand()%M<low_vwl_num)? 1 : 0;
                        }
                        input_v[w] = k;

                        generate_model_config_file(input_v, crx_state, r_wire, w);
                        if (system("./sim")<0) 
                        {
                            cout << "error" << endl;
                            return 0;
                        }
                        // if (system("hspice out > hspice.out")<0)
                        // {
                        //     cout << "error" << endl;
                        //     return 0;
                        // }
                        ifstream in("fastmode.out");
                        nodal_v_2 = read_data(in);
                    }

                    for (int wl=0; wl<M; ++wl)
                        for (int bl=0; bl<N; ++bl)
                        {
                            nodal_v_1.first[wl][bl] += nodal_v_2.first[wl][bl];
                            nodal_v_1.second[wl][bl] += nodal_v_2.second[wl][bl];
                            nodal_v_1.first[wl][bl] /= 2;
                            nodal_v_1.second[wl][bl] /= 2;

                            if (wl==w)
                            {
                                table_up[i][j][k][w][bl] = nodal_v_1.first[wl][bl];
                                table_down[i][j][k][w][bl] = nodal_v_1.second[wl][bl];
                            }
                        }
                }
            }
        }
    }

    print("table.out");
    return 0;
}

void print(string str)
{
    ofstream out(str);
    int vwl_number = 1 << bit_vwl;
    int g_number = 1 << bit_g;
    int input_states = 1 << bit_input;

    for (int i=0; i<vwl_number; ++i)
        for (int j=0; j<g_number; ++j)
            for (int k=0; k<input_states; ++k)
                for (int w=0; w<M; ++w)
                    for (int y=0; y<N; ++y)
                    {
                        out << table_up[i][j][k][w][y] << ' ' << table_down[i][j][k][w][y] << ' ';
                    }
}

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

void get_params(int argc, char *argv[])
{
    if (argc>3)
        bit_vwl = stoi(string(argv[3]));
    if (argc>4)
        bit_g = stoi(string(argv[4]));
    if (argc>5)
        bit_input = stoi(string(argv[5]));
    if (argc>1)
        M = stoi(string(argv[1]));
    if (argc>2)
        N = stoi(string(argv[2]));
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

void generate_model_config_file(vec input_v, Mat crx_states, double r_wire, int only_one_line, string hspice_file_name, bool enable_fast_solve, bool enable_hspice, bool enable_fast_solve_table)
{
    ofstream out("config");

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

    if (only_one_line>=0)
        out << "senseCellV "+to_string(only_one_line)+" -1" << endl;
    else
        out << "senseCellV -1 -1" << endl;

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