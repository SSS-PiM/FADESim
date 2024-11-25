#ifndef __FAST_IR_SOLVER__
#define __FAST_IR_SOLVER__
#include <sys/time.h>
#include <cstring>
// #include <armadillo>
#include "2dArray.h"

struct time_record
{
    struct timeval t1, t2;
    time_record()
    {
        gettimeofday(&t1, NULL);
    }

    ~time_record()
    {
        gettimeofday(&t2, NULL);
        cout << "time = " << (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0 << " s" << std::endl;
    }
};

// M is row size, N is col size
int ir_drop_fastSolve_cpu_singleMat(vector<double> &vin, vector<double> &vout, vector<vector<double>> &g, double g_wire, double r_load, vector<double> &out, vector<vector<double>> &vup, vector<vector<double>> &vdown, vector<vector<double>> &iarr, int M, int N, int times, bool enable_break, double break_th)
{
    if (!enable_break)
    {
        double iwl[M], ibl[N];

        for (int k=0; k<times; ++k)
        {
            memset(iwl, 0, sizeof(iwl));
            memset(ibl, 0, sizeof(ibl));

            for (int i=0; i<M; ++i)
            {
                for (int j=0; j<N; ++j)
                {
                    iarr[i][j] = (vup[i][j]-vdown[i][j])*g[i][j];
                    iwl[i] += iarr[i][j];
                    ibl[j] += iarr[i][j];
                }
            }

            for (int i=0; i<M; ++i)
            {
                vup[i][0] = vin[i]-iwl[i]*r_load;
                iwl[i] -= iarr[i][0];
                for (int j=1; j<N; ++j)
                {
                    vup[i][j] = vup[i][j-1]-iwl[i]/g_wire;
                    iwl[i] -= iarr[i][j];
                }
            }

            for (int j=0; j<N; ++j)
            {
                vdown[M-1][j] = ibl[j]*r_load+vout[j];
                ibl[j] -= iarr[M-1][j];
            }

            for (int i=M-2; i>=0; --i)
            {
                for (int j=0; j<N; ++j)
                {
                    vdown[i][j] = vdown[i+1][j]+ibl[j]/g_wire;
                    ibl[j] -= iarr[i][j];
                }
            }
        }

        for (int j=0; j<N; ++j)
        {
            out[j] = 0;
            for (int i=0; i<M; ++i)
            {
                iarr[i][j] = (vup[i][j]-vdown[i][j])*g[i][j];
                out[j] += iarr[i][j];
            }
        }

        return times;
    }
    else
    {
        
        double mx=0;
        double iwl[M], ibl[N];
        int record=times;

        auto check_set = [&](double &x, double newx) -> void
        {
            mx = std::max(mx, std::abs(newx-x));
            x = newx;
        };

        for (int k=0; k<times; ++k)
        {
            memset(iwl, 0, sizeof(iwl));
            memset(ibl, 0, sizeof(ibl));

            mx = 0;

            for (int i=0; i<M; ++i)
            {
                for (int j=0; j<N; ++j)
                {
                    iarr[i][j] = (vup[i][j]-vdown[i][j])*g[i][j];
                    iwl[i] += iarr[i][j];
                    ibl[j] += iarr[i][j];
                }
            }

            for (int i=0; i<M; ++i)
            {
                check_set(vup[i][0], vin[i]-iwl[i]*r_load);
                // vup[i][0] = vin[i]-iwl[i]*r_load;
                iwl[i] -= iarr[i][0];
                for (int j=1; j<N; ++j)
                {
                    check_set(vup[i][j], vup[i][j-1]-iwl[i]/g_wire);
                    // vup[i][j] = vup[i][j-1]-iwl[i]/g_wire;
                    iwl[i] -= iarr[i][j];
                }
            }

            for (int j=0; j<N; ++j)
            {
                check_set(vdown[M-1][j], ibl[j]*r_load+vout[j]);
                // vdown[M-1][j] = ibl[j]*r_load+vout[j];
                ibl[j] -= iarr[M-1][j];
            }

            for (int i=M-2; i>=0; --i)
            {
                for (int j=0; j<N; ++j)
                {
                    check_set(vdown[i][j], vdown[i+1][j]+ibl[j]/g_wire);
                    // vdown[i][j] = vdown[i+1][j]+ibl[j]/g_wire;
                    ibl[j] -= iarr[i][j];
                }
            }

            if (mx<break_th)
            {
                record = k;
                break;
            }
        }

        for (int j=0; j<N; ++j)
        {
            out[j] = 0;
            for (int i=0; i<M; ++i)
            {
                iarr[i][j] = (vup[i][j]-vdown[i][j])*g[i][j];
                out[j] += iarr[i][j];
            }
        }
        return record+1;
    }

}


void ir_drop_fastsolve_cpu_singleMat_in_aihwkit(vector<double> &vin, vector<double> &out, vector<vector<int>> &weights, double g_wire, double g_max, double ir_drop_beta, int M, int N)
{
    vector<double> tmp_in(M), tmp_out(N), tmp_c(N);
    double a_scale = M / (g_wire/g_max);

    for (int j=0; j<N; ++j)
    {
        double accum = 0.0;
        for (int i=0; i<M; ++i)
            accum += vin[i]*weights[i][j];
        double a = a_scale * accum;
        tmp_c[j] = a* (a* (0.05*a-0.2) + 0.5);
    }

    for (int i=0; i<M; ++i)
    {
        double p = (1.0-1.0*i/M);
        tmp_in[i] = vin[i] * (1-p*p);
    }

    for (int j=0; j<N; ++j)
    {
        out[j] = 0;
        for (int i=0; i<M; ++i)
        {
            out[j] += weights[i][j] * vin[i]; //ideal
            tmp_out[j] += weights[i][j] * tmp_in[i];
        }
    }

    for (int j=0; j<N; ++j)
    {
        out[j] -= tmp_out[j]*tmp_c[j]*ir_drop_beta;
    }
}

/* This method comes from sci china, Efficient evaluation model including interconnect 
resistance effect for large scale RRAM crossbar array matrix computing,
*/
void ir_drop_fastsolve_cpu_singleMat_in_scichina(vector<double> &vin, vector<double> Vout, vector<double> &Iout, vector<vector<double>> &R, double r_wire, double Rs, int M, int N)
{
    vector<vector<double>> Reqv_up(M, vector<double>(N));
    vector<vector<double>> Reqv_down(M, vector<double>(N));
    vector<vector<double>> Reqv(M, vector<double>(N));
    vector<vector<double>> f(M, vector<double>(N));
    vector<vector<double>> RR(M, vector<double>(N));
    vector<vector<double>> RRR(M, vector<double>(N));
    vector<vector<double>> V_wl(M, vector<double>(N));

    vector<double> g_sum_for_bl(N);
    for (int j=0; j<N; ++j)
    {
        for (int i=0; i<M; ++i)
            g_sum_for_bl[j]+=1.0/R[i][j];
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
        Vout[j] = 0;
        for (int i=0; i<M; ++i)
        {
            Vout[j] += V_wl[i][j]*(Reqv[i][j])/(Reqv[i][j]+R[i][j])*f[i][j];
        }
        Iout[j] = Vout[j]/Rs;
    }
}

/*
*
* this method comes from MLP_Neurosim_V3.0
*/
void ir_drop_fastsolve_cpu_singleMat_in_neurosim(vector<double> &vin, vector<double> &Iout, vector<vector<double>> &R, double r_wire, int M, int N)
{
    for (int j=0; j<N; ++j)
    {
        Iout[j] = 0;
        for (int i=0; i<M; ++i)
        {
            double total_wire_resistance = (j+1)*r_wire+(M-i)*r_wire;
            Iout[j] += vin[i]/(R[i][j]+total_wire_resistance);
        }
    }
}


// use armadillo lib to compute eigval to get best omega.
// see our paper for details. best omega is a value realted to eigval.
// Due to the fact that many people have not installed ARMA before, I will comment this function by default. 
double best_omega_for_GSmethod(vector<double> &vin, vector<vector<double>> &g, double g_wire, int M, int N)
{
    // B = I- D^-1 * A, D = diag(A)
/*     int siz = M*N;
    arma::mat B(siz*2, siz*2);
    for (int i=0; i<M; ++i)
    {
        for (int j=0; j<N; ++j)
        {
            int x = i*N+j;
            int l = (j==0? -1 : x-1), r = (j==N-1? -1 : x+1);
            int o =  x+siz;
            double sumg = (j==N-1? g_wire+g[i][j] : g_wire*2+g[i][j]);
            if (l!=-1)
                B(x, l) = g_wire/sumg;
            if (r!=-1)
                B(x, r) = g_wire/sumg; 
            B(x, o) = g[i][j]/sumg;
        }
    }

    for (int i=0; i<M; ++i)
    {
        for (int j=0; j<N; ++j)
        {
            int x = i*N+j+siz;
            int l = (i==0? -1 : x-N), r = (i==M-1? -1 : x+N);
            int o = x-siz;
            double sumg = (i==0? g_wire+g[i][j] : g_wire*2+g[i][j]);

            if (l!=-1)
                B(x, l) = g_wire/sumg;
            if (r!=-1)
                B(x, r) = g_wire/sumg;
            B(x, o) = g[i][j]/sumg;
        }
    }

    cout << endl;
    // B.print();
    auto eigval = arma::eig_gen(B);
    double mx = eigval.at(0).real();
    for (auto &i : eigval)
        mx = std::max(std::abs(i.real()), mx);
    return 2/(1+std::sqrt(1-mx*mx)); */
}

#endif