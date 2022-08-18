//#ifndef HEADER_H_INCLUDED
//#define HEADER_H_INCLUDED

#include <iostream>
#include <random>
#include <cstdlib>
#include <math.h>
#include "Constants.h"
#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <stdlib.h>
#include <fstream>
#include <cstdlib>
#include <time.h>
#include <ctime>

using namespace std;



void quickSort(double *arr, int left, int right)
{

    int i = left, j = right;
    double tmp;
    double pivot = arr[(left + right) / (int)2];
    while (i <= j)
    {

        while (arr[i] < pivot)
            i++;

        while (arr[j] > pivot)
            j--;

        if (i <= j)
        {
            tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
            i++;
            j--;
        }
    };

    if (left < j)
        quickSort(arr, left, j);

    if (i < right)
        quickSort(arr, i, right);

}

void bubble_sort(double *a, int lenght)
{
    for (int j=0; j<lenght -1; j++)
    {
        for (int i=0; i<lenght-j-1; i++)
        {
            if (a[i] > a[i+1])
            {
                double b = a[i];
                a[i] = a[i+1];
                a[i+1]=b;
            }
        }
    }
}

void bubble_sort_indecies(double *a, int *index, int lenght)
{
    for (int j=0; j<lenght-1; j++)
    {
        for (int i=0; i<lenght-j-1; i++)
        {
            if (a[i] > a[i+1])
            {
                double b = a[i];
                int b_ = index[i];
                a[i] = a[i+1];
                index[i] = index[i+1];

                a[i+1]=b;
                index[i+1] =b_;
            }
        }
    }
}

void initializePopulation(double **x, double **y, int pop_size, int N, int a, int b)
{
    double ba = b-a;
    for (int i=0; i!=pop_size; i++)
    {
        for (int j=0; j!=N; j++)
        {
            x[i][j] = y[i][j] = RANDOM()*ba+a;
        }
    }
}

void initializeHistory(double **History_F, double **History_CR, int H, int M)
{
    for (int i=0; i!=M; i++)
    {
        for (int j=0; j!=H; j++)
        {
            History_F[i][j]  = 0.5;
            History_CR[i][j] = 0.5;
        }
    }
}

void chooseCrossoverIndecies(int &r1, int &r2, int pbest, int pop_size, int *A, int p)
{
 
    r1 = RANDOM()*(pop_size-1);
    r2 = RANDOM()*(pop_size-1+A[p]);


    while (r1 == r2 || r1 == pbest || r2 == pbest)
    {
        r1 = RANDOM()*(pop_size-1);
        r2 = RANDOM()*(pop_size-1+A[p]);
    }
}



void findBestIndex(double **fitness, int &pbest, int pop_size, int piece_int, int p)
{
    double *f_sort = new double [pop_size];
    int *index_sort = new int [pop_size];

    for (int i=0; i!=pop_size; i++)
    {
        f_sort[i] = fitness[p][i];
    }

    bubble_sort(f_sort, pop_size);

    for (int i=0; i!=pop_size; i++)
    {
        for (int j=0; j!=pop_size; j++)
        {
            if (f_sort[i]==fitness[p][j])
            {
                index_sort[i]=j;
            }
        }
    }

    pbest = index_sort[int(RANDOM()*piece_int)];

    delete [] f_sort;
    delete [] index_sort;
}

double findMinElement(double *fitness, int pop_size)
{
    int index = 0;
    double minn = fitness[index];

    for (int i=1; i!=pop_size; i++)
    {
        if (fitness[i]<minn)
        {
            minn = fitness[i];
            index = i;
        }
    }

    return minn;
}



void generation_CR(double &CR, double **H, int r, int p)
{
    CR = randn(H[p][r],0.1);
    while (CR>1)
    {
        CR=randn(H[p][r],0.1);
    }
    if (CR<0)
    {
        CR=0;
    }
}

void generation_F(double &F, double **H, int r, int p)
{
    F=randc(H[p][r],0.1);
    while (F<-0 || F>1)
    {
        F=randc(H[p][r],0.1);
    }
}

void reset_k(int *k, int H, int p)
{
    if (k[p]>=H)
    {
        k[p]=0;
    }
}

void Algorithm_1(double *delta_f, double *w, double *S_CR, double *S_F, double **HISTORY_CR, double **HISTORY_F, int *k, int &success, int H, int p)
{

    if (success == 0)
    {
        k[p]++;
        reset_k(k,H,p);
        return;
    }

    double sum1 = 0.0;
    double sum2 = 0.0;

    for (int i=0; i!=success; i++)
    {
        sum1+=delta_f[i];
    }

    if (sum1 == 0)
    {
        success = 0;
        k[p]++;
        reset_k(k,H,p);
        return;
    }

    sum1 = 0.0;
    sum2 = 0.0;
    for (int i=0; i!=success; i++)
    {
        sum1+=S_CR[i];
        sum2+=S_F[i];
    }

    if (sum1==0.0 && sum2==0.0)
    {
        k[p]++;
        success = 0;
        reset_k(k,H,p);
        return;
    }

    sum1 = 0.0;
    sum2 = 0.0;
    for (int i=0; i!=success; i++)
    {
        sum1+=S_CR[i];
        sum2+=S_F[i];
    }

    if (sum1==0.0 && sum2!=0.0)
    {
        sum1 = 0.0;
        sum2 = 0.0;
        for (int i=0; i!=success; i++)
        {
            sum1+=delta_f[i];
        }

        for (int i=0; i!=success; i++)
        {
            w[i]=delta_f[i]/sum1;
        }
        sum1 = 0.0;
        sum2 = 0.0;

        for (int i=0; i!=success; i++)
        {
            sum1+=w[i]*S_F[i]*S_F[i];
            sum2+=w[i]*S_F[i];
        }
        double meanF = sum1/sum2;

        HISTORY_F[p][k[p]] = meanF;
        if (HISTORY_F[p][k[p]] != HISTORY_F[p][k[p]])
        {
            HISTORY_F[p][k[p]] = 0.5;
        }
        HISTORY_CR[p][k[p]] = 0.0;
        k[p]++;
        success=0;
        reset_k(k, H, p);
        return;
    }

    sum1 = 0.0;
    sum2 = 0.0;
    for (int i=0; i!=success; i++)
    {
        sum1+=S_CR[i];
        sum2+=S_F[i];
    }

    if (sum1!=0.0 && sum2==0.0)
    {

        sum2 = 0.0;
        for (int i=0; i!=success; i++)
        {
            sum2+=delta_f[i];
        }
        for (int i=0; i!=success; i++)
        {
            w[i]=delta_f[i]/sum2;
        }
        sum1 = 0.0;
        sum2 = 0.0;

        for (int i=0; i!=success; i++)
        {
            sum1+=w[i]*S_CR[i];
        }
        double meanCR = sum1;

        HISTORY_CR[p][k[p]] = meanCR;
        if (HISTORY_CR[p][k[p]] != HISTORY_CR[p][k[p]])
        {
            HISTORY_CR[p][k[p]]=0.5;
        }
        HISTORY_F[p][k[p]] = 0;
        k++;
        success=0;
        reset_k(k,H,p);
        return;
    }


    sum1 = 0.0;
    sum2 = 0.0;
    double sum3 = 0.0;

    for (int i=0; i!=success; i++)
    {
        sum1+=S_CR[i];
        sum2+=S_F[i];
    }

    for (int i=0; i!=success; i++)
    {
        sum3+=delta_f[i];
    }

    if (sum1!=0.0 && sum2!=0.0 && sum3!=0.0 && success!=0)
    {
        sum1 = 0.0;

        for (int i=0; i!=success; i++)
        {
            sum1+=delta_f[i];
        }

        for (int i=0; i!=success; i++)
        {
            w[i]=delta_f[i]/sum1;
        }
        sum1 = 0.0;
        sum2 = 0.0;

        for (int i=0; i!=success; i++)
        {
            sum1+=w[i]*S_F[i]*S_F[i];
            sum2+=w[i]*S_F[i];
        }
        double meanF = sum1/sum2;

        HISTORY_F[p][k[p]] = meanF;
        if (HISTORY_F[p][k[p]]!=HISTORY_F[p][k[p]])
        {
            HISTORY_F[p][k[p]] =0.5;
        }
        sum1 = 0.0;
        sum2 = 0.0;

        for (int i=0; i!=success; i++)
        {
            sum1+=w[i]*S_CR[i];
        }

        double meanCR = sum1;

        HISTORY_CR[p][k[p]] = meanCR;
        if (HISTORY_CR[p][k[p]] != HISTORY_CR[p][k[p]])
        {
            HISTORY_CR[p][k[p]] =0.5;
        }
        k[p]++;
        success=0;
        reset_k(k,H,p);
        return;
    }
}


void check_out_borders(double **u, double **population, int i, int N, int a, int b, int *range, int p, int *indeces)
{
    for (int j=range[p]; j!=range[p+1]; j++)
    {
        while (u[i][indeces[j]]<a)
        {
            u[i][indeces[j]] = (a+population[i][indeces[j]])/2;
        }
        while (u[i][indeces[j]]>b)
        {
            u[i][indeces[j]] = (b+population[i][indeces[j]])/2;
        }
    }
}

void updateArchive(double **archive, double **population, int i, int archive_size, int *A, int *range, int p, int *indeces)
{
    if (A[p]>=archive_size)
    {
        int index = RANDOM()*(archive_size-1);
        for (int j=range[p]; j!=range[p+1]; j++)
        {
            archive[index][indeces[j]] = population[i][indeces[j]];
        }
        return;
    }

    if (A[p]<archive_size)
    {
        for (int j=range[p]; j!=range[p+1]; j++)
        {
            archive[A[p]][indeces[j]] = population[i][indeces[j]];
        }
        A[p]++;

        return;
    }
}

void LPSR(double **population, double *fitness, int LPSR, int &pop_size, int N)
{
    double *f_sort = new double [pop_size];
    int *index_sort = new int [pop_size];
    double *fitness_new = new double [pop_size];

    double **population_new = new double* [pop_size];
    for (int count = 0; count < pop_size; count++)
        population_new[count] = new double [N];

    for (int i=0; i!=pop_size; i++)
    {
        index_sort[i] = i;
        f_sort[i] = fitness[i];
    }

    bubble_sort_indecies(f_sort, index_sort, pop_size);


    for (int i=0; i!=pop_size; i++)
    {
        for (int j=0; j!=N; j++)
        {
            population_new[i][j] = population[index_sort[i]][j];

        }
        fitness_new[i] = fitness[index_sort[i]];




    }

    for (int i=0; i!=pop_size; i++)
    {
        for (int j=0; j!=N; j++)
        {
            population[i][j]=population_new[i][j];
        }
        fitness[i]=fitness_new[i];
    }



    pop_size = LPSR;







    delete [] f_sort;
    delete [] index_sort;
    delete [] fitness_new;

    for (int count = 0; count < pop_size; count++)
        delete [] population_new[count];
    delete [] population_new;
}




double mean_stat(double **x, int gener, int R)
{
    double sum;
    sum = 0.0;

    for (int i = 0; i<R; i++)
    {
        sum += x[i][gener];
    }

    sum = sum / (double)R;

    return sum;
}


double min_stat(double **x, int gener, int R)
{
    double sum;

    sum = x[0][gener];

    for (int i = 1; i<R; i++)
    {
        if (x[i][gener]<sum)
        {
            sum = x[i][gener];
        }
    }
    return sum;
}

double max_stat(double **x, int gener, int R)
{
    double sum;

    sum = x[0][gener];

    for (int i = 1; i<R; i++)
    {
        if (x[i][gener]>sum)
        {
            sum = x[i][gener];
        }
    }


    return sum;
}

double median_stat(double **x, int gener, int R)
{
    double answ;
    answ = 0.0;
    double *fitness = new double[R];

    for (int i = 0; i<R; i++)
    {
        fitness[i] = x[i][gener];
    }

    quickSort(fitness, 0, R-1);

    R = (R - 1) / 2;
    answ = fitness[R];

    delete[]fitness;

    return answ;
}

double stddev_stat(double **x, int gener, int R, double M)
{
    double sum;
    sum = 0.0;

    for (int i = 0; i<R; i++)
    {
        sum += pow((M - x[i][gener]), 2);
    }

    R = R - 1;
    sum = sum / (double)R;
    sum = sqrt(sum);

    return sum;

}


void rnd_indecies(int *vec, int pop, int M)
{
    for (int i=0; i!=M; i++)
    {
        vec[i] = RANDOM()*(pop-1);
    }
}

void indecesSuccession(int *x, int N)
{
    for (int i = 0; i != N; i++)
    {
        x[i] = i;
    }
}

void randperm(int *x, int N)
{
    int a, temp;
    for (int i = 0; i != N; i++)
    {
        temp = x[i];
        a = RANDOM()*(N - 1);
        while (a < 0 || a > (N-1) || a == i || a == N)
        {
            a = RANDOM()*(N - 1);
        }

        x[i] = x[a];
        x[a] = temp;
    }
}

void find_best_part_index (int *best_indecies, double **cc_fitness, int p, int pop_size)
{
    best_indecies[p]=0;
    double min_fitness = cc_fitness[p][0];
    for (int j=0; j!=pop_size; j++)
    {
        if (cc_fitness[p][j]<min_fitness)
        {
            min_fitness = cc_fitness[p][j];
            best_indecies[p] = j;
        }
    }
}

double find_best_fitness_value (double **fitness_cc, int M, int pop_size)
{
    double minn = fitness_cc[0][0];

    for (int i=0; i!=M; i++)
    {
        for (int j=0; j!=pop_size; j++)
        {
            if (fitness_cc[i][j]<minn)
            {
                minn = fitness_cc[i][j];
            }
        }
    }
    return minn;
}


void variable_condition(int *indeces, int N)
{
    int var1 = 0;
    int var2 = 0;

    for (int i = 0; i != N; i++)
    {
        for (int j=0; j!= N; j++)
        {
            if (indeces[i] == j)
            {
                var1++;
            }
        }
        var2++;
    }

    if (var1 == var2)
    {
        cout << "Replacement of indecies is done correctly!" << endl;
    }

    if (var1 != var2)
    {
        cout << "PROBLEM WITH REPLACEMENT OF INDECES IN GROUPS" << endl;
        cin.get();
        cin.get();
    }
}


int max_number(int *x, int lenght)
{
    int a = x[0];

    for(int i=1; i!=lenght; i++)
    {
        if (a<x[i])
        {
            a = x[i];
        }
    }

    return a;



}

void rankify(double* A, double *R, int n)
{

    for (int i = 0; i < n; i++)
    {
        int r = 1, s = 1;

        for (int j = 0; j < n; j++)
        {
            if (j != i && A[j] < A[i])
                r += 1;

            if (j != i && A[j] == A[i])
                s += 1;
        }

        R[i] = r + (float)(s - 1) / (float) 2;

    }

    for (int i = 0; i < n; i++)

        for (int i=0; i!=n; i++)
        {
            A[i] = R[i];
        }

}



void random_performance(double *performance, int N, int &CC_index, int power)
{
        double rnd = RANDOM();
        int index = -1;
        double *performance2 =new double [N];
        double sum = 0.0;
        for (int i=0; i!=N; i++)
        {
            performance2[i] = pow(E, power*performance[i]);
            if (isinf(performance2[i])==1)
            {
                performance2[i] = 1e300;
            }
        }
                        for (int i=0;i!=N;i++)
        {cout<<performance2[i]<<endl;}
        cout<<endl;

        for (int i=0; i!=N; i++)
        {
            sum+=performance2[i];
        }
        for (int i=0; i!=N; i++)
        {
            performance2[i]=performance2[i]/sum;
        }
                                for (int i=0;i!=N;i++)
        {cout<<performance2[i]<<endl;}



        for (int i=1; i!=N; i++)
        {
            performance2[i]+=performance2[i-1];
        }


        for (int i=0; i!=N-1; i++)
        {

            if (rnd > performance2[i] && rnd <= performance2[i+1])
            {
                index = i+1;
            }
        }

        if (rnd >=0 && rnd <=performance2[0])
        {
            index = 0;
        }
        if (rnd >=performance2[N-1])
        {
            index = N-1;
        }
        CC_index = index;

        delete [] performance2;



}


