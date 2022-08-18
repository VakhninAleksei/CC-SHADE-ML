#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <stdlib.h>
#include <fstream>
#include <cstdlib>
#include <time.h>
#include <ctime>
#include "Header.h"
#include "lsgocec2013benchmarks.h"
#include "Constants.h"
using namespace std;

const int N = 1000; 
double a = -100.0, b = 100.0, best_solution = 1e300, THE_BEST_solution = 1e300; 

int const FEV_global = 3e6, krantost = FEV_global/100;
int ID, M, max_CC, R = 25, pop_size, FEV;
const int pop_size_init = 100;
string name_of_func; 

bool tournament_selection = 0;

const int H = 6;
double piece = 0.1;

int r1, r2, pbest, trigger_data_stat; 
int*indeces = new int [N];

int main(int argc, char** argv)
{
    int pop_size_pool[3] = {25, 50, 100};//{25, 50, 100, 150, 200};
    int subcomponents_pool[4] = {5, 10, 20, 50};//{1, 2, 5, 10, 20, 50, 100, 200, 500, 1000};
    double performance_pop_size[3] = {1.0,1.0,1.0};
    double performance_CC[4] = {1.0,1.0,1.0,1.0};

    int pop_size_max = max_number(pop_size_pool, 3);
    const int archive_size = pop_size_max;
    int M = max_number(subcomponents_pool, 4);

    int world_size, world_rank, name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(NULL, NULL); // Initialize the MPI environment
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get the number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get the rank of the process
    MPI_Get_processor_name(processor_name, &name_len);// Get the name of the processor

    int params_1[15] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    double params_2[1] =  {7}; 

    int param_1_counter = sizeof(params_1)/sizeof(params_1[0]);
    int param_2_counter = sizeof(params_2)/sizeof(params_2[0]);


    int all_params = param_1_counter*param_2_counter;

    int *thread_number = new int [all_params];

    int j=0;

    for (int i=0; i!=all_params; i++)
    {
        thread_number[i] = j;
        j++;
        if (j > world_size-1)
        {
            j=0;
        }
    }

    int thread_index = 0;
    for (int p1=0; p1!=param_1_counter; p1++)
    {
        for (int p2=0; p2!=param_2_counter; p2++)
         {


        if (world_rank == thread_number[thread_index])
        {
            srand(time(0));
            ID = params_1[p1];
            double power = params_2[p2];
            M = max_number(subcomponents_pool, 4);
            pop_size = pop_size_max;


           
            int S = N / M; 
            int *range = new int[M + 1]; 
            range[0] = 0;
            range[M] = N;

            for (int i = 1; i < M; i++)
            {
                range[i] = range[i - 1] + S;   
            }


            double *Ovector = new double[N]; 
            int *Pvector = new int[N]; 

            double **r25 = new double*[25];
            for (int count = 0; count < 25; count++)
            {
                r25[count] = new double[25];
            }

            double **r50 = new double*[50];
            for (int count = 0; count < 50; count++)
            {
                r50[count] = new double[50];
            }

            double **r100 = new double*[100];
            for (int count = 0; count < 100; count++)
            {
                r100[count] = new double[100];
            }

            int s_size = 0;

            if (ID == 4 || ID == 5 || ID == 6 || ID == 7)   
            {
                s_size = 7;
            }

            if (ID == 8 || ID == 9 || ID == 10 || ID == 11 || ID == 13 || ID == 14)   
            {
                s_size = 20;
            }

            int *s = new int[s_size];
            double *w = new double[s_size];

            if (ID == 4 || ID == 5 || ID == 6 || ID == 7 || ID == 8 || ID == 9 || ID == 10 || ID == 11 || ID == 13 || ID == 14)
            {
                Pvector = readPermVector(N, ID);
                r25 = readR(25, ID);
                r50 = readR(50, ID);
                r100 = readR(100, ID);
                s = readS(s_size, ID);
                w = readW(s_size, ID);
            }

            double **OvectorVec = new double*[s_size];
            for (int count = 0; count < s_size; count++)
                OvectorVec[count] = new double[s[count]];

            if (ID == 14)
            {
                OvectorVec = readOvectorVec(N, s_size, s, ID);
            }
            if (ID != 14)
            {
                Ovector = readOvector(N, ID);
            }

            select_borders(a, b, name_of_func, ID);

          
            const int H = 6;
            int *k = new int [M]; 
            int *A = new int [M]; 
            double piece = 0.1;

            int r1, r2, pbest, trigger; 

            string name ="Experiment_results/INFO_" + to_string(ID) +"_power_"+to_string(power)+  "_.txt"; 
            ofstream fout(name);
            fout << "CC-SHADE_ML" << "LSGO CEC'2013"<< endl<< "ID: " << ID << endl<<"FEV: "<<FEV_global<<endl<<"Name_of benchmark problem: "<<name_of_func<<endl;
            fout << "R: "<< R << endl <<"Population size: "<< pop_size<<endl<< "N: " << N <<endl << "a: " << a << endl << "b: " << b << endl;
            fout << "M: " << M <<endl<<"H: "<<H<<endl<<"Archive_size: "<<archive_size<<endl<<"Piece: "<<piece<<endl;

            double **population = new double* [pop_size]; 
            for (int count = 0; count < pop_size; count++)
                population[count] = new double [N];

            double **population_new = new double* [pop_size];
            for (int count = 0; count < pop_size; count++)
                population_new[count] = new double [N];

            double **archive= new double* [archive_size];
            for (int count = 0; count < archive_size; count++)
                archive[count] = new double [N];

            double *solution = new double [N];

            double **u = new double *[pop_size]; 
            for (int count = 0; count < pop_size; count++)
                u[count] = new double [N];

            double **data_stat = new double *[R];
            for (int count = 0; count < R; count++)
                data_stat[count] = new double [100];

            double **data_stat_CC = new double *[R];
            for (int count = 0; count < R; count++)
                data_stat_CC[count] = new double [100];

            double **data_stat_pop_size = new double *[R];
            for (int count = 0; count < R; count++)
                data_stat_pop_size[count] = new double [100];

            double **HISTORY_F = new double *[M];
            for (int count = 0; count < M; count++)
                HISTORY_F[count] = new double [H];

            double **HISTORY_CR = new double *[M];
            for (int count = 0; count < M; count++)
                HISTORY_CR[count] = new double [H];

            int *r = new int [pop_size];
            double *F = new double [pop_size];
            double *CR = new double [pop_size];

            double *fitness = new double [pop_size];
            double *fitness_new = new double [pop_size];

            double *S_CR = new double [pop_size];
            double *S_F = new double [pop_size];
            double *delta_f = new double [pop_size];
            double *W = new double [pop_size];


            double **best_solution_ever = new double *[R];
            for (int count =0; count <R; count++)
                best_solution_ever[count] = new double [N];
            double *best_fitness_ever = new double [R];

            
            int *cc_best_individual_index = new int[M];

            double **fitness_cc = new double *[M]; 
            for (int count =0; count <M; count++)
            {
                fitness_cc[count] = new double [pop_size];
            }

            double **fitness_cc_new = new double *[M];
            for (int count =0; count <M; count++)
            {
                fitness_cc_new[count] = new double [pop_size];
            }


           

            double best_fitness_before, best_fitness_after;
            for (int z=0; z!=R; z++) 
            {
                FEV = FEV_global;
                trigger = 0;
                best_solution = 1e300;
                int success = 0;

                initializePopulation(population, population_new, max_number(pop_size_pool, 3) , N, a, b);

                for (int i=0; i!=4; i++)
                {
                    performance_CC[i] = 1.0;
                }
                for (int i=0; i!=3; i++)
                {
                    performance_pop_size[i] = 1.0;
                }

                double *X_best_solution = new double [N];
                for (int i=0; i!=N; i++)
                {
                    solution[i] = 0.0;
                }
                while (FEV>0)
                {
                    randperm(indeces, N);
                    int CC_index = 0;
                    int pop_size_index = 0;

                    random_performance(performance_CC, 4, CC_index, power);
                    random_performance(performance_pop_size,3, pop_size_index, power);

                    pop_size = pop_size_pool[pop_size_index];

                    int piece_int = pop_size*piece; 
                    M = subcomponents_pool[CC_index];
                    S = N / M; 

                    range[0] = 0;
                    range[M] = N;

                    for (int i = 1; i < M; i++)
                    {
                        range[i] = range[i - 1] + S;   
                    }

                    cout<<"M: "<<M<<endl;

                    for (int i=0; i!=4; i++)
                    {
                        cout<<performance_CC[i]<<" ";
                    }
                    cout<<endl;




                    indecesSuccession(indeces, N); 
                    randperm(indeces,N);


                    for (int i=0; i!=M; i++)
                    {
                        A[i] = 0;
                    }
                    for (int i=0; i!=M; i++)
                    {
                        k[i] = 0;
                    }

                    for (int i=0; i!=M; i++)
                    {
                        k[i] = 0;
                    }

                    for (int i=0; i!=pop_size; i++)
                    {
                        S_CR[i] = S_F[i] = delta_f[i] = 0.0;
                    }

                    initializeHistory(HISTORY_F, HISTORY_CR, H, M);

                    rnd_indecies(cc_best_individual_index, pop_size, M);

                    for (int p=0; p!=M; p++)
                    {
                        for (int i=0; i!=pop_size; i++)
                        {
                            for (int j=range[p]; j!=range[p+1]; j++)
                            {
                                solution[indeces[j]] = population[i][indeces[j]];
                            }

                            for (int p_cc = 0; p_cc < M; p_cc++)
                            {
                                if (p !=p_cc)
                                {
                                    for (int j = range[p_cc]; j < range[p_cc + 1]; j++) 
                                    {
                                        solution[indeces[j]] = population[cc_best_individual_index[p_cc]][indeces[j]];
                                    }
                                }
                            }

                            fitness_cc[p][i] = fitness_cc_new[p][i] = benchmark_func(solution, Ovector, OvectorVec, Pvector, r25, r50, r100, s, w, N, ID);;
                            FEV--;
                            if (FEV % krantost == 0 )
                            {
                                std::cout<<name_of_func<<endl;
                                std::cout<<"Current best solution value: "<<best_solution<<endl;
                                std::cout<<"Current FEV: "<<FEV<<endl;
                                std::cout<<z+1<<" out of "<<R<<" RUNS"<<endl;
                                std::cout<<"Number of subcomponents: "<<M<<endl;
                                std::cout<<"Current pop_size: "<<pop_size<<endl;
                                std::cout<<"Archive_size: "<<archive_size<<endl<<endl;
                                
                                if (trigger<=99)
                                {
                                    data_stat[z][trigger] = best_solution;
                                    data_stat_CC[z][trigger] = M;
                                    data_stat_pop_size[z][trigger] = pop_size;
                                    trigger++;
                                }
                            }
                            if (fitness_cc[p][i] < best_solution)
                            {
                                best_solution = fitness_cc[p][i];
                                for (int k=0; k!=N; k++)
                                {
                                    X_best_solution[k] = solution[k];
                                }
                            }

                        }

                        find_best_part_index(cc_best_individual_index,fitness_cc, p, pop_size);
                    }
                    if (find_best_fitness_value(fitness_cc, M, pop_size)< best_solution)
                    {
                        best_solution = find_best_fitness_value(fitness_cc, M, pop_size);
                    }


                    int FEV_cycle = 3e6/50;
                    best_fitness_before = best_solution;
                    while (FEV_cycle>0)
                    {
                        for (int p=0; p!=M; p++)
                        {
                            for (int i=0; i!=pop_size; i++) 
                            {
                                findBestIndex(fitness_cc, pbest, pop_size, piece_int, p); 
                                r[i] = RANDOM() * (H-1);
                                generation_CR(CR[i], HISTORY_CR, r[i], p);
                                generation_F(F[i], HISTORY_F, r[i], p);
                                chooseCrossoverIndecies(r1, r2, pbest, pop_size, A, p);

                                if (r2<pop_size) 
                                {
                                    for (int j=range[p]; j!=range[p+1]; j++)
                                    {
                                        u[i][indeces[j]] = population[i][indeces[j]]+F[i]*(population[pbest][indeces[j]]-population[i][indeces[j]])+
                                        F[i]*(population[r1][indeces[j]]-population[r2][indeces[j]]);
                                    }
                                }

                                if (r2>=pop_size) 
                                {
                                    r2-=pop_size;

                                    for (int j=range[p]; j!=range[p+1]; j++) 
                                    {
                                        u[i][indeces[j]] = population[i][indeces[j]]+F[i]*(population[pbest][indeces[j]]-population[i][indeces[j]])+F[i]*(population[r1][indeces[j]]-archive[r2][indeces[j]]);
                                    }
                                }

                                int jrand = RANDOM()*((range[p+1]-range[p])+range[p]);

                                for (int j=range[p]; j!=range[p+1]; j++)
                                {
                                    if(RANDOM()<=CR[i] || j==jrand)
                                    {
                                    }
                                    else
                                    {
                                        u[i][indeces[j]] = population[i][indeces[j]];
                                    }
                                }
                                check_out_borders(u, population, i, N, a, b, range, p, indeces);
                            }

                            for (int i=0; i!=pop_size; i++)
                            {
                                for (int j=range[p]; j!=range[p+1]; j++)
                                {
                                    solution[indeces[j]] = u[i][indeces[j]];
                                }
                                for (int p_cc = 0; p_cc < M; p_cc++)
                                {
                                    if (p != p_cc)
                                    {
                                        for (int j = range[p_cc]; j < range[p_cc + 1]; j++)
                                        {
                                            solution[indeces[j]] = population[cc_best_individual_index[p_cc]][indeces[j]];
                                        }
                                    }
                                }

                                double test_function = benchmark_func(solution, Ovector, OvectorVec, Pvector, r25, r50, r100, s, w, N, ID);;
                                FEV--;
                                FEV_cycle--;

                                if (FEV % krantost == 0 )
                                {
                                    std::cout<<name_of_func<<endl;
                                    std::cout<<"Current best solution value: "<<best_solution<<endl;
                                    std::cout<<"Current FEV: "<<FEV<<endl;
                                    std::cout<<z+1<<" out of "<<R<<" RUNS"<<endl;
                                    std::cout<<"Number of subcomponents: "<<M<<endl;
                                    std::cout<<"Current pop_size: "<<pop_size<<endl;
                                    std::cout<<"Archive_size: "<<archive_size<<endl<<endl;
                                    
                                    if (trigger<=99)
                                    {
                                        data_stat[z][trigger] = best_solution;
                                        data_stat_CC[z][trigger] = M;
                                        data_stat_pop_size[z][trigger] = pop_size;
                                        trigger++;

                                    }
                                }

                                if (test_function < fitness_cc[p][i])
                                {
                                    for (int j=range[p]; j!=range[p+1]; j++) 
                                    {
                                        population_new[i][indeces[j]] = u[i][indeces[j]];
                                    }

                                    updateArchive(archive, population, i, archive_size, A,range, p, indeces);
                                    fitness_cc_new[p][i] = test_function;
                                    delta_f[success] = sqrt ((test_function-fitness_cc[p][i])*(test_function-fitness_cc[p][i]));
                                    S_F[success] = F[i];
                                    S_CR[success] = CR[i];
                                    success++;
                                }

                            } 

                            Algorithm_1(delta_f, W, S_CR, S_F, HISTORY_CR, HISTORY_F, k, success, H, p);

                            for (int i=0; i!=pop_size; i++)
                            {
                                for (int j=range[p]; j!=range[p+1]; j++)
                                {
                                    population[i][indeces[j]] = population_new[i][indeces[j]];
                                }
                                fitness_cc[p][i] = fitness_cc_new[p][i];
                            }

                            double min_population_fitness_value = find_best_fitness_value(fitness_cc, M, pop_size);

                            if (min_population_fitness_value < best_solution)
                            {
                                best_solution = min_population_fitness_value;
                                for (int q=0; q!=N; q++)
                                {
                                    best_solution_ever[z][q] = solution[q];
                                }
                                best_fitness_ever[z] = min_population_fitness_value;
                            }

                            find_best_part_index(cc_best_individual_index, fitness_cc, p, pop_size);

                        } 
                    }
                    
                    best_fitness_after = best_solution;

                    performance_CC[CC_index] = performance_pop_size[pop_size_index] = (best_fitness_before-best_fitness_after)/best_fitness_after;

                    if (performance_CC[CC_index]==INF)
                    {
                        performance_CC[CC_index] = 1e308;
                        cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
                    }
                    if (performance_CC[CC_index]<1e-4)
                    {
                        performance_CC[CC_index] = 1e-4;
                    }

                    if (performance_pop_size[pop_size_index]==INF)
                    {
                        performance_pop_size[pop_size_index] = 1e308;
                    }

                    if (performance_pop_size[pop_size_index]<1e-4)
                    {
                        performance_pop_size[pop_size_index] = 1e-4;
                    }

                  

                } 


            } 
            fout<<endl;
           
            for (int i=0; i!=R; i++)
            {
                for (int j=0; j!=100; j++)
                {
                    if (j<99)
                    {
                        fout<<data_stat[i][j]<<", ";
                    }

                    if (j == 99)
                    {
                        fout<<data_stat[i][j];
                    }
                }
                fout<<endl;
            }
            fout << endl;
            fout << "Average:" << endl;
            for (int i = 0; i != 100; i++) 
            {
                double sum = 0.0;

                for (int j = 0; j != R; j++)
                {
                    sum += data_stat[j][i];
                }

                if (i<99)
                {
                    fout << sum / R << ", ";
                }
                if (i==99)
                {
                    fout << sum / R;
                }
            }
            fout << endl;
            fout << endl;
            fout << "Best decisions" << endl;

            for (int i = 0; i < R; i++) 
            {
                data_stat[i][99] = best_fitness_ever[i];
                if (i!=(R-1))
                {
                    fout << data_stat[i][99] << ", ";
                }
                if (i == R-1)
                {
                    fout << data_stat[i][99];
                }
            }
            fout << endl;
            fout << endl;
            int sr1 = 4;
            fout << endl;
            fout << "1.2e+5: " << endl;
            fout << "BEST: " << min_stat(data_stat, sr1 - 1, R) << endl;
            fout << "MEDIAN: " << median_stat(data_stat, sr1 - 1, R) << endl;
            fout << "WORST: " << max_stat(data_stat, sr1 - 1, R) << endl;
            fout << "MEAN: " << mean_stat(data_stat, sr1 - 1, R) << endl;
            fout << "STDEV: " << stddev_stat(data_stat, sr1 - 1, R, mean_stat(data_stat, sr1 - 1, R)) << endl;
            fout << endl;

            int sr2 = 20;
            fout << endl;
            fout << "6.0e+5: " << endl;
            fout << "BEST: " << min_stat(data_stat, sr2 - 1, R) << endl;
            fout << "MEDIAN: " << median_stat(data_stat, sr2 - 1, R) << endl;
            fout << "WORST: " << max_stat(data_stat, sr2 - 1, R) << endl;
            fout << "MEAN: " << mean_stat(data_stat, sr2 - 1, R) << endl;
            fout << "STDEV: " << stddev_stat(data_stat, sr2 - 1, R, mean_stat(data_stat, sr2 - 1, R)) << endl;
            fout << endl;


            int sr3 = 100;
            fout << endl;
            fout << "3.0e+6: " << endl;
            fout << "BEST: " << min_stat(data_stat, sr3 - 1, R) << endl;
            fout << "MEDIAN: " << median_stat(data_stat, sr3 - 1, R) << endl;
            fout << "WORST: " << max_stat(data_stat, sr3 - 1, R) << endl;
            fout << "MEAN: " << mean_stat(data_stat, sr3 - 1, R) << endl;
            fout << "STDEV: " << stddev_stat(data_stat, sr3 - 1, R, mean_stat(data_stat, sr3 - 1, R)) << endl;
            fout << endl;
            fout << endl;

            fout << endl;
            fout << min_stat(data_stat, sr3 - 1, R) << endl;
            fout << median_stat(data_stat, sr3 - 1, R) << endl;
            fout << max_stat(data_stat, sr3 - 1, R) << endl;
            fout << mean_stat(data_stat, sr3 - 1, R) << endl;
            fout << stddev_stat(data_stat, sr3 - 1, R, mean_stat(data_stat, sr3 - 1, R)) << endl;

            fout<<endl;
            fout<<"Pop_size:"<<endl;
          
            for (int i=0; i!=R; i++)
            {
                for (int j=0; j!=100; j++)
                {
                    if (j<99)
                    {
                        fout<<data_stat_pop_size[i][j]<<", ";
                    }

                    if (j == 99)
                    {
                        fout<<data_stat_pop_size[i][j];
                    }
                }
                fout<<endl;
            }
            fout << endl;
            fout << "Average_pop_size:" << endl;
            for (int i = 0; i != 100; i++)
            {
                double sum = 0.0;

                for (int j = 0; j != R; j++)
                {
                    sum += data_stat_pop_size[j][i];
                }

                if (i<99)
                {
                    fout << sum / R << ", ";
                }
                if (i==99)
                {
                    fout << sum / R;
                }
            }


                        fout<<endl;
            fout<<"CC:"<<endl;
            
            for (int i=0; i!=R; i++)
            {
                for (int j=0; j!=100; j++)
                {
                    if (j<99)
                    {
                        fout<<data_stat_CC[i][j]<<", ";
                    }

                    if (j == 99)
                    {
                        fout<<data_stat_CC[i][j];
                    }
                }
                fout<<endl;
            }
            fout << endl;
            fout << "Average_CC:" << endl;
            for (int i = 0; i != 100; i++)
            {
                double sum = 0.0;

                for (int j = 0; j != R; j++)
                {
                    sum += data_stat_CC[j][i];
                }

                if (i<99)
                {
                    fout << sum / R << ", ";
                }
                if (i==99)
                {
                    fout << sum / R;
                }
            }












            fout.close(); 

            for (int count = 0; count < pop_size; count++)
                delete [] population[count];

            for (int count = 0; count < pop_size; count++)
                delete [] population_new[count];

            for (int count = 0; count < archive_size; count++)
                delete [] archive[count];

            for (int count = 0; count < pop_size; count++)
            {
                delete [] u[count];
            }


        } 
        thread_index++;
        
        }
    }

    MPI_Finalize();
}
