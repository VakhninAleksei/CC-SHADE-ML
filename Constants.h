#ifndef CONSTANTS_H_INCLUDED
#define CONSTANTS_H_INCLUDED

extern double  E = 2.7182818284590452353602874713526625;
extern double PI = 3.1415926535897932384626433832795029;


double randn(double M, double s){
        return M+s*sqrt(-2.0*log((double)(rand())/RAND_MAX))*sin(2.0*PI*(double)(rand())/RAND_MAX);
    }

    double randc(double a, double b){
        return a+b*tan(PI*((double)(rand())/RAND_MAX - 0.5));

    }

double RANDOM()
    {
        double x=-1;
        while (x<0.0 || x > 1.0)
            {
                x = (double)(rand())/(double)RAND_MAX;
            }
        return x;
}

#endif // CONSTANTS_H_INCLUDED
