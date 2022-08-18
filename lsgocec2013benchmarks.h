//#ifndef HEAD_H_INCLUDED
#define HEAD_H_INCLUDED

//#if !defined MYHEDER_H
#define _CRT_SECURE_NO_WARNINGS
#define MYHEDER_H

#include <sstream>
#include <vector>
#include <fstream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <ctime>
#include <random>
#include <time.h>
#include <stdlib.h>
#include <random>
using namespace std;

#define INF 1.0e99
#define EPS 1.0e-14


int sign(double x)
{
	if (x > 0) return 1;
	if (x < 0) return -1;
	return 0;
}

double hat(double x)
{
	if (x == 0)
	{
		return 0;
	}
	else
	{
		return log(abs(x));
	}
}


double c1(double x)
{
	if (x>0)
	{
		return 10;
	}
	else
	{
		return 5.5;
	}
}

double c2(double x)
{
	if (x>0)
	{
		return 7.9;
	}
	else
	{
		return 3.1;
	}
}


void transform_osz(double* z, int dim)
{
	for (int i = 0; i < dim; ++i)
	{
		z[i] = sign(z[i]) * exp(hat(z[i]) + 0.049 * (sin(c1(z[i]) * hat(z[i])) + sin(c2(z[i])* hat(z[i]))));
	}
}

void transform_asy(double* z, double beta, int dim)
{
	for (int i = 0; i < dim; i++)
	{
		if (z[i]>0)
		{
			z[i] = pow(z[i], 1 + beta * i / ((double)(dim - 1)) * sqrt(z[i]));
		}
	}
}

void Lambda(double* z, double alpha, int dim)
{
	for (int i = 0; i < dim; ++i)
	{
		z[i] = z[i] * pow(alpha, 0.5 * i / ((double)(dim - 1)));
	}
}


double* readOvector(int dimension, int ID)
{
	// read O vector from file in csv format
	double * d = new double[dimension];
	stringstream ss;
	ss << "cdatafiles/" << "F" << ID << "-xopt.txt";
	ifstream file(ss.str());
	string value;
	string line;
	int c = 0;

	if (file.is_open())
	{
		stringstream iss;
		while (getline(file, line))
		{
			iss << line;
			while (getline(iss, value, ','))
			{
				d[c++] = stod(value);
			}
			iss.clear();
			// if (c==dimension)
			//   {
			//     break;
			//   }
			// printf("%d\n",c);
		}
		file.close();
	}
	else
	{
		cout << "Cannot open datafiles" << endl;
	}
	return d;
}

double** readOvectorVec(int dimension, int s_size, int *s, int ID) 
{
	double** d = (double**)malloc(s_size * sizeof(double*));
	stringstream ss;
	ss << "cdatafiles/" << "F" << ID << "-xopt.txt";
	ifstream file(ss.str());
	string value;
	string line;
	int c = 0;                      // index over 1 to dim
	int i = -1;                      // index over 1 to s_size
	int up = 0;                   // current upper bound for one group

	if (file.is_open())
	{
		stringstream iss;
		while (getline(file, line))
		{
			if (c == up)             // out (start) of one group
			{
				// printf("=\n");
				i++;
				d[i] = (double*)malloc(s[i] * sizeof(double));
				up += s[i];
			}
			iss << line;
			while (getline(iss, value, ','))
			{
				// printf("c=%d\ts=%d\ti=%d\tup=%d\tindex=%d\n",c,s[i],i,up,c-(up-s[i]));
				d[i][c - (up - s[i])] = stod(value);
				// printf("1\n");
				c++;
			}
			iss.clear();
			// printf("2\n");
		}
		file.close();
	}
	else
	{
		cout << "Cannot open datafiles" << endl;
	}
	return d;
}

int* readPermVector(int dimension, int ID) {
	int* d;

	d = new int[dimension];

	stringstream ss;
	ss << "cdatafiles/" << "F" << ID << "-p.txt";
	ifstream file(ss.str());
	int c = 0;
	string value;

	if (file.is_open())
	{
		while (getline(file, value, ','))
		{
			d[c++] = stod(value) - 1;
		}
	}
	return(d);
}


double** readR(int sub_dim, int ID)
{
	double** m;
	m = new double*[sub_dim];
	for (int i = 0; i< sub_dim; i++)
	{
		m[i] = new double[sub_dim];
	}

	stringstream ss;
	ss << "cdatafiles/" << "F" << ID << "-R" << sub_dim << ".txt";
	// cout<<ss.str()<<endl;

	ifstream file(ss.str());
	string value;
	string line;
	int i = 0;
	int j;

	if (file.is_open())
	{
		stringstream iss;
		while (getline(file, line))
		{
			j = 0;
			iss << line;
			while (getline(iss, value, ','))
			{
				// printf("%d,%d\t%f\n", i,j, stod(value));
				m[i][j] = stod(value);
				// printf("done\n");
				j++;
			}
			iss.clear();
			i++;
		}
		file.close();
	}
	else
	{
		cout << "Cannot open datafiles" << endl;
	}
	return m;
}

int* readS(int num, int ID)
{
	int *s = new int[num];

	stringstream ss;
	ss << "cdatafiles/" << "F" << ID << "-s.txt";
	ifstream file(ss.str());
	int c = 0;
	string value;
	if (file.is_open())
	{
		while (getline(file, value))
		{
			// cout<<stod(value)<<endl;
			s[c++] = stod(value);
		}
	}
	return s;
}

double* readW(int num, int ID)
{
	double *w = new double[num];

	stringstream ss;
	ss << "cdatafiles/" << "F" << ID << "-w.txt";
	ifstream file(ss.str());
	int c = 0;
	string value;
	if (file.is_open())
	{
		while (getline(file, value))
		{
			// cout<<stod(value)<<endl;
			w[c++] = stod(value);
		}
	}
	return w;
}


void multiply(double*vector, double**matrix, int dim, double *result) {

	for (int i = 0; i < dim; i++) {
		result[i] = 0;

		for (int j = 0; j < dim; j++) {
			result[i] += vector[j] * matrix[i][j];
		}
	}
}


void rotateVector(int i, int c, double *anotherz, double *anotherz1, int *Pvector, int *s, double **r25, double **r50, double **r100, int dim)
{
	double* z = new double[s[i]];

	for (int j = c; j < c + s[i]; j++)
	{
		z[j - c] = anotherz[Pvector[j]];
	}

	if (s[i] == 25)
	{
		multiply(z, r25, s[i], anotherz1);
	}
	else if (s[i] == 50)
	{
		multiply(z, r50, s[i], anotherz1);
	}
	else if (s[i] == 100)
	{
		multiply(z, r100, s[i], anotherz1);
	}
	else
	{
		cout << "size of rotation matrix out of range" << endl;
	}
	delete[]z;
}

void rotateVectorConform(int i, int c, double *anotherz, double *anotherz1, int *Pvector, int *s, double **r25, double **r50, double **r100, int overlap, int dim)
{
	double* z = new double[s[i]];

	for (int j = c - i*overlap; j < c + s[i] - i*overlap; ++j)
	{
		z[j - (c - i*overlap)] = anotherz[Pvector[j]];
	}

	if (s[i] == 25)
	{
		multiply(z, r25, s[i], anotherz1);
	}
	else if (s[i] == 50)
	{
		multiply(z, r50, s[i], anotherz1);

	}
	else if (s[i] == 100)
	{
		multiply(z, r100, s[i], anotherz1);
	}
	else
	{
		cout << "size of rotation matrix out of range" << endl;
	}

	delete[]z;
}


void rotateVectorConflict(int i, int c, double *anotherz, double *anotherz1, int *Pvector, double **OvectorVec, int *s, double **r25, double **r50, double **r100, int overlap, int dim) 
{
	double* z = new double[s[i]];

	for (int j = c - i*overlap; j < c + s[i] - i*overlap; ++j)
	{
		z[j - (c - i*overlap)] = anotherz[Pvector[j]] - OvectorVec[i][j - (c - i*overlap)];
	}

	if (s[i] == 25)
	{
		multiply(z, r25, s[i], anotherz1);
	}
	else if (s[i] == 50)
	{
		multiply(z, r50, s[i], anotherz1);
	}
	else if (s[i] == 100)
	{
		multiply(z, r100, s[i], anotherz1);
	}
	else
	{
		cout << "size of rotation matrix out of range" << endl;
	}

	delete[]z;
}


double elliptic(double *x, int dim)
{
	double result = 0.0;
	int    i;
	transform_osz(x, dim);
	for (i = 0; i<dim; i++)
	{
		// printf("%f\n", pow(1.0e6,  i/((double)(dim - 1)) ));
		result += pow(1.0e6, i / ((double)(dim - 1))) * x[i] * x[i];
	}
	return(result);
}


double rastrigin(double *x, int dim)
{
	double sum = 0.0;
	int    i;
	// T_{osz}
	transform_osz(x, dim);
	// T_{asy}^{0.2}
	transform_asy(x, 0.2, dim);
	// lambda
	Lambda(x, 10, dim);

	for (i = dim - 1; i >= 0; i--) {
		sum += x[i] * x[i] - 10.0 * cos(2 * PI * x[i]) + 10.0;
	}
	return(sum);
}


double ackley(double *x, int dim)
{
	double sum1 = 0.0;
	double sum2 = 0.0;
	double sum = 0.0;
	int    i;
	// T_{osz}
	transform_osz(x, dim);
	// T_{asy}^{0.2}
	transform_asy(x, 0.2, dim);
	// lambda
	Lambda(x, 10, dim);
	for (int i = 0; i< dim; i++)
	{
		sum1 += (x[i] * x[i]);
		sum2 += cos(2.0 * PI * x[i]);
	}
	sum = -20.0 * exp(-0.2 * sqrt(sum1 / dim)) - exp(sum2 / dim) + 20.0 + E;
	return (sum);
}


double schwefel(double *x, int dim)
{
	int    j;
	double s1 = 0;
	double s2 = 0;

	// T_{osz}
	transform_osz(x, dim);
	// T_{asy}^{0.2}
	transform_asy(x, 0.2, dim);

	for (j = 0; j < dim; j++) {
		s1 += x[j];
		s2 += (s1 * s1);
	}
	return(s2);
}


double sphere(double*x, int dim) {
	double sum = 0;
	int    i;

	for (i = dim - 1; i >= 0; i--) {
		sum += pow(x[i], 2);
	}

	return(sum);
}

double rosenbrock(double*x, int dim) {
	int    j;
	double oz, t;
	double s = 0.0;
	j = dim - 1;

	for (--j; j >= 0; j--) {
		oz = x[j + 1];
		t = ((x[j] * x[j]) - oz);
		s += (100.0 * t * t);
		t = (x[j] - 1.0);
		s += (t * t);
	}
	return(s);
}

double shifted_elliptic(double *, double *, int); // F1 Shifted Elliptic Function
double shifted_rastrigin(double *, double *, int); // F2 Shifted Rastrigin's Function
double shifted_ackley(double *, double *, int); // F3 Shifted Ackley's Function
double nonsep_7_shifted_rotated_elliptic(double*, double *, int *, double **, double **, double **, int *, double *, int); // F4 7nonseparable, 1 separable Shifted and Rotated Elliptic Function
double nonsep_7_shifted_rotated_rastrigin(double*, double *, int *, double **, double **, double **, int *, double *, int); // F5 7nonseparable, 1 separable Shifted and Rotated Rastrigin Function
double nonsep_7_shifted_rotated_ackley(double*, double *, int *, double **, double **, double **, int *, double *, int); // F6 7nonseparable, 1 separable Shifted and Rotated Ackley Function
double nonsep_7_shifted_schwefel(double*, double *, int *, double **, double **, double **, int *, double *, int); // F7 7nonseparable, 1 separable Shifted and shifted Schwefel Function
double nonsep_20_shifted_elliptic(double*, double *, int *, double **, double **, double **, int *, double *, int); // F8 20 nonseparable,  Shifted and shifted Elliptic Function
double nonsep_20_shifted_rastrigin(double*, double *, int *, double **, double **, double **, int *, double *, int); // F9 20 nonseparable,  Shifted and shifted Rastrigin Function
double nonsep_20_shifted_ackley(double*, double *, int *, double **, double **, double **, int *, double *, int); // F10 20 nonseparable,  Shifted and shifted Ackley Function
double nonsep_20_shifted_schwefel(double*, double *, int *, double **, double **, double **, int *, double *, int); // F11 20 nonseparable,  Shifted and shifted Schewefel Function
double shifted_rosenbrock(double *, double *, int);  // F12 Shifted Rosenbrock Function
double nonsep_20_shifted_schwefel_overlapping(double*, double *, int *, double **, double **, double **, int *, double *, int); // F13 20 nonseparable,  Shifted and shifted Overlapping Schewefel Function
double schwefel_overlapping_ver2(double*, double **, int *, double **, double **, double **, int *, double *, int); // F14 20 nonseparable,  Shifted Overlapping Schewefel Function
double shifted_schwefel(double *, double *, int);  // F15 Shifted Rosenbrock Function

double benchmark_func(double *x, double *Ovector, double **OvectorVec, int *Pvector, double **r25, double **r50, double **r100, int *s, double *w, int dim, int ID)
{
	double function = 0.0;

	switch (ID)
	{
	case 1:
		function = shifted_elliptic(x, Ovector, dim);
		break;

	case 2:
		function = shifted_rastrigin(x, Ovector, dim);
		break;

	case 3:
		function = shifted_ackley(x, Ovector, dim);
		break;

	case 4:
		function = nonsep_7_shifted_rotated_elliptic(x, Ovector, Pvector, r25, r50, r100, s, w, dim);
		break;

	case 5:
		function = nonsep_7_shifted_rotated_rastrigin(x, Ovector, Pvector, r25, r50, r100, s, w, dim);
		break;

	case 6:
		function = nonsep_7_shifted_rotated_ackley(x, Ovector, Pvector, r25, r50, r100, s, w, dim);
		break;

	case 7:
		function = nonsep_7_shifted_schwefel(x, Ovector, Pvector, r25, r50, r100, s, w, dim);
		break;

	case 8:
		function = nonsep_20_shifted_elliptic(x, Ovector, Pvector, r25, r50, r100, s, w, dim);
		break;

	case 9:
		function = nonsep_20_shifted_rastrigin(x, Ovector, Pvector, r25, r50, r100, s, w, dim);
		break;

	case 10:
		function = nonsep_20_shifted_ackley(x, Ovector, Pvector, r25, r50, r100, s, w, dim);
		break;

	case 11:
		function = nonsep_20_shifted_schwefel(x, Ovector, Pvector, r25, r50, r100, s, w, dim);
		break;

	case 12:
		function = shifted_rosenbrock(x, Ovector, dim);
		break;

	case 13:
		function = nonsep_20_shifted_schwefel_overlapping(x, Ovector, Pvector, r25, r50, r100, s, w, dim);
		break;

	case 14:
		function = schwefel_overlapping_ver2(x, OvectorVec, Pvector, r25, r50, r100, s, w, dim);
		break;

	case 15:
		function = shifted_schwefel(x, Ovector, dim);
		break;

	}
	return function;
}


double shifted_elliptic(double *x, double *Ovector, int dim) {

	double result = 0.0;
	double *anotherz = new double[dim];

	for (int i = 0; i != dim; i++)
	{
		anotherz[i] = x[i];
	}

	for (int i = 0; i != dim; i++) {
		anotherz[i] = anotherz[i] - Ovector[i];
	}

	transform_osz(anotherz, dim);

	for (int i = 0; i!=dim; i++)
	{
		result += pow(1.0e6, i / ((double)(dim - 1))) * anotherz[i] * anotherz[i];
	}

	delete[] anotherz;
	return(result);
}


double shifted_rastrigin(double*x, double *Ovector, int dim) {

	double sum = 0.0;
	int i;
	double *anotherz = new double[dim];
	for (int i = 0; i < dim; i++)
	{
		anotherz[i] = x[i];
	}

	for (i = 0; i< dim; i++) {
		anotherz[i] = anotherz[i] - Ovector[i];
	}

	sum = rastrigin(anotherz, dim);

	delete[] anotherz;
	return(sum);
}

double shifted_ackley(double*x, double *Ovector, int dim) {

	double result = 0.0;
	double *anotherz = new double[dim];

	for (int i = 0; i < dim; i++)
	{
		anotherz[i] = x[i];
	}

	for (int i = 0; i< dim; i++)
	{
		anotherz[i] = anotherz[i] - Ovector[i];
	}

	result = ackley(anotherz, dim);

	delete[] anotherz;
	return(result);
}

double nonsep_7_shifted_rotated_elliptic(double*x, double *Ovector, int *Pvector, double **r25, double **r50, double **r100, int *s, double *w, int dim)
{
	double result = 0.0;
	int s_size = 7;
	double *anotherz = new double[dim];
	double *anotherz1 = new double[dim];

	for (int i = 0; i < dim; i++)
	{
		anotherz[i] = x[i];
	}

	for (int i = 0; i< dim; i++)
	{
		anotherz[i] = anotherz[i] - Ovector[i];
	}

	int c = 0;

	for (int i = 0; i < s_size; i++)
	{
		rotateVector(i, c, anotherz, anotherz1, Pvector, s, r25, r50, r100, dim);
		result += w[i] * elliptic(anotherz1, s[i]);
		c = c + s[i];
	}

	double* z = new double[dim - c];

	for (int i = c; i < dim; i++)
	{
		z[i - c] = anotherz[Pvector[i]];
	}

	result += elliptic(z, dim - c);

	delete[] z;
	delete[] anotherz;
	delete[] anotherz1;
	return(result);
}


double nonsep_7_shifted_rotated_rastrigin(double*x, double *Ovector, int *Pvector, double **r25, double **r50, double **r100, int *s, double *w, int dim)
{
	double result = 0.0;
	int s_size = 7;
	double *anotherz = new double[dim];
	double *anotherz1 = new double[dim];

	for (int i = 0; i < dim; i++)
	{
		anotherz[i] = x[i];
	}

	for (int i = 0; i< dim; i++) 
	{
		anotherz[i] = anotherz[i] - Ovector[i];
	}

	int c = 0;
	for (int i = 0; i < s_size; i++)
	{
		rotateVector(i, c, anotherz, anotherz1, Pvector, s, r25, r50, r100, dim);
		result += w[i] * rastrigin(anotherz1, s[i]);
		c = c + s[i];
	}

	double* z = new double[dim - c];

	for (int i = c; i < dim; i++)
	{
		z[i - c] = anotherz[Pvector[i]];
	}

	result += rastrigin(z, dim - c);

	delete[] z;
	delete[] anotherz;
	delete[] anotherz1;

	return(result);

}


double nonsep_7_shifted_rotated_ackley(double*x, double *Ovector, int *Pvector, double **r25, double **r50, double **r100, int *s, double *w, int dim)
{
	double result = 0.0;
	int s_size = 7;
	double *anotherz = new double[dim];
	double *anotherz1 = new double[dim];

	for (int i = 0; i < dim; i++)
	{
		anotherz[i] = x[i];
	}

	for (int i = 0; i< dim; i++) 
	{
		anotherz[i] = anotherz[i] - Ovector[i];
	}

	int c = 0;

	for (int i = 0; i < s_size; i++)
	{
		rotateVector(i, c, anotherz, anotherz1, Pvector, s, r25, r50, r100, dim);
		result += w[i] * ackley(anotherz1, s[i]);
		c = c + s[i];
	}

	double* z = new double[dim - c];

	for (int i = c; i < dim; i++)
	{
		z[i - c] = anotherz[Pvector[i]];
	}

	result += ackley(z, dim - c);

	delete[] z;
	delete[] anotherz;
	delete[] anotherz1;

	return(result);

}

double nonsep_7_shifted_schwefel(double*x, double *Ovector, int *Pvector, double **r25, double **r50, double **r100, int *s, double *w, int dim)
{

	double result = 0.0;
	int s_size = 7;
	double *anotherz = new double[dim];
	double *anotherz1 = new double[dim];

	for (int i = 0; i < dim; i++)
	{
		anotherz[i] = x[i];
	}

	for (int i = 0; i< dim; i++) 
	{
		anotherz[i] = anotherz[i] - Ovector[i];
	}

	int c = 0;

	for (int i = 0; i < s_size; i++)
	{
		rotateVector(i, c, anotherz, anotherz1, Pvector, s, r25, r50, r100, dim);
		result += w[i] * schwefel(anotherz1, s[i]);
		c = c + s[i];
	}

	double* z = new double[dim - c];

	for (int i = c; i < dim; i++)
	{
		z[i - c] = anotherz[Pvector[i]];
	}

	result += sphere(z, dim - c);

	delete[] z;
	delete[] anotherz;
	delete[] anotherz1;

	return(result);
}

double  nonsep_20_shifted_elliptic(double*x, double *Ovector, int *Pvector, double **r25, double **r50, double **r100, int *s, double *w, int dim)
{
	double result = 0.0;
	int s_size = 20;
	double *anotherz = new double[dim];
	double *anotherz1 = new double[dim];

	for (int i = 0; i < dim; i++)
	{
		anotherz[i] = x[i];
	}

	for (int i = 0; i< dim; i++)
	{
		anotherz[i] = anotherz[i] - Ovector[i];
	}

	int c = 0;

	for (int i = 0; i < s_size; i++)
	{
		rotateVector(i, c, anotherz, anotherz1, Pvector, s, r25, r50, r100, dim);
		result += w[i] * elliptic(anotherz1, s[i]);
		c = c + s[i];
	}

	delete[] anotherz;
	delete[] anotherz1;

	return (result);
}

double nonsep_20_shifted_rastrigin(double*x, double *Ovector, int *Pvector, double **r25, double **r50, double **r100, int *s, double *w, int dim)
{

	double result = 0.0;
	int s_size = 20;

	double *anotherz = new double[dim];
	double *anotherz1 = new double[dim];

	for (int i = 0; i < dim; i++)
	{
		anotherz[i] = x[i];
	}

	for (int i = 0; i< dim; i++) 
	{
		anotherz[i] = anotherz[i] - Ovector[i];
	}

	int c = 0;

	for (int i = 0; i < s_size; i++)
	{
		rotateVector(i, c, anotherz, anotherz1, Pvector, s, r25, r50, r100, dim);
		result += w[i] * rastrigin(anotherz1, s[i]);
		c = c + s[i];
	}

	delete[] anotherz;
	delete[] anotherz1;

	return (result);
}

double nonsep_20_shifted_ackley(double*x, double *Ovector, int *Pvector, double **r25, double **r50, double **r100, int *s, double *w, int dim)
{
	double result = 0.0;
	int s_size = 20;

	double *anotherz = new double[dim];
	double *anotherz1 = new double[dim];

	for (int i = 0; i < dim; i++)
	{
		anotherz[i] = x[i];
	}

	for (int i = 0; i< dim; i++)
	{
		anotherz[i] = anotherz[i] - Ovector[i];
	}

	int c = 0;

	for (int i = 0; i < s_size; i++)
	{
		rotateVector(i, c, anotherz, anotherz1, Pvector, s, r25, r50, r100, dim);
		result += w[i] * ackley(anotherz1, s[i]);
		c = c + s[i];
	}

	delete[] anotherz;
	delete[] anotherz1;

	return (result);
}

double nonsep_20_shifted_schwefel(double*x, double *Ovector, int *Pvector, double **r25, double **r50, double **r100, int *s, double *w, int dim)
{

	double result = 0.0;
	int s_size = 20;

	double *anotherz = new double[dim];
	double *anotherz1 = new double[dim];

	for (int i = 0; i < dim; i++)
	{
		anotherz[i] = x[i];
	}

	for (int i = 0; i< dim; i++) 
	{
		anotherz[i] = anotherz[i] - Ovector[i];
	}

	int c = 0;

	for (int i = 0; i < s_size; i++)
	{
		rotateVector(i, c, anotherz, anotherz1, Pvector, s, r25, r50, r100, dim);
		result += w[i] * schwefel(anotherz1, s[i]);
		c = c + s[i];
	}

	delete[] anotherz;
	delete[] anotherz1;

	return (result);
}

double shifted_rosenbrock(double *x, double *Ovector, int dim) {

	double result = 0.0;
	double *anotherz = new double[dim];
	for (int i = 0; i < dim; i++)
	{
		anotherz[i] = x[i];
	}

	for (int i = 0; i< dim; i++) {
		anotherz[i] = anotherz[i] - Ovector[i];
	}


	result = rosenbrock(anotherz, dim);

	delete[] anotherz;
	return(result);
}

double nonsep_20_shifted_schwefel_overlapping(double*x, double *Ovector, int *Pvector, double **r25, double **r50, double **r100, int *s, double *w, int dim)
{
	double result = 0.0;
	int s_size = 20;
	dim = 905;       
	int overlap = 5;

	double *anotherz = new double[dim];
	double *anotherz1 = new double[dim];

	for (int i = 0; i < dim; i++)
	{
		anotherz[i] = x[i];
	}

	for (int i = 0; i < dim; i++)
	{
		anotherz[i] = anotherz[i] - Ovector[i];
	}

	int c = 0;

	for (int i = 0; i < s_size; i++)
	{
		rotateVectorConform(i, c, anotherz, anotherz1, Pvector, s, r25, r50, r100, overlap, dim);
		result += w[i] * schwefel(anotherz1, s[i]);
		c = c + s[i];
	}

	delete[] anotherz;
	delete[] anotherz1;
	return (result);
}

double schwefel_overlapping_ver2(double *x, double **OvectorVec, int *Pvector, double **r25, double **r50, double ** r100, int *s, double *w, int dim)
{
	double result = 0.0;
	int s_size = 20;
	dim = 905;        
	int overlap = 5;
	int i;
	double *anotherz1 = new double[dim];
	double *anotherz = new double[dim];

	for (int i = 0; i < dim; i++)
	{
		anotherz[i] = x[i];
	}

	int c = 0;
	for (i = 0; i < s_size; i++)
	{
		rotateVectorConflict(i, c, anotherz, anotherz1, Pvector, OvectorVec, s, r25, r50, r100, overlap, dim);
		result += w[i] * schwefel(anotherz1, s[i]);
		c = c + s[i];
	}

	delete[] anotherz1;
	delete[] anotherz;
	return(result);
}

double shifted_schwefel(double *x, double *Ovector, int dim) {

	double result = 0.0;
	double *anotherz = new double[dim];

	for (int i = 0; i < dim; i++)
	{
		anotherz[i] = x[i];
	}

	for (int i = 0; i< dim; i++) {
		anotherz[i] = anotherz[i] - Ovector[i];
	}

	result = schwefel(anotherz, dim);
	delete[] anotherz;
	return(result);
}

void select_borders(double &a, double &b, string &name_of_func, int ID)

{
	a = 0.0;
	b = 0.0;

	switch (ID)
	{

	case 1:
		a = -100.0;
		b = 100.0;
		name_of_func = "F1_Shifted Elliptic Function";
		break;

	case 2:
		a = -5.0;
		b = 5.0;
		name_of_func = "F2_Shifted Rastrigin's Function";
		break;

	case 3:
		a = -32.0;
		b = 32.0;
		name_of_func = "F3_ Shifted Ackley's Function";
		break;

	case 4:
		a = -100.0;
		b = 100.0;
		name_of_func = "F4_7-nonsep, 1-sep Shifted and Rotated Elliptic Function";
		break;

	case 5:
		a = -5.0;
		b = 5.0;
		name_of_func = "F5_7-nonsep, 1-sep Shifted and Rotated Rastrifin's Function";
		break;

	case 6:
		a = -32.0;
		b = 32.0;
		name_of_func = "F6_7-nonsep, 1-sep Shifted and Rotated Ackley's Function";
		break;

	case 7:
		a = -100.0;
		b = 100.0;
		name_of_func = "F7_7-nonsep, 1-sep Shifted and Rotated Schwefel's Function";
		break;

	case 8:
		a = -100.0;
		b = 100.0;
		name_of_func = "F8_20-nonsep Shifted and Rotated Elliptic Function";
		break;

	case 9:
		a = -5.0;
		b = 5.0;
		name_of_func = "F9_20-nonsep Shifted and Rotated Rastrigin's Function";
		break;

	case 10:
		a = -32.0;
		b = 32.0;
		name_of_func = "F10_20-nonsep Shifted and Rotated Ackley's Function";
		break;

	case 11:
		a = -100.0;
		b = 100.0;
		name_of_func = "F11_20-nonsep Shifted Schwefel's Function";
		break;

	case 12:
		a = -100.0;
		b = 100.0;
		name_of_func = "F12_Shifted Rosenbrock's Function";
		break;

	case 13:
		a = -100.0;
		b = 100.0;
		name_of_func = "F13_Shifted Schwefel's Function with Conforming Overlappins subcomps";
		break;

	case 14:
		a = -100.0;
		b = 100.0;
		name_of_func = "F14_Shifted Schwefel's Function with Conforming Overlappins subcomps";
		break;

	case 15:
		a = -100.0;
		b = 100.0;
		name_of_func = "F15_Shifted Schwefel's Function";
		break;
	}

	if (a == 0.0 || b == 0.0 || ID == 0)
	{
		cout << "Wrong ID!" << endl;
		system("pause");
	}

}
