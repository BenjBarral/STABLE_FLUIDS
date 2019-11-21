//
//  main.cpp
//
//
//  Created by Benjamin Barral.
//

/// C++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <ctime>
/// OpenCv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/// Eigen
#include <eigen3/Eigen/Dense>
#include<eigen3/Eigen/IterativeLinearSolvers>

using namespace cv;
using namespace std;
using namespace Eigen;

/// IMAGE PROCESSING PARAMETERS AND VARIABLES
// Image parameters : display resolution, processing resolution
const int disp_width = 500, disp_height = 500, proc_width = 75,  proc_height = 75;
int num_particles = proc_width * proc_height;

float frameRate = 60.;
float dt = 1. / frameRate;

int Lx = 600, Ly = 600;
float dx = float(Lx) / float(proc_height), dy = float(Ly) / float(proc_height);
float conv=(float) ((1440./0.3)*(1440./0.3));


float visc = 120. * pow(10,-6) * conv;
double Cx = -visc * dt / (dx*dx);
double Cy = -visc * dt / (dy*dy);
int boundary_condition = 1;

// Matrices
VectorXd V_x_0(num_particles), V_x_1(num_particles), V_y_0(num_particles), V_y_1(num_particles);
VectorXd F_x(num_particles), F_y(num_particles);

// Image
Mat fluid_grid = Mat::zeros(proc_height,proc_width,CV_8UC3), prev_grid = Mat::zeros(proc_height,proc_width,CV_8UC3);
float g = 130, b = 200;

void AddForce(){
    V_x_0 = V_x_0 + dt * F_x;
    V_y_0 = V_y_0 + dt * F_y;
}

Vector2d Trace(int i, int j){
    double x = (j + 0.5) * dx;
    double y = (i + 0.5) * dy;
    double ux = V_x_0[i * proc_height + j]; // VX[0][i*n+j];
    double uy = V_y_0[i * proc_height + j]; // VY[0][i*n+j];
    x = x - dt * ux;
    y = y - dt * uy;
    return Vector2d(x,y);
}

void LinearInterpolate(double x, double y, int i0, int j0, bool setTexture){
    //int j=(int)((x-dx/2)/dx);
    //double t1=(x-(j+0.5)*dx)/dx;
    int j = (x>0) ? (int)((x -dx/2.) / dx) : (int)((x - dx/2.) / dx) - 1.;
    double t1 = (x- (double(j) + 0.5) * dx) / dx;
    if (j >= proc_height){
        j = j % proc_height;
        if (boundary_condition == 1) return;
    }
    if (j < 0){
        if (boundary_condition == 1){
            if (j == -1 && t1 > 0.5) t1 = 1.;
            else return;
        }
        int q = j / proc_height;
        j = (j + (-q+1) * proc_height) % proc_height;
    }
    t1 = (j == proc_height-1 && boundary_condition==1) ? 0. : t1;
    //int i=(int)((y-dy/2)/dy);
    //double t=(y-(i+0.5)*dy)/dy;
    int i = (y > 0) ? (int)((y - dy/2.) / dy) : (int)((y - dy/2.) / dy) - 1.;
    double t = (y - (double(i) + 0.5) * dy) / dy;
    if (i >= proc_height){
        i = i % proc_height;
        if (boundary_condition == 1) return;
    }
    if (i < 0){
        if (boundary_condition == 1){
            if (i==-1 && t>0.5) t=1.;
            else return;
        }
        int q = i / proc_height;
        i = (i + (-q + 1) * proc_height) % proc_height ;
    }
    t=(i == proc_height-1 && boundary_condition==1)? 0. : t;
    int i2 = (i==proc_height-1)? 0:i+1;
    int j2 = (j==proc_height-1)? 0:j+1;
    double s1a = V_x_0[i*proc_height+j], s1b = V_x_0[i*proc_height+j2]; //  VX[0][i*n+j],s1b= VX[0][i*n+j2];
    double s2a = V_x_0[i2*proc_height+j], s2b = V_x_0[i2*proc_height+j2]; // VX[0][i2*n+j],s2b=VX[0][i2*n+j2];
    double s1 = t1*s1b+(1.-t1)*s1a;
    double s2 = t1*s2b+(1.-t1)*s2a;
    double uxBis = t*s2 + (1.-t)*s1;
    V_x_1[i0*proc_height+j0]= uxBis; //    VX[1][i0*n+j0]= uxBis;
    
    
    s1a = V_y_0[i * proc_width + j]; // VY[0][i*n+j];
    s1b = V_y_0[i * proc_width + j2]; // VY[0][i*n+j2];
    s2a = V_y_0[i2 * proc_width + j]; // VY[0][i2*n+j];
    s2b = V_y_0[i2 * proc_width + j2]; // VY[0][i2*n+j2];
    s1 = t1*s1b + (1.-t1)*s1a;
    s2 = t1*s2b + (1.-t1)*s2a;
    double uyBis = t*s2 + (1.-t)*s1;
    V_y_1[i0*proc_height+j0]= uyBis; //   VY[1][i0*n+j0]= uyBis;
    
    
    if (setTexture){
        Vec3f s1a = prev_grid.at<Vec3b>(i,j);
        Vec3f s1b = prev_grid.at<Vec3b>(i,j2);
        Vec3f s2a = prev_grid.at<Vec3b>(i2,j);
        Vec3f s2b = prev_grid.at<Vec3b>(i2,j2);
        
        Vec3f s1 = t1 * s1b + (1.-t1) * s1a;
        Vec3f s2 = t1 * s2b + (1.-t1) * s2a;
        Vec3f preCol = (t*s2 + (1.-t)*s1);
        
        Vec3b color;
        for (int k = 0; k < 3; k++){
            int col = round(preCol.val[k]);
            color.val[k] = col;
        }
        fluid_grid.at<Vec3b>(i0,j0) = color;
    }
}

void Transport(){
    Vector2d p;
    fluid_grid.copyTo(prev_grid); //int[][][] textures0=displayGrid.textures.clone();
    for (int i = 0; i < proc_height; i++){
        for (int j = 0; j < proc_height; j++){
            p=Trace(i,j);
            LinearInterpolate(p[0],p[1], i,j, true);
        }
    }
}

void Swap(){
    VectorXd temp = V_x_0;
    V_x_0 = V_x_1;
    V_x_1 = temp;
    temp = V_y_0;
    V_y_0 = V_y_1;
    V_y_1 = temp;
}

VectorXd DX(VectorXd U, int n, double dx, bool tilde) {
    int N = n*n;
    VectorXd U2(num_particles); // = new double[N];
    for(int i = 0; i < num_particles;i++) {
        int l = i%proc_width;
        int k = i/proc_width;
        if (l != 0 && (!tilde || l!=proc_width-1)) U2[i] -= U[i-1];   //AJOUTE
        if (l!=n-1 && (!tilde || l!=0)) U2[i] += U[i+1];
        
        U2[i] = U2[i] / (2.*dx);
    }
    return U2;
}

VectorXd DY(VectorXd U, int n, double dy, bool tilde) {
    int N = n*n;
    VectorXd U2(num_particles); // = new double[N];
    for(int i = 0; i < num_particles;i++) {
        int l = i%proc_width;
        int k = i/proc_width;
        if (k!=0 && (!tilde || k!=proc_width-1) ) U2[i]-=U[i-proc_width];
        if (k!=n-1 && (!tilde || k!=0)) U2[i]+=U[i+proc_width];
        
        U2[i] = U2[i] / (2.*dy);
    }
    return U2;
}

void Project(SparseMatrix<double> laplacian, const BiCGSTAB< SparseMatrix<double> >& solver_laplacian){
    VectorXd q = DX(V_x_0, proc_height, dx, false) + DY(V_y_0, proc_width, dy, false);
    
    //double[] q = DX(VX[0], proc_height, dx, false) + DY(VY[0],proc,dy, false);
   VectorXd Q = solver_laplacian.solve(q);
    V_x_0 = V_x_0 - DX(Q, proc_height, dx, true);
    V_y_0 = V_y_0 - DY(Q, proc_height, dx, true);
    
    //VX[0]=minus(VX[0],ComputeMatricesMTJ_BC2.DX(Q,n,dx,true));
    //VY[0]=minus(VY[0],ComputeMatricesMTJ_BC2.DY(Q,n,dy,true));
}

void Diffuse(const BiCGSTAB< SparseMatrix<double> >& solver_diff_X, const BiCGSTAB< SparseMatrix<double> >& solver_diff_Y){
    V_x_0 = solver_diff_X.solve(V_x_0);// VX[0] = solver_diff_X.solve(VX[0]);
    V_y_0 = solver_diff_Y.solve(V_y_0); //VY[0]=solverDiffY.solve(VY[0]);
}

void Routine(const BiCGSTAB< SparseMatrix<double> >& solver_diff_X, const BiCGSTAB< SparseMatrix<double> >& solver_diff_Y,
             SparseMatrix<double> laplacian, const BiCGSTAB< SparseMatrix<double> >& solver_laplacian){   // the routine of the solver algorith
    AddForce();
    Transport();
    Swap();
    Diffuse(solver_diff_X, solver_diff_Y);
    Project(laplacian, solver_laplacian);
}

int main()
{
    for (int i = 0; i < proc_height; i++){
        int r = int(5. * double(i) * (50. / double(proc_height)))%255 ;
        for (int j = 0; j < proc_width; j++) {
            fluid_grid.at<Vec3b>(i,j) = Vec3b(r,g,b);
        }
    }
    
    for (int i = 0; i < proc_height/2; i++){
        for (int j = 0; j < proc_height; j++){
            int K = i*proc_width + j;
            F_y[K] = 0.1;
        }
    }
    
    /// VIDEO : Create windows
    namedWindow( "Fluid simulator", CV_WINDOW_AUTOSIZE );
    
    int key = waitKey(3);
    
    
    SparseMatrix<double> diffusion_X(num_particles,num_particles), diffusion_Y(num_particles,num_particles),
    laplacian(num_particles,num_particles);
    //Compute DiffusionX
    for(int i = 0; i < num_particles; i++) {
        int k = i / proc_height;
        int l = i %proc_height;
        if(k !=0 && k != proc_height-1) diffusion_X.coeffRef(i, i) = 1. - 2. * Cx - 2. * Cy;  //DX.set(i,i,1-2*Cx -2*Cy);
        else diffusion_X.coeffRef(i, i) = 1. - 2. * Cx - Cy;  // DX.set(i,i,1-2*Cx-Cy);
        
        if(l != 0) diffusion_X.coeffRef(i, i-1) = Cx; // DX.set(i,i-1,Cx);
        //else diffusion_X.coeffRef(i, i+proc_width-1) = 0.; // DX.set(i,i+n-1,0.);
        if(l != proc_width-1) diffusion_X.coeffRef(i, i+1) = Cx; //DX.set(i,i+1,Cx);
        //else diffusion_X.coeffRef(i, i-n+1) = 0.; DX.set(i,i-n+1,0.);
        
        if(k != 0) diffusion_X.coeffRef(i, i-proc_height) = Cy; // DX.set(i,i-n,Cy);
        //else DX.set(i,N-n+l,0.);
        if(k != proc_height-1) diffusion_X.coeffRef(i, i+proc_height) = Cy;// DX.set(i, i + proc_height,Cy);
        //else DX.set(i,l,0.);
    }
    
    for(int i = 0; i < num_particles; i++) {
        int k = i / proc_height;
        int l = i % proc_height;
        if(l != 0 && l != proc_height-1) diffusion_Y.coeffRef(i, i) = 1. - 2. * Cx - 2. * Cy; // DY.set(i,i,1 -2*Cx -2*Cy);
        else diffusion_Y.coeffRef(i, i) = 1. - Cx - 2. * Cy; // DY.set(i,i,1 -Cx -2*Cy);
        
        if(l != 0) diffusion_Y.coeffRef(i, i-1) = Cx; // DY.set(i,i-1,Cx);
        //else DY.set(i,i+n-1,0.);
        if(l != proc_height-1) diffusion_Y.coeffRef(i,i+1) = Cx; // DY.set(i,i+1,Cx);
        //else DY.set(i,i-n+1,0.);
        
        if(k != 0) diffusion_Y.coeffRef(i, i-proc_height) = Cy; // DY.set(i,i-n,Cy);
        //else DY.set(i,N-n+l,0.);
        if(k != proc_height-1) diffusion_Y.coeffRef(i, i+proc_height) = Cy; // DY.set(i,i+n,Cy);
        //else DY.set(i,l,0.);
    }
    
    Cx = 1. / (dx*dx);
    Cy = 1. / (dy*dy);
    //Compute L
    for(int i = 0; i < num_particles;i++) {
        int k = i / proc_height;
        int l = i % proc_height;
        laplacian.coeffRef(i,i) = -2. * Cx - 2. * Cy; //L.set(i,i,-2*Cx -2*Cy);
        
        if(l != 0 && l != proc_height-1) {
            laplacian.coeffRef(i, i-1) = Cx; // L.set(i,i-1,Cx);
            laplacian.coeffRef(i, i+1) = Cx; // L.set(i,i+1,Cx);
        }
        else {
            if(l == 0) laplacian.coeffRef(i, i+1) = 2. * Cx; // L.set(i,i+1,2*Cx);
            //else L.set(i, i-n+1, 0.);
            if(l == proc_height-1) laplacian.coeffRef(i, i-1) = 2. * Cx; // L.set(i,i-1,2*Cx);
            //else L.set(i, i+n-1, 0.);
        }
        
        if(k != 0 && k != proc_height-1) {
            laplacian.coeffRef(i, i - proc_height) = Cy; // L.set(i,i-n,Cy);
            laplacian.coeffRef(i, i + proc_height) = Cy; // L.set(i,i+n,Cy);
        }
        else {
            if(k != proc_height-1) laplacian.coeffRef(i, i+proc_height) = Cy; // L.set(i,i+n,Cy);
            //else  L.set(i,l,0.);
            if(k != 0) laplacian.coeffRef(i, i-proc_height) = Cy; // L.set(i,i-n,Cy);
            //else L.set(i,N-n+l,0.);
        }
    }
    
    cout << "Matrices" << endl;
    
    BiCGSTAB< SparseMatrix<double> > solver_diff_X;
    solver_diff_X.compute(diffusion_X);
    cout << "Solver Diff X created" << endl;
    BiCGSTAB< SparseMatrix<double> > solver_diff_Y;
    solver_diff_Y.compute(diffusion_Y);
    cout << "Solver Diff X created" << endl;
    BiCGSTAB< SparseMatrix<double> > solver_laplacian;
    solver_laplacian.compute(laplacian);
    cout << "Solver Laplacian created" << endl;
    
    
    while (key != 32) {
        clock_t begin = clock();
        
        Routine(solver_diff_X, solver_diff_Y, laplacian, solver_laplacian);
        imshow( "Delaunay", fluid_grid);
        
        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        cout << "Time : " << elapsed_secs << endl;
        /// Exit if space bar is pressed
        key = waitKey(3);
    }
}
