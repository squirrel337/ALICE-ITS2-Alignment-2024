#include "TGraphErrors.h"
#include "TF1.h"
#include "TRandom.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TMath.h"

//For fitting
#include <TGraph2D.h>
#include <TRandom2.h>
#include <TStyle.h>
#include <TF2.h>
#include <TH1.h>
#include <Math/Functor.h>
#include <TPolyLine3D.h>
#include <Math/Vector3D.h>
#include <Fit/Fitter.h>
#include <cassert>

// define the parametric line equation
void line(double t, const double *p, double &x, double &y, double &z) {
  // a parametric line is define from 6 parameters but 4 are independent
  // x0,y0,z0,z1,y1,z1 which are the coordinates of two points on the line
  // can choose z0 = 0 if line not parallel to x-y plane and z1 = 1;
  x = p[0] + p[1]*t;
  y = p[2] + p[3]*t;
  z = p[4] + p[5]*t;
}


bool first_line = true;

// function Object to be minimized
struct SumDistance2_line {
  // the TGraph is a data member of the object
  TGraph2D *fGraph;

  SumDistance2_line(TGraph2D *g) : fGraph(g) {}

  // calculate distance line-point
  double distance2(double x,double y,double z, const double *p) {
    // distance line point is D= | (xp-x0) cross  ux |
    // where ux is direction of line and x0 is a point in the line (like t = 0)
    ROOT::Math::XYZVector xp(x,y,z);
    ROOT::Math::XYZVector x0(p[0], p[2], p[4] );
    ROOT::Math::XYZVector x1(p[0] + p[1], p[2] + p[3], p[4] + p[5] );
    ROOT::Math::XYZVector u = (x1-x0).Unit();
    double d2 = ((xp-x0).Cross(u)).Mag2();
    return d2;
  }

  // implementation of the function to be minimized
  double operator() (const double *par) {
    assert(fGraph != 0);
    double * x = fGraph->GetX();
    double * y = fGraph->GetY();
    double * z = fGraph->GetZ();
    int npoints = fGraph->GetN();
    double sum = 0;
    for (int i  = 0; i < npoints; ++i) {
       double d = distance2(x[i],y[i],z[i],par);
       sum += d;
    }
    if (first_line) {
       //std::cout << "Total Initial distance square = " << sum << std::endl;
    }
    first_line = false;
    return sum;
  }
};

void line3Dfit(double* input, double* par, double &MSEvalue, int hitentries=3){
  gStyle->SetOptStat(0);
  gStyle->SetOptFit();

  TGraph2D * gr = new TGraph2D();
  // Fill the 2D graph
 
  double temp_par[6];  
  for(int a=0; a<6; a++){
     temp_par[a]=0;
  } 

  int last = hitentries-1;
  double i[hitentries],j[hitentries],k[hitentries]; 
  for(int a=0; a<hitentries;a++){
    i[a]=input[(3*a)+0];
    j[a]=input[(3*a)+1];
    k[a]=input[(3*a)+2];
    gr->SetPoint(a,i[a],j[a],k[a]);
    temp_par[0]+=(double)i[a]/hitentries;
    temp_par[2]+=(double)j[a]/hitentries;
    temp_par[4]+=(double)k[a]/hitentries;    
  }

  double i1, j1, k1;
  double i2, j2, k2;

  if(hitentries==3){
     i1 = 0.5*(i[0]+i[1]); 		j1 = 0.5*(j[0]+j[1]); 			k1 = 0.5*(k[0]+k[1]);  
     i2 = 0.5*(i[last-1]+i[last]); 	j2 = 0.5*(j[last-1]+j[last]);		k2 = 0.5*(k[last-1]+k[last]);  
  } else {
     i1 = 0.5*(i[last]+i[0]); 		j1 = 0.5*(j[last]+j[0]); 		k1 = 0.5*(k[last]+k[0]);  
     i2 = 0.5*(i[0]+i[1]); 		j2 = 0.5*(j[0]+j[1]);			k2 = 0.5*(k[0]+k[1]);    
  }
  temp_par[1] = i2 - i1;		temp_par[3] = j2 - j1;			temp_par[5] = k2 - k1;

  // fit the graph now
  ROOT::Fit::Fitter  fitter;

  // make the functor objet
  SumDistance2_line sdist(gr);
  ROOT::Math::Functor fcn(sdist,6);
  // set the function and the initial parameter values
  double pStart[6] = {temp_par[0],temp_par[1],temp_par[2],temp_par[3],temp_par[4],temp_par[5]};
  fitter.SetFCN(fcn,pStart);
  // set step sizes different than default ones (0.3 times parameter values)
  for (int a = 0; a < 6; ++a) fitter.Config().ParSettings(a).SetStepSize(0.005);
  bool ok = fitter.FitFCN();
  if (!ok) {
    par[0]=0;
    par[1]=0;
    par[2]=0;
    par[3]=0;
    par[4]=0;
    par[5]=0;
    MSEvalue = -1;

    //Error("line3Dfit","Line3D Fit failed");

  } else {
    const ROOT::Fit::FitResult & result = fitter.Result();
    //std::cout << "Total final distance square " << result.MinFcnValue() << std::endl;
    //result.Print(std::cout);
    // get fit parameters
    const double * parFit = result.GetParams();
    par[0]=parFit[0];
    par[1]=parFit[1];
    par[2]=parFit[2];
    par[3]=parFit[3];
    par[4]=parFit[4];
    par[5]=parFit[5];
    //Fit =ok;
    MSEvalue = result.MinFcnValue();
  }

  delete gr;
}



