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

#include "./DetectorConstant.h"

#define MinRes 0
#define CM2M   1.0 // 1e-5 	
#define CFStepSize0 0.3
#define CFStepSize1 1.0e-5	
#define CFStepSize2 1.0e-5
#define CFStepSize3 1.0e-5

#define CTolerance 1.0e-8

double Sigma_MEAS[2][8] = {{45,45,45,55,55,55,55,5},
                           {40,40,40,40,40,40,40,5}}; //um unit
double Sigma_MSC[2][8]  = {{30,30,30,110,110,110,110,20},
                           {25,25,25,75,75,75,75,20}}; //um unit

double GetSigma(double R, int L, double B, int axis){
   //R : cm
   //L : 
   //T : T
   if(L<0 || L>8) return 1;
   double aL   = Sigma_MEAS[axis][L]*1e-4; //um -> cm
   double bL   = Sigma_MSC[axis][L]*1e-4;  //um -> cm
   double Beff = 0.3*B;
   double sigma = std::sqrt( std::pow(aL,2) + std::pow(bL,2)/(std::pow(Beff,2)*std::pow(R*1e-2,2)));

   

   return sigma;
}


// define the parametric line equation
void circle(double t, const double *p, double r, double z0, double &x, double &y, double &z) {

  x = p[0] + r*TMath::Cos(t);
  y = p[1] + r*TMath::Sin(t);
  z = p[2] + z0*t;
}

void circle3Dfit_Z(double* z, double* beta, double* parz, double Radius, bool vertex =false, int hitentries=4);
void VertexCorrection(double* z, double z_target, double* beta, double beta_target, double& parz, double Radius, int tothits);

bool first_circle = true;

// function Object to be minimized
struct SumDistance2_circle {
  // the TGraph is a data member of the object
  TGraph2D *fHits;
  double thetaR;
  SumDistance2_circle(TGraph2D *h, double tR) : fHits(h), thetaR(tR) { }

  // calculate distance line-point
  double distance2a(double x, double y, const double *p, double tR, int charge) {

    double R = std::abs(1/p[0]);
    
    double Xc  = p[0]>0 ? R*std::cos(p[1]+tR + 0.5*TMath::Pi()) : R*std::cos(p[1]+tR - 0.5*TMath::Pi());
    double Yc  = p[0]>0 ? R*std::sin(p[1]+tR + 0.5*TMath::Pi()) : R*std::sin(p[1]+tR - 0.5*TMath::Pi());   
    
    double dx = x - (Xc + p[2]);
    double dy = y - (Yc + p[3]);    
    double dxy = R - std::sqrt(dx*dx + dy*dy);
    
    double d2 = dxy*dxy;
    return d2;
  }

  // implementation of the function to be minimized
  double operator() (const double *par) { //const double -> double
    assert(fHits != 0);    
    double * x    = fHits->GetX();
    double * y    = fHits->GetY(); 
    double * lay  = fHits->GetZ(); 
            
    int nhits    = fHits->GetN();  
    int last        = nhits -1;
    double sum      = 0;

    double charge    = par[0]>0 ? +1 : -1;
    double RecRadius = std::abs(1/par[0]);

    double Sigma_tot[8];// = {150,150,150,500,500,500,500,100}; //um unit
    double sum_Sigma_tot = 0;
    for(int l = 0; l < nhits; l++){
      Sigma_tot[l] = GetSigma(RecRadius, lay[l], DET_MAG,1);
      sum_Sigma_tot += 1/(std::pow(Sigma_tot[l],2));
    } 
    
    for (int l = 0; l < nhits; l++) {
       double d = distance2a(x[l], y[l], par, thetaR, charge)/(std::pow(Sigma_tot[l],2));  
       sum += d;
    }
    sum = sum / sum_Sigma_tot;
    //std::cout << "Total Initial distance square = " << sum << std::endl;
    if (first_circle) {
       //std::cout << "Total Initial distance square = " << sum << std::endl;
    }
    first_circle = false;
    return sum;
  }
};

void Getds1FITdcs1(int layer, double* z, double* beta, double RecRadius, int tothits, double* value){

   //std::cout<<"Getds1FITdcs1 Target Layer"<<layer<<" with "<<tothits<<" hits"<<std::endl;   
   double R_layer = beta[layer];
   double Z_layer = z[layer];
   
   double R[tothits];
   double w[tothits];
   for(int l = 0; l < tothits; l++){
      R[l] = beta[l];
      w[l] = 1/std::pow(GetSigma(RecRadius, l, DET_MAG,0)*1e+4,2);
   } 
   
   double Wsum   = 0;  
   double Rsum   = 0;
   double RWsum  = 0;
   double R2sum  = 0;
   double R2Wsum = 0;
   for(int l = 0; l< tothits; l++){
      //std::cout<<"Getds1FITdcs1 "<<l<<" "<<beta[l]<<" "<<RecRadius<<" -> "<<R[l]<<std::endl;
      Wsum   += w[l];
      Rsum   += R[l];
      RWsum  += R[l]*w[l];
      R2sum  += R[l]*R[l]; 
      R2Wsum += R[l]*R[l]*w[l];      
   }
   
   for(int l = 0; l< tothits; l++){
      double tA = w[l]*R[l]*Wsum - w[l]*RWsum;
      double tB = R2Wsum*Wsum - RWsum*RWsum;
      double tC = w[l]*R2Wsum - w[l]*R[l]*RWsum;
      double tD = R2Wsum*Wsum - RWsum*RWsum;
      value[l] = (tA/tB)*R_layer + tC/tD;
   }
}

void Getds1FITdcs3(int layer, double* z, double* beta, double RecRadius, int tothits, double* value){

   //std::cout<<"Getds1FITdcs3 Target Layer"<<layer<<" with "<<tothits<<" hits"<<std::endl;   
   double R_layer = beta[layer];
   double Z_layer = z[layer];

   double R[tothits];
   double w[tothits];
   for(int l = 0; l < tothits; l++){
      R[l] = beta[l];   
      w[l] = 1/std::pow(GetSigma(RecRadius, l, DET_MAG,0)*1e+4,2);
   } 
   
   double Wsum   = 0;     
   double Rsum   = 0;
   double RWsum  = 0;
   
   double R2sum  = 0;
   double R2Wsum = 0;
      
   double Zsum   = 0;
   double ZWsum  = 0;
   
   double RZsum  = 0;
   double RZWsum = 0;
   for(int l = 0; l< tothits; l++){
      //std::cout<<"Getds1FITdcs3 "<<l<<" "<<beta[l]<<" "<<RecRadius<<" -> "<<R[l]<<std::endl;
      Wsum   += w[l];
      Rsum   += R[l];
      RWsum  += R[l]*w[l];

      R2sum  += R[l]*R[l]; 
      R2Wsum += R[l]*R[l]*w[l]; 
      
      Zsum   += z[l];
      ZWsum  += z[l]*w[l];
      
      RZsum  += R[l]*z[l];
      RZWsum += R[l]*z[l]*w[l];      
   }
   double sign = beta[tothits] > 0 ? +1 : -1 ;
   for(int l = 0; l< tothits; l++){
      double tA = (w[l]*z[l]*Wsum - w[l]*ZWsum)*(R2Wsum*Wsum - RWsum*RWsum) - (RZWsum*Wsum - RWsum*ZWsum)*(2.0*R[l]*w[l]*Wsum - 2.0*w[l]*RWsum);
      double tB = (R2Wsum*Wsum - RWsum*RWsum)*(R2Wsum*Wsum - RWsum*RWsum);
      
      double tC = RZWsum*Wsum - RWsum*ZWsum;
      double tD = R2Wsum*Wsum - RWsum*RWsum;
      double tE = (layer==l) ? 1 : 0;
      
      double tF = (2.0*w[l]*R[l]*ZWsum - (w[l]*RZWsum + w[l]*z[l]*RWsum))*(R2Wsum*Wsum - RWsum*RWsum) - (R2Wsum*ZWsum - RWsum*RZWsum)*(2.0*R[l]*w[l]*Wsum - 2.0*w[l]*RWsum);    
      double tG = (R2Wsum*Wsum - RWsum*RWsum)*(R2Wsum*Wsum - RWsum*RWsum);
      value[l] = (sign/RecRadius)*((tA/tB)*R_layer + (tC/tD)*tE + (tF/tG));
      //std::cout<<"value["<<l<<"] "<<tA<<" "<<tB<<" "<<beta[l]<<" "<<RecRadius<<" "<<tC<<" "<<tD<<" "<<tE<<" "<<tF<<" "<<tG<<std::endl;
   }
}

void circle3Dfit(double* input, double* par, double &MSEvalue, std::vector<bool> hitUpdate, int step = 0){

  int hitentries = 0;
  for(int a=0; a<hitUpdate.size(); a++){
     if(hitUpdate[a]==true) hitentries++;
  }
  gStyle->SetOptStat(0);
  gStyle->SetOptFit();

  TGraph2D * hr = new TGraph2D();  
  // Fill the 2D graph
 
  double frphiX = 0;
  double frphiY = 0;
  int    nfrdet = 0;
  for(int a=0; a<hitUpdate.size();a++){
    if(hitUpdate[a]==false) continue;
    if(a!=2) continue;
    
    frphiX += input[(3*a)+0] - input[(3*(hitUpdate.size()-1))+0];
    frphiY += input[(3*a)+1] - input[(3*(hitUpdate.size()-1))+1];
    nfrdet++;
  }
  frphiX /=nfrdet;
  frphiY /=nfrdet;
  
  //std::cout<<" frphiXY "<<frphiX<<" "<<frphiY<<" "<<nfrdet<<std::endl;
  
  double FitFrame = std::atan2(frphiY,frphiX) - TMath::Pi()/4.;
  TMatrixD RotF(2,2);
  RotF[0] = { TMath::Cos(FitFrame),	TMath::Sin(FitFrame)};
  RotF[1] = {-TMath::Sin(FitFrame),	TMath::Cos(FitFrame)};    

  TMatrixD RotFInv(2,2);
  RotFInv[0] = { TMath::Cos(FitFrame), -TMath::Sin(FitFrame)};
  RotFInv[1] = { TMath::Sin(FitFrame),	TMath::Cos(FitFrame)};  
 
 
  
  double i[hitentries],j[hitentries],k[hitentries], l[hitentries], d[hitentries], s[hitentries], theta[hitentries]; 
  double irot[hitentries],jrot[hitentries];
  TVector3 vecX[hitentries];
  TVector3 vecXmid[hitentries-1];
  double InvSlope[hitentries-1];
  TMatrixD MatC;  
  int fa=0;  
    int index[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  for(int a=0; a<hitUpdate.size();a++){
    if(hitUpdate[a]==false) continue;
    i[fa]=input[(3*a)+0] - input[(3*(hitUpdate.size()-1))+0];
    j[fa]=input[(3*a)+1] - input[(3*(hitUpdate.size()-1))+1];
    k[fa]=input[(3*a)+2];
    vecX[fa].SetXYZ(i[fa], j[fa], 0);
    l[fa]=0;
    s[fa]=0;
    d[fa]=0; 
    theta[fa]=0;
    
    TMatrixD gloX[2];
    gloX[0].ResizeTo(1,2);
    gloX[0][0] = { i[fa], j[fa]};
    gloX[0].T();
    gloX[1].ResizeTo(2,1);
    gloX[1] = RotF * gloX[0];
    gloX[1].T();  
    
    irot[fa] = gloX[1][0][0];
    jrot[fa] = gloX[1][0][1];
    hr->SetPoint(fa, irot[fa], jrot[fa], a);    
    index[a]=fa;    
    //std::cout<<"[CircleFIT N DEBUG20221104] layer "<<a<<" "<<i[fa]<<" "<<j[fa]<<" "<<k[a]<<std::endl;
    //std::cout<<"[CircleFIT R DEBUG20221104] layer "<<a<<" "<<irot[fa]<<" "<<jrot[fa]<<std::endl;    
    fa++;    
  }

  int cntR[] = {0, 0};
  std::vector<TVector3> initR;
  //std::vector<double> initR;  

  //012 + (2)34(5) + 56
  int hit1[] = {0, 1, 2};
  int hit2[] = {3, 4, 5, 2}; bool hit_mid = false;
  int hit3[] = {6, 5};
  for(int i1 = 0; i1 < 3; i1++){
    // i1 -> hit1[i1]
    for(int i2 = 0; i2 < 4; i2++){
      // i2 -> hit2[i2]
      if(hit_mid==true && i2>= 2) continue;
      for(int i3 = 0; i3 < 2; i3++){
        // i3 -> hit3[i3]
        if(hit1[i1]==hit2[i2] || hit2[i2]==hit3[i3]) continue;
        //if(hit_mid==true) continue;
        if(hitUpdate[hit1[i1]]==false) continue;
        if(hitUpdate[hit2[i2]]==false) continue;
        if(hitUpdate[hit3[i3]]==false) continue;
        double hitX[3] = {i[index[hit1[i1]]], i[index[hit2[i2]]], i[index[hit3[i3]]]};
        double hitY[3] = {j[index[hit1[i1]]], j[index[hit2[i2]]], j[index[hit3[i3]]]};

        double d12 = -(hitX[1] - hitX[0])/(hitY[1] - hitY[0]);
        double d23 = -(hitX[2] - hitX[1])/(hitY[2] - hitY[1]);

        double x12 = 0.5*(hitX[0] + hitX[1]);
        double x23 = 0.5*(hitX[1] + hitX[2]);
        double y12 = 0.5*(hitY[0] + hitY[1]);
        double y23 = 0.5*(hitY[1] + hitY[2]);

        double CenterX = ((-d23*x23 + d12*x12) + (y23 - y12))/(-d23 + d12);
        double CenterY = d12*(CenterX - x12) + y12;

        double temp_R = std::sqrt(std::pow(CenterX - hitX[0],2) + std::pow(CenterY - j[hit1[i1]],2));
        initR.push_back(TVector3(CenterX,CenterY,temp_R));
        if(i2<2) {
           hit_mid=true; // mid hit is successfully used. Do not find inner or outer hits for initial radius searching
        }
        cntR[0]++;
      }  
    }
  }
  if(initR.size()==0) {
    initR.push_back(TVector3(0,0,10000));
    cntR[0]++;
  }

  double mean_X[] = {0, 0};
  double mean_Y[] = {0, 0};
  double mean_R[] = {0, 0};
  for(int i = 0; i < cntR[0]; i++) {
    mean_X[0] += initR[i].X()/(double)cntR[0];
    mean_Y[0] += initR[i].Y()/(double)cntR[0];
    mean_R[0] += initR[i](2)/(double)cntR[0];
  }

  for(int i = 0; i < cntR[0]; i++) {
    if(std::abs(mean_R[0] - initR[i](2)) < mean_R[0]) {
      mean_X[1] += initR[i].X();
      mean_Y[1] += initR[i].Y();
      mean_R[1] += initR[i](2);
      cntR[1]++;
    }
  }
  mean_R[1] /= cntR[1];

  //std::cout<<"(fit)R mean       : "<<mean_R[1]<<std::endl;  
  mean_R[1] *= std::pow(sqrt(10),step);
  if(mean_R[1]<1.0e+1) mean_R[1] = 1.0e+1;
  if(mean_R[1]>1.0e+6) mean_R[1] = 1.0e+6;
  //std::cout<<"(fit)R mean("<<step<<") : "<<mean_R[1]<<std::endl;  

  double thetaR = std::atan2( jrot[0], irot[0]);
  
  double temp_parA[4];
  temp_parA[0] = + 1/mean_R[1];
  temp_parA[1] = 0;  

  double temp_parB[4];
  temp_parB[0] = - 1/mean_R[1];
  temp_parB[1] = 0;  

  for(int iTol = 0; iTol < 4; iTol++){

    ROOT::Math::MinimizerOptions minOpt;
    //minOpt.SetMaxIterations(1);
    //minOpt.SetPrecision(CTolerance);
    if(step==0) minOpt.SetTolerance(CTolerance);
    else minOpt.SetTolerance(std::pow(10,iTol)*CTolerance);
    //fitConfig.SetMinimizerOptions(minOpt);

    // fit the graph now
    ROOT::Fit::Fitter  fitterA;
    fitterA.Config().SetMinimizerOptions(minOpt);
    ROOT::Fit::Fitter  fitterB;  
    fitterB.Config().SetMinimizerOptions(minOpt);  

    // make the functor objet
    SumDistance2_circle sdist(hr, thetaR);
    ROOT::Math::Functor fcn(sdist,4);

    // set the function and the initial parameter values
    double pStartA[4] = {temp_parA[0],temp_parA[1], 0, 0};
    fitterA.SetFCN(fcn,pStartA);
    // set step sizes different than default ones (0.3 times parameter values)
    //for (int a = 0; a < 2; ++a) 
    fitterA.Config().ParSettings(0).SetStepSize(CFStepSize0);
    fitterA.Config().ParSettings(1).SetStepSize(CFStepSize1);
    fitterA.Config().ParSettings(2).SetStepSize(CFStepSize2);
    fitterA.Config().ParSettings(3).SetStepSize(CFStepSize3);  

    fitterA.Config().ParSettings(0).SetLimits(+1.0e-10, +1.0e-1); // + side
    bool okA = fitterA.FitFCN();
  
    double pStartB[4] = {temp_parB[0],temp_parB[1], 0, 0};  
    fitterB.SetFCN(fcn,pStartB);
    // set step sizes different than default ones (0.3 times parameter values)
    //for (int b = 0; b < 2; ++b) 
    fitterB.Config().ParSettings(0).SetStepSize(CFStepSize0);
    fitterB.Config().ParSettings(1).SetStepSize(CFStepSize1);
    fitterB.Config().ParSettings(2).SetStepSize(CFStepSize2);
    fitterB.Config().ParSettings(3).SetStepSize(CFStepSize3);  

    fitterB.Config().ParSettings(0).SetLimits(-1.0e-1, -1.0e-10); // - side
    bool okB = fitterB.FitFCN();
  
    if (!okA) {
      if(!okB) {
        const ROOT::Fit::FitResult & resultA = fitterA.Result();  
        const double * parFitA = resultA.GetParams();        
        double MSEvalueA = resultA.MinFcnValue();        
        int    ncallsA   = resultA.NCalls();
        const ROOT::Fit::FitResult & resultB = fitterB.Result();
        const double * parFitB = resultB.GetParams();
        double MSEvalueB = resultB.MinFcnValue(); 
        int    ncallsB   = resultB.NCalls();     
      
        //std::cout<<"TAG1 FitA : "<<okA<<" "<<MSEvalueA<<" "<<ncallsA<<" ("<<1/parFitA[0]<<
        //              ") FitB : "<<okB<<" "<<MSEvalueB<<" "<<ncallsB<<" ("<<1/parFitB[0]<<")"<<std::endl;

        TMatrixD vxyA[2];
        vxyA[0].ResizeTo(1,2);
        vxyA[0][0] = { parFitA[2], parFitA[3]};
        vxyA[0].T();
        vxyA[1].ResizeTo(2,1);
        vxyA[1] = RotFInv * vxyA[0];
        vxyA[1].T();   
        
        TMatrixD vxyB[2];
        vxyB[0].ResizeTo(1,2);
        vxyB[0][0] = { parFitB[2], parFitB[3]};
        vxyB[0].T();
        vxyB[1].ResizeTo(2,1);
        vxyB[1] = RotFInv * vxyB[0];      
        vxyB[1].T();

        //std::cout<<" DEBUG20221104 A "<<parFitA[2]<<" "<<parFitA[3]<<" "<<vxyA[1][0][0]<<" "<<vxyA[1][0][1]<<std::endl;
        //std::cout<<" DEBUG20221104 B "<<parFitB[2]<<" "<<parFitB[3]<<" "<<vxyB[1][0][0]<<" "<<vxyB[1][0][1]<<std::endl;
        
        if(MSEvalueA<MSEvalueB){
          MSEvalue = MSEvalueA;
          par[0]=parFitA[0];
          par[1]=parFitA[1];  
          par[2]=vxyA[1][0][0];
          par[3]=vxyA[1][0][1];
          par[4]= thetaR + FitFrame;       
          par[5]= iTol*10 + okA;
        } else {
          MSEvalue = MSEvalueB;
          par[0]=parFitB[0];
          par[1]=parFitB[1];  
          par[2]=vxyB[1][0][0];
          par[3]=vxyB[1][0][1]; 
          par[4]= thetaR + FitFrame;   
          par[5]= iTol*10 + okB;    
        }     
  
      } else {
        const ROOT::Fit::FitResult & resultB = fitterB.Result();
        const double * parFitB = resultB.GetParams();
        MSEvalue = resultB.MinFcnValue();  
        int    ncallsB   = resultB.NCalls();     
  
        //std::cout<<"TAG2 FitA : "<<okA<<" "<<-1<<" "<<-1<<
        //               " FitB : "<<okB<<" "<<MSEvalue<<" "<<ncallsB<<" ("<<1/parFitB[0]<<")"<<std::endl;
    
        TMatrixD vxyB[2];
        vxyB[0].ResizeTo(1,2);
        vxyB[0][0] = { parFitB[2], parFitB[3]};
        vxyB[0].T();
        vxyB[1].ResizeTo(2,1);
        vxyB[1] = RotFInv * vxyB[0];     
        vxyB[1].T();
        
        par[0]=parFitB[0];
        par[1]=parFitB[1];  
        par[2]=vxyB[1][0][0];
        par[3]=vxyB[1][0][1]; 
        par[4]= thetaR + FitFrame;   
        par[5]= iTol*10 + okB;      
        break;    
      }

    } else {
      if(!okB){
        const ROOT::Fit::FitResult & resultA = fitterA.Result();
        const double * parFitA = resultA.GetParams();
        MSEvalue = resultA.MinFcnValue(); 
        int    ncallsA   = resultA.NCalls();
  
        //std::cout<<"TAG3 FitA : "<<okA<<" "<<MSEvalue<<" "<<ncallsA<<" ("<<1/parFitA[0]<<
        //              ") FitB : "<<okB<<" "<<-1<<" "<<-1<<std::endl;

        TMatrixD vxyA[2];
        vxyA[0].ResizeTo(1,2);
        vxyA[0][0] = { parFitA[2], parFitA[3]};
        vxyA[0].T();
        vxyA[1].ResizeTo(2,1);
        vxyA[1] = RotFInv * vxyA[0];
        vxyA[1].T();
        
        par[0]=parFitA[0];
        par[1]=parFitA[1];  
        par[2]=vxyA[1][0][0];
        par[3]=vxyA[1][0][1]; 
        par[4]= thetaR + FitFrame;       
        par[5]= iTol*10 + okA;
        break;
      } else { 
        const ROOT::Fit::FitResult & resultA = fitterA.Result();  
        const double * parFitA = resultA.GetParams();        
        double MSEvalueA = resultA.MinFcnValue();    
        int    ncallsA   = resultA.NCalls();          
        const ROOT::Fit::FitResult & resultB = fitterB.Result();
        const double * parFitB = resultB.GetParams();
        double MSEvalueB = resultB.MinFcnValue();  
        int    ncallsB   = resultB.NCalls();    
    
        //std::cout<<"TAG4 FitA : "<<okA<<" "<<MSEvalueA<<" "<<ncallsA<<" ("<<1/parFitA[0]<<
        //              ") FitB : "<<okB<<" "<<MSEvalueB<<" "<<ncallsB<<" ("<<1/parFitB[0]<<")"<<std::endl;

        TMatrixD vxyA[2];
        vxyA[0].ResizeTo(1,2);
        vxyA[0][0] = { parFitA[2], parFitA[3]};
        vxyA[0].T();
        vxyA[1].ResizeTo(2,1);
        vxyA[1] = RotFInv * vxyA[0];
        vxyA[1].T();
        
        TMatrixD vxyB[2];
        vxyB[0].ResizeTo(1,2);
        vxyB[0][0] = { parFitB[2], parFitB[3]};
        vxyB[0].T();
        vxyB[1].ResizeTo(2,1);
        vxyB[1] = RotFInv * vxyB[0]; 
        vxyB[1].T();
        
        if(MSEvalueA<MSEvalueB){
          MSEvalue = MSEvalueA;
          par[0]=parFitA[0];
          par[1]=parFitA[1];  
          par[2]=vxyA[1][0][0];
          par[3]=vxyA[1][0][1];  
          par[4]= thetaR + FitFrame;     
          par[5]= iTol*10 + okA;  
        } else {
          MSEvalue = MSEvalueB;
          par[0]=parFitB[0];
          par[1]=parFitB[1];  
          par[2]=vxyB[1][0][0];
          par[3]=vxyB[1][0][1];  
          par[4]= thetaR + FitFrame;       
          par[5]= iTol*10 + okB;
        }
        break;  
      }
    }
  }  
  delete hr;
}

void circle3Dfit_Z(double* zIn, double* betaIn, double* parz, double Radius, bool vertex, std::vector<bool> hitUpdate){

  int hitentries = 0;
  for(int a=0; a<hitUpdate.size(); a++){
     if(hitUpdate[a]==true) hitentries++;
  }
  if(vertex==true) hitentries++;  
  
  double z[hitentries], beta[hitentries]; 
  int index[hitentries];

  int fa=0;
  for(int a=0; a<hitUpdate.size(); a++){
    if(hitUpdate[a]==true){
      z[fa]=zIn[a];
      beta[fa]=betaIn[a];
      index[fa]=a;
      fa++;    
    }
  }
  if(vertex==true){
    z[fa]=zIn[hitUpdate.size()];
    beta[fa]=betaIn[hitUpdate.size()];
    index[fa]=hitUpdate.size();
    fa++; 
  }

  //std::cout<<"circle3Dfit_Z with Nhits : "<<hitentries<<std::endl;
  int tothits = hitentries;  
  
  int last = tothits-1;
  TMatrixD MatA(tothits,2);
  TMatrixD MatAT(2,tothits);
  TMatrixD MatZ(tothits,1);   
  double beta2sum = 0;
  double betasum = 0;
  double nhits = 0;  

  double Sigma_tot[tothits];// = {150,150,150,500,500,500,500,100}; //um unit
  for(int l = 0; l < tothits; l++){
     Sigma_tot[l] = GetSigma(Radius, index[l], DET_MAG,0);
  } 
  if(vertex==true){
    MatAT[0][0] = beta[last]/std::pow(Sigma_tot[last]*1e-4/Radius,2);
    MatAT[1][0] = 1/std::pow(Sigma_tot[last]*1e-4/Radius,2);
    
    MatZ[0]  = { z[last]};
  
    beta2sum += beta[last]*beta[last]/std::pow(Sigma_tot[last]*1e-4/Radius,2);
    betasum  += beta[last]/std::pow(Sigma_tot[last]*1e-4/Radius,2);
    nhits += 1/std::pow(Sigma_tot[last]*1e-4/Radius,2); 
  
    for(int i = 1; i < tothits; i++){

      MatAT[0][i]  = beta[i-1]/std::pow(Sigma_tot[i-1]*1e-4/Radius,2);
      MatAT[1][i]  = 1/std::pow(Sigma_tot[i-1]*1e-4/Radius,2); 

      MatZ[i]  = { z[i-1]};  

      beta2sum += beta[i-1]*beta[i-1]/std::pow(Sigma_tot[i-1]*1e-4/Radius,2);
      betasum  += beta[i-1]/std::pow(Sigma_tot[i-1]*1e-4/Radius,2);  
      nhits += 1/std::pow(Sigma_tot[i-1]*1e-4/Radius,2);            
    }
  } else {
    for(int i = 0; i < tothits; i++){
      MatAT[0][i]  = beta[i]/std::pow(Sigma_tot[i]*1e-4/Radius,2);
      MatAT[1][i]  = 1/std::pow(Sigma_tot[i]*1e-4/Radius,2); 

      MatZ[i]  = { z[i]};  

      beta2sum += beta[i]*beta[i]/std::pow(Sigma_tot[i]*1e-4/Radius,2);
      betasum  += beta[i]/std::pow(Sigma_tot[i]*1e-4/Radius,2);  
      nhits += 1/std::pow(Sigma_tot[i]*1e-4/Radius,2);            
    }
  } 
  TMatrixD MatATAInv(2,2);   
  double DetATA = nhits*beta2sum - betasum*betasum;
  MatATAInv[0] = {nhits/DetATA, -betasum/DetATA};
  MatATAInv[1] = {-betasum/DetATA, beta2sum/DetATA};
     
  TMatrixD MatP(2,1);
    
  MatP = MatATAInv * MatAT * MatZ;
  
  parz[0] = MatP[0][0];
  parz[1] = MatP[1][0]; 

}

void VertexCorrection(double* z, double z_target, double* beta, double beta_target, double& parz, double Radius, int tothits){ 

   int last = tothits-1;
   TMatrixD MatA(tothits,1);
   TMatrixD MatAT(1,tothits);
   TMatrixD MatZ(tothits,1);   
   double beta2sum = 0;

   double Sigma_tot[tothits];
   for(int l = 0; l < tothits; l++){
      Sigma_tot[l] = GetSigma(Radius, l, DET_MAG,0);
   } 
   
   for(int i = 0; i < tothits; i++){
      //std::cout<<"VertexCorrection : beta["<<i<<"] beta_target "<<beta[i]<<" "<<beta_target<<std::endl;
      //std::cout<<"VertexCorrection :    z["<<i<<"]    z_target "<<z[i]   <<" "<<z_target   <<std::endl;      
      MatAT[0][i]  = (beta[i] - beta_target)/std::pow(Sigma_tot[i]*1e-4/Radius,2);
      MatZ[i]  = { z[i] - z_target};  
      beta2sum += (beta[i] - beta_target)*(beta[i] - beta_target)/std::pow(Sigma_tot[i]*1e-4/Radius,2);
   }

  TMatrixD MatATAInv(1,1);   
   double DetATA = beta2sum;
   MatATAInv[0] = {1/DetATA};
  
   TMatrixD MatP(1,1);   
   MatP = MatATAInv*MatAT*MatZ;
   parz = MatP[0][0];
}
