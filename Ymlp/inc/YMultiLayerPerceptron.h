// @(#)root/mlp:$Id$
// Author: Christophe.Delaere@cern.ch   20/07/03

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_YMultiLayerPerceptron
#define ROOT_YMultiLayerPerceptron

#include "TObject.h"
#include "TString.h"
#include "TObjArray.h"
#include "TMatrixD.h"
#include "TMultiLayerPerceptron.h"
#include "YNeuron.h"
#include "DetectorConstant.h"
#include "YO2Compat.h"
#include "YSensorCorrection.h"
#include "DataInputStructure.h"


class TTree;
class TEventList;
class TTreeFormula;
class TTreeFormulaManager;

//#define nTrackMax 3  
#ifndef FIT_FUNCTION
#define FIT_FUNCTION

void F_LINEAR(TF1* fLINEAR, double min, double max, double p0, double p1){
  //std::cout<<"F_LINEAR["<<name<<"] : "<<min<<" "<<max<<" "<<p0<<" "<<p1<<std::endl;
  
  //fLINEAR= new TF1(name,"pol1");
  fLINEAR->SetRange(min,max);
  fLINEAR->SetParameter(0,p0); 	// 0.5*(X[nhitlayers-1]+X[0])
  fLINEAR->SetParameter(1,p1);	// slopeXY
}

void F_POL2(TF1 *FPOL2, double min, double max, double p0, double p1, double p2){
  //std::cout<<"F_POL2 : "<<min<<" "<<max<<" "<<p0<<" "<<p1<<" "<<p2<<std::endl;
  
  //FPOL2= new TF1(name,"pol2");
  FPOL2->SetRange(min,max);
  FPOL2->SetParameter(0,p0); 	// intercept at the origin
  FPOL2->SetParameter(1,p1);	// slope at the origin
  FPOL2->SetParameter(2,p2);	// radius of curvature = -1/(2R)
}

#endif
//____________________________________________________________________
//
// YMultiLayerPerceptron
//
// This class decribes a Neural network.
// There are facilities to train the network and use the output.
//
// The input layer is made of inactive neurons (returning the
// normalized input), hidden layers are made of sigmoids and output
// neurons are linear.
//
// The basic input is a TTree and two (training and test) TEventLists.
// For classification jobs, a branch (maybe in a TFriend) must contain
// the expected output.
// 6 learning methods are available: kStochastic, kBatch,
// kSteepestDescent, kRibierePolak, kFletcherReeves and kBFGS.
//
// This implementation is *inspired* from the mlpfit package from
// J.Schwindling et al.
//
//____________________________________________________________________

struct YVertexFitParameter {
   double z_meas[nLAYER+1]; //Event Vertex z
   double beta[nLAYER+1];
   double parz[2];
   double Radius;
   int    chipID[nLAYER+1];   
   bool   valid;
};

struct YOffsetTuneByMean {

   YOffsetTuneByMean() {
      fds1_s1 = nullptr;
      fds1_s2 = nullptr;   
      fds2_s1 = nullptr;   
      fds2_s2 = nullptr;
   };

   YOffsetTuneByMean(TF1* f11, TF1* f12, TF1* f21, TF1* f22) {
      fds1_s1 = f11;
      fds1_s2 = f12;   
      fds2_s1 = f21;   
      fds2_s2 = f22;
   };  
   
   TF1*      fds1_s1;
   TF1*      fds1_s2;   
   TF1*      fds2_s1;   
   TF1*      fds2_s2;

   double ftunePAR[17]; //#ChipID p1A p1B p1C p1D q1A q1B q1C q1D q1E q1F q1G q1H r1A r1B r1C r1D r1E
}; 

struct YImpactParameter {
   double yX, yAlpha;
   double yP[5]; // Y, Z, sin(phi), tg(lambda), q/pT
   char yAbsCharge = 1;
   float ip[2];

   void Print(){
      std::cout<<" [TrackParameterization by XYZ and pXYZ]"
               <<" X "<<yX
               <<" Alpha "<<yAlpha
               <<" Y "<<yP[0]
               <<" Z "<<yP[1]
               <<" snp "<<yP[2]
               <<" tgl "<<yP[3]
               <<" QpT "<<yP[4]
               <<" ipD "<<ip[0]
               <<" ipZ "<<ip[1]<<std::endl;                                             
   }
   
   void TrackParametrization(double* xyz, double* pxpypz, int charge, bool sectorAlpha=true) {
     // construct track param from kinematics

     // Alpha of the frame is defined as:
     // sectorAlpha == false : -> angle of pt direction
     // sectorAlpha == true  : -> angle of the sector from X,Y coordinate for r>1
     //                           angle of pt direction for r==0
     //
     //
     double kSafe = 1e-5;
     double radPos2 = xyz[0] * xyz[0] + xyz[1] * xyz[1];
     double alp = 0;
     if (sectorAlpha || radPos2 < 1) {
       alp = std::atan2(pxpypz[1], pxpypz[0]);
     } else {
       alp = std::atan2(xyz[1], xyz[0]);
     }
     if (sectorAlpha) {
       alp = YO2::Angle2Alpha(alp);
     }
     //
     double sn, cs;
     YO2::SinCos(alp, sn, cs);
     // protection against cosp<0
     if (cs * pxpypz[0] + sn * pxpypz[1] < 0) {
       LOG(debug) << "alpha from phiPos() will invalidate this track parameters, overriding to alpha from phi()";
       alp = std::atan2(pxpypz[1], pxpypz[0]);
       if (sectorAlpha) {
         alp = YO2::Angle2Alpha(alp);
       }
       YO2::SinCos(alp, sn, cs);
     }

     // protection:  avoid alpha being too close to 0 or +-pi/2
     if (std::abs(sn) < 2 * kSafe) {
       if (alp > 0) {
         alp += alp < YO2::kPIHalf ? 2 * kSafe : -2 * kSafe;
       } else {
         alp += alp > -YO2::kPIHalf ? -2 * kSafe : 2 * kSafe;
       }
       YO2::SinCos(alp, sn, cs);
     } else if (std::abs(cs) < 2 * kSafe) {
       if (alp > 0) {
         alp += alp > YO2::kPIHalf ? 2 * kSafe : -2 * kSafe;
       } else {
         alp += alp > -YO2::kPIHalf ? 2 * kSafe : -2 * kSafe;
       }
       YO2::SinCos(alp, sn, cs);
     }
     // get the vertex of origin and the momentum
     std::array<double, 3> ver{xyz[0], xyz[1], xyz[2]};
     std::array<double, 3> mom{pxpypz[0], pxpypz[1], pxpypz[2]};
     //
     // Rotate to the local coordinate system
     YO2::RotateZ(ver, -alp);
     YO2::RotateZ(mom, -alp);
     //
     double ptI = 1.f / sqrt(mom[0] * mom[0] + mom[1] * mom[1]);
     
     yX = ver[0];
     yAlpha = alp;
     yP[0] = ver[1];
     yP[1] = ver[2];
     yP[2] = mom[1] * ptI;
     yP[3] = mom[2] * ptI;
     yAbsCharge = std::abs(charge);
     yP[4] = charge ? ptI * charge : ptI;
     //mPID = pid;
     //
     if (std::abs(1 - yP[2]) < kSafe) {
       yP[2] = 1.f - kSafe; // Protection
     } else if (std::abs(-1 - yP[2]) < kSafe) {
       yP[2] = -1.f + kSafe; // Protection
     }
     //
   };
   
   void getImpactParams(float x, float y, float z, float bz) {
      //------------------------------------------------------------------
      // This function calculates the transverse and longitudinal impact parameters
      // with respect to a point with global coordinates (x,y,0)
      // in the magnetic field "bz" (kG)
      //------------------------------------------------------------------
      float f1 = yP[2], r1 = TMath::Sqrt((1. - f1) * (1. + f1));
      float xt = yX, yt = yP[0];
      float sn = TMath::Sin(yAlpha), cs = TMath::Cos(yAlpha);
      float a = x * cs + y * sn;
      y = -x * sn + y * cs;
      x = a;
      xt -= x;
      yt -= y;

      float rp4 = yP[4] * bz * YO2::kB2C;//getCurvature(bz);
      float Almost0 = 1e-12;
      if ((TMath::Abs(bz) < Almost0) || (TMath::Abs(rp4) < Almost0)) {
        ip[0] = -(xt * f1 - yt * r1);
        ip[1] = yP[1] + (ip[0] * f1 - xt) / r1 * yP[3] - z;
        return;
      }

      sn = rp4 * xt - f1;
      cs = rp4 * yt + r1;
      a = 2 * (xt * f1 - yt * r1) - rp4 * (xt * xt + yt * yt);
      float rr = TMath::Sqrt(sn * sn + cs * cs);
      ip[0] = -a / (1 + rr);
      float f2 = -sn / rr, r2 = TMath::Sqrt((1. - f2) * (1. + f2));
      ip[1] = yP[1] + yP[3] / rp4 * TMath::ASin(f2 * r1 - f1 * r2) - z;
    };   
   
};


struct YResidualMonitor {

   YResidualMonitor() {
      vdcaX= -9999;
      vdcaY= -9999;
      vdcaZ= -9999;   
      vtxX= -9999;
      vtxY= -9999;
      vtxZ= -9999;
      vtxevtX= -9999;
      vtxevtY= -9999;
      vtxevtZ= -9999;       
      vtxfitX= -9999;
      vtxfitY= -9999;
      vtxfitZ= -9999;
      pT= -9999;
      p= -9999;
      px= -9999;
      py= -9999;
      pz= -9999;  
      cuvR = 0;
      theta= -9999;
      phi= -9999;  
      eta= -9999;      
      for(int lay = 0; lay < nLAYER+1; lay++){
         fs1[lay]     = -9999;
         fs2[lay]     = -9999;
         fds1[lay]    = -9999;
         fds2[lay]    = -9999;
         fgX[lay]     = -9999;
         fgY[lay]     = -9999;
         fgZ[lay]     = -9999;
         fgR[lay]     = -9999;
         fgPhi[lay]   = -9999;
         fdgX[lay]    = -9999;
         fdgY[lay]    = -9999;
         fdgZ[lay]    = -9999;        
         fcs1[lay]    = -9999;
         fcs2[lay]    = -9999;
         fcs3[lay]    = -9999;             
         fchipID[lay] = -1; 
         flayer[lay]  = -1;
         fhb[lay]     = -1;
         fstv[lay]    = -1;         
         fhs[lay]     = -1;
         fmd[lay]     = -1;
         flchip[lay]  = -1;
         fschip[lay]  = -1;
         fhschip[lay] = -1;
         fmchip[lay]  = -1;         
         
         fntracks[lay] = 0;
         fstatus[lay]  = 0;              
      }
   };

   void registerDCA(double vx, double vy, double vz){
      vdcaX=vx;
      vdcaY=vy;
      vdcaZ=vz;
   }

   void registerVertex(double vx, double vy, double vz){
      vtxX=vx;
      vtxY=vy;
      vtxZ=vz;
   }

   void registerVertexEvent(double vx, double vy, double vz){
      vtxevtX=vx;
      vtxevtY=vy;
      vtxevtZ=vz;
   }

   void registerVertexFit(double vx, double vy, double vz){
      vtxfitX=vx;
      vtxfitY=vy;
      vtxfitZ=vz;
   }
   
   void registerCurvature(double R){
      cuvR = R;
   }

   void registerMomentum(double vpt, double vphi, double vtheta){
      pT=vpt;

      phi    = vphi;
      theta  = vtheta;
      eta    = -std::log(std::tan(theta/2.));
      p   = pT*std::cosh(eta);
      pz  = pT*std::sinh(eta);

      px  = pT*std::cos(phi);
      py  = pT*std::sin(phi);
   };

   void registerChipUpdateStatus(int layer, int ntracks, short status){
      fntracks[layer] = ntracks;
      fstatus[layer]  = status;  
   };

   void registerResidual(int layer, double s1, double s2, double ds1, double ds2, double gX, double gY, double gZ, double gR, double gPhi){
      if(layer<0 || layer>nLAYER+1) return;
      fs1[layer]     = s1;
      fs2[layer]     = s2;
      fds1[layer]    = ds1;
      fds2[layer]    = ds2;
      fgX[layer]     = gX;
      fgY[layer]     = gY;
      fgZ[layer]     = gZ;
      fgR[layer]     = gR;
      fgPhi[layer]   = gPhi;
   };

   void registerchipInfo(int layer, int chipID, int hb, int stv, int hs, int md, int lchip, int schip, int hschip, int mchip){
      if(layer<0 || layer>nLAYER+1) return;  
      flayer[layer]  = layer;
      fchipID[layer] = chipID;
      fhb[layer]     = hb;
      fstv[layer]    = stv;         
      fhs[layer]     = hs;
      fmd[layer]     = md;
      flchip[layer]  = lchip;
      fschip[layer]  = schip;
      fhschip[layer] = hschip;
      fmchip[layer]  = mchip;
   };

   void registerCorrectionFunction(int layer, double gx, double gy, double gz, double cs1, double cs2, double cs3){
      if(layer<0 || layer>nLAYER+1) return;   
      fdgX[layer]     = gx;
      fdgY[layer]     = gy;
      fdgZ[layer]     = gz;        
      fcs1[layer]     = cs1;
      fcs2[layer]     = cs2;
      fcs3[layer]     = cs3;      
   };

   void registerImpactParameters(YImpactParameter yIP){
      fyX = yIP.yX;
      fyAlpha = yIP.yAlpha;
      fyP[0] = yIP.yP[0];
      fyP[1] = yIP.yP[1];
      fyP[2] = yIP.yP[2];
      fyP[3] = yIP.yP[3];
      fyP[4] = yIP.yP[4];                        
      fyAbsCharge = yIP.yAbsCharge;
      fip[0] = yIP.ip[0];
      fip[1] = yIP.ip[1];
   };

   void clear(){
      vdcaX= -9999;
      vdcaY= -9999;
      vdcaZ= -9999;
      vtxX= -9999;
      vtxY= -9999;
      vtxZ= -9999;
      vtxevtX= -9999;
      vtxevtY= -9999;
      vtxevtZ= -9999;          
      vtxfitX= -9999;
      vtxfitY= -9999;
      vtxfitZ= -9999;      
      pT= -9999;
      p= -9999;
      px= -9999;
      py= -9999;
      pz= -9999;
      cuvR = 0;  
      theta= -9999;
      phi= -9999; 
      eta= -9999;
      for(int lay = 0; lay < nLAYER+1; lay++){
         fs1[lay]     = -9999;
         fs2[lay]     = -9999;
         fds1[lay]    = -9999;
         fds2[lay]    = -9999;
         fgX[lay]     = -9999;
         fgY[lay]     = -9999;
         fgZ[lay]     = -9999;
         fgR[lay]     = -9999;
         fgPhi[lay]   = -9999;
         fdgX[lay]    = -9999;
         fdgY[lay]    = -9999;
         fdgZ[lay]    = -9999;        
         fcs1[lay]    = -9999;
         fcs2[lay]    = -9999;
         fcs3[lay]    = -9999;             
         fchipID[lay] = -1;
         flayer[lay]  = -1;
         fhb[lay]     = -1;
         fstv[lay]    = -1;         
         fhs[lay]     = -1;
         fmd[lay]     = -1;
         flchip[lay]  = -1;
         fschip[lay]  = -1;
         fhschip[lay] = -1;
         fmchip[lay]  = -1;
         fntracks[lay] = 0;
         fstatus[lay]  = 0;       
      }
   };
   
   double vdcaX, vdcaY, vdcaZ;
   double vtxX, vtxY, vtxZ;
   double vtxevtX, vtxevtY, vtxevtZ;
   double vtxfitX, vtxfitY, vtxfitZ;
   double pT, p, px, py, pz;
   double cuvR;
   double theta, phi, eta;
   double fs1[nLAYER+1],  fs2[nLAYER+1];
   double fds1[nLAYER+1], fds2[nLAYER+1];
   int fntracks[nLAYER+1];
   short fstatus[nLAYER+1];
   double fgX[nLAYER+1], fgY[nLAYER+1], fgZ[nLAYER+1], fgR[nLAYER+1], fgPhi[nLAYER+1];
   int fchipID[nLAYER+1];
   int flayer[nLAYER+1], fhb[nLAYER+1], fstv[nLAYER+1], fhs[nLAYER+1], fmd[nLAYER+1];
   int flchip[nLAYER+1], fschip[nLAYER+1], fhschip[nLAYER+1], fmchip[nLAYER+1];
   double fdgX[nLAYER+1], fdgY[nLAYER+1], fdgZ[nLAYER+1], fcs1[nLAYER+1], fcs2[nLAYER+1], fcs3[nLAYER+1];
   
   //YImpactParameter yIP;
   double fyX, fyAlpha;
   double fyP[5]; // Y, Z, sin(phi), tg(lambda), q/pT
   char fyAbsCharge = 1;
   float fip[2];
};
// Detector 0, Detector(IB/OB1/OB2) 1, HalfStave 2, ChipZ 3, Layer 4, HalfBarrel 5 Stave 6 SubStave 7 // sensor -1 // fix -2
//int kDetectorUnit[7] = {-2, -2, -2, 3, 3, 3, 3};
int kDetectorUnit[7] = {-1, -1, -1, -1, -1, -1, -1};
//int kDetectorUnit[7] = {-1, -1, -1, -1, -1, -1, -1};
struct YDetectorUnit {

   YDetectorUnit() = default;
   YDetectorUnit(short lev, short idx) :
   level(lev),
   index(idx)
   {
      fSCNetwork = new YSensorCorrection(lev, idx);
   };
   
   void showchipIDs()
   {
      std::cout<<" >> ";
      for(int ich = 0; ich < chipID.size(); ich++)
      {
         std::cout<<chipID[ich]<<" ";
      }
      std::cout<<std::endl;
   };
      
   short level; // Detector 0, Detector(IB/OB1/OB2) 1, HalfStave 2, ChipZ 3, Layer 4, HalfBarrel 5 Stave 6 SubStave 7 // sensor -1 // fix -2
   short index; // index for level
   std::vector<YDetectorUnit*> SubUnit;			// vs map ?
   std::vector<int> chipID;   				//  
   YSensorCorrection* fSCNetwork;   
};

class YMultiLayerPerceptron : public TMultiLayerPerceptron {
 friend class YMLPAnalyzer;

 public:
   enum ELearningMethod { kStochastic, kBatch, kBatchDetectorUnitUser, kSteepestDescent,
                          kRibierePolak, kFletcherReeves, kBFGS, kOffsetTuneByMean };
   enum EDataSet { kTraining, kTest };
   YMultiLayerPerceptron();
   YMultiLayerPerceptron(const char* layout, TTree* data = 0,
                         const char* training = "Entry$%2==0",
                         const char* test = "",
                         YNeuron::ENeuronType type = YNeuron::kSigmoid,
                         const char* extF = "", const char* extD  = "");
   YMultiLayerPerceptron(const char* layout,
                         const char* weight, TTree* data = 0,
                         const char* training = "Entry$%2==0",
                         const char* test = "",
                         YNeuron::ENeuronType type = YNeuron::kSigmoid,
                         const char* extF = "", const char* extD  = "");
   YMultiLayerPerceptron(const char* layout, TTree* data,
                         TEventList* training,
                         TEventList* test,
                         YNeuron::ENeuronType type = YNeuron::kSigmoid,
                         const char* extF = "", const char* extD  = "");                               
   YMultiLayerPerceptron(const char* layout,
                         const char* weight, TTree* data,
                         TEventList* training,
                         TEventList* test,
                         YNeuron::ENeuronType type = YNeuron::kSigmoid,
                         const char* extF = "", const char* extD  = "");
   virtual ~YMultiLayerPerceptron();
   YSensorCorrection** SCNetwork() {return fSCNetwork;};   
   
   void SetData(TTree*);
   void SetTrainingDataSet(TEventList* train);
   void SetTestDataSet(TEventList* test);
   void SetTrainingDataSet(const char* train);
   void SetTestDataSet(const char* test);
   void SetLearningMethod(YMultiLayerPerceptron::ELearningMethod method);
   void SetEventWeight(const char*);
   void Train(Int_t nEpoch, Option_t* option = "text", Double_t minE=0);
   void EvaluateCost(int step, int core);
   
   Double_t GetCost(Int_t ntrack);// const;
   Double_t GetCost(YMultiLayerPerceptron::EDataSet set);// const;

   void SetFitModel(int x);				//[DetectorAlignment] kLineFit
   void SetRandomSeed(int x);
   void SetLayerTrain(int x);  
   
   void BetaLinearization(double* beta, TVector3* dirXc, std::vector<bool> hitUpdate);
   void GetProjectionPoints(TVector3 vecCircle_center, double RecRadius, TVector3* vecSensorNorm, TVector3* vecXc_meas, TVector3* vecXc_proj, TVector3* vecXc_norm);
   void BuildDerivativesXY();
   
   void ComputeDCDw(); //const
   void ComputeDCDwDetectorUnit(); //const   
   void Init_Randomize() const;     
   void Init_RandomizeSensorCorrection() const;
   void Init_RandomizeSensorCorrectionDetectorUnit() const;   
   void SetNpronged(int nprong);
   void SetWeightMonitoring(const char* WeightName, int step);
   void SetPrevUSL(const char* PrevUSLName);
   void SetPrevWeight(const char* PrevWeightName);
   void SetPrevWeightDetectorUnit(const char* PrevWeightDetectorName);

   void SetEta(Double_t eta);
   void SetEpsilon(Double_t eps);
   void SetDelta(Double_t delta);
   void SetEtaDecay(Double_t ed);
   void SetTau(Double_t tau);
   void SetReset(Int_t reset);
   inline Double_t GetEta()      const { return fEta; }
   inline Double_t GetEpsilon()  const { return fEpsilon; }
   inline Double_t GetDelta()    const { return fDelta; }
   inline Double_t GetEtaDecay() const { return fEtaDecay; }
   YMultiLayerPerceptron::ELearningMethod GetLearningMethod() const { return fLearningMethod; }
   inline Double_t GetTau()      const { return fTau; }
   inline Int_t GetReset()       const { return fReset; }
   inline TString GetStructure() const { return fStructure; }
   inline YNeuron::ENeuronType GetType() const { return fType; }

   Bool_t DumpUpdateSensorList(Option_t * filename = "-") const;
   Bool_t DumpWeights(Option_t* filename = "-") const;
   Bool_t DumpWeightsDetectorUnit(Option_t* filename = "-") const;
   Bool_t PrepareDumpResiduals();
   Bool_t DumpResiduals(int epoch);   
   Bool_t DumpCostGradients(int epoch, Double_t** bufferArr, double threshold = 2) const;
   Bool_t PrintCurrentWeights();
   Bool_t LoadOffsetSlopeCorrectionParameters(Option_t * filename = "");

   Bool_t LoadUpdateSensorList(Option_t * filename = "");
   Bool_t LoadWeights(Option_t* filename = "");
   Bool_t LoadWeightsDetectorUnit(Option_t* filename = "");
   void SetWeightsDetectorUnitUser();
   Bool_t LoadWeights(int icLayer, int icAxis, int icOrder, double icValue); //Layer Axis Order value   
   void SetNetworkUpdateState(std::vector<bool> chiplist);   
   void InitDetectorUnitNetworkUpdateState();
   void SetDetectorUnitNetworkUpdateStateUser();
   void SetSplitReferenceSensor(int layer, int chipIDinlayer);
   
   TVector3 GetCorrectedS2Vector(int chipID);   
   TVector3 GetCorrectedNormalVector(int chipID);
   
   Double_t Evaluate(Int_t index, Double_t* params) const;
   Double_t Evaluate(Int_t index, Double_t* params, Int_t chipID) const;   
   void Export(Option_t* filename = "NNfunction", Option_t* language = "C++") const;
   virtual void Draw(Option_t *option="");
   virtual void DrawNonCrossing(Option_t *option="");
 protected:
   void AttachData();
   void AttachDataToSensorCorrectionNetwork(int chipID);
   void BuildNetwork();  
   void BuildSensorCorrectionNetwork();
   void BuildDetectorUnitNetwork();
   void PrintDetectorUnitNetwork();
   void GetEntry(Int_t) const;
   // it's a choice not to force learning function being const, even if possible
   void MLP_Stochastic(Double_t*);
   void MLP_StochasticArr(Double_t**);   
   void MLP_Batch(Double_t*);
   void MLP_BatchArr(Double_t**);
   void MLP_BatchDetectorUnit(Double_t**, Double_t*, Double_t**, Double_t***, Double_t****, Double_t*****, Double_t******, Double_t*******, Double_t********);
#ifdef MONITORSENSORUNITprofile
   void MLP_OffsetTuneByMean();   
#endif
   void CalculateSCWeights(int sensorID, YSensorCorrection* scn, double* dcdwARR, bool nUpdate = true, bool sUpdate = true);   
   
   Bool_t LineSearch(Double_t*, Double_t*);
   Bool_t LineSearchArr(Double_t**, Double_t**);   
   void SteepestDir(Double_t*);
   void SteepestDirArr(Double_t**);  
   void ConjugateGradientsDir(Double_t*, Double_t);
   void SetGammaDelta(TMatrixD&, TMatrixD&, Double_t*);
   bool GetBFGSH(TMatrixD&, TMatrixD &, TMatrixD&);
   void BFGSDir(TMatrixD&, Double_t*);
   Double_t DerivDir(Double_t*);
   Double_t GetCost_Sensor(int fit);// const;					//[DetectorAlignment] kLineFit
   Double_t GetCost_Beam(int fit, int ntrack);// const;				//[DetectorAlignment]    
   
   Double_t GetCost_Sensor_LineFit();// const;					//[DetectorAlignment] kLineFit
   Double_t GetCost_Beam_LineFit(int track);// const;				//[DetectorAlignment] 
   Double_t GetCost_Vertex_LineFit(); 
   void     InitVertex_LineFit();
   void     AddVertex_LineFit(double* z, double* beta);
   
   Double_t GetCost_Sensor_CircleFit();// const;				//[DetectorAlignment] kLineFit
   Double_t GetCost_Beam_CircleFit(int track);// const;				//[DetectorAlignment] 
   Double_t GetCost_Vertex_CircleFit();  
   void     InitVertex_CircleFit();             
   void     AddVertex_CircleFit(int track, double* z, double* beta, double* parz, double Radius, bool valid);

   void     CalculateEventDcdw(int ntracks);
   void     InitSCNeuronDcdw();
   void     SetSCNeuronDcdw(int i, int j, double x);
   Double_t GetSCNeuronDcdw(int i, int j) const { return fSCNeuronDcdw[i][j]; } 
   
   void EventCheck();

   void PrepareChipsToNetwork(int mode=0);
   
   void MonitorResidual(int epoch);
   void ComputeTrackProfile();   
   void MonitorTracksBySensor1D();   
   void MonitorTracksByHalfStave1D();      
   void MonitorTracksBySensor();
   bool DumpNTracksBySensor();
   
   void UpdateVertexByAlignment();   
   bool TrackerFit(double* input, std::vector<bool> hitUpdate, TVector3* fXFit, double* fdXY, double* fdZR, double* fparXY, double* fparZR, double* mPAR);
 private:
   YMultiLayerPerceptron(const YMultiLayerPerceptron&); // Not implemented
   YMultiLayerPerceptron& operator=(const YMultiLayerPerceptron&); // Not implemented
   void ExpandStructure();
   void BuildFirstLayer(TString&);
   void BuildHiddenLayers(TString&);
   void BuildHiddenLayersNonCrossing(TString&);   
   void BuildOneHiddenLayer(const TString& sNumNodes, Int_t& layer,
                            Int_t& prevStart, Int_t& prevStop,
                            Bool_t lastLayer); 
   void BuildOneHiddenLayerNonCrossing(const TString& sNumNodes, Int_t& layer,
                            Int_t& prevStart, Int_t& prevStop,
                            Bool_t lastLayer);                                                                                            
   void BuildLastLayer(TString&, Int_t);
   void BuildLastLayerNonCrossing(TString&, Int_t);   
   void BuildAddition();
   vector<Int_t*> EventIndex(int type);// const;   
   void Shuffle(Int_t*, Int_t) const;
   void MLP_Line(Double_t*, Double_t*, Double_t);
   void MLP_LineArr(Double_t**, Double_t**, Double_t);
   
   void InitEventLoss();
   void AddEventLoss(); 
   
   void SetEventLoss(int type); //0 : total, 1 : training, 2 : test
   int  GetEventLoss(int type); //0 : total, 1 : training, 2 : test
   
   void InitTrackLoss();
   void AddTrackLoss(); 
   
   void SetTrackLoss(int type); //0 : total, 1 : training, 2 : test
   int  GetTrackLoss(int type); //0 : total, 1 : training, 2 : test
  
   TTree* fData;                   //! pointer to the tree used as datasource
   Int_t fCurrentTree;             //! index of the current tree in a chain
   Double_t fCurrentTreeWeight;    //! weight of the current tree in a chain
   
   TObjArray fNetwork;             // Collection of all the neurons in the network
   TObjArray fFirstLayer;          // Collection of the input neurons; subset of fNetwork
   TObjArray fLastLayer;           // Collection of the output neurons; subset of fNetwork
   TObjArray fSynapses;            // Collection of all the synapses in the network
   TObjArray fAddition;
   
   YSensorCorrection** fSCNetwork;   
   int* 	       NetworkChips;
   vector<bool>        fWUpdatebyMean;        
   vector<bool>        fWUpdate;     
   YDetectorUnit*      fDetectorUnit;
    
   int nChipsPerHicIB;//  = nHicPerStave[0]*nChipsPerHic[0];
   int nChipsPerHicOB1;// = nHicPerStave[3]*nChipsPerHic[3];
   int nChipsPerHicOB2;// = nHicPerStave[5]*nChipsPerHic[5];   

   int nHalfStaveIB;//  = NSubStave[0];
   int nHalfStaveOB1;// = NSubStave[3];
   int nHalfStaveOB2;// = NSubStave[5];

   vector<Double_t>    fSCNeuronDcdw[3*nLAYER]; //[layer][track] <cs1, cs2, cs3> 

   TString fStructure;             // String containing the network structure
   vector<int> vStructure;
   Int_t nSynapses;
   Int_t nNetwork;
   TString fWeight;                // String containing the event weight
   YNeuron::ENeuronType fType;     // Type of hidden neurons
   YNeuron::ENeuronType fOutType;  // Type of output neurons
   TString fextF;                  // String containing the function name
   TString fextD;                  // String containing the derivative name
   TEventList *fTraining;          //! EventList defining the events in the training dataset
   TEventList *fTest;              //! EventList defining the events in the test dataset
   TH1D* fCostMonitor;
   TH2D* fBeamXY;
   TH2D* fBeamZR;
   TH2D* fVertexFitXY;
   TH2D* fVertexFitZR;   
   TH2D* fVertexXY;
   TH2D* fVertexZR;

   TFile* ResidualMonitor;
   TTree* fResidualMonitor;
   YResidualMonitor* b_resmonitor;

   //Layer Unit   
   TH1D* fChi2Layer[nLAYER];
   TH2D* fpTvsResLayer[nLAYER][2];   
   TH2D* fpTvsChiLayer[nLAYER][2];
   
   //Half Stave Unit
#ifdef MONITORHALFSTAVEUNIT   
   TH2D***** fResidualsVsZLayerHBHS;   		// nLAYER nHB nHS 2   
   TH2D***** fResidualsVsPhiLayerHBHS;  	// nLAYER nHB nHS 2
   TH2D***** fpTvsResLayerHBHS;   		// nLAYER nHB nHS 2
   TH2D***** fpTvsChiLayerHBHS;			// nLAYER nHB nHS 2

   TProfile***** fProfileVsZLayerHBHS;   	// nLAYER nHB nHS 2   
   TProfile***** fProfileVsPhiLayerHBHS;   	// nLAYER nHB nHS 2
   TProfile***** fSensorCenterVsZLayerHBHS;   	// nLAYER nHB nHS 2   
   TProfile***** fSensorCenterVsPhiLayerHBHS;   // nLAYER nHB nHS 2
#endif
   
#ifdef MONITORONLYUPDATES
   TH1D* fUPDATESENSORS;
   TH1C* fUPDATETRACKS;
#endif   
   double fCostChargeSymSum[20];
   int    fCostChargeSymNtr[20];
   TH2D* fCostChargeSym; //pT vs nTRK by pstv
   TH1D* fChargeSymMonitorPositive; 
   TH1D* fChargeSymMonitorNegative; 
   
   ELearningMethod fLearningMethod; //! The Learning Method
   TTreeFormula* fEventWeight;     //! formula representing the event weight
   TTreeFormulaManager* fManager;  //! TTreeFormulaManager for the weight and neurons
   //TTreeFormulaManager* fManagerArr[nLAYER][nSTAVELAYER2][nChips];  //! TTreeFormulaManager for the weight and neurons 

   Double_t fEta;                  //! Eta - used in stochastic minimisation - Default=0.1
   Double_t fEpsilon;              //! Epsilon - used in stochastic minimisation - Default=0.
   Double_t fDelta;                //! Delta - used in stochastic minimisation - Default=0.
   Double_t fEtaDecay;             //! EtaDecay - Eta *= EtaDecay at each epoch - Default=1.
   Double_t fTau;                  //! Tau - used in line search - Default=3.
   Double_t fLastAlpha;            //! internal parameter used in line search
   Int_t fReset;                   //! number of epochs between two resets of the search direction to the steepest descent - Default=50
   Bool_t fTrainingOwner;          //! internal flag whether one has to delete fTraining or not
   Bool_t fTestOwner;              //! internal flag whether one has to delete fTest or not
   
   EventData* fDataClass;
   vector<Int_t*> fEventIndex;
   vector<Int_t*> fTrainingIndex;
   vector<Int_t*> fTestIndex;   

   static constexpr int NSubStave2[nLAYER] = { 1, 1, 1, 2, 2, 2, 2 };
   const int NSubStave[nLAYER] = { 1, 1, 1, 2, 2, 2, 2 };
   const int NStaves[nLAYER] = { 12, 16, 20, 24, 30, 42, 48 };
   const int nHicPerStave[nLAYER] = { 1, 1, 1, 8, 8, 14, 14 };
   const int nChipsPerHic[nLAYER] = { 9, 9, 9, 14, 14, 14, 14 };
   const int ChipBoundary[nLAYER + 1] = { 0, 108, 252, 432, 3120, 6480, 14712, 24120 };
   const int StaveBoundary[nLAYER + 1] = { 0, 12, 28, 48, 72, 102, 144, 192 };

   int nSensorsbyLayer[nLAYER] 		= {	ChipBoundary[1] - ChipBoundary[0],
						ChipBoundary[2] - ChipBoundary[1],
						ChipBoundary[3] - ChipBoundary[2],
						ChipBoundary[4] - ChipBoundary[3],
						ChipBoundary[5] - ChipBoundary[4],
						ChipBoundary[6] - ChipBoundary[5],
						ChipBoundary[7] - ChipBoundary[6]	};	

   double** paramRparr;
   double** paramRperp;
   double** paramQparr;
   double** paramQperp;

   Int_t   fSplitReferenceSensor;		// layer + chipIDinlayer -> ChipID
						   
   Int_t   fTotNEvents;
   Int_t   fTotNTraining; 
   Int_t   fTotNTest;        
   
   Int_t   fTotNEventsLoss;
   Int_t   fTotNTrainingLoss; 
   Int_t   fTotNTestLoss; 
   Int_t   fEventLoss;
   
   Int_t   fTotNTracksLoss;
   Int_t   fTotNTrainingTrackLoss; 
   Int_t   fTotNTestTrackLoss; 
   Int_t   fTrackLoss;

   YVertexFitParameter fvertex[nTrackMax];
   std::vector<YOffsetTuneByMean*> fOffsetTuning;
    
   Int_t   fFitModel;		   //[DetectorAlignment] kLineFit // 1 : Line , 2 : Circle
   Int_t   fRandomSeed;
   Int_t   fLayerTrain;
   Int_t   fNpronged;
   
   TString fWeightName;
   Int_t   fWeightStep;
     
   Int_t   fCurrentEpoch;         
      
   TString fPrevUSLName;   
   TString fPrevWeightName;
   TString fPrevWeightDetectorUnitName;
   
   // TRACKER FIT
   TF1* fXY;
   TF1* fZR;
   TVector3 fvertex_TRKF;
   TVector3* fvertex_track_TRKF;
   
   ClassDef(YMultiLayerPerceptron, 4) // a Neural Network
};

#endif
