#ifndef ROOT_YSensorCorrection
#define ROOT_YSensorCorrection

#include "TObject.h"
#include "TString.h"
#include "TObjArray.h"
#include "TMatrixD.h"
#include "DetectorConstant.h"
#include "TMultiLayerPerceptron.h"
#include "YNeuron.h"
#include "YSynapse.h"
//____________________________________________________________________
//
// YSensorCorrection
//____________________________________________________________________

class YSensorCorrection : public TObject {
 public:
   YSensorCorrection();
   YSensorCorrection(int ChipID); //layer, chipID (for IB and OB)
   YSensorCorrection(int level, int index);
     
   virtual ~YSensorCorrection();
   void ExpandStructure();
   void BuildNetwork();
   void BuildFirstLayer(TString&);
   void BuildHiddenLayers(TString&);
   void BuildOneHiddenLayer(const TString& sNumNodes, Int_t& layer,
                            Int_t& prevStart, Int_t& prevStop,
                            Bool_t lastLayer);                                                                     
   void BuildLastLayer(TString&, Int_t);
   void BuildAddition();   
   TObjArray& GetNetwork() {return fNetwork;};
   TObjArray& GetFirstLayer() {return fFirstLayer;};
   TObjArray& GetLastLayer() {return fLastLayer;};
   TObjArray& GetSynapses() {return fSynapses;};
   TObjArray& GetAddition() {return fAddition;};
   void SetnEvents(int x) {fnEvents=x;};
   int GetnEvents() {return fnEvents;};
   void SetUpdateState(short mode) { fUpdate = mode; };
   short GetUpdateState() { return fUpdate; };

   void InitResProfile();
   
#ifdef MONITORSENSORUNITpT   
   void InitpTvsRes();
   void FillpTvsRes(int axis, double pt, double res);
   TH2D*  GetpTvsResHisto(int axis) { return hpTvsRes[axis];};  
   
   void InitpTvsChi();
   void FillpTvsChi(int axis, double pt, double chi);
   TH2D*  GetpTvsChiHisto(int axis) { return hpTvsChi[axis];};   
   
   void InitChi2();
   void FillChi2(double value);
   TH1D*  GetChi2Histo() { return hChi2;};
#endif   

   void   SetMntDir(short md) {fmntdir=md;};
   short  GetMntDir()  {return fmntdir;};
   
   double aR(int i) {return faR[i];};
   double aT(int i) {return faT[i];};
   
   void SetaR(int i, double xR) { faR[i] = xR;};
   void SetaT(int i, double xT) { faT[i] = xT;};  

#ifdef MONITORSENSORUNITprofile

   void ProfileOptimizerbyMinBin(int minbin=10);

   TProfile* hpds1_s1;
   TProfile* hpds1_s2;
   TProfile* hpds2_s1;
   TProfile* hpds2_s2;
   
   int GetEntriesProfile(TProfile* hp){
      int entries = 0;
      for(int nb = 0; nb < hp->GetNbinsX(); nb++){
         entries += hp->GetBinEntries(nb+1);
      }
      return entries;
   };
#endif   
   
   TH1D*     hNtracksByRejection;
     
 private:
   //TTree* fData;                   //! pointer to the tree used as datasource
   TString fStructure; 		   // String containing the network structure
 
   TObjArray fNetwork;             // Collection of all the neurons in the network
   TObjArray fFirstLayer;          // Collection of the input neurons; subset of fNetwork
   TObjArray fLastLayer;           // Collection of the output neurons; subset of fNetwork
   TObjArray fSynapses;            // Collection of all the synapses in the network
   TObjArray fAddition;
   
   double faR[3];
   double faT[3];

   int	fnEvents;		   // 
   short fUpdate;

   YNeuron::ENeuronType fType;     // Type of hidden neurons
   YNeuron::ENeuronType fOutType;  // Type of output neurons
   TString fextF;                  // String containing the function name
   TString fextD;                  // String containing the derivative name
   //TEventList *fTraining;          //! EventList defining the events in the training dataset
   //TEventList *fTest;              //! EventList defining the events in the test dataset

   //TTreeFormula* fEventWeight;     //! formula representing the event weight
   //TTreeFormulaManager* fManager;  //! TTreeFormulaManager for the weight and neurons  

#ifdef MONITORSENSORUNITpT   
   TH2D*   hpTvsRes[2];
   TH2D*   hpTvsChi[2];
   TH1D*   hChi2;
#endif

   short   fmntdir;
   
   ClassDef(YSensorCorrection, 1)  // simple weighted bidirectionnal connection between 2 neurons
};

#endif
