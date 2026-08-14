// @(#)location
// Author: J.H.Kim

/*************************************************************************
 *   Yonsei Univ.                                                        *
 *                                                                       *
 *                                                                       *
 *                                                                       *
 *                                                                       *
 *************************************************************************/

#ifndef ROOT_YAlignment
#define ROOT_YAlignment

#include "TObject.h"
#include "TString.h"
//#include "LineFitError.h"
//#include "CircleFitError.h"
#include "YSensorSet.h"
#include "DetectorConstant.h"
#include "YMultiLayerPerceptron.h"
#include "YFitModel.h"

//____________________________________________________________________
//
// YAlignment
//
// Description
//
//--------------------------------------------------------------------

struct sEventIndex {
   int event;
   int ntracks;
};

class YAlignment {

public:
   YAlignment();
   YAlignment(vector<YSensorSet> sensorset);
   virtual ~YAlignment() {};
  
   void SetEpoch(int epoch);
   int 	GetEpoch() {return fEpoch;}   
   void SetStep(int step);
   int 	GetStep() {return fStep;}   
   //void SetMode(int mode);
   //int  GetMode() {return fMode;}
   void SetDataMC(int x);
   int  GetDataMC() { return fDataMC;}
   void SetHiddenLayer(vector<int> &hiddenlayer); 

   void SetPrevUSL(TString);
   void SetPrevWeight(TString);
   void SetPrevWeightDU(TString);   
   void SetNormalizationShift(float value) { fNorm_shift = value; }

   void SetSourceDataName(TString);
   void SetSourceData(TFile*);
   void SetSourceTreeName(TString);
   void SetSourceTree(TTree*);
 
   void EventIndex(TTree* tree);   
   void EventCheck(TTree* tree);   
   double GetEventWeight(EventData* event, double* z_loc_max, double* z_loc_min);

   void PrepareData(int nentries = 10000, int parallel = 0, bool build = true, TString selectedevents = "");
   void EndOfPrepareData();
   void LoadData(int nentries = 10000, int parallel = 0, bool build = true);
   
   void SetSplitReferenceSensor(int layer, int chipIDinlayer);
   void InitNetworkUpdateList();   
   void SetNetworkUpdateListLayerStave(int layer, int stave);
   void LoadNetworkUpdateList(bool userdefined = false);   
   void TrainMLP(YMultiLayerPerceptron::ELearningMethod method = YMultiLayerPerceptron::kSteepestDescent, bool removeDataTree = false);   
   void EvaluateCostMLP(int step, int Ncores, YMultiLayerPerceptron::ELearningMethod method = YMultiLayerPerceptron::kSteepestDescent);
   void AnalyzeMLP(int step, bool build = true, bool bfield = true);   
   
   void ReconstructGeometries(int res_level = 5);
   void GetGeometry(TString Network_dir, TNtuple* fNtuple, int res_level);
   void TotalizeGeometry(TNtuple* fNtupleI, TNtuple* fNtupleT, TNtuple* fNtupleC, TNtuple* fNtuple);

   void ComputeMSE();
   void ComputeOffset();
   void ComputeCorrectionFunction();   
   
   void AddSensorSet(YSensorSet sensorset);
   
private:
   TString				fSourceDataName;		// String containing the source file name 
   TFile*				fSourceData;			// Source file
   TString				fSourceTreeName;		// String containing the source tree    
   TTree* 				fSourceTree;                 	// pointer to the tree used as source data
   TTree* 				fInputTree;			// pointer to the tree used as input data in the training
   TFile*				fOutputFile;			// Output file
   TFile*				fAnalyzeFile;			// Output file

   TFile*				fGeomFile;			// Geometry File; 

   vector<Int_t*> fSourceIndex;
   vector<Int_t*> fEventIndex;
   vector<Int_t*> fEventTraining;
   vector<Int_t*> fEventTest;  
   vector<Int_t*> fEventVaild;   
   
   YMultiLayerPerceptron*		fMLPNetwork;             	// MLP Network by YMultiLayerPerceptron
   
   int					fSplitReferenceSensor;		// layer + chipIDinlayer -> ChipID
   vector<bool>				fNetworkUpdateList;		// NetworkUpdate list
   vector<YSensorSet> 			fSensorset;
   int 					fEpoch;				// Maximum Epoch in a Step    
   int 					fStep;				// Epoch Step, (Total Epoch) = (fStep)*(fEpoch)
   int			                fMode;				// n-pronged track in the training
   int					fDataMC;			// 0 : Real data, 1 : MC data, Default =1
   TString			        fHiddenlayer;			// String containing the hidden layer structure  
   TString				fNetworkStructure;		// Network Structure, input layer + hidden layer + output layer
   
   TString				fPrevUSL;			// Prev Update Sensors List
   TString				fPrevWeights;			// Prev weight set
   TString				fPrevWeightsDU;			// Prev weightDU set
   
   TString 				fDirectory_name;		// = "MLPTrain"; 
   TString				fAnalyze_Directory_name;	// = "MLPTrain/XXXXanalyse";
   TString 				fXXXXtrain_Directory_name;	// = "MLPTrain/XXXXtrain";
   TString 				fweights_Directory_name; 	// = "MLPTrain/weights/weights";    
   TString 				flosscurve_Directory_name; 	// = "MLPTrain/LossCurve/LossCurve";    
   
   float				fNorm_shift;			// Normalization [0, 1] -> [0-fNorm_shift, 1-fNorm_shift]

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

   // List of branches
   TBranch        *b_event_fUniqueID;   //!
   TBranch        *b_event_fBits;   //!
   TBranch        *b_event_Track_;   //!
   TBranch        *b_Track_fUniqueID;   //!
   TBranch        *b_Track_fBits;   //!
   TBranch        *b_Track_s1;   //!
   TBranch        *b_Track_s2;   //!
   TBranch        *b_Track_s3;   //!
   TBranch        *b_Track_row;   //!
   TBranch        *b_Track_col;   //!
   TBranch        *b_Track_Layer;   //!
   TBranch        *b_Track_HalfBarrel;   //!
   TBranch        *b_Track_Stave;   //!
   TBranch        *b_Track_HalfStave;   //!
   TBranch        *b_Track_Module;   //!
   TBranch        *b_Track_Chip;   //!
   TBranch        *b_Track_ChipID;   //!
   TBranch        *b_Track_index;   //!
   TBranch        *b_Track_ncluster;   //!
   TBranch        *b_Track_Det;   //!
   TBranch        *b_event_X1;   //!
   TBranch        *b_event_X2;   //!
   TBranch        *b_event_X3;   //!
   TBranch        *b_event_P1;   //!
   TBranch        *b_event_P2;   //!
   TBranch        *b_event_P3;   //!
   TBranch        *b_event_evno;   //!
   TBranch        *b_event_ntracks;   //!

};

#endif
