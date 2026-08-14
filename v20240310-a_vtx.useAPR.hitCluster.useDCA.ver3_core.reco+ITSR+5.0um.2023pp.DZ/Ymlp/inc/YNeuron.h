// @(#)root/mlp:$Id$
// Author: Christophe.Delaere@cern.ch   20/07/03

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_YNeuron
#define ROOT_YNeuron

#include "TNamed.h"
#include "TObjArray.h"
#include "TNeuron.h"
#include "DetectorConstant.h"
#include "YFitModel.h"
//#include "LineFitError.h"
//#include "CircleFitError.h"
//#include "TVector3.h"				//[DetectorAlignment] kLineFit
	
class TTreeFormula;
class YSynapse;
class TBranch;
class TTree;
class TFormula;

//____________________________________________________________________
//
// YNeuron
//
// This class decribes an elementary neuron, which is the basic
// element for a Neural Network.
// A network is build connecting neurons by synapses.
// There are different types of neurons: linear (a+bx),
// sigmoid (1/(1+exp(-x)), tanh or gaussian.
// In a Multi Layer Perceptron, the input layer is made of
// inactive neurons (returning the normalized input), hidden layers
// are made of sigmoids and output neurons are linear.
//
// This implementation contains several methods to compute the value,
// the derivative, the DcDw, ...
// Values are stored in local buffers. The SetNewEvent() method is
// there to inform buffered values are outdated.
//
//____________________________________________________________________

class YNeuron : public TNeuron {
   friend class YSynapse;

 public:
   enum ENeuronType { kOff, kLinear, kSigmoid, kTanh, kGauss, kLFETanh, kSoftmax, kExternal }; 

   YNeuron(ENeuronType type = kSigmoid,
           const char* name = "", const char* title = "",
           const char* extF = "", const char* extD  = "" );
   virtual ~YNeuron() {}
   inline YSynapse* GetPre(Int_t n) const { return (YSynapse*) fpre.At(n); }
   inline YSynapse* GetPost(Int_t n) const { return (YSynapse*) fpost.At(n); }
   inline YNeuron* GetInLayer(Int_t n) const { return (YNeuron*) flayer.At(n); }
   TTreeFormula* UseBranch(TTree*, const char*);
   TTreeFormula* UseBranchAddition(TTree*, const char*);   
   Double_t GetInput() const;
   Double_t GetValue() const;
   Double_t GetDerivative() const;
   Double_t GetCost(int fit) const; // 1 : line , 2 : circle
   Double_t GetTarget() const;
   Double_t GetDcDw() const;
   Double_t GetBranch() const;
   Double_t GetBranchAddition() const;   
   ENeuronType GetType() const;
   void SetWeight(Double_t w);
   inline Double_t GetWeight() const { return fWeight; }
   //Double_t GetWeight() const { return fWeight; }   
   void SetNormalisation(Double_t mean, Double_t RMS);
   inline const Double_t* GetNormalisation() const { return fNorm; }
   void SetNewEvent() const;
   void SetDCDw(Double_t in);
   inline Double_t GetDCDw() const { return fDCDw; }
   //Double_t GetDCDw() const { return fDCDw; }   
   void ForceExternalValue(Double_t value);
   void AddInLayer(YNeuron*);

   void SetLFEmemory(int nprong=1);
   void ClearLFEmemory();
   
   void SetLFEinput(int i, int j, double x);
   Double_t GetLFEinput(int i, int j) const { return fLFEinput[i][j]; } //i : index, j : track
   void SetLFEoutput(int i, int j, double x);
   Double_t GetLFEoutput(int i, int j) const { return fLFEoutput[i][j]; }   
   void SetLFEextended(int i, int j, double x);
   Double_t GetLFEextended(int i, int j) const { return fLFEextended[i][j]; }   
   void SetLFEaddition(int i, int j, double x);
   Double_t GetLFEaddition(int i, int j) const { return fLFEaddition[i][j]; }    

   void InitSCNeuronDcdw();
   void DumpSCNeuronDcdw(vector<Double_t> *NeuronDcdw);
   void SetSCNeuronDcdw(int i, int j, double x);
   Double_t GetSCNeuronDcdw(int i, int j) const { return fSCNeuronDcdw[i][j]; }    
   
   void SetNeuronIndex(int i);
   Int_t GetNeuronIndex() const {return fIndex;}

   void SetLFENodeIndex(int i);
   Double_t GetLFENodeIndex() const { return fLFENodeIndex; }   

   void SetLFEnprong(int i);
   Double_t GetLFEnprong() const { return fLFEnprong; }  
   
   Double_t GetCost_Sensor(int fit) const;
   Double_t GetCost_Beam(int fit) const;

   Double_t GetCost_Sensor_LineFit() const;
   Double_t GetCost_Beam_LineFit() const;
   
   Double_t GetCost_Sensor_CircleFit() const;
   Double_t GetCost_Beam_CircleFit() const;   
   
   void SetFitModel(int x);

   void SetOutputRange(double x);
   Double_t GetOutputRange(){ return fOutputRange; }

   void SetLFELayerTrain(double x);
   Int_t GetLFELayerTrain() const { return fLFELayerTrain; }

 protected:
   Double_t Sigmoid(Double_t x) const;
   Double_t DSigmoid(Double_t x) const;
   void AddPre(YSynapse*);
   void AddPost(YSynapse*);

 private:
   YNeuron(const YNeuron&); // Not implemented
   YNeuron& operator=(const YNeuron&); // Not implemented

   TObjArray fpre;        // pointers to the previous level in a network
   TObjArray fpost;       // pointers to the next level in a network
   TObjArray flayer;      // pointers to the current level in a network (neurons, not synapses)
   Double_t fWeight;      // weight used for computation
   Double_t fNorm[2];     // normalisation to mean=0, RMS=1.
   ENeuronType fType;     // neuron type
   TFormula* fExtF;       // function   (external mode)
   TFormula* fExtD;       // derivative (external mode)
   //buffers
   //should be mutable when supported by all compilers
   //TTreeFormula* fFormula;//! formula to be used for inputs and outputs
   vector<TTreeFormula*> fFormula;//! formula to be used for inputs and outputs   
   vector<TTreeFormula*> fFormulaAddition;//! formula to be used for inputs and outputs      
   Int_t fIndex;          //! index in the formula
   Bool_t fNewInput;      //! do we need to compute fInput again ?
   Double_t fInput;       //! buffer containing the last neuron input
   Bool_t fNewValue;      //! do we need to compute fValue again ?
   Double_t fValue;       //! buffer containing the last neuron output
   Bool_t fNewDeriv;      //! do we need to compute fDerivative again ?
   Double_t fDerivative;  //! buffer containing the last neuron derivative
   Bool_t fNewDcDw;       //! do we need to compute fDcDw again ?
   Double_t fDcDw;        //! buffer containing the last derivative of the error
   Double_t fDCDw;        //! buffer containing the sum over all examples of DcDw  

   vector<Double_t> fLFEinput[3*nLAYER]; 		//[DetectorAlignment] kLineFit	
   vector<Double_t> fLFEoutput[3*nLAYER];  
   vector<Double_t> fLFEextended[3*nLAYER];
   vector<Double_t> fLFEaddition[8]; 
   vector<Double_t> fSCNeuronDcdw[3*nLAYER]; //[layer][track] <cs1, cs2, cs3> 

   Int_t    fLFEnprong;
   Int_t    fLFENodeIndex;
   Double_t fLineFit_Error;		//[DetectorAlignment] kLineFit
   //Double_t fMSE;			//[DetectorAlignment] kLineFit	
   Int_t    fFitModel;		//[DetectorAlignment] kLineFit	
   Double_t fOutputRange;
   Int_t    fLFELayerTrain;

 public:   
   bool fSCNetworkNeuron;

   ClassDef(YNeuron, 4)   // Neuron for MultiLayerPerceptrons
};

#endif
