
//#include "../inc/maincoordinate.h"
#include "../inc/YMultiLayerPerceptron.h"
#include "../inc/YSynapse.h"
#include "../inc/YNeuron.h"
#include "../inc/YSensorCorrection.h"
#include "TClass.h"
#include "TTree.h"
#include "TEventList.h"
#include "TRandom3.h"
#include "TTimeStamp.h"
#include "TRegexp.h"
#include "TCanvas.h"
#include "TH2.h"
#include "TGraph.h"
#include "TLegend.h"
#include "TMultiGraph.h"
#include "TDirectory.h"
#include "TSystem.h"
#include "Riostream.h"
#include "TMath.h"
//#include "TTreeFormula.h"
//#include "TTreeFormulaManager.h"
#include "TMarker.h"
#include "TLine.h"
#include "TText.h"
#include "TObjString.h"
#include <stdlib.h>
#define Sensor1 1.0
#define Sensor2 1.0
#define Sensor3 1.0

//#define YMLPDEBUG

ClassImp(YSensorCorrection);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

YSensorCorrection::YSensorCorrection()
{
   std::cout<<"YSensorCorrection Default"<<std::endl;	//[DetectorAlignment] kLineFit
   fNetwork.SetOwner(true);
   fFirstLayer.SetOwner(false);
   fLastLayer.SetOwner(false);
   fSynapses.SetOwner(true);  
   fAddition.SetOwner(true);    
   fUpdate = false;
   fmntdir = +1;
   
   faR[0] = 0;
   faR[1] = 0;
   faR[2] = 0;   
   faT[0] = 0;
   faT[1] = 0;
   faT[2] = 0;   
}

YSensorCorrection::YSensorCorrection(int chipID)
{
   //std::cout<<"YSensorCorrection chipID "<<chipID<<std::endl;	//[DetectorAlignment] kLineFit
   fNetwork.SetOwner(true);
   fFirstLayer.SetOwner(false);
   fLastLayer.SetOwner(false);
   fSynapses.SetOwner(true);
   fAddition.SetOwner(true);       
   fUpdate = false;   
   fType = YNeuron::kLinear;   // Type of hidden neurons
   fOutType = YNeuron::kLinear;  // Type of output neurons
   fextF="";                  
   fextD="";      
   fStructure = "s1,s2:0:cs1,cs2,cs3";
   BuildNetwork();
   
   int l = yGEOM->GetLayer(chipID);

#ifdef MONITORSENSORUNITpT   
   int    nbinsI = l<3 ? MONITOR_NbinsIB : MONITOR_NbinsOB;
   double rangeI = l<3 ? MONITOR_RangeIB : MONITOR_RangeOB;    
   for(int a=0; a<2;a++){  
      hpTvsRes[a] = new TH2D(Form("hL%dS%dpTvsResChipID%d",l,a+1,chipID),Form("pT vs #Deltas_{%d} Monitoring Layer %d ChipID %d ; ds(cm) ; pT(GeV)",a+1,l,chipID),nbinsI,-rangeI,+rangeI,50,0,5);   
      hpTvsChi[a] = new TH2D(Form("hL%dS%dpTvsChiChipID%d",l,a+1,chipID),Form("pT vs #chi s_{%d} Monitoring Layer %d ChipID %d ; #chi; pT(GeV)",a+1,l,chipID),MONITOR_NbinsChi,-10,+10,50,0,5);         
   }   
   hChi2 = new TH1D(Form("hChi2ChipID%d",chipID),Form("Chi2 Monitoring Layer %d",l),100,0,20);
#endif

#ifdef MONITORSENSORUNITprofile
   int    nbinsII = l<3 ? 100 : 4;
   double rangeII = 0.5;   
   hpds1_s1 = new TProfile(Form("hpds1_s1_Chip%d",chipID), Form("Profile ds1 vs s1 Chip%d",chipID),nbinsII,-rangeII,+rangeII,-rangeI,+rangeI);
   hpds1_s2 = new TProfile(Form("hpds1_s2_Chip%d",chipID), Form("Profile ds1 vs s2 Chip%d",chipID),nbinsII,-rangeII,+rangeII,-rangeI,+rangeI);
   hpds2_s1 = new TProfile(Form("hpds2_s1_Chip%d",chipID), Form("Profile ds2 vs s1 Chip%d",chipID),nbinsII,-rangeII,+rangeII,-rangeI,+rangeI);
   hpds2_s2 = new TProfile(Form("hpds2_s2_Chip%d",chipID), Form("Profile ds2 vs s2 Chip%d",chipID),nbinsII,-rangeII,+rangeII,-rangeI,+rangeI);   
#endif     

   int    nSteps = 10; 
   hNtracksByRejection = new TH1D(Form("hL%dNtrackMonitorChipID%d",l,chipID),Form("N_{track} Monitoring Layer %d ChipID %d ; Step ; N_{track}",l,chipID),nSteps,0,nSteps);   
   
   fmntdir = +1;   
   
   faR[0] = 0;
   faR[1] = 0;
   faR[2] = 0;   
   faT[0] = 0;
   faT[1] = 0;
   faT[2] = 0;   
}

YSensorCorrection::YSensorCorrection(int level, int index)
{
   fNetwork.SetOwner(true);
   fFirstLayer.SetOwner(false);
   fLastLayer.SetOwner(false);
   fSynapses.SetOwner(true);
   fAddition.SetOwner(true);       
   fUpdate = false;   
   fType = YNeuron::kLinear;   // Type of hidden neurons
   fOutType = YNeuron::kLinear;  // Type of output neurons
   fextF="";                  
   fextD="";      
   fStructure = "s1,s2:0:cs1,cs2,cs3";
   BuildNetwork();
   fmntdir = +1;   
   
   faR[0] = 0;
   faR[1] = 0;
   faR[2] = 0;   
   faT[0] = 0;
   faT[1] = 0;
   faT[2] = 0;   
}

YSensorCorrection::~YSensorCorrection()
{
   /*
   TH2D*   hpTvsRes[2];
   TH2D*   hpTvsChi[2];      
   TH1D*   hChi2;
   */
#ifdef MONITORSENSORUNITpT      
   for(int a=0; a<2;a++){
      delete hpTvsRes[a];
      delete hpTvsChi[a];
   }
#endif

#ifdef MONITORSENSORUNITprofile
   delete hpds1_s1;
   delete hpds1_s2;
   delete hpds2_s1;
   delete hpds2_s2;
#endif
   delete hNtracksByRejection;
}


void YSensorCorrection::ExpandStructure()
{
   TString input  = TString(fStructure(0, fStructure.First(':')));
   const TObjArray *inpL = input.Tokenize(", ");
   Int_t nneurons = inpL->GetLast()+1;
   TString hiddenAndOutput = TString(
         fStructure(fStructure.First(':') + 1,
                    fStructure.Length() - fStructure.First(':')));
   TString newInput;
   Int_t i = 0;
   // loop on input neurons
   for (i = 0; i<nneurons; i++) {
      const TString name = ((TObjString *)inpL->At(i))->GetString();
      if(i) newInput += ",";
      newInput += name;
   }
   delete inpL;

   // Save the result
   fStructure = newInput + ":" + hiddenAndOutput;
}

void YSensorCorrection::BuildNetwork()
{
   ExpandStructure();
   TString input = "s1,s2" ;
   TString hidden = TString(
           fStructure(fStructure.First(':') + 1,
                      fStructure.Last(':') - fStructure.First(':') - 1));
   TString output = "cs1,cs2,cs3" ;  
   //std::cout<<"BuildNetwork input "<<input<<" "<<hidden<<" "<<output<<std::endl;
   Int_t bll = atoi(TString(
           hidden(hidden.Last(':') + 1,
                  hidden.Length() - (hidden.Last(':') + 1))).Data());
   if (input.Length() == 0) {
      Error("BuildNetwork()","malformed structure. No input layer.");
      return;
   }
   if (output.Length() == 0) {
      Error("BuildNetwork()","malformed structure. No output layer.");
      return;
   }
   
   BuildFirstLayer(input);
   if(hidden=='0'){
      //std::cout<<"BuildHiddenLayers No Hidden Layer"<<std::endl;   
      BuildLastLayer(output, 2);
   } else {
      std::cout<<"BuildHiddenLayers"<<std::endl;
      BuildHiddenLayers(hidden);
      BuildLastLayer(output, bll);    
   }
   BuildAddition();
}

void YSensorCorrection::BuildFirstLayer(TString& input)
{
   //std::cout<<"YSensorCorrection::BuildFirstLayer "<<std::endl;
   const TObjArray *inpL = input.Tokenize(", ");
   const Int_t nneurons =inpL->GetLast()+1;
   YNeuron *neuron = 0;
   Int_t i = 0;
   for (i = 0; i<nneurons; i++) {
      const TString name = ((TObjString *)inpL->At(i))->GetString();
      neuron = new YNeuron(YNeuron::kOff, name);
      fFirstLayer.AddLast(neuron);
      fNetwork.AddLast(neuron);
   }
   delete inpL;
}

void YSensorCorrection::BuildHiddenLayers(TString& hidden)
{
   //std::cout<<"YSensorCorrection::BuildHiddenLayers "<<std::endl;
   Int_t beg = 0;
   Int_t end = hidden.Index(":", beg + 1);
   Int_t prevStart = 0;
   Int_t prevStop = fNetwork.GetEntriesFast();
   Int_t layer = 1;
   while (end != -1) {
      BuildOneHiddenLayer(hidden(beg, end - beg), layer, prevStart, prevStop, false);
      beg = end + 1;
      end = hidden.Index(":", beg + 1);
   }
   BuildOneHiddenLayer(hidden(beg, hidden.Length() - beg), layer, prevStart, prevStop, true);
}

void YSensorCorrection::BuildOneHiddenLayer(const TString& sNumNodes, Int_t& layer,
                            Int_t& prevStart, Int_t& prevStop,
                            Bool_t lastLayer)
{
   YNeuron *neuron = 0;
   YSynapse *synapse = 0;
   TString name;
   if (!sNumNodes.IsAlnum() || sNumNodes.IsAlpha()) {
      Error("BuildOneHiddenLayer",
            "The specification '%s' for hidden layer %d must contain only numbers!",
            sNumNodes.Data(), layer - 1);
   } else {
      Int_t num = atoi(sNumNodes.Data());
      for (Int_t i = 0; i < num; i++) {
         name.Form("HiddenL%d:N%d",layer,i);
         neuron = new YNeuron(fType, name, "", (const char*)fextF, (const char*)fextD);
         fNetwork.AddLast(neuron);
         for (Int_t j = prevStart; j < prevStop; j++) {
            synapse = new YSynapse((YNeuron *) fNetwork[j], neuron);
            fSynapses.AddLast(synapse);
         }
      }

      if (!lastLayer) {
         // tell each neuron which ones are in its own layer (for Softmax)
         Int_t nEntries = fNetwork.GetEntriesFast();
         for (Int_t i = prevStop; i < nEntries; i++) {
            neuron = (YNeuron *) fNetwork[i];
            for (Int_t j = prevStop; j < nEntries; j++)
               neuron->AddInLayer((YNeuron *) fNetwork[j]);
         }
      }

      prevStart = prevStop;
      prevStop = fNetwork.GetEntriesFast();
      layer++;
   }                          
}                                                                     
void YSensorCorrection::BuildLastLayer(TString& output, Int_t prev)
{
   //std::cout<<"YSensorCorrection::BuildLastLayer "<<std::endl;
   Int_t nneurons = output.CountChar(',')+1;
   if (fStructure.EndsWith("!")) {
      fStructure = TString(fStructure(0, fStructure.Length() - 1));  // remove "!"
      if (nneurons == 1)
         fOutType = YNeuron::kSigmoid;
      else
         fOutType = YNeuron::kSoftmax;
   }
   Int_t prevStop = fNetwork.GetEntriesFast();
   Int_t prevStart = prevStop - prev;
   Ssiz_t pos = 0;
   YNeuron *neuron;
   YSynapse *synapse;
   TString name;
   Int_t i,j;
   for (i = 0; i<nneurons; i++) {
      Ssiz_t nextpos=output.Index(",",pos);
      if (nextpos!=kNPOS)
         name=output(pos,nextpos-pos);
      else name=output(pos,output.Length());
      pos+=nextpos+1;
      neuron = new YNeuron(fOutType, name);
      for (j = prevStart; j < prevStop; j++) {
         synapse = new YSynapse((YNeuron *) fNetwork[j], neuron);
         fSynapses.AddLast(synapse);
      }
      fLastLayer.AddLast(neuron);
      fNetwork.AddLast(neuron);
   }
   // tell each neuron which ones are in its own layer (for Softmax)
   Int_t nEntries = fNetwork.GetEntriesFast();
   for (i = prevStop; i < nEntries; i++) {
      neuron = (YNeuron *) fNetwork[i];
      for (j = prevStop; j < nEntries; j++)
         neuron->AddInLayer((YNeuron *) fNetwork[j]);
   }
}

void YSensorCorrection::BuildAddition(){

   YNeuron *neuron = 0;

   TString name;
   name.Form("Addition");
   neuron = new YNeuron(fType, name, "", (const char*)fextF, (const char*)fextD);
   fAddition.AddLast(neuron);

}   

void YSensorCorrection::InitResProfile(){
#ifdef MONITORSENSORUNITprofile
   hpds1_s1->Reset();
   hpds1_s2->Reset();
   hpds2_s1->Reset();
   hpds2_s2->Reset();  
#endif
   hNtracksByRejection->Reset(); 
}

#ifdef MONITORSENSORUNITpT   
void YSensorCorrection::InitpTvsRes(){
   for(int a=0; a<2;a++){
      hpTvsRes[a]->Reset();
   }
}

void YSensorCorrection::FillpTvsRes(int axis, double pt, double res){
   hpTvsRes[axis]->Fill(res, pt);
}

void YSensorCorrection::InitpTvsChi(){
   for(int a=0; a<2;a++){
      hpTvsChi[a]->Reset();
   }
}

void YSensorCorrection::FillpTvsChi(int axis, double pt, double chi){
   hpTvsChi[axis]->Fill(chi, pt);
}

void YSensorCorrection::InitChi2(){
   hChi2->Reset();
}

void YSensorCorrection::FillChi2(double value){
   hChi2->Fill(value);
}
#endif

#ifdef MONITORSENSORUNITprofile
void YSensorCorrection::ProfileOptimizerbyMinBin(int minbin=4){
   for(int nb = 0; nb < hpds1_s1->GetNbinsX(); nb++ ){
      int count1 = hpds1_s1->GetBinEntries(nb+1);
      if(count1<minbin){
         hpds1_s1->SetBinEntries(nb+1,0);
         hpds2_s1->SetBinEntries(nb+1,0);
      }
   }

   for(int nb = 0; nb < hpds2_s2->GetNbinsX(); nb++ ){
      int count2 = hpds2_s2->GetBinEntries(nb+1);
      if(count2<minbin){
         hpds1_s2->SetBinEntries(nb+1,0);
         hpds2_s2->SetBinEntries(nb+1,0);        
      }
   }   
}
#endif
