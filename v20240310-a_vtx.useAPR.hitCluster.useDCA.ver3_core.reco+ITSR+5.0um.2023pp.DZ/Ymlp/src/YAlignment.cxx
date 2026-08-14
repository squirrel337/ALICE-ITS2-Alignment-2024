// @(#)location
// Author: J.H.Kim

/*************************************************************************
 *   Yonsei Univ.                                                        *
 *                                                                       *
 *                                                                       *
 *                                                                       *
 *                                                                       *
 *************************************************************************/

///////////////////////////////////////////////////////////////////////////
//
// YAlignment
//
//
///////////////////////////////////////////////////////////////////////////

#include "../inc/YAlignment.h"
#include "../inc/maincoordinate.h"

#include "../src/YSensorSet.cxx"
#include "../src/YNeuron.cxx"
#include "../src/YSynapse.cxx"
#include "../src/YSensorCorrection.cxx"
#include "../src/YMLPAnalyzer.cxx"
#include "../src/YMultiLayerPerceptron.cxx"
#include "../src/YFitModel.cxx"

//#define YALIGNDEBUG

////////////////////////////////////////////////////////////////////////////////
/// Defualt Constructor YAlignment

YAlignment::YAlignment() 
{
   std::cout<<"Default Constructor YAlignment "<<std::endl;
   fSourceData = 0;
   fSourceTree = 0;   
   fEpoch = 100;
   fStep = 0;
   //fMode = 1;
   fDataMC = 1;   
   fNorm_shift = 0.5;
   fPrevUSL="";   
   fPrevWeights="";
   fPrevWeightsDU=""; 
   fHiddenlayer = "";   
   InitNetworkUpdateList();
   fSplitReferenceSensor = -1;

   fDirectory_name = "MLPTrain";
   //fAnalyze_Directory_name = "MLPTrain/XXXXanalyse";   
   fXXXXtrain_Directory_name = "MLPTrain/XXXXtrain";
   fweights_Directory_name   = "MLPTrain/weights";    
   flosscurve_Directory_name = "MLPTrain/LossCurve"; 
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor YAlignment type 1

YAlignment::YAlignment(vector<YSensorSet> sensorset) 
{
   std::cout<<"Constructor YAlignment with sensor set"<<std::endl;
   std::cout<<"Please Select Sensors"<<std::endl;  
   fSourceData = 0;
   fSourceTree = 0;      
   fEpoch = 100;
   fStep = 0;
   //fMode = 1;  
   fDataMC = 1;
   fNorm_shift = 0.5;
   fPrevUSL="";   
   fPrevWeights="";
   fPrevWeightsDU="";   
   fHiddenlayer = "";
   for(int i=0; i<sensorset.size(); i++){
      fSensorset.push_back(sensorset[i]);
      for(int j=0; j<sensorset.size(); j++){
         std::cout<<"SensorSet["<<i<<"]: layer "<<fSensorset[i].Getlayer(j)
                                     <<" stave "<<fSensorset[i].Getstave(j)
                                     <<" chip "<<fSensorset[i].Getchip(j);                                     
      }
   }
   InitNetworkUpdateList();
   fSplitReferenceSensor = -1;   

   fDirectory_name = "MLPTrain";
   //fAnalyze_Directory_name = "MLPTrain/XXXXanalyse";   
   fXXXXtrain_Directory_name = "MLPTrain/XXXXtrain";
   fweights_Directory_name   = "MLPTrain/weights";    
   flosscurve_Directory_name = "MLPTrain/LossCurve"; 
}

////////////////////////////////////////////////////////////////////////////////
/// SetEpoch

void YAlignment::SetEpoch(int epoch)
{
   fEpoch=epoch; 

}

////////////////////////////////////////////////////////////////////////////////
/// SetStep

void YAlignment::SetStep(int step)
{
   fStep=step; 
}     

////////////////////////////////////////////////////////////////////////////////
/// SetMode

//void YAlignment::SetMode(int mode) 
//{ 
//   fMode = mode; /
//}

////////////////////////////////////////////////////////////////////////////////
/// SetDataMC

void YAlignment::SetDataMC(int x)
{ 
   fDataMC = x;
}

////////////////////////////////////////////////////////////////////////////////
/// SetHiddenLayer

void YAlignment::SetHiddenLayer(vector<int> &hiddenlayer)
{

   for(int i=0; i<hiddenlayer.size(); i++){
      fHiddenlayer += ":" + TString::Itoa(hiddenlayer[i],10) ;
   }
   fHiddenlayer += ":";
   std::cout<<"Set Hiddenlayer Structure "<<fHiddenlayer<<std::endl; 
}

void YAlignment::SetPrevUSL(TString usl)
{
   fPrevUSL = usl;
   std::cout<<"Set PrevUSL "<<fPrevUSL<<std::endl; 
}

////////////////////////////////////////////////////////////////////////////////
/// SetPrevWeight

void YAlignment::SetPrevWeight(TString w)
{
   fPrevWeights = w;
   std::cout<<"Set PrevWeight "<<fPrevWeights<<std::endl; 
}

void YAlignment::SetPrevWeightDU(TString wDU)
{
   fPrevWeightsDU = wDU;
   std::cout<<"Set PrevWeight DU"<<fPrevWeightsDU<<std::endl; 
}

////////////////////////////////////////////////////////////////////////////////
/// SetSourceDataName

void YAlignment::SetSourceDataName(TString name)
{
   fSourceDataName = name;
}

////////////////////////////////////////////////////////////////////////////////
/// SetSourceData

void YAlignment::SetSourceData(TFile* data)
{
   if (fSourceData) {
      std::cerr << "Error: source data already defined." << std::endl;
      return;
   }
   fSourceData = data;
}

////////////////////////////////////////////////////////////////////////////////
/// SetSourceTreeName

void YAlignment::SetSourceTreeName(TString name)
{
   fSourceTreeName = name;
}

////////////////////////////////////////////////////////////////////////////////
/// SetSourceTree

void YAlignment::SetSourceTree(TTree* tree)
{
   if (fSourceTree) {
      std::cerr << "Error: source tree already defined." << std::endl;
      return;
   }
   fSourceTree = tree;
}

////////////////////////////////////////////////////////////////////////////////
/// EventIndexing

void YAlignment::EventIndex(TTree* tree)
{
 
   if(tree==fSourceTree){
      EventData* b_event = new EventData();  tree->SetBranchAddress("event",      &b_event); 
      for (int i = 0; i < tree->GetEntriesFast(); i++) {
         tree->GetEntry(i); 
         int index[2];
         index[0] = (int)i;
         index[1] = (int)b_event->GetNtracks();
         fSourceIndex.push_back(new int [2]);  
         fSourceIndex[fSourceIndex.size()-1][0] = index[0];
         fSourceIndex[fSourceIndex.size()-1][1] = index[1];         
#ifdef YALIGNDEBUG 
         std::cout<<"YAlignment::EventIndex(S) "<<i<<" "<<index[0]<<" "<<index[1]<<std::endl;                 
         std::cout<<"YAlignment::EventIndex(S) "<<fSourceIndex.size()-1<<" "<<fSourceIndex[fSourceIndex.size()-1][0]<<" "<<fSourceIndex[fSourceIndex.size()-1][1]<<std::endl;
#endif                          
         //i = i + index[1];

      }   
   } else {
   
      EventData* b_event = new EventData();  tree->SetBranchAddress("event",      &b_event);
      for (int i = 0; i < tree->GetEntriesFast(); i++) {
         tree->GetEntry(i); 
         int index[2];
         index[0] = (int)i;
         index[1] = (int) b_event->GetNtracks();
         fEventIndex.push_back(new int [2]);
         fEventIndex[fEventIndex.size()-1][0] = index[0];
         fEventIndex[fEventIndex.size()-1][1] = index[1]; 
#ifdef YALIGNDEBUG 
         std::cout<<"YAlignment::EventIndex "<<fEventIndex.size()-1<<" "<<fEventIndex[fEventIndex.size()-1][0]<<" "<<fEventIndex[fEventIndex.size()-1][1]<<std::endl;      
#endif               
         //"(evno%10)>=0&&(evno%10)<6","(evno%10)>=6&&(evno%10)<8"
         if((index[0]%10)>=0&&(index[0]%10)<6){
            fEventTraining.push_back(new int [2]);    
            fEventTraining[fEventTraining.size()-1][0] = index[0];
            fEventTraining[fEventTraining.size()-1][1] = index[1];  
         } else if((index[0]%10)>=6&&(index[0]%10)<8){
            fEventTest.push_back(new int [2]); 
            fEventTest[fEventTest.size()-1][0] = index[0];
            fEventTest[fEventTest.size()-1][1] = index[1];               
         } else if((index[0]%10)>=8&&(index[0]%10)<10){
            fEventVaild.push_back(new int [2]); 
            fEventVaild[fEventVaild.size()-1][0] = index[0];
            fEventVaild[fEventVaild.size()-1][1] = index[1];                    
         }            
      }    
   }   
}  

void YAlignment::EventCheck(TTree* tree)
{
   std::cout<<"[YAlignment] EventCheck"<<std::endl;
   EventData* b_event = new EventData();  tree->SetBranchAddress("event",      &b_event);
   for (int i = 0; i < tree->GetEntriesFast(); i++) {
      tree->GetEntry(i); 
      std::cout<<"[YAlignment] Event : "<<i<<" nTracks : "<<b_event->GetNtracks()<<std::endl;
      for(int j=0; j<b_event->GetNtracks() ;j++){ 
         std::cout<<"[YAlignment] Track : "<<j<<std::endl;
         TrackData *b_track = (TrackData *) b_event->GetTrack()->At(j);      
         for(int k = 0; k<nLAYER; k++){      
            std::cout<<"[YAlignment] layer stave chip ID "<<k<<" "<<b_track->Stave[k]<<" "<<b_track->Chip[k]<<" "<<b_track->ChipID[k]<<" "<<std::endl;
         }  
      }   
   }
}
////////////////////////////////////////////////////////////////////////////////
/// PrepareData

void YAlignment::PrepareData(int nentries=10000, int parallel = 0, bool build = true, TString selectedevents = "")
{
   std::cout<<"YAlignment::PrepareData START"<<std::endl;
   if(build==true){
      gSystem->mkdir(fDirectory_name);
      //gSystem->mkdir(fAnalyze_Directory_name);   
      gSystem->mkdir(fXXXXtrain_Directory_name);
      gSystem->mkdir(fweights_Directory_name);
      gSystem->mkdir(flosscurve_Directory_name);    
   }

   TString s_X1 = "X1";		
   TString s_X2 = "X2";		
   TString s_X3 = "X3";		
   TString s_P1 = "P1";		
   TString s_P2 = "P2";		
   TString s_P3 = "P3";	
   TString s_evno = "evno";
   TString s_NT   = "ntracks";

   TString fInputlayer  = "";
   TString fOutputlayer = "";
   TString fAddlayer = "";
   TString Comma = ",";
   for(int l =0; l< nLAYER; l++){
      fInputlayer  += Comma + "s1["    + TString::Itoa(l,10) + "]" 
                    + Comma + "s2["    + TString::Itoa(l,10) + "]";
      fOutputlayer += Comma + "Stave[" + TString::Itoa(l,10) + "]"  
                    + Comma + "Chip["  + TString::Itoa(l,10) + "]"  
                    + Comma + "ChipID["+ TString::Itoa(l,10) + "]";                      
   }
                  
   fAddlayer += "," + s_X1 + "," + s_X2 + "," + s_X3
              + "," + s_P1 + "," + s_P2 + "," + s_P3
              + "," + s_evno + "," + s_NT;                   
   
   fInputlayer.Replace(0,1,"");
   fOutputlayer.Replace(0,1,"");
   fAddlayer.Replace(0,1,"");
   
   fNetworkStructure = fInputlayer + fHiddenlayer + fOutputlayer + "|" + fAddlayer;
   std::cout<<"fNetworkStructure : "<<fNetworkStructure<<std::endl;
   
   if(selectedevents!="") {
      fXXXXtrain_Directory_name = selectedevents;
   } else {

// Source data
      fSourceData = new TFile(fSourceDataName,"read");
      fSourceTree = (TTree *) fSourceData->Get(fSourceTreeName); 
   
      //std::cout<<"[Debug]Step A"<<std::endl; 
      EventIndex(fSourceTree);
      //std::cout<<"[Debug]Step B"<<std::endl; 
  
      EventData* s_event = new EventData();  
      fSourceTree->SetBranchAddress("event",      &s_event); 

      //Total # of Initial Condition nRseed by Stave & Chip Choice
      
      //int NLastLayer = 9;    
      //int Nfitparam = 6;         
               
      //std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
      //std::chrono::duration<double> sec = std::chrono::system_clock::now() - start;
      //std::cout << "Running Time : " << sec.count() << " seconds" << std::endl;  
      TString sInputTree = "";

      TString Colon = ":";
      for(int l =0; l< nLAYER; l++){
         sInputTree  += Colon + "s1["    + TString::Itoa(l,10) + "]"
                      + Colon + "s2["    + TString::Itoa(l,10) + "]"
                      + Colon + "s3["    + TString::Itoa(l,10) + "]";                     
      }
      for(int l =0; l< nLAYER; l++){
         sInputTree += Colon + "Stave[" + TString::Itoa(l,10) + "]"  
                     + Colon + "Chip["  + TString::Itoa(l,10) + "]"  
                     + Colon + "ChipID["+ TString::Itoa(l,10) + "]";                      
      }                

      sInputTree += ":" + s_X1 + ":" + s_X2 + ":" + s_X3
                  + ":" + s_P1 + ":" + s_P2 + ":" + s_P3;                   
   
      sInputTree += ":" + s_evno + ":" + s_NT;
   
      sInputTree.Replace(0,1,"");
      //int Size_sInputTree = 27; //3 + 9 + 9+ 6
   
      std::cout<<"fMode = "<<nTrackMax<<std::endl;    
      std::cout<<"InputTree Structure "<< sInputTree <<std::endl;

      fOutputFile = nullptr;   
      fOutputFile = new TFile("XXXXtrain.root","recreate");

      fInputTree = nullptr;
      fInputTree = new TTree("InputTree","InputTree");    

      EventData* b_event = new EventData();  
      fInputTree->Branch("event",      &b_event);

      int pinput    = 0;
      int poutput   = pinput  + 9;
      int paddition = poutput + 9;	    
      int pntrack   = paddition + 6; 
   
      int sel_ievent = 0;    

      //double input_Max[2][nSensors];
      //double input_Min[2][nSensors];
      //double norm[2][nSensors];

      double** input_Max;
      double** input_Min;
      double** norm;
   
      input_Max = new double *[2];
      input_Min = new double *[2];  
      norm      = new double *[2];    
      for(int axis = 0; axis <2; axis++){
         input_Max[axis] = new double [nSensors];
         input_Min[axis] = new double [nSensors];  
         norm[axis]      = new double [nSensors];  
         for(int iID=0; iID<nSensors; iID++){  
            input_Max[axis][iID] = 0;
            input_Min[axis][iID] = 0; 
            norm[axis][iID]      = 0; 
         }    
      }

      double z_loc_max[nLAYER];
      double z_loc_min[nLAYER];

      for(int layer=0; layer<nLAYER; layer++){
         z_loc_max[layer] = 0;
         z_loc_min[layer] = 0;    
      }

      for(int iID=0; iID<nSensors; iID++){
         double ip1 = yGEOM->GToS(iID,yGEOM->LToG(iID,0,0)(0),	
 				      yGEOM->LToG(iID,0,0)(1),
                              	      yGEOM->LToG(iID,0,0)(2))(0);
         double fp1 = yGEOM->GToS(iID,yGEOM->LToG(iID,511,1023)(0),
	                              yGEOM->LToG(iID,511,1023)(1),
	      	     	              yGEOM->LToG(iID,511,1023)(2))(0); 
         double ip2 = yGEOM->GToS(iID,yGEOM->LToG(iID,0,0)(0),
                                      yGEOM->LToG(iID,0,0)(1),
                                      yGEOM->LToG(iID,0,0)(2))(1);
         double fp2 = yGEOM->GToS(iID,yGEOM->LToG(iID,511,1023)(0),	
                                      yGEOM->LToG(iID,511,1023)(1),
                                      yGEOM->LToG(iID,511,1023)(2))(1);               
         input_Max[0][iID]=std::max(ip1,fp1);
         input_Min[0][iID]=std::min(ip1,fp1); 
         input_Max[1][iID]=std::max(ip2,fp2);
         input_Min[1][iID]=std::min(ip2,fp2);             
         norm[0][iID]=input_Max[0][iID]-input_Min[0][iID];          
         norm[1][iID]=input_Max[1][iID]-input_Min[1][iID];  
#ifdef YALIGNDEBUG                                              
         std::cout<<"Sensor Boundary ChipID["<<iID<<"] "<<input_Max[0][iID]<<" "<<input_Min[0][iID]<<" "
                                                        <<input_Max[1][iID]<<" "<<input_Min[1][iID]<<" Norm (s1, s2) "<<norm[0][iID]<<" "<<norm[1][iID]<<std::endl;
#endif

         int layer = yGEOM->GetLayer(iID); 
         double z_loc = yGEOM->LToG(iID,256,512)(2);
         z_loc_max[layer] = std::max(z_loc_max[layer],z_loc);
         z_loc_min[layer] = std::min(z_loc_min[layer],z_loc);
      }  
                                    //0  1    2    3    4     5     6      7
      int SensorBoundary[nLAYER + 1] = { 0, 108, 252, 432, 3120, 6480, 14712, 24120 };	

      int check_track = 0;
      for(int a=0; a<nentries/*fSourceIndex.size()*/; a++){

         //fSourceTree->GetEntry(a);
         //if(!fSourceTree->GetEntry(a)) break;

         fSourceTree->GetEntry(a + parallel*nentries);
         if(!fSourceTree->GetEntry(a + parallel*nentries)) break;

         b_event->GetTrack()->Clear();   
         b_event->SetNtracks(0);  
         //std::cout<<"fSourceIndex : "<<a<<" ntracks : "<<s_event->GetNtracks()<<" check_track : "<<check_track<<std::endl;
         if(s_event->GetNtracks()<nTrackMax) { 
            //std::cout<<" [SKIP] Fewer Tracks than Targets"<<std::endl;
            continue;      
         }    
         if(s_event->GetNvtx()>1) {
            //std::cout<<" [SKIP] More than One Event Associated in One TimeFrame"<<std::endl;
            continue;  
         }
         //double b_WE = GetEventWeight(s_event, z_loc_max, z_loc_min)/s_event->GetNtracks();
         //if((double)b_WE<3.0) {
            //std::cout<<" [SKIP] Event Weight Cut Applied"<<std::endl;
            //continue; 
         //}      
      
         int itrack = 0;    
         double vtxZ_event = s_event->GetX3();
         int    Ntracks    = s_event->GetNtracks();
         double vtxZ_trackArr[Ntracks];    
         int    vtxZ_indexArr[Ntracks];
      
         vector<int> vtxZ[Ntracks];
         for(int x=0; x<Ntracks ;x++){ 
            TrackData *s_track = (TrackData *) s_event->GetTrack()->At(x);             
            vtxZ_trackArr[x] = vtxZ_event;//s_track->tv3_X0;
            vtxZ_indexArr[x] = -1;
         }
         bool vertexclustering =false;
      
         if(vertexclustering==true){
            bool ascending = (a%2==0) ? true : false;
            if(Ntracks>=6) yFIT->Clustering(vtxZ_trackArr,vtxZ_indexArr,Ntracks, YFitModel::kDBSCAN, 0.1, ascending);
            else yFIT->Clustering(vtxZ_trackArr,vtxZ_indexArr,Ntracks, YFitModel::kDBSCAN, 0.5, ascending);

            for(int x=0; x<Ntracks ;x++){ 
               if(vtxZ_indexArr[x]>=0) vtxZ[vtxZ_indexArr[x]].push_back(x);
            }
         } else {
            for(int x=0; x<Ntracks ;x++){ 
               if(std::abs(vtxZ_event-vtxZ_trackArr[x])<0.25) vtxZ[0].push_back(x);
            }
         }
#ifdef YALIGNVTXDEBUG       
         std::cout<<"Check Vertex Clustering"<<std::endl;
#endif      
         struct vtxGroup{
            int index[nTrackMax];
         };

         vector<vtxGroup> vGroup;
         TRandom3 rnd(a + 1 + parallel*nentries);      
         if(a%2==0){
#ifdef YALIGNVTXDEBUG       
            std::cout<<"vtx grouping (a)"<<std::endl;
#endif         
            for(int x=0; x<Ntracks ;x++){ 
               if(vtxZ[x].size()==0) continue;

               //shuffle
               int vtxIndex[vtxZ[x].size()];
               int f1, f2;
               int fntr = vtxZ[x].size() - 1;
               for(int f = 0; f < vtxZ[x].size(); f++) {
                  vtxIndex[f] = f;
               }      
               for(int f = 0; f < vtxZ[x].size(); f++) {
                  f1 = (int) (rnd.Rndm() * fntr);
                  f2 = vtxIndex[f1];
                  vtxIndex[f1] = vtxIndex[f];
                  vtxIndex[f] = f2;
               }
               //shuffle

               int Idx = 0; 
               vtxGroup b_vGroup;
               int vcnt = 0;
               for(int f=0; f<vtxZ[x].size();f++){
                  int v = vtxIndex[f];
#ifdef YALIGNVTXDEBUG 
                  std::cout<<"v["<<x<<"]-["<<f<<"]:["<<v<<"], Index["<<vtxZ[x][v]<<"], Vz = "<<vtxZ_trackArr[vtxZ[x][v]]<<" dVz = "<<vtxZ_event-vtxZ_trackArr[vtxZ[x][v]]<<std::endl;
#endif
                  b_vGroup.index[Idx] = vtxZ[x][v];
                  if(vcnt%nTrackMax==(nTrackMax-1)) {
                     vGroup.push_back(b_vGroup);
                     for(int ntr = 0; ntr < nTrackMax; ntr++) b_vGroup.index[ntr] = 0;   
                     Idx=0;                          
                  } else Idx++;
                  vcnt++;
               }
            }
         } else {
#ifdef YALIGNVTXDEBUG
            std::cout<<"vtx grouping (b)"<<std::endl;  
#endif             
            for(int x=0; x<Ntracks ;x++){ 
               if(vtxZ[x].size()==0) continue;
            
               //shuffle
               int vtxIndex[vtxZ[x].size()];
               int f1, f2;
               int fntr = vtxZ[x].size() - 1;
               for(int f = 0; f < vtxZ[x].size(); f++) {
                  vtxIndex[f] = f;
               }      
               for(int f = 0; f < vtxZ[x].size(); f++) {
                  f1 = (int) (rnd.Rndm() * fntr);
                  f2 = vtxIndex[f1];
                  vtxIndex[f1] = vtxIndex[f];
                  vtxIndex[f] = f2;
               }
               //shuffle
            
               int Idx = 0; 
               vtxGroup b_vGroup;
               //for(int v=0; v<vtxZ[x].size();v++){
               int vcnt = 0;            
               for(int f=vtxZ[x].size()-1; f>=0; f--){   
                  int v = vtxIndex[f];   
#ifdef YALIGNVTXDEBUG                                        
                  std::cout<<"v["<<x<<"]-["<<f<<"]:["<<v<<"], Index["<<vtxZ[x][v]<<"], Vz = "<<vtxZ_trackArr[vtxZ[x][v]]<<" dVz = "<<vtxZ_event-vtxZ_trackArr[vtxZ[x][v]]<<std::endl;
#endif               
                  b_vGroup.index[Idx] = vtxZ[x][v];
                  if(vcnt%nTrackMax==(nTrackMax-1)) {
                     vGroup.push_back(b_vGroup);
                     for(int ntr = 0; ntr < nTrackMax; ntr++) b_vGroup.index[ntr] = 0;      
                     Idx=0;                          
                  } else Idx++;
                  vcnt++;
               }
            }   
         }
#ifdef YALIGNVTXDEBUG       
         for(int g=0; g<vGroup.size() ;g++){ 
            for(int v=0; v<nTrackMax;v++){
               std::cout<<"*v["<<g<<"]-["<<v<<"], Index["<<vGroup[g].index[v]<<"], Vz = "<<vtxZ_trackArr[vGroup[g].index[v]]<<" dVz = "<<vtxZ_event-vtxZ_trackArr[vGroup[g].index[v]]<<std::endl;
            }
         }
#endif
         //continue;      
  
         for(int g=0; g<vGroup.size() ;g++){ 
            bool IsEventSelect = false; 
            bool IsTrackSelect = false;     
            int itrack=0;
            b_event->GetTrack()->Clear();
            b_event->SetNtracks(0);           
            for(int v=0; v<nTrackMax;v++){
               int x = vGroup[g].index[v];
            
               TrackData *s_track = (TrackData *) s_event->GetTrack()->At(x);                 
               double vtxZ_track = s_track->tv3_X0;  
              
               int totselect = 0;             
               bool validhit[] = {true, true, true, true, true, true, true}; 
               for(int c = 0; c < nLAYER; c++){      
#ifdef YALIGNDEBUG 
                  std::cout<<" LAYER(G  Check) "<<x<<" "<<c<<" "<<s_track->Layer[c]<<" "<<s_track->s1[c]<<" "<<s_track->s2[c]<<" "<<s_track->s3[c]<<std::endl;
#endif
                  if(s_track->ChipID[c]<0) {
                     validhit[c] = false;       
                     continue;
                  }            
                                    
                  if(std::abs(s_track->s1[c])>1000 || std::abs(s_track->s2[c])>1000 || std::abs(s_track->s3[c])>1000) continue;
                  TVector3 sensorS  = yGEOM->GToS(s_track->ChipID[c],yGEOM->LToG(s_track->ChipID[c],s_track->row[c] + 0.5,s_track->col[c] + 0.5)(0),
                                                                     yGEOM->LToG(s_track->ChipID[c],s_track->row[c] + 0.5,s_track->col[c] + 0.5)(1),
                                                                     yGEOM->LToG(s_track->ChipID[c],s_track->row[c] + 0.5,s_track->col[c] + 0.5)(2));

                  TVector3 sensorSG = yGEOM->SToG(s_track->ChipID[c],sensorS(0),sensorS(1),sensorS(2));
#ifdef YALIGNDEBUG
                  std::cout<<" LAYER(S  Check) "<<x<<" "<<c<<" "<<s_track->Layer[c]<<" "<<sensorS(0)<<" "<<sensorS(1)<<" "<<sensorS(2)<<std::endl;
                  std::cout<<" LAYER(SG Check) "<<x<<" "<<c<<" "<<s_track->Layer[c]<<" "<<sensorSG(0)<<" "<<sensorSG(1)<<" "<<sensorSG(2)<<std::endl;            
#endif

                  // s1 axis 
                  if(sensorS(0)<input_Min[0][s_track->ChipID[c]]||
                     sensorS(0)>input_Max[0][s_track->ChipID[c]]) {
                     validhit[c] = false;
                     //std::cout<<" *** Track Unselect by Sensor Range Error s1 *** "<<std::endl;
                  }
                  // s2 axis 
                  if(sensorS(1)<input_Min[1][s_track->ChipID[c]]||
                     sensorS(1)>input_Max[1][s_track->ChipID[c]]) {
                     validhit[c] = false;
                     //std::cout<<" *** Track Unselect by Sensor Range Error s2 *** "<<std::endl;
                  }
                  // s3 axis 
                  if((double)sensorS(2)<(double)-1.0e-4||
                     (double)sensorS(2)>(double)+1.0e-4) {
                     validhit[c] = false;
                     //std::cout<<" *** Track Unselect by Sensor Range Error s3 *** "<<std::endl;
                  }
               
                  if(validhit[c]==true) totselect++;
               }       
               if(totselect<5) continue;
               b_event->AddOneTrack();  
               TrackData *b_track = (TrackData *) b_event->GetTrack()->At(itrack++);      
               b_track->p		= s_track->p;
               b_track->pt		= s_track->pt;
               b_track->theta	= s_track->theta;
               b_track->phi	= s_track->phi;
               b_track->eta	= s_track->eta;
               b_track->tv1	= s_track->tv1;
               b_track->tv2      	= s_track->tv2;
               b_track->tv3      = s_track->tv3;
               b_track->tv1_X0   = s_track->tv1_X0;	
               b_track->tv2_X0   = s_track->tv2_X0;	
               b_track->tv3_X0   = s_track->tv3_X0;
               b_track->tv1_DCA  = s_track->tv1_DCA;	
               b_track->tv2_DCA  = s_track->tv2_DCA; 	
               b_track->tv3_DCA  = s_track->tv3_DCA;  
            
               for(int c = 0; c < nLAYER; c++){  
                  if(validhit[c]==true){                    
                                       
                     TVector3 sensorS = yGEOM->GToS(s_track->ChipID[c],yGEOM->LToG(s_track->ChipID[c],s_track->row[c] + 0.5,s_track->col[c] + 0.5)(0),
                                                                       yGEOM->LToG(s_track->ChipID[c],s_track->row[c] + 0.5,s_track->col[c] + 0.5)(1),
                                                                       yGEOM->LToG(s_track->ChipID[c],s_track->row[c] + 0.5,s_track->col[c] + 0.5)(2));

#ifdef YALIGNDEBUG 
                     std::cout<<"ChipID "<<c<<" "<<s_track->ChipID[c]<<std::endl;
                     std::cout<<" LAYER(I) "<<c<<" "<<sensorS.X()<<" "<<sensorS.Y()<<" "<<sensorS.Z()<<std::endl;
                     std::cout<<" LAYER(IO) "<<c<<" "<<sensorS.X()-input_Min[0][s_track->ChipID[c]]<<" "<<sensorS.Y()-input_Min[1][s_track->ChipID[c]]<<" "<<sensorS.Z()<<std::endl;
#endif
                     b_track->s1[c]	= (float)((sensorS(0)-input_Min[0][s_track->ChipID[c]])/norm[0][s_track->ChipID[c]] - fNorm_shift);
                     b_track->s2[c]	= (float)((sensorS(1)-input_Min[1][s_track->ChipID[c]])/norm[1][s_track->ChipID[c]] - fNorm_shift);
                     b_track->s3[c]	= (float)sensorS(2);  
                     b_track->Stave[c]   = (int)s_track->Stave[c];
                     b_track->Chip[c]    = (int)s_track->Chip[c];
                     b_track->ChipID[c]  = (int)s_track->ChipID[c];

                     b_track->row[c]     = (float)s_track->row[c];
                     b_track->col[c]     = (float)s_track->col[c];  

                     int layer       = yGEOM->GetLayer(s_track->ChipID[c]);
                     int staveID     = yGEOM->GetStave(s_track->ChipID[c]);
                     int hs          = yGEOM->GetHalfStave(s_track->ChipID[c]); 
                     int chipIDstave = yGEOM->GetChipIdInStave(s_track->ChipID[c]);  
               
                     // skimming
                     if(fSplitReferenceSensor==-1){
                        if(fNetworkUpdateList[s_track->ChipID[c]]==true) IsTrackSelect = true;    
                     } else {
                        if(fNetworkUpdateList[s_track->ChipID[c]]==true && fSplitReferenceSensor==s_track->ChipID[c]) IsTrackSelect = true;    
                     }
                  } else {
                     b_track->s1[c]	= (float)-9999;
                     b_track->s2[c]	= (float)-9999;
                     b_track->s3[c]	= (float)-9999;
                     b_track->Stave[c]   = (int)-9999;
                     b_track->Chip[c]    = (int)-9999;
                     b_track->ChipID[c]  = (int)-9999;
                  
                     b_track->row[c]     = (float)-9999;
                     b_track->col[c]     = (float)-9999;                     
                  }
#ifdef YALIGNDEBUG
                  std::cout<<" LAYER(N) "<<c<<" "<<b_track->s1[c]<<" "<<b_track->s2[c]<<" "<<b_track->s3[c]<<" "<<b_track->Stave[c]<<" "<<b_track->Chip[c]<<" "<<b_track->ChipID[c]<<std::endl;
                  std::cout<<" LAYER(G) "<<c<<" "<<s_track->s1[c]<<" "<<s_track->s2[c]<<" "<<s_track->s3[c]<<std::endl;
                  std::cout<<" LAYER(P) "<<c<<" "<<yGEOM->LToG(b_track->ChipID[c],s_track->row[c] + 0.5,s_track->col[c] + 0.5)(0)<<" "
                                                 <<yGEOM->LToG(b_track->ChipID[c],s_track->row[c] + 0.5,s_track->col[c] + 0.5)(1)<<" "
                                                 <<yGEOM->LToG(b_track->ChipID[c],s_track->row[c] + 0.5,s_track->col[c] + 0.5)(2)<<std::endl;
#endif
               }   
            }

            if(itrack==nTrackMax) IsEventSelect = true; 
            if(IsEventSelect==true && IsTrackSelect==true) {
#ifdef YALIGNDEBUG 
               std::cout<<"Event Selected "<<sel_ievent<<std::endl;
#endif
               b_event->SetEvno(sel_ievent++);
               b_event->SetWE(0);
               b_event->SetNtracks(itrack);  
               b_event->SetNvtx(s_event->GetNvtx());
               b_event->SetX1(s_event->GetX1()); 
               b_event->SetX2(s_event->GetX2());
               b_event->SetX3(s_event->GetX3());      
               b_event->SetP1(s_event->GetP1()); 
               b_event->SetP2(s_event->GetP2());
               b_event->SetP3(s_event->GetP3());
          
               fInputTree->Fill();      
            }            
         }          
      }
#ifdef YALIGNDEBUG
      std::cout<<"Selected Events : "<<fInputTree->GetEntriesFast()<<std::endl;
#endif
      EndOfPrepareData();
      std::cout<<"EventIndex InputTree"<<std::endl;
      EventIndex(fInputTree);
      std::cout<<"End EventIndex fInputTree"<<std::endl;
   } 

}


////////////////////////////////////////////////////////////////////////////////
/// EndOfPrepareData

void YAlignment::EndOfPrepareData(){
   fOutputFile->cd();
   fOutputFile->Write();  
   TString XXXXtrain_name = "XXXXtrain.root";    
   TString fExec_argument = "mv " + XXXXtrain_name + " " + fXXXXtrain_Directory_name + "/" + XXXXtrain_name;
   gSystem->Exec(fExec_argument);
}

////////////////////////////////////////////////////////////////////////////////
/// GetEventWeight

double YAlignment::GetEventWeight(EventData* event, double* z_loc_max, double* z_loc_min)
{

   int hitentries = nLAYER;
   int trackentries = event->GetNtracks();
  
   double Weight_Event = 0;      
   for(int imode = 0; imode < trackentries; imode++){

      TrackData *b_track = (TrackData *) event->GetTrack()->At(imode);  
      int chipID[nLAYER+1]; 
 
      double Weight_Track;
      double Weight_Hit[nLAYER];       
    
      chipID[nLAYER] = -1;      
      Weight_Track = 0;                                                                                                          
      for(int layer = 0; layer<nLAYER; layer++){  
         chipID[layer]  = (int)b_track->ChipID[layer]; 
         Weight_Hit[layer] = 0;
      }     
                    
      for(int layer = 0; layer < nLAYER; layer++){           
         double c0 = 0.1;
         double c1 = 0.9*(2/TMath::Abs(z_loc_max[layer]-z_loc_min[layer]));
         double z_loc = yGEOM->LToG(chipID[layer],256,512)(2);
             
         Weight_Hit[layer] = c0 + c1*TMath::Abs(z_loc);
         Weight_Track     += Weight_Hit[layer];
      }            
      Weight_Event += Weight_Track;
   }
   return Weight_Event;
}

////////////////////////////////////////////////////////////////////////////////
/// LoadData

void YAlignment::LoadData(int nentries=10000, int parallel = 0, bool build = true)
{
   std::cout<<"YAlignment:LoadData START"<<std::endl;
   fDirectory_name = "Data";
   fXXXXtrain_Directory_name = "Data/XXXXtrain";
   TString XXXXtrain_name    = "XXXXtrain.root";    
   fAnalyze_Directory_name   = "Data/XXXXanalyse"; 
   TString XXXXanalyse_name  = "XXXXanalyse.root";  

   if(build==true){
      gSystem->mkdir(fDirectory_name);
      gSystem->mkdir(fAnalyze_Directory_name);   
      gSystem->mkdir(fXXXXtrain_Directory_name);   
   }
// Source data
   fSourceData = new TFile(fSourceDataName,"read");
   fSourceTree = (TTree *) fSourceData->Get(fSourceTreeName); 

   //std::cout<<"[Debug]Step A"<<std::endl; 
   EventIndex(fSourceTree);
   //std::cout<<"[Debug]Step B"<<std::endl; 
  
   EventData* s_event = new EventData();  
   fSourceTree->SetBranchAddress("event",      &s_event); 

   TString s_X1 = "X1";		
   TString s_X2 = "X2";		
   TString s_X3 = "X3";		
   TString s_P1 = "P1";		
   TString s_P2 = "P2";		
   TString s_P3 = "P3";	
   TString s_evno = "evno";
   TString s_NT   = "ntracks";

   TString sInputTree = "";

   TString Colon = ":";
   for(int l =0; l< nLAYER; l++){
      sInputTree  += Colon + "s1["    + TString::Itoa(l,10) + "]"
                   + Colon + "s2["    + TString::Itoa(l,10) + "]"
                   + Colon + "s3["    + TString::Itoa(l,10) + "]";                     
   }
   for(int l =0; l< nLAYER; l++){
      sInputTree += Colon + "Stave[" + TString::Itoa(l,10) + "]"  
                  + Colon + "Chip["  + TString::Itoa(l,10) + "]"  
                  + Colon + "ChipID["+ TString::Itoa(l,10) + "]";                      
   }                

   sInputTree += ":" + s_X1 + ":" + s_X2 + ":" + s_X3
               + ":" + s_P1 + ":" + s_P2 + ":" + s_P3;                   
   
   sInputTree += ":" + s_evno + ":" + s_NT;
   
   sInputTree.Replace(0,1,"");
   
   std::cout<<"fMode = "<<nTrackMax<<std::endl;    
   std::cout<<"InputTree Structure "<< sInputTree <<std::endl;

   fOutputFile = nullptr;   
   fOutputFile = new TFile(XXXXtrain_name,"recreate");

   fInputTree = nullptr;
   fInputTree = new TTree("InputTree","InputTree");    

   EventData* b_event = new EventData();  
   fInputTree->Branch("event",      &b_event);
   
   int pinput    = 0;
   int poutput   = pinput  + 9;
   int paddition = poutput + 9;	    
   int pntrack   = paddition + 6; 
   
   int sel_ievent = 0;    

   double** input_Max;
   double** input_Min;
   double** norm;
   
   input_Max = new double *[2];
   input_Min = new double *[2];  
   norm      = new double *[2];    
   for(int axis = 0; axis <2; axis++){
      input_Max[axis] = new double [nSensors];
      input_Min[axis] = new double [nSensors];  
      norm[axis]      = new double [nSensors];  
      for(int iID=0; iID<nSensors; iID++){  
         input_Max[axis][iID] = 0;
         input_Min[axis][iID] = 0; 
         norm[axis][iID]      = 0; 
      }    
   }

   double z_loc_max[nLAYER];
   double z_loc_min[nLAYER];

   for(int layer=0; layer<nLAYER; layer++){
      z_loc_max[layer] = 0;
      z_loc_min[layer] = 0;    
   }

   for(int iID=0; iID<nSensors; iID++){
      double ip1 = yGEOM->GToS(iID,yGEOM->LToG(iID,0,0)(0),	
 				   yGEOM->LToG(iID,0,0)(1),
                        	   yGEOM->LToG(iID,0,0)(2))(0);
      double fp1 = yGEOM->GToS(iID,yGEOM->LToG(iID,511,1023)(0),
	                           yGEOM->LToG(iID,511,1023)(1),
		     	           yGEOM->LToG(iID,511,1023)(2))(0); 
      double ip2 = yGEOM->GToS(iID,yGEOM->LToG(iID,0,0)(0),
                                   yGEOM->LToG(iID,0,0)(1),
                                   yGEOM->LToG(iID,0,0)(2))(1);
      double fp2 = yGEOM->GToS(iID,yGEOM->LToG(iID,511,1023)(0),	
                                   yGEOM->LToG(iID,511,1023)(1),
                                   yGEOM->LToG(iID,511,1023)(2))(1);               
      input_Max[0][iID]=std::max(ip1,fp1);
      input_Min[0][iID]=std::min(ip1,fp1); 
      input_Max[1][iID]=std::max(ip2,fp2);
      input_Min[1][iID]=std::min(ip2,fp2);             
      norm[0][iID]=input_Max[0][iID]-input_Min[0][iID];          
      norm[1][iID]=input_Max[1][iID]-input_Min[1][iID];  
#ifdef YALIGNDEBUG                                              
      std::cout<<"Sensor Boundary ChipID["<<iID<<"] "<<input_Max[0][iID]<<" "<<input_Min[0][iID]<<" "
                                                     <<input_Max[1][iID]<<" "<<input_Min[1][iID]<<" Norm (s1, s2) "<<norm[0][iID]<<" "<<norm[1][iID]<<std::endl;
#endif

      int layer = yGEOM->GetLayer(iID); 
      double z_loc = yGEOM->LToG(iID,256,512)(2);
      z_loc_max[layer] = std::max(z_loc_max[layer],z_loc);
      z_loc_min[layer] = std::min(z_loc_min[layer],z_loc);
   }  

#ifdef YALIGNDEBUG
   std::cout<<"[YAlignment] Sensor Z Range"<<std::endl;
   for(int layer=0; layer<nLAYER; layer++){
      std::cout<<"layer "<<layer<<" max "<<z_loc_max[layer]<<" min "<<z_loc_min[layer]<<std::endl;    
   }
#endif
                                    //0  1    2    3    4     5     6      7
   int SensorBoundary[nLAYER + 1] = { 0, 108, 252, 432, 3120, 6480, 14712, 24120 };	

   for(int a=0; a<nentries; a++){
      fSourceTree->GetEntry(a);
      if(!fSourceTree->GetEntry(a)) break;
      b_event->GetTrack()->Clear();   
      b_event->SetNtracks(0);  
      std::cout<<"fSourceIndex : "<<a<<" ntracks : "<<s_event->GetNtracks()<<" "<<fSourceTree->GetEntry(a)<<std::endl;
      if(s_event->GetNtracks()<1) { 
         std::cout<<" [SKIP] Fewer Tracks than Targets"<<std::endl;
         continue;      
      }    
      
      int itrack = 0;    
      bool IsEventSelect = false;
      double vtxZ_event = s_event->GetX3();
 
      for(int x=0; x<s_event->GetNtracks() ;x++){ 
         if(itrack>=1) {
            IsEventSelect = true;
         } 
         TrackData *s_track = (TrackData *) s_event->GetTrack()->At(x);             
         int totselect = 0;         
         bool IsTrackSelect = true;
         double vtxZ_track = s_track->tv3_X0;
         if(TMath::Abs(vtxZ_event-vtxZ_track)>0.5){
            IsTrackSelect = false;
            std::cout<<"Mismatch VtxZ Event "<<vtxZ_event<<" Track "<<vtxZ_track<<" Deviation "<<vtxZ_event-vtxZ_track<<std::endl;
            continue;
         }                  
         for(int c = 0; c < nLAYER; c++){      
#ifdef YALIGNDEBUG 
            std::cout<<" LAYER(G Check) "<<x<<" "<<c<<" "<<s_track->Layer[c]<<" "<<s_track->s1[c]<<" "<<s_track->s2[c]<<" "<<s_track->s3[c]<<std::endl;
#endif

            if(s_track->ChipID[c]<0) continue;
            if(s_track->Layer[c]>=0) totselect++;            
            if(std::abs(s_track->s1[c])>1000 || std::abs(s_track->s2[c])>1000 || std::abs(s_track->s3[c])>1000) continue;
            TVector3 sensorS  = yGEOM->GToS(s_track->ChipID[c],s_track->s1[c],s_track->s2[c],s_track->s3[c]);
            //if(s_track->ChipID[c]==4){ //layer 0, stave 0
            //   std::cout<<" LAYER(D) "<<c<<" "<<sensorS.X()<<" "<<sensorS.Y()<<" "<<sensorS.Z()<<std::endl;
            //   TVector3 dsensorS(0.05,0,0);
            //   sensorS = sensorS - dsensorS;
            //}
            TVector3 sensorSG = yGEOM->SToG(s_track->ChipID[c],sensorS(0),sensorS(1),sensorS(2));
#ifdef YALIGNDEBUG
            std::cout<<" LAYER(S Check) "<<x<<" "<<c<<" "<<s_track->Layer[c]<<" "<<sensorS(0)<<" "<<sensorS(1)<<" "<<sensorS(2)<<std::endl;
#endif
            // s1 axis 
            if(sensorS(0)<input_Min[0][s_track->ChipID[c]]||
               sensorS(0)>input_Max[0][s_track->ChipID[c]]) {
               IsTrackSelect = false;
               //std::cout<<" *** Track Unselect by Sensor Range Error s1 *** "<<std::endl;
               //break;
            }
            // s2 axis 
            if(sensorS(1)<input_Min[1][s_track->ChipID[c]]||
               sensorS(1)>input_Max[1][s_track->ChipID[c]]) {
               IsTrackSelect = false;
               //std::cout<<" *** Track Unselect by Sensor Range Error s2 *** "<<std::endl;
               //break;
            }
            // s3 axis 
            if((double)sensorS(2)<(double)-5.0e-5||
               (double)sensorS(2)>(double)+5.0e-5) {
               IsTrackSelect = false;
               //std::cout<<" *** Track Unselect by Sensor Range Error s3 *** "<<std::endl;
               //break;
            }
    
         }

         if((totselect<5)||(IsTrackSelect==false)){
            if(itrack>=2) IsEventSelect = true;      
            continue;
         }
         b_event->AddOneTrack();  
         TrackData *b_track = (TrackData *) b_event->GetTrack()->At(itrack++);      
         b_track->p	=s_track->p;
         b_track->pt	=s_track->pt;
         b_track->theta	=s_track->theta;
         b_track->phi	=s_track->phi;
         b_track->eta	=s_track->eta;
         b_track->tv1	   = s_track->tv1;
         b_track->tv2      = s_track->tv2;
         b_track->tv3      = s_track->tv3;
         b_track->tv1_X0   = s_track->tv1_X0;	
         b_track->tv2_X0   = s_track->tv2_X0;	
         b_track->tv3_X0   = s_track->tv3_X0;
         b_track->tv1_DCA  = s_track->tv1_DCA;	
         b_track->tv2_DCA  = s_track->tv2_DCA; 	
         b_track->tv3_DCA  = s_track->tv3_DCA;          
         //b_track->SetIndex(j); 
         //std::cout<<"sEvent : "<<sel_ievent<<" tEvent : "<<a<<std::endl;
         for(int c = 0; c < nLAYER; c++){      
            TVector3 sensorS = yGEOM->GToS(s_track->ChipID[c],s_track->s1[c],s_track->s2[c],s_track->s3[c]);
            //if(s_track->ChipID[c]<9){ //layer 0, stave 0
            //   std::cout<<" LAYER(D) "<<c<<" "<<sensorS.X()<<" "<<sensorS.Y()<<" "<<sensorS.Z()<<std::endl;
            //   TVector3 dsensorS(0.05,0,0); //for an example.
            //   sensorS = sensorS - dsensorS;
            //}
            //std::cout<<"ChipID "<<c<<" "<<s_track->ChipID[c]<<std::endl;
            //std::cout<<" LAYER(I) "<<c<<" "<<sensorS.X()<<" "<<sensorS.Y()<<" "<<sensorS.Z()<<std::endl;
            b_track->s1[c]	= (float)((sensorS(0)-input_Min[0][s_track->ChipID[c]])/norm[0][s_track->ChipID[c]] - fNorm_shift);
            b_track->s2[c]	= (float)((sensorS(1)-input_Min[1][s_track->ChipID[c]])/norm[1][s_track->ChipID[c]] - fNorm_shift);
            b_track->s3[c]	= (float)sensorS(2);  
            b_track->Stave[c]   = (int)s_track->Stave[c];
            b_track->Chip[c]    = (int)s_track->Chip[c];
            b_track->ChipID[c]  = (int)s_track->ChipID[c];
            //std::cout<<" LAYER(N) "<<c<<" "<<b_track->s1[c]<<" "<<b_track->s2[c]<<" "<<b_track->s3[c]<<" "<<b_track->Stave[c]<<" "<<b_track->Chip[c]<<" "<<b_track->ChipID[c]<<std::endl;
            //std::cout<<" LAYER(G) "<<c<<" "<<s_track->s1[c]<<" "<<s_track->s2[c]<<" "<<s_track->s3[c]<<std::endl;
            //std::cout<<" LAYER(P) "<<c<<" "<<yGEOM->LToG(b_track->ChipID[c],s_track->row[c] + 0.5,s_track->col[c] + 0.5)(0)<<" "
            //                               <<yGEOM->LToG(b_track->ChipID[c],s_track->row[c] + 0.5,s_track->col[c] + 0.5)(1)<<" "
            //                               <<yGEOM->LToG(b_track->ChipID[c],s_track->row[c] + 0.5,s_track->col[c] + 0.5)(2)<<std::endl;
         }                                      
      } 
      if(IsEventSelect==true) {
         //std::cout<<"Event Selected "<<sel_ievent<<std::endl;
         b_event->SetEvno(sel_ievent++);
        
         b_event->SetNvtx(s_event->GetNvtx());
         b_event->SetX1(s_event->GetX1()); 
         b_event->SetX2(s_event->GetX2());
         b_event->SetX3(s_event->GetX3());      
         b_event->SetP1(s_event->GetP1()); 
         b_event->SetP2(s_event->GetP2());
         b_event->SetP3(s_event->GetP3());
          
         fInputTree->Fill();      
      }          
   }

   //std::cout<<"Selected Events : "<<fInputTree->GetEntriesFast()<<std::endl;
   TString fInputlayer  = "";
   TString fOutputlayer = "";
   TString fAddlayer = "";
   TString Comma = ",";
   for(int l =0; l< nLAYER; l++){
      fInputlayer  += Comma + "s1["    + TString::Itoa(l,10) + "]" 
                    + Comma + "s2["    + TString::Itoa(l,10) + "]";
      fOutputlayer += Comma + "Stave[" + TString::Itoa(l,10) + "]"  
                    + Comma + "Chip["  + TString::Itoa(l,10) + "]"  
                    + Comma + "ChipID["+ TString::Itoa(l,10) + "]";                      
   }
                  

   fAddlayer += "," + s_X1 + "," + s_X2 + "," + s_X3
              + "," + s_P1 + "," + s_P2 + "," + s_P3
              + "," + s_evno + "," + s_NT;                   
   
   fInputlayer.Replace(0,1,"");
   fOutputlayer.Replace(0,1,"");
   fAddlayer.Replace(0,1,"");
   
   fNetworkStructure = fInputlayer + fHiddenlayer + fOutputlayer + "|" + fAddlayer;
   std::cout<<"fNetworkStructure : "<<fNetworkStructure<<std::endl;
   EventIndex(fInputTree);
   std::cout<<"End EventIndex fInputTree"<<std::endl;

   EndOfPrepareData();
   TString fTrainFileName = fXXXXtrain_Directory_name + "/XXXXtrain.root";
   TFile*  fTrainData = new TFile(fTrainFileName,"read");
   fInputTree = (TTree *) fTrainData->Get("InputTree"); 

   b_event = new EventData();  
   fInputTree->SetBranchAddress("event",      &b_event); 

   fMLPNetwork = new YMultiLayerPerceptron(fNetworkStructure,fInputTree,"(evno%10)>=0&&(evno%10)<6","(evno%10)>=6&&(evno%10)<8");
   fMLPNetwork->SetNpronged(nTrackMax);  
   fMLPNetwork->SetFitModel(FITMODEL);
 
   fAnalyzeFile = nullptr;   
   fAnalyzeFile = new TFile(XXXXanalyse_name,"recreate");   


   //chipID:layer:halfbarrel:stave:halfstave:module:lchip:schip:hschip:mchip:gSx:gSy:gSr:gSphi:gSz:gS1:gS2:gS3


   TNtuple *fAnalysis_E  
            = new TNtuple("fAnalysis_E",	"fAnalysis_E",
            "epoch:evno:layer:halfbarrel:stave:halfstave:module:lchip:schip:hschip:mchip:Sx:Sy:Sr:Sphi:Sz:chipID:imode:\
             Cs1:Cs2:Cs3:Csd1:Csd2:Csd3:Sigma_s1:Sigma_s2:CMSE:R:P:PT:Theta:Phi:Eta:WH:WT:WE:itrack:ntracks:nvtx");
   TNtuple *fAnalysis_S  
            = new TNtuple("fAnalysis_S",	"fAnalysis_S",	
            "epoch:evno:layer:halfbarrel:stave:halfstave:module:lchip:schip:hschip:mchip:Sx:Sy:Sr:Sphi:Sz:chipID:imode:Cs1:Cs2:Cs3:Csd1:Csd2:Csd3:CMSE:R:P:PT:Theta:Phi:Eta:WH:WT");
   TNtuple *fAnalysis_G  
            = new TNtuple("fAnalysis_G",	"fAnalysis_G",	
            "epoch:evno:layer:chipID:imode:Cgx:Cgy:Cgz:Cgdx:Cgdy:Cgdz:CMSE:R:P:PT:Theta:Phi:Eta");                       
          
   std::cout<<"nInput Tree : "<< fInputTree->GetEntriesFast() <<std::endl;           
   fMLPNetwork->Init_Randomize();
   fMLPNetwork->Init_RandomizeSensorCorrection();
   fMLPNetwork->PrintCurrentWeights();        

   for(int ientry = 0; ientry < fInputTree->GetEntriesFast(); ientry++ ){

      fInputTree->GetEntry(ientry);      
      int hitentries = nLAYER;
      int trackentries = b_event->GetNtracks();
      int vtxentries   = b_event->GetNvtx();
      
      double Weight_Event = 0;
      int nentries_track = 35;      
      vector<Float_t> ntuple_track[nentries_track]; 
      for(int imode = 0; imode < trackentries; imode++){

         TrackData *b_track = (TrackData *) b_event->GetTrack()->At(imode);  
           
         double mlpInput[nLAYER][2];
         double mlpOutput[nLAYER][3];
         double mlpExtended[nLAYER][3];   
         double mlpX[3];  

         double mlp_p        = b_track->p;
         double mlp_pt       = b_track->pt;
         double mlp_theta    = b_track->theta;
         double mlp_phi      = b_track->phi;
         double mlp_eta      = b_track->eta;
         
         std::vector<bool> hitUpdate;
         std::vector<bool> hitUpdate_Z; 
         for(Int_t layer = 0; layer<nLAYER ;layer++){                  
   
            mlpInput[layer][0]   = b_track->s1[layer]; 
            mlpInput[layer][1]   = b_track->s2[layer];
            
            mlpExtended[layer][0]= b_track->Stave[layer];    
            mlpExtended[layer][1]= b_track->Chip[layer];  
            mlpExtended[layer][2]= b_track->ChipID[layer];

            bool layUpdate = b_track->ChipID[layer] < 0 ? false : true;
            hitUpdate.push_back(layUpdate);
            hitUpdate_Z.push_back(layUpdate);            
         }
         hitUpdate.push_back(true);
         
         mlpX[0] = b_event->GetX1();       
         mlpX[1] = b_event->GetX2();       
         mlpX[2] = b_event->GetX3();    
                           
         int stave[nLAYER];
         int chip[nLAYER];    
         int chipID[nLAYER+1]; 

         int sensor_Layer[nLAYER];
         int sensor_HalfBarrel[nLAYER];
         int sensor_Stave[nLAYER];
         int sensor_HalfStave[nLAYER];
         int sensor_Module[nLAYER];
         int sensor_ChipIdInLayer[nLAYER];
         int sensor_ChipIdInStave[nLAYER];
         int sensor_ChipIdInHalfStave[nLAYER];
         int sensor_ChipIdInModule[nLAYER];
 
         double Weight_Track;
         double Weight_Hit[nLAYER];       
    
         chipID[nLAYER] = -1;      
         Weight_Track = 0;                                                                                                          
         for(int layer = 0; layer<nLAYER; layer++){  
            stave[layer] = (int)mlpExtended[layer][0];
            chip[layer]  = (int)mlpExtended[layer][1]; 
            chipID[layer]  = (int)mlpExtended[layer][2]; 

            sensor_Layer[layer]			= yGEOM->GetLayer(chipID[layer]); 
            sensor_HalfBarrel[layer]		= yGEOM->GetHalfBarrel(chipID[layer]);  
            sensor_Stave[layer]			= yGEOM->GetStave(chipID[layer]); 
            sensor_HalfStave[layer]		= yGEOM->GetHalfStave(chipID[layer]);  
            sensor_Module[layer]		= yGEOM->GetModule(chipID[layer]);  
            sensor_ChipIdInLayer[layer]		= yGEOM->GetChipIdInLayer(chipID[layer]); 
            sensor_ChipIdInStave[layer]		= yGEOM->GetChipIdInStave(chipID[layer]); 
            sensor_ChipIdInHalfStave[layer]	= yGEOM->GetChipIdInHalfStave(chipID[layer]); 
            sensor_ChipIdInModule[layer]	= yGEOM->GetChipIdInModule(chipID[layer]); 

            for(int axis = 0; axis<3 ;axis++){
               mlpOutput[layer][axis] = fMLPNetwork->Evaluate(axis, mlpInput[layer],chipID[layer]);
            }  
            Weight_Hit[layer] = 0;
         }     
              
         double pos_S1[2][nLAYER];  // uncorrected : 0, corrected : 1
         double pos_S2[2][nLAYER]; 
         double pos_S3[2][nLAYER];  

         double pos_GX[2][nLAYER];  // uncorrected : 0, corrected : 1
         double pos_GY[2][nLAYER]; 
         double pos_GZ[2][nLAYER]; 
            
         double pos_Sx[nLAYER], pos_Sy[nLAYER], pos_Sr[nLAYER], pos_Sphi[nLAYER], pos_Sz[nLAYER];

         double pos_GC[3*(nLAYER+1)];
#ifdef YALIGNDEBUG   
         std::cout<< "[LoadData] Filling " <<" - "<<ientry<<" - "<<imode<<" :: "<<std::endl;  
#endif          
         TVector3 vecX[nLAYER+1];
         TVector3 vecXmid[nLAYER];
         double InvSlope[nLAYER];            

         for(Int_t layer = 0; layer<nLAYER ;layer++){  

            pos_S1[0][layer]  = (mlpInput[layer][0] + fNorm_shift)*norm[0][chipID[layer]] + input_Min[0][chipID[layer]]; 
            pos_S2[0][layer]  = (mlpInput[layer][1] + fNorm_shift)*norm[1][chipID[layer]] + input_Min[1][chipID[layer]];
            pos_S3[0][layer]  = 0; 
            pos_S1[1][layer] = (mlpInput[layer][0] + fNorm_shift)*norm[0][chipID[layer]] + input_Min[0][chipID[layer]] + mlpOutput[layer][0]; 
            pos_S2[1][layer] = (mlpInput[layer][1] + fNorm_shift)*norm[1][chipID[layer]] + input_Min[1][chipID[layer]] + mlpOutput[layer][1];
            pos_S3[1][layer] = mlpOutput[layer][2];             
                       
            pos_GX[0][layer] = yGEOM->SToG(chipID[layer],pos_S1[0][layer],pos_S2[0][layer],pos_S3[0][layer]).X();
            pos_GY[0][layer] = yGEOM->SToG(chipID[layer],pos_S1[0][layer],pos_S2[0][layer],pos_S3[0][layer]).Y();
            pos_GZ[0][layer] = yGEOM->SToG(chipID[layer],pos_S1[0][layer],pos_S2[0][layer],pos_S3[0][layer]).Z();                               
            pos_GX[1][layer] = yGEOM->SToG(chipID[layer],pos_S1[1][layer],pos_S2[1][layer],pos_S3[1][layer]).X();
            pos_GY[1][layer] = yGEOM->SToG(chipID[layer],pos_S1[1][layer],pos_S2[1][layer],pos_S3[1][layer]).Y();
            pos_GZ[1][layer] = yGEOM->SToG(chipID[layer],pos_S1[1][layer],pos_S2[1][layer],pos_S3[1][layer]).Z();     

            pos_GC[(3*layer)+0] = pos_GX[1][layer];
            pos_GC[(3*layer)+1] = pos_GY[1][layer];
            pos_GC[(3*layer)+2] = pos_GZ[1][layer];       
              
            vecX[layer].SetXYZ(pos_GC[(3*layer)+0],pos_GC[(3*layer)+1],pos_GC[(3*layer)+2]);
#ifdef YALIGNDEBUG
            std::cout << pos_GC[(3*layer)+0] <<" "<< pos_GC[(3*layer)+1] <<" "<< pos_GC[(3*layer)+2] <<std::endl;            
            std::cout << vecX[layer].X() <<" "<< vecX[layer].Y() <<" "<< vecX[layer].Z() <<std::endl;                               
#endif
            //std::cout << pos_S1[0][layer] <<" "<< pos_S2[0][layer] <<" "<< pos_S3[0][layer] <<std::endl;
            //std::cout << mlpOutput[layer][0] <<" "<<  mlpOutput[layer][1] <<" "<< mlpOutput[layer][2] <<std::endl;
            //std::cout << pos_S1[1][layer] <<" "<< pos_S2[1][layer] <<" "<< pos_S3[1][layer] <<std::endl;     

            pos_Sx[layer]   = yGEOM->LToG(chipID[layer],256,512).X();
            pos_Sy[layer]   = yGEOM->LToG(chipID[layer],256,512).Y();
            pos_Sr[layer]   = TMath::Sqrt(pos_Sx[layer]*pos_Sx[layer] + pos_Sy[layer]*pos_Sy[layer]);
            pos_Sphi[layer] = TMath::ATan2(pos_Sy[layer],pos_Sx[layer]);
            pos_Sphi[layer] = ( pos_Sphi[layer] >= 0 ) ? pos_Sphi[layer] : 2*TMath::ATan2(0,-1) + pos_Sphi[layer];
            pos_Sz[layer]   = yGEOM->LToG(chipID[layer],256,512).Z();              
         }
           
         //std::cout<<std::endl;          
         pos_GC[(3*nLAYER)+0] = 0;
         pos_GC[(3*nLAYER)+1] = 0;
         pos_GC[(3*nLAYER)+2] = mlpX[2];  
         vecX[nLAYER].SetXYZ(pos_GC[(3*nLAYER)+0],pos_GC[(3*nLAYER)+1],pos_GC[(3*nLAYER)+2]);
#ifdef YALIGNDEBUG
         std::cout << pos_GC[(3*nLAYER)+0] <<" "<< pos_GC[(3*nLAYER)+1] <<" "<< pos_GC[(3*nLAYER)+2] <<std::endl;          
         std::cout << vecX[nLAYER].X() <<" "<< vecX[nLAYER].Y() <<" "<< vecX[nLAYER].Z() <<std::endl;     
#endif                                     
/*
         double MSEvalue;
         double fitpar[5];

         for(int j = 0; j < 5; j++){
            fitpar[j]=0.0;
         }
         MSEvalue =0;
         circle3Dfit(pos_GC, fitpar, MSEvalue, hitUpdate); 
*/
         double MSEvalue;
         double fitpar[6];

         for(int j = 0; j < 6; j++){
            fitpar[j]=0.0;
         }
         MSEvalue =0;
         circle3Dfit(pos_GC, fitpar, MSEvalue, hitUpdate, 0);

         double min_MSEvalue_Scale = ((int)fitpar[5])%10==0 ? 1e+10 : MSEvalue;
         double MSEvalueD(0), fitparD[6];
         int search_strategy[] = {-2, +2, +4};
         if(((int)fitpar[5])%10>=0 || MSEvalue>1.0e-4) {
            for(int isch = 0; isch < 3; isch++){
               for(int j = 0; j < 6; j++){
                  fitparD[j]=0.0;
               }
   
               circle3Dfit(pos_GC, fitparD, MSEvalueD, hitUpdate, search_strategy[isch]);


               if(min_MSEvalue_Scale>MSEvalueD && ((int)fitparD[5])%10==1){
                  min_MSEvalue_Scale = MSEvalueD;
                  MSEvalue = MSEvalueD;
                  for(int j = 0; j < 6; j++){
                     fitpar[j]=fitparD[j];
                  }
               }
            }
            if(((int)fitpar[5])%10==0) {
               std::cout<<"Event::"<<ientry<<" FITERROR"<<std::endl;
               //exit(0);
            }
         }
 

#ifdef YALIGNDEBUG       
         std::cout<<"fitpar : ";
         for(int j = 0; j < 6; j++){
            std::cout<<fitpar[j]<<" ";
         }      
         std::cout<<MSEvalue<<std::endl; 
#endif         
         double RecRadius = fitpar[0]>0 ? std::abs(1/(CM2M*(fitpar[0] + MinRes))) : std::abs(1/(CM2M*(fitpar[0] - MinRes)));
         double CircleXc  = fitpar[0]>0 ? RecRadius*std::cos(fitpar[1]+fitpar[4] + 0.5*TMath::Pi()) : RecRadius*std::cos(fitpar[1]+fitpar[4] - 0.5*TMath::Pi());
         double CircleYc  = fitpar[0]>0 ? RecRadius*std::sin(fitpar[1]+fitpar[4] + 0.5*TMath::Pi()) : RecRadius*std::sin(fitpar[1]+fitpar[4] - 0.5*TMath::Pi()); 
                          
         TVector3 vecXc(CircleXc, CircleYc, 0);
         TVector3 dirXr[nLAYER+1];
         for(int a=0; a<nLAYER+1;a++){
            dirXr[a] = vecX[a] - vecXc;
            //dirXr[a].Print();
         }

         double beta[nLAYER+1];
         double z_meas[nLAYER+1];      
         for(int l = 0; l < nLAYER+1; l++){    
            beta[l] = std::atan2(dirXr[l].Y(), dirXr[l].X());         
            //beta[l] = std::atan2(dirXr[l].Y(), dirXr[l].X()) > 0 ? std::atan2(dirXr[l].Y(), dirXr[l].X()) : 2*std::atan2(0,-1) + std::atan2(dirXr[l].Y(), dirXr[l].X());
            z_meas[l] = vecX[l].Z();               
         }            
         //beta linearization
         for(int l = 0; l < nLAYER; l++){   
         
            double linear_beta_arr[5];
            double linear_beta_dev = 2*std::atan2(0,-1);
            for(int lc = 0; lc < 5; lc++){
               linear_beta_arr[lc] = 2*std::atan2(0,-1)*(lc-2) + std::atan2(dirXr[l].Y(), dirXr[l].X());
               if(linear_beta_dev > TMath::Abs(linear_beta_arr[lc] - beta[(l+nLAYER)%(nLAYER+1)])) {
                  beta[l] = linear_beta_arr[lc];
                  linear_beta_dev = TMath::Abs(linear_beta_arr[lc] - beta[(l+nLAYER)%(nLAYER+1)]);
               }
            }              
         }     
         
         
         double parz[2] = {0 , 0};
         circle3Dfit_Z(z_meas, beta, parz, RecRadius, VERTEXFIT, hitUpdate_Z);
#ifdef YALIGNDEBUG               
         std::cout<<" [Verification] Circle(Xc, Yc, R) = "<<CircleXc<<" "<<CircleYc<<" "<<RecRadius<<std::endl;     
#endif            
         double pos_1[nLAYER+1][2],pos_2[nLAYER+1][2],pos_3[nLAYER+1][2];
         double est_1[nLAYER+1][2],est_2[nLAYER+1][2],est_3[nLAYER+1][2];             
         double stddev_1[nLAYER],stddev_2[nLAYER];
         
         double Cost_Beam=0;                         
         for(int layer = 0; layer < nLAYER + 1; layer++){           
            //"evno:layer:Cgx:Cgy:Cgz:Cgdx:Cgdy:Cgdz:CMSE"); 
#ifdef YALIGNDEBUG  
            std::cout<<" YAlignment ::  layer ["<<layer<<"] beta = "<<beta[layer]<<std::endl; 
#endif              
            //corrected
            pos_1[layer][0] = pos_GC[(3*layer)+0]; //alpha
            pos_2[layer][0] = pos_GC[(3*layer)+1]; //beta
            pos_3[layer][0] = pos_GC[(3*layer)+2]; //gamma                          
            est_1[layer][0] = RecRadius*std::cos(beta[layer]) + CircleXc;
            est_2[layer][0] = RecRadius*std::sin(beta[layer]) + CircleYc;
            est_3[layer][0] = (parz[0])*(beta[layer]) + (parz[1]);            
#ifdef YALIGNDEBUG                       
            std::cout<<" YAlignment :: Glayer pos1 est1 "<<pos_1[layer][0]<<" "<<est_1[layer][0]<<std::endl;
            std::cout<<" YAlignment :: Glayer pos2 est2 "<<pos_2[layer][0]<<" "<<est_2[layer][0]<<std::endl;
            std::cout<<" YAlignment :: Glayer pos3 est3 "<<pos_3[layer][0]<<" "<<est_3[layer][0]<<std::endl;                              
#endif           

            stddev_1[layer] = GetSigma(RecRadius, layer, DET_MAG, 0);
            stddev_2[layer] = GetSigma(RecRadius, layer, DET_MAG, 1);

            Cost_Beam += std::pow(pos_1[layer][0]-est_1[layer][0],2) + std::pow(pos_2[layer][0]-est_2[layer][0],2) + std::pow(pos_3[layer][0]-est_3[layer][0],2);           
               
            if(layer>=nLAYER) continue;    
                                                                
            pos_1[layer][1] = yGEOM->GToS(chipID[layer],pos_1[layer][0],pos_2[layer][0],pos_3[layer][0])(0);
            pos_2[layer][1] = yGEOM->GToS(chipID[layer],pos_1[layer][0],pos_2[layer][0],pos_3[layer][0])(1);
            pos_3[layer][1] = yGEOM->GToS(chipID[layer],pos_1[layer][0],pos_2[layer][0],pos_3[layer][0])(2);
            est_1[layer][1] = yGEOM->GToS(chipID[layer],est_1[layer][0],est_2[layer][0],est_3[layer][0])(0);
            est_2[layer][1] = yGEOM->GToS(chipID[layer],est_1[layer][0],est_2[layer][0],est_3[layer][0])(1);
            est_3[layer][1] = yGEOM->GToS(chipID[layer],est_1[layer][0],est_2[layer][0],est_3[layer][0])(2);
#ifdef YALIGNDEBUG        
            std::cout<<" YAlignment :: Slayer pos1 est1 "<<pos_1[layer][1]<<" "<<est_1[layer][1]<<std::endl;
            //if(TMath::Abs(pos_1[layer][1]-est_1[layer][1])>0.5) std::cout<<" ZFitError :: Slayer pos1 est1 "<<pos_1[layer][1] - est_1[layer][1]<<std::endl;
            std::cout<<" YAlignment :: Slayer pos2 est2 "<<pos_2[layer][1]<<" "<<est_2[layer][1]<<std::endl;
            std::cout<<" YAlignment :: Slayer pos3 est3 "<<pos_3[layer][1]<<" "<<est_3[layer][1]<<std::endl;   
#endif         
            //z_loc_max[layer]
            //z_loc_min[layer]
            double c0 = 0.1;
            double c1 = 0.9*(2/TMath::Abs(z_loc_max[layer]-z_loc_min[layer]));
            double z_loc = yGEOM->LToG(chipID[layer],256,512)(2);
             
            Weight_Hit[layer] = c0 + c1*TMath::Abs(z_loc);
            Weight_Track     += Weight_Hit[layer];
         }

         for(int layer = 0; layer < nLAYER + 1; layer++){           
            //epoch:evno:layer:chipID:imode:Cs1:Cs2:Cs3:Csd1:Csd2:Csd3:CMSE:R:P:PT
            Float_t TGlobal[] = {(float)-1,(float)ientry,(float)layer,(float)chipID[layer],(float)imode,
                                 (float)pos_1[layer][0], 		
                                 (float)pos_2[layer][0], 		
                                 (float)pos_3[layer][0],
                                 (float)(est_1[layer][0]-pos_1[layer][0]),
                                 (float)(est_2[layer][0]-pos_2[layer][0]),	
                                 (float)(est_3[layer][0]-pos_3[layer][0]), (float)Cost_Beam, (float)RecRadius, (float) mlp_p, (float)mlp_pt, (float)mlp_theta, (float)mlp_phi, (float)mlp_eta};

            fAnalysis_G->Fill(TGlobal);          
               
            if(layer>=nLAYER) continue;    

            //epoch:evno:layer:halfbarrel:stave:halfstave:module:lchip:schip:hschip:mchip:Sx:Sy:Sr:Sphi:Sz:chipID:imode:Cs1:Cs2:Cs3:Csd1:Csd2:Csd3:CMSE:R:P:PT:Theta:Phi:Eta:WH:WT
            Float_t TSensor[]  = {(float)-1,(float)ientry,
				 (float)sensor_Layer[layer], (float)sensor_HalfBarrel[layer],
			         (float)sensor_Stave[layer], (float)sensor_HalfStave[layer],
				 (float)sensor_Module[layer],
				 (float)sensor_ChipIdInLayer[layer], (float)sensor_ChipIdInStave[layer], (float)sensor_ChipIdInHalfStave[layer], (float)sensor_ChipIdInModule[layer],
                                 (float)pos_Sx[layer],(float)pos_Sy[layer],(float)pos_Sr[layer],(float)pos_Sphi[layer],(float)pos_Sz[layer],                                 
                                 (float)chipID[layer],(float)imode,
                                 (float)pos_1[layer][1], 		
                                 (float)pos_2[layer][1], 		
                                 (float)pos_3[layer][1],
                                 (float)(est_1[layer][1]-pos_1[layer][1]),	
                                 (float)(est_2[layer][1]-pos_2[layer][1]), 	
                                 (float)(est_3[layer][1]-pos_3[layer][1]), 
                                 (float)stddev_1[layer],
                                 (float)stddev_2[layer],
                                 (float)Cost_Beam, (float)RecRadius, (float) mlp_p, (float)mlp_pt, (float)mlp_theta, (float)mlp_phi, (float)mlp_eta,
                                 (float)Weight_Hit[layer], (float)Weight_Track};

            fAnalysis_S->Fill(TSensor);   
            for(int el = 0; el<nentries_track; el++){
               ntuple_track[el].push_back(TSensor[el]);
            }
         }            
         Weight_Event += Weight_Track;   
      }
      //epoch:evno:ntracks:WE
      //epoch:evno:layer:halfbarrel:stave:halfstave:module:lchip:schip:hschip:mchip:Sx:Sy:Sr:Sphi:Sz:chipID:imode:Cs1:Cs2:Cs3:Csd1:Csd2:Csd3:CMSE:R:P:PT:Theta:Phi:Eta:WH:WT:WE:itrack:ntracks
      for(int t=0; t<(int)ntuple_track[0].size(); t++){
         Float_t TEvent[nentries_track+4];
         //std::cout<<"[YAlignment] fAnalysis_E "<<t<<" ";
         for(int el=0; el<nentries_track; el++){
            TEvent[el] = (float)ntuple_track[el][t];
            //std::cout<<ntuple_track[el][t]<<" ";
         } 
         //std::cout<<std::endl;
         TEvent[nentries_track + 0] = Weight_Event;
         TEvent[nentries_track + 1] = (int)(t%(nLAYER));
         TEvent[nentries_track + 2] = trackentries;
         TEvent[nentries_track + 3] = vtxentries;         
         fAnalysis_E->Fill(TEvent); 
      }
   }
   fAnalyzeFile->cd();
   fAnalyzeFile->Write();  
   fOutputFile->cd();
   fOutputFile->Write();   
   TString fExec_argument1 = "mv " + XXXXanalyse_name + " ./" + fAnalyze_Directory_name + "/.";
   std::cout<<fExec_argument1<<std::endl;
   TString fExec_argument2 = "mv " + XXXXtrain_name + " ./" + fXXXXtrain_Directory_name + "/.";
   std::cout<<fExec_argument2<<std::endl;
   gSystem->Exec((TString)fExec_argument1);
   gSystem->Exec((TString)fExec_argument2);
}

////////////////////////////////////////////////////////////////////////////////
/// SetSplitRefernceSensor

void YAlignment::SetSplitReferenceSensor(int layer, int chipIDinlayer)
{
   if(layer>=0 && layer < nLAYER) {
      fSplitReferenceSensor = ChipBoundary[layer] + chipIDinlayer;
   } else {
      fSplitReferenceSensor = -1;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// InitNetworkUpdateList

void YAlignment::InitNetworkUpdateList()
{
   for(int s=0; s<nSensors; s++){
      fNetworkUpdateList.push_back(false);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// SetNetworkUpdateListLayerStave

void YAlignment::SetNetworkUpdateListLayerStave(int layer, int stave)
{
   for(int s=0; s<nSensorsbyLayer[layer]; s++){
      int chipID = s + ChipBoundary[layer];
      int stv = yGEOM->GetStave(chipID);  
      if(stave==stv) fNetworkUpdateList[chipID] = true;
   }      
}

////////////////////////////////////////////////////////////////////////////////
/// LoadNetworkUpdateList

void YAlignment::LoadNetworkUpdateList(bool userdefined = false)
{
   if(userdefined==false){
      for(int s=0; s<nSensors; s++){
         fNetworkUpdateList[s] = true;
      }
   } else {
      //Load Sensor Lists//
      int nselectedchips = 0;
      for(int s=0; s<nSensors; s++){
         fNetworkUpdateList[s] = false;
      }
      // read File
      std::string filePath = "SensorList.txt";   
      ifstream openFile(filePath.data());
      if( openFile.is_open() ){
         std::string line;
         while(getline(openFile, line)){
            std::cout << line << std::endl;
            if(line[0]!='#') {
               int chipID = std::stoi(line);
               fNetworkUpdateList[chipID] = true;
               nselectedchips++;
            }
         }
         openFile.close();
      } 
   }

}

////////////////////////////////////////////////////////////////////////////////
/// TrainMLP

void YAlignment::TrainMLP(YMultiLayerPerceptron::ELearningMethod method = YMultiLayerPerceptron::kSteepestDescent, bool removeDataTree = false)
{

   TString fTrainFileName = fXXXXtrain_Directory_name + "/XXXXtrain.root";
   TFile*  fTrainData = new TFile(fTrainFileName,"read");
   fInputTree = (TTree *) fTrainData->Get("InputTree"); 

   EventData* b_event = new EventData();  
   fInputTree->SetBranchAddress("event",      &b_event); 

   fMLPNetwork = new YMultiLayerPerceptron(fNetworkStructure,fInputTree,"(evno%10)>=0&&(evno%10)<6","(evno%10)>=6&&(evno%10)<8");
   std::cout<<" YAlignment::TrainMLP SetNetworkUpdateState"<<std::endl;
   fMLPNetwork->SetNetworkUpdateState(fNetworkUpdateList);
   std::cout<<" YAlignment::TrainMLP SetNpronged = "<<nTrackMax<<std::endl;   
   fMLPNetwork->SetNpronged(nTrackMax);  
   std::cout<<" YAlignment::TrainMLP SetFitModel : "<<FITMODEL<<std::endl;
   fMLPNetwork->SetFitModel(FITMODEL);
   if(fSplitReferenceSensor==-1){
      fMLPNetwork->SetSplitReferenceSensor(-1, 0);
   } else {
      int rlayer = yGEOM->GetLayer(fSplitReferenceSensor);
      int rchipIDinlayer = yGEOM->GetChipIdInLayer(fSplitReferenceSensor);
      fMLPNetwork->SetSplitReferenceSensor(rlayer, rchipIDinlayer);
   }
   
   TString weights_name = "weights.txt";  
   TString weightsDU_name = "weightsDU.txt";   
   TString losscurve_name = "LossCurve.gif";            
   double epoch_tau = 0.05;                                                    
   fMLPNetwork->SetTau(100*exp(-2*epoch_tau*fStep));  
   fMLPNetwork->SetWeightMonitoring(weights_name, int(fEpoch/1)); 
   fMLPNetwork->SetLearningMethod(method);
   if(fPrevWeights!="") {
      fMLPNetwork->SetPrevUSL(fPrevUSL);      
      fMLPNetwork->SetPrevWeight(fPrevWeights);   
      fMLPNetwork->SetPrevWeightDetectorUnit(fPrevWeightsDU);   
   }
   fMLPNetwork->Train(fEpoch,"graph, text, update=1");          
   fMLPNetwork->DumpWeights(weights_name);
   fMLPNetwork->DumpWeightsDetectorUnit(weightsDU_name);
   weights_name.Resize(weights_name.Sizeof()-5);
   TString fExec_argument1 = "mv " + weights_name + "* " + fweights_Directory_name + "/";
   TString fExec_argument2 = "mv " + losscurve_name + " " + flosscurve_Directory_name + "/" + losscurve_name;
   gSystem->Exec(fExec_argument1);
   gSystem->Exec(fExec_argument2);

   TString fExec_argument3 = "rm -rf MLPTrain/XXXXtrain";
   gSystem->Exec(fExec_argument3);   
}   


////////////////////////////////////////////////////////////////////////////////
/// EvaluateCostMLP

void YAlignment::EvaluateCostMLP(int step, int core, YMultiLayerPerceptron::ELearningMethod method = YMultiLayerPerceptron::kSteepestDescent)
{
   fMLPNetwork = new YMultiLayerPerceptron(fNetworkStructure,fInputTree,"(evno%10)>=0&&(evno%10)<6","(evno%10)>=6&&(evno%10)<8"); 

   fMLPNetwork->SetNpronged(nTrackMax);  
   fMLPNetwork->SetFitModel(FITMODEL); 
   fMLPNetwork->SetTau(3);  
   fMLPNetwork->SetLearningMethod(method);

   TString prevWeightSet = "./MLPTrain_Step" + TString::Itoa(step,10) + "/weights/weights_core" + TString::Itoa(core,10) +".txt";
   fMLPNetwork->SetPrevWeight(prevWeightSet);
   fMLPNetwork->EvaluateCost(step, core);
   
}

////////////////////////////////////////////////////////////////////////////////
/// AnalyzenMLP

void YAlignment::AnalyzeMLP(int step = 0, bool build = true, bool bfield = true)
{
   std::cout<<"YAlignment::AnalyzeMLP"<<std::endl;
   fDirectory_name = "MLPTrain";
   fAnalyze_Directory_name = "MLPTrain/XXXXanalyse"; 
   TString XXXXanalyse_name = "XXXXanalyse.root";  
   if(build==true){ 
      gSystem->mkdir(fDirectory_name);  
      gSystem->mkdir(fAnalyze_Directory_name);  
   }
   
   TString fTrainFileName = fDirectory_name + "_Step" + TString::Itoa(step,10) + "/XXXXtrain/XXXXtrain.root";
   TString fExec_argument0 = "ls " + fTrainFileName;
   std::cout<<fExec_argument0<<std::endl;
   gSystem->Exec(fExec_argument0);   
   TFile*  fTrainData = new TFile(fTrainFileName,"read");
   fInputTree = (TTree *) fTrainData->Get("InputTree"); 

   fMLPNetwork = new YMultiLayerPerceptron(fNetworkStructure,fInputTree,"(evno%10)>=0&&(evno%10)<6","(evno%10)>=6&&(evno%10)<8");
   fMLPNetwork->SetNpronged(nTrackMax);  
   fMLPNetwork->SetFitModel(FITMODEL);
 
   fAnalyzeFile = nullptr;   
   fAnalyzeFile = new TFile(XXXXanalyse_name,"recreate");   
   
   int nentries = fInputTree->GetEntriesFast();     

   double** input_Max;
   double** input_Min;
   double** norm;
   
   input_Max = new double *[2];
   input_Min = new double *[2];  
   norm      = new double *[2];    
   for(int axis = 0; axis <2; axis++){
      input_Max[axis] = new double [nSensors];
      input_Min[axis] = new double [nSensors];  
      norm[axis]      = new double [nSensors];  
      for(int iID=0; iID<nSensors; iID++){  
         input_Max[axis][iID] = 0;
         input_Min[axis][iID] = 0; 
         norm[axis][iID]      = 0; 
      }    
   }

   for(int iID=0; iID<nSensors; iID++){
      double ip1 = yGEOM->GToS(iID,yGEOM->LToG(iID,0,0)(0),	
 				   yGEOM->LToG(iID,0,0)(1),
                        	   yGEOM->LToG(iID,0,0)(2))(0);
      double fp1 = yGEOM->GToS(iID,yGEOM->LToG(iID,511,1023)(0),
	                           yGEOM->LToG(iID,511,1023)(1),
		     	           yGEOM->LToG(iID,511,1023)(2))(0); 
      double ip2 = yGEOM->GToS(iID,yGEOM->LToG(iID,0,0)(0),
                                   yGEOM->LToG(iID,0,0)(1),
                                   yGEOM->LToG(iID,0,0)(2))(1);
      double fp2 = yGEOM->GToS(iID,yGEOM->LToG(iID,511,1023)(0),	
                                   yGEOM->LToG(iID,511,1023)(1),
                                   yGEOM->LToG(iID,511,1023)(2))(1);               
      input_Max[0][iID]=std::max(ip1,fp1);
      input_Min[0][iID]=std::min(ip1,fp1); 
      input_Max[1][iID]=std::max(ip2,fp2);
      input_Min[1][iID]=std::min(ip2,fp2);             
      norm[0][iID]=input_Max[0][iID]-input_Min[0][iID];          
      norm[1][iID]=input_Max[1][iID]-input_Min[1][iID];   
#ifdef YALIGNDEBUG                                                         
      std::cout<<"Sensor Boundary ChipID["<<iID<<"] "<<input_Max[0][iID]<<" "<<input_Min[0][iID]<<" "
                                                     <<input_Max[1][iID]<<" "<<input_Min[1][iID]<<" Norm (s1, s2) "<<norm[0][iID]<<" "<<norm[1][iID]<<std::endl;
#endif         
   }  

                                    //0  1    2    3    4     5     6      7
   int SensorBoundary[nLAYER + 1] = { 0, 108, 252, 432, 3120, 6480, 14712, 24120 };	
          
   TNtuple *fAnalysis_S  
            = new TNtuple("fAnalysis_S",	"fAnalysis_S",	
            "epoch:evno:layer:halfbarrel:stave:halfstave:module:lchip:schip:hschip:mchip:Sx:Sy:Sr:Sphi:Sz:chipID:imode:Cs1:Cs2:Cs3:Csd1:Csd2:Csd3:Sigma_s1:Sigma_s2:CMSE:R:P:PT:Theta:Phi:Eta");   
   TNtuple *fAnalysis_G  
            = new TNtuple("fAnalysis_G",	"fAnalysis_G",	
            "epoch:evno:layer:halfbarrel:stave:halfstave:module:lchip:schip:hschip:mchip:chipID:imode:Cgx:Cgy:Cgz:Cgdx:Cgdy:Cgdz:CMSE:R:P:PT:Theta:Phi:Eta");          
   TNtuple *fAnalysis_V  
            = new TNtuple("fAnalysis_V",	"fAnalysis_V",	
            "epoch:evno:imode:Vz_track:Sigma_t:Vz_event:Vz_mean:Vz_target");  

   EventData* b_event = new EventData();     
            
   std::cout<<"nInput Tree : "<< nentries <<std::endl;           
   for(int iepoch = 0; iepoch < fEpoch; iepoch++){
      TString prev_weight   = "./MLPTrain_Step" + TString::Itoa(step,10) + "/weights/weights_Epoch_At_" + TString::Itoa(iepoch,10) +".txt";
      TString prev_weightDU = "./MLPTrain_Step" + TString::Itoa(step,10) + "/weights/weightsDU_Epoch_At_" + TString::Itoa(iepoch,10) +".txt";
      SetPrevWeight(prev_weight);
      fMLPNetwork->SetPrevWeight(prev_weight); 
      fMLPNetwork->SetPrevWeightDetectorUnit(prev_weightDU); 
      fMLPNetwork->Init_Randomize();
      fMLPNetwork->Init_RandomizeSensorCorrection();
      fMLPNetwork->LoadWeights(fPrevWeights);
      fMLPNetwork->PrintCurrentWeights();        
      fInputTree->SetBranchAddress("event",      &b_event);      
      
      for(int ientry = 0; ientry < nentries; ientry++ ){
      
         fInputTree->GetEntry(ientry);    // err evt 11288       
         int hitentries = nLAYER;
         int trackentries = b_event->GetNtracks();
         std::cout<<"Analyze Epoch "<<iepoch<<" "<<ientry<<" "<<trackentries<<std::endl;
         YVertexFitParameter vtxfit[trackentries];     
         for(int imode = 0; imode < trackentries; imode++){

            TrackData *b_track = (TrackData *) b_event->GetTrack()->At(imode);  
                   
            double mlpInput[nLAYER][2];
            double mlpOutput[nLAYER][3];
            double mlpExtended[nLAYER][3];   
            double mlpX[3];  

            double mlp_p        = b_track->p;
            double mlp_pt       = b_track->pt;
            double mlp_theta    = b_track->theta;
            double mlp_phi      = b_track->phi;
            double mlp_eta      = b_track->eta;

            std::vector<bool> hitUpdate;
            std::vector<bool> hitUpdate_Z; 
            for(Int_t layer = 0; layer<nLAYER ;layer++){                  
   
               mlpInput[layer][0]   = b_track->s1[layer]; 
               mlpInput[layer][1]   = b_track->s2[layer];
            
               mlpExtended[layer][0]= b_track->Stave[layer];    
               mlpExtended[layer][1]= b_track->Chip[layer];  
               mlpExtended[layer][2]= b_track->ChipID[layer];
               bool layUpdate = b_track->ChipID[layer] < 0 ? false : true;
               hitUpdate.push_back(layUpdate);
               hitUpdate_Z.push_back(layUpdate);   
               std::cout<<"  hitUpdate["<<layer<<"] = "<<layUpdate<<std::endl;               
            }    
            hitUpdate.push_back(true);
                            
            mlpX[0] = b_event->GetX1();       
            mlpX[1] = b_event->GetX2();       
            mlpX[2] = b_event->GetX3();    
                           
            int stave[nLAYER];
            int chip[nLAYER];    
            int chipID[nLAYER+1];  
            chipID[nLAYER] = -1;     
            
            int sensor_Layer[nLAYER];
            int sensor_HalfBarrel[nLAYER];
            int sensor_Stave[nLAYER];
            int sensor_HalfStave[nLAYER];
            int sensor_Module[nLAYER];
            int sensor_ChipIdInLayer[nLAYER];
            int sensor_ChipIdInStave[nLAYER];
            int sensor_ChipIdInHalfStave[nLAYER];
            int sensor_ChipIdInModule[nLAYER];           
                                                                                                                         
            for(int layer = 0; layer<nLAYER; layer++){  
               sensor_Layer[layer]		= -1;
               sensor_HalfBarrel[layer]		= -1;
               sensor_Stave[layer]		= -1;
               sensor_HalfStave[layer]		= -1;
               sensor_Module[layer]		= -1;
               sensor_ChipIdInLayer[layer]	= -1;
               sensor_ChipIdInStave[layer]	= -1;
               sensor_ChipIdInHalfStave[layer]	= -1;
               sensor_ChipIdInModule[layer]	= -1;
                           
               if(hitUpdate[layer]==false) continue;                   
               stave[layer] = (int)mlpExtended[layer][0];
               chip[layer]  = (int)mlpExtended[layer][1]; 
               chipID[layer]  = (int)mlpExtended[layer][2]; 
               
               sensor_Layer[layer]		= yGEOM->GetLayer(chipID[layer]); 
               sensor_HalfBarrel[layer]		= yGEOM->GetHalfBarrel(chipID[layer]);  
               sensor_Stave[layer]		= yGEOM->GetStave(chipID[layer]); 
               sensor_HalfStave[layer]		= yGEOM->GetHalfStave(chipID[layer]);  
               sensor_Module[layer]		= yGEOM->GetModule(chipID[layer]);  
               sensor_ChipIdInLayer[layer]	= yGEOM->GetChipIdInLayer(chipID[layer]); 
               sensor_ChipIdInStave[layer]	= yGEOM->GetChipIdInStave(chipID[layer]); 
               sensor_ChipIdInHalfStave[layer]	= yGEOM->GetChipIdInHalfStave(chipID[layer]); 
               sensor_ChipIdInModule[layer]	= yGEOM->GetChipIdInModule(chipID[layer]); 
               
               for(int axis = 0; axis<3 ;axis++){
                  mlpOutput[layer][axis] = fMLPNetwork->Evaluate(axis, mlpInput[layer],chipID[layer]);
               }  
            }     

              
            double pos_S1[2][nLAYER];  // uncorrected : 0, corrected : 1
            double pos_S2[2][nLAYER]; 
            double pos_S3[2][nLAYER];  

            double pos_GX[2][nLAYER];  // uncorrected : 0, corrected : 1
            double pos_GY[2][nLAYER]; 
            double pos_GZ[2][nLAYER]; 

            double pos_Sx[nLAYER], pos_Sy[nLAYER], pos_Sr[nLAYER], pos_Sphi[nLAYER], pos_Sz[nLAYER];
            
            double pos_GC[3*(nLAYER+1)];
            
            //std::cout<< iepoch <<" - "<<ientry<<" - "<<imode<<" :: "<<std::endl;         
            TVector3 vecX[nLAYER+1];
            TVector3 vecXmid[nLAYER];
            double InvSlope[nLAYER];            

            for(int ln = 0; ln < nLAYER+1; ln++){    
               if(hitUpdate[ln]==false) continue;                   
               vtxfit[imode].z_meas[ln] = 0;
               vtxfit[imode].beta[ln]   = 0;
            }
      
            vtxfit[imode].parz[0]=0;
            vtxfit[imode].parz[1]=0;
            vtxfit[imode].Radius =0;
            vtxfit[imode].valid  =false;

            for(Int_t layer = 0; layer<nLAYER ;layer++){  

               pos_S1[0][layer] = hitUpdate[layer]==true ? (mlpInput[layer][0] + fNorm_shift)*norm[0][chipID[layer]] + input_Min[0][chipID[layer]] : -9999; 
               pos_S2[0][layer] = hitUpdate[layer]==true ? (mlpInput[layer][1] + fNorm_shift)*norm[1][chipID[layer]] + input_Min[1][chipID[layer]] : -9999;
               pos_S3[0][layer] = hitUpdate[layer]==true ? 0 : -9999; 
               pos_S1[1][layer] = hitUpdate[layer]==true ? (mlpInput[layer][0] + fNorm_shift)*norm[0][chipID[layer]] + input_Min[0][chipID[layer]] + mlpOutput[layer][0] : -9999; 
               pos_S2[1][layer] = hitUpdate[layer]==true ? (mlpInput[layer][1] + fNorm_shift)*norm[1][chipID[layer]] + input_Min[1][chipID[layer]] + mlpOutput[layer][1] : -9999;
               pos_S3[1][layer] = hitUpdate[layer]==true ? mlpOutput[layer][2] : -9999;             
                       
               pos_GX[0][layer] = hitUpdate[layer]==true ? yGEOM->SToG(chipID[layer],pos_S1[0][layer],pos_S2[0][layer],pos_S3[0][layer]).X() : -9999;
               pos_GY[0][layer] = hitUpdate[layer]==true ? yGEOM->SToG(chipID[layer],pos_S1[0][layer],pos_S2[0][layer],pos_S3[0][layer]).Y() : -9999;
               pos_GZ[0][layer] = hitUpdate[layer]==true ? yGEOM->SToG(chipID[layer],pos_S1[0][layer],pos_S2[0][layer],pos_S3[0][layer]).Z() : -9999;                               
               pos_GX[1][layer] = hitUpdate[layer]==true ? yGEOM->SToG(chipID[layer],pos_S1[1][layer],pos_S2[1][layer],pos_S3[1][layer]).X() : -9999;
               pos_GY[1][layer] = hitUpdate[layer]==true ? yGEOM->SToG(chipID[layer],pos_S1[1][layer],pos_S2[1][layer],pos_S3[1][layer]).Y() : -9999;
               pos_GZ[1][layer] = hitUpdate[layer]==true ? yGEOM->SToG(chipID[layer],pos_S1[1][layer],pos_S2[1][layer],pos_S3[1][layer]).Z() : -9999;     

               pos_GC[(3*layer)+0] = pos_GX[1][layer];
               pos_GC[(3*layer)+1] = pos_GY[1][layer];
               pos_GC[(3*layer)+2] = pos_GZ[1][layer];       
              
               vecX[layer].SetXYZ(pos_GC[(3*layer)+0],pos_GC[(3*layer)+1],pos_GC[(3*layer)+2]);
               std::cout << pos_GC[(3*layer)+0] <<" "<< pos_GC[(3*layer)+1] <<" "<< pos_GC[(3*layer)+2] <<std::endl;            
               std::cout << vecX[layer].X() <<" "<< vecX[layer].Y() <<" "<< vecX[layer].Z() <<std::endl;                               
               std::cout << pos_S1[0][layer] <<" "<< pos_S2[0][layer] <<" "<< pos_S3[0][layer] <<std::endl;
               std::cout << mlpOutput[layer][0] <<" "<<  mlpOutput[layer][1] <<" "<< mlpOutput[layer][2] <<std::endl;
               std::cout << pos_S1[1][layer] <<" "<< pos_S2[1][layer] <<" "<< pos_S3[1][layer] <<std::endl;  

               pos_Sx[layer]   = hitUpdate[layer]==true ? yGEOM->LToG(chipID[layer],256,512).X() : -9999;
               pos_Sy[layer]   = hitUpdate[layer]==true ? yGEOM->LToG(chipID[layer],256,512).Y() : -9999;
               pos_Sr[layer]   = hitUpdate[layer]==true ? TMath::Sqrt(pos_Sx[layer]*pos_Sx[layer] + pos_Sy[layer]*pos_Sy[layer]) : -9999;
               pos_Sphi[layer] = hitUpdate[layer]==true ? TMath::ATan2(pos_Sy[layer],pos_Sx[layer]) : -9999;
               pos_Sphi[layer] = hitUpdate[layer]==true ? ( pos_Sphi[layer] >= 0 ) ? pos_Sphi[layer] : 2*TMath::ATan2(0,-1) + pos_Sphi[layer] : -9999;
               pos_Sz[layer]   = hitUpdate[layer]==true ? yGEOM->LToG(chipID[layer],256,512).Z() : -9999;
            }
            
            std::cout<<std::endl;          
            pos_GC[(3*nLAYER)+0] = 0;
            pos_GC[(3*nLAYER)+1] = 0;
            pos_GC[(3*nLAYER)+2] = mlpX[2];  
            vecX[nLAYER].SetXYZ(pos_GC[(3*nLAYER)+0],pos_GC[(3*nLAYER)+1],pos_GC[(3*nLAYER)+2]);
            std::cout << pos_GC[(3*nLAYER)+0] <<" "<< pos_GC[(3*nLAYER)+1] <<" "<< pos_GC[(3*nLAYER)+2] <<std::endl;          
            std::cout << vecX[nLAYER].X() <<" "<< vecX[nLAYER].Y() <<" "<< vecX[nLAYER].Z() <<std::endl;     
                                              
         double MSEvalue;
         double fitpar[6];

         for(int j = 0; j < 6; j++){
            fitpar[j]=0.0;
         }
         MSEvalue =0;
         circle3Dfit(pos_GC, fitpar, MSEvalue, hitUpdate, 0);

         double min_MSEvalue_Scale = ((int)fitpar[5])%10==0 ? 1e+10 : MSEvalue;
         double MSEvalueD(0), fitparD[6];
         int search_strategy[] = {-2, +2, +4};
         if(((int)fitpar[5])%10>=0 || MSEvalue>1.0e-4) {
            for(int isch = 0; isch < 3; isch++){
               for(int j = 0; j < 6; j++){
                  fitparD[j]=0.0;
               }
   
               circle3Dfit(pos_GC, fitparD, MSEvalueD, hitUpdate, search_strategy[isch]);


               if(min_MSEvalue_Scale>MSEvalueD && ((int)fitparD[5])%10==1){
                  min_MSEvalue_Scale = MSEvalueD;
                  MSEvalue = MSEvalueD;
                  for(int j = 0; j < 6; j++){
                     fitpar[j]=fitparD[j];
                  }
               }
            }
            if(((int)fitpar[5])%10==0) {
               std::cout<<"Event::"<<ientry<<" FITERROR"<<std::endl;
               //exit(0);
            }
         }
#ifdef YALIGNDEBUG         
            std::cout<<"fitpar : ";
            for(int j = 0; j < 6; j++){
               std::cout<<fitpar[j]<<" ";
            }      
            std::cout<<MSEvalue<<std::endl; 
#endif         
            double RecRadius = fitpar[0]>0 ? std::abs(1/(CM2M*(fitpar[0] + MinRes))) : std::abs(1/(CM2M*(fitpar[0] - MinRes)));
            double CircleXc  = fitpar[0]>0 ? RecRadius*std::cos(fitpar[1]+fitpar[4] + 0.5*TMath::Pi()) : RecRadius*std::cos(fitpar[1]+fitpar[4] - 0.5*TMath::Pi());
            double CircleYc  = fitpar[0]>0 ? RecRadius*std::sin(fitpar[1]+fitpar[4] + 0.5*TMath::Pi()) : RecRadius*std::sin(fitpar[1]+fitpar[4] - 0.5*TMath::Pi()); 
                          
            TVector3 vecXc(CircleXc, CircleYc, 0);
            TVector3 dirXr[nLAYER+1];
            for(int a=0; a<nLAYER+1;a++){
               dirXr[a] = TVector3(-9999,-9999,-9999);
               if(hitUpdate[a]==false) continue;                   
               dirXr[a] = vecX[a] - vecXc;
               //dirXr[a].Print();
            }

            double beta[nLAYER+1];
            double z_meas[nLAYER+1];      
            for(int l = 0; l < nLAYER+1; l++){    
               beta[l] = -9999;
               if(hitUpdate[l]==false) continue;                   
               beta[l] = std::atan2(dirXr[l].Y(), dirXr[l].X());// > 0 ? std::atan2(dirXr[l].Y(), dirXr[l].X()) : 2*std::atan2(0,-1) + std::atan2(dirXr[l].Y(), dirXr[l].X());
               z_meas[l] = vecX[l].Z();               
            }            

            //beta linearization
            double beta_dum[nLAYER+1];
            int dum_index[nLAYER+1];
            int lin_dum =0; 
            for(int l = 0; l < nLAYER+1; l++){   
               dum_index[l] = -1;
               if(hitUpdate[l]==false) continue;   
               beta_dum[lin_dum]  = beta[l];
               dum_index[lin_dum] = l;
               lin_dum++;
            }       
            lin_dum = lin_dum - 1;
            
            std::cout<<"beta linerization Before "<<std::endl;
            for(int l = 0; l < nLAYER+1; l++){ 
               std::cout<<"beta["<<l<<"]= "<<beta[l]<<std::endl;
            }         
            for(int ld = 0; ld < lin_dum+1; ld++){ 
               std::cout<<"beta_dum["<<ld<<"]= "<<beta_dum[ld]<<std::endl;
            }
            
            for(int ld = 0; ld < lin_dum; ld++){   
               double linear_beta_arr[5];
               double linear_beta_dev = 2*std::atan2(0,-1);
               for(int lc = 0; lc < 5; lc++){
                  linear_beta_arr[lc] = 2*std::atan2(0,-1)*(lc-2) + std::atan2(dirXr[dum_index[ld]].Y(), dirXr[dum_index[ld]].X());
                  std::cout<<ld<<" "<<lc<<" "<<linear_beta_arr[lc]<<" "<<beta_dum[(ld+lin_dum)%(lin_dum+1)]<<" "<<TMath::Abs(linear_beta_arr[lc] - beta_dum[(ld+lin_dum)%(lin_dum+1)])<<std::endl;
                  if(linear_beta_dev > TMath::Abs(linear_beta_arr[lc] - beta_dum[(ld+lin_dum)%(lin_dum+1)])) {
                     beta[dum_index[ld]] = linear_beta_arr[lc];
                     beta_dum[ld] = linear_beta_arr[lc];
                     linear_beta_dev = TMath::Abs(linear_beta_arr[lc] - beta_dum[(ld+lin_dum)%(lin_dum+1)]);
                  }
               }              
            }   
            std::cout<<"beta linerization After "<<std::endl;
            for(int l = 0; l < nLAYER+1; l++){ 
               std::cout<<"beta["<<l<<"]= "<<beta[l]<<std::endl;
            }     
            
            double parz[2] = {0 , 0};
            circle3Dfit_Z(z_meas, beta, parz, RecRadius, VERTEXFIT, hitUpdate_Z);
            for(int ln = 0; ln < nLAYER+1; ln++){    
               if(hitUpdate[ln]==false) continue;                   
               vtxfit[imode].z_meas[ln] = z_meas[ln];
               vtxfit[imode].beta[ln]   = beta[ln];
            }
            vtxfit[imode].parz[0]=parz[0];
            vtxfit[imode].parz[1]=parz[1];
            vtxfit[imode].Radius =RecRadius;
            vtxfit[imode].valid  =true;            
            
            //std::cout<<" [Verification] Circle(Xc, Yc, R) = "<<CircleXc<<" "<<CircleYc<<" "<<RecRadius<<std::endl;     
            
            double pos_1[nLAYER+1][2],pos_2[nLAYER+1][2],pos_3[nLAYER+1][2];
            double est_1[nLAYER+1][2],est_2[nLAYER+1][2],est_3[nLAYER+1][2];             
            double stddev_1[nLAYER],stddev_2[nLAYER];
         
            double Cost_Beam=0;                         
            for(int layer = 0; layer < nLAYER + 1; layer++){     
               if(hitUpdate[layer]==false) continue;       
               //"evno:layer:Cgx:Cgy:Cgz:Cgdx:Cgdy:Cgdz:CMSE"); 
               std::cout<<" YAlignment ::  layer ["<<layer<<"] beta = "<<beta[layer]<<" "<<chipID[layer]<<" "<<sensor_ChipIdInModule[layer]<<std::endl;      
               //corrected
               pos_1[layer][0] = pos_GC[(3*layer)+0]; //alpha
               pos_2[layer][0] = pos_GC[(3*layer)+1]; //beta
               pos_3[layer][0] = pos_GC[(3*layer)+2]; //gamma                          
               est_1[layer][0] = RecRadius*std::cos(beta[layer]) + CircleXc;
               est_2[layer][0] = RecRadius*std::sin(beta[layer]) + CircleYc;
               est_3[layer][0] = (parz[0])*(beta[layer]) + (parz[1]);                              
               std::cout<<" YAlignment :: Glayer pos1 est1 "<<pos_1[layer][0]<<" "<<est_1[layer][0]<<std::endl;
               std::cout<<" YAlignment :: Glayer pos2 est2 "<<pos_2[layer][0]<<" "<<est_2[layer][0]<<std::endl;
               std::cout<<" YAlignment :: Glayer pos3 est3 "<<pos_3[layer][0]<<" "<<est_3[layer][0]<<std::endl;                              

               stddev_1[layer] = GetSigma(RecRadius, layer, DET_MAG, 0);
               stddev_2[layer] = GetSigma(RecRadius, layer, DET_MAG, 1);
         
               Cost_Beam += std::pow(pos_1[layer][0]-est_1[layer][0],2) + std::pow(pos_2[layer][0]-est_2[layer][0],2) + std::pow(pos_3[layer][0]-est_3[layer][0],2);           
               
               if(layer>=nLAYER) continue;    
                                                                
               pos_1[layer][1] = yGEOM->GToS(chipID[layer],pos_1[layer][0],pos_2[layer][0],pos_3[layer][0])(0);
               pos_2[layer][1] = yGEOM->GToS(chipID[layer],pos_1[layer][0],pos_2[layer][0],pos_3[layer][0])(1);
               pos_3[layer][1] = yGEOM->GToS(chipID[layer],pos_1[layer][0],pos_2[layer][0],pos_3[layer][0])(2);
               est_1[layer][1] = yGEOM->GToS(chipID[layer],est_1[layer][0],est_2[layer][0],est_3[layer][0])(0);
               est_2[layer][1] = yGEOM->GToS(chipID[layer],est_1[layer][0],est_2[layer][0],est_3[layer][0])(1);
               est_3[layer][1] = yGEOM->GToS(chipID[layer],est_1[layer][0],est_2[layer][0],est_3[layer][0])(2);
               std::cout<<" YAlignment :: Slayer pos1 est1 "<<pos_1[layer][1]<<" "<<est_1[layer][1]<<std::endl;
               std::cout<<" YAlignment :: Slayer pos2 est2 "<<pos_2[layer][1]<<" "<<est_2[layer][1]<<std::endl;
               std::cout<<" YAlignment :: Slayer pos3 est3 "<<pos_3[layer][1]<<" "<<est_3[layer][1]<<std::endl;
               
               if(std::abs(est_1[layer][0]-pos_1[layer][0])>1 ||
                  std::abs(est_2[layer][0]-pos_2[layer][0])>1 ||
                  std::abs(est_3[layer][0]-pos_3[layer][0])>1) std::cout<< "Abnormal Report !"<<std::endl;                               
            }    
            for(int layer = 0; layer < nLAYER + 1; layer++){           
               if(hitUpdate[layer]==false) continue;                   
               //"epoch:evno:layer:halfbarrel:stave:halfstave:module:lchip:schip:hschip:mchip:Sx:Sy:Sr:Sphi:Sz:chipID:imode:Cs1:Cs2:Cs3:Csd1:Csd2:Csd3:Sigma_s1:Sigma_s2:CMSE:R:P:PT:Theta:Phi:Eta");   
               Float_t TGlobal[] = {(float)iepoch,(float)ientry,(float)layer,
               		            (float)sensor_HalfBarrel[layer],
			            (float)sensor_Stave[layer], (float)sensor_HalfStave[layer],
				    (float)sensor_Module[layer],
				    (float)sensor_ChipIdInLayer[layer], (float)sensor_ChipIdInStave[layer], (float)sensor_ChipIdInHalfStave[layer], (float)sensor_ChipIdInModule[layer],
                                    (float)chipID[layer],(float)imode,
                                    (float)pos_1[layer][0], 		
                                    (float)pos_2[layer][0], 		
                                    (float)pos_3[layer][0],
                                    (float)(est_1[layer][0]-pos_1[layer][0]),
                                    (float)(est_2[layer][0]-pos_2[layer][0]),	
                                    (float)(est_3[layer][0]-pos_3[layer][0]), (float)Cost_Beam, (float)RecRadius, (float) mlp_p, (float)mlp_pt, (float)mlp_theta, (float)mlp_phi, (float)mlp_eta};

               fAnalysis_G->Fill(TGlobal);          
               
               if(layer>=nLAYER) continue;    
               //epoch:evno:layer:
               //halfbarrel:
               //stave:halfstave:
               //module:
               //lchip:schip:hschip:mchip:
               //Sx:Sy:Sr:Sphi:Sz:
               //chipID:imode:
               //Cs1:Cs2:Cs3:
               //Csd1:Csd2:Csd3:
               //Sigma_s1:Sigma_s2:
               //CMSE:R:P:PT:Theta:Phi:Eta
               Float_t TStave[]  = {(float)iepoch,(float)ientry,(float)layer,
				    (float)sensor_HalfBarrel[layer],
			            (float)sensor_Stave[layer], (float)sensor_HalfStave[layer],
				    (float)sensor_Module[layer],
				    (float)sensor_ChipIdInLayer[layer], (float)sensor_ChipIdInStave[layer], (float)sensor_ChipIdInHalfStave[layer], (float)sensor_ChipIdInModule[layer],               
                                    (float)pos_Sx[layer],(float)pos_Sy[layer],(float)pos_Sr[layer],(float)pos_Sphi[layer],(float)pos_Sz[layer],
                                    (float)chipID[layer],(float)imode,
                                    (float)pos_1[layer][1], 		
                                    (float)pos_2[layer][1], 		
                                    (float)pos_3[layer][1],
                                    (float)(est_1[layer][1]-pos_1[layer][1]),	
                                    (float)(est_2[layer][1]-pos_2[layer][1]), 	
                                    (float)(est_3[layer][1]-pos_3[layer][1]), 
                                    (float)stddev_1[layer],
                                    (float)stddev_2[layer],
                                    (float)Cost_Beam, (float)RecRadius, (float) mlp_p, (float)mlp_pt, (float)mlp_theta, (float)mlp_phi, (float)mlp_eta};

               fAnalysis_S->Fill(TStave);   
            }                                            
         } 
         
         ///
         
         double z_event = 0;
         bool   valid_event = false;
         int    valid_nhits = 0;
         
         double z_track[trackentries];
         double z_track_mean   = 0;
         double z_track_weight = 0;
         for(int track = 0; track < trackentries; track++){
            z_track[track]        =0;   
            if(vtxfit[track].valid==false) continue;
            
            if(valid_event==false){      
               z_event = vtxfit[track].z_meas[nLAYER];
               valid_event = true;
            }
            z_track[track] = (vtxfit[track].parz[0])*(vtxfit[track].beta[nLAYER]) + (vtxfit[track].parz[1]);       
#ifdef YVTXFITALIGN         
            std::cout<<"Track "<<track<<std::endl;
               
            std::cout<<" z :: "; 
      
            for(int l = 0; l< nLAYER; l++){
               std::cout<< vtxfit[track].z_meas[l] <<" ";
            }
            std::cout<<std::endl;
            std::cout<<" beta :: ";
            for(int l = 0; l< nLAYER; l++){
               std::cout<< vtxfit[track].beta[l] <<" ";
            }
            std::cout<<std::endl;
       
            std::cout<<" parz :: "<<vtxfit[track].parz[0]<<" "<<vtxfit[track].parz[1]<<std::endl;
            std::cout<<" R :: "<<vtxfit[track].Radius<<" "<<vtxfit[track].valid<<std::endl;
            
            std::cout<< " z_event "<<vtxfit[track].z_meas[nLAYER]<<std::endl;
      
            std::cout<< " z_track "<<z_track[track]<<std::endl;
#endif      
            z_track_mean   += z_track[track]/std::pow(GetSigma(vtxfit[track].Radius, nLAYER, DET_MAG, 0),2);
            z_track_weight += 1/std::pow(GetSigma(vtxfit[track].Radius, nLAYER, DET_MAG, 0),2);
            valid_nhits++;
         }  
         z_track_mean = z_track_mean/z_track_weight;
#ifdef YVTXFITALIGN         
         std::cout<<"z_track(mean) "<<z_track_mean<<std::endl;
#endif  

         double z_track_target[trackentries];
         double beta_target[trackentries];
         for(int track1 = 0; track1 < trackentries; track1++){
            z_track_target[track1] = 0;
            beta_target[track1] = 0;
            if(vtxfit[track1].valid==false) continue;
            double z_track_mean_target   = 0;
            double z_track_weight_target = 0; 
            if(valid_nhits>1){
               for(int track2 = 0; track2 < trackentries; track2++){
                  if(track1==track2) continue;
                  if(vtxfit[track2].valid==false) continue; 
                  z_track_mean_target   += z_track[track2]/std::pow(GetSigma(vtxfit[track2].Radius, nLAYER, DET_MAG, 0),2);
                  z_track_weight_target += 1/std::pow(GetSigma(vtxfit[track2].Radius, nLAYER, DET_MAG, 0),2);
               }
               z_track_target[track1] = z_track_mean_target/z_track_weight_target;
               beta_target[track1]    = vtxfit[track1].beta[nLAYER];
            } else {
               z_track_target[track1] = z_track[track1];
               beta_target[track1]    = vtxfit[track1].beta[nLAYER];     
            }
#ifdef YVTXFITALIGN         
            std::cout<<"z_event[" <<track1<<"]       "<<z_event<<std::endl;
            std::cout<<"z_track[" <<track1<<"]       "<<z_track[track1]<<std::endl;
            std::cout<<"z_target["<<track1<<"](mean) "<<z_track_target[track1]<<std::endl;
#endif      
         }  
         for(int imode = 0; imode < trackentries; imode++){
            //epoch:evno:imode:Vz_track:Vz_event:Vz_mean:Vz_target
            if(vtxfit[imode].valid==false) continue;            
            Float_t TVertex[]  = {(float)iepoch,(float)ientry,(float)imode,
                                  (float)z_track[imode], (float)GetSigma(vtxfit[imode].Radius, nLAYER, DET_MAG, 0),(float)z_event, (float)z_track_mean, (float)z_track_target[imode]};

            fAnalysis_V->Fill(TVertex);    
         }                 
         ///        
      }
   }
   fAnalyzeFile->cd();
   fAnalyzeFile->Write();  
   
   TString fExec_argument1 = "mv " + XXXXanalyse_name + " " + fAnalyze_Directory_name + "/" + XXXXanalyse_name;

   gSystem->Exec(fExec_argument1);
      
}

////////////////////////////////////////////////////////////////////////////////
/// GetGeometries

void YAlignment::ReconstructGeometries(int res_level = 4)
{


   //   10    9    8    7    6    5    4    3    2    1 
   //------------------------------------------------------

   //  512  256  128   64   32   16    8    4    2    1  row
   // 1024  512  256  128   64   32   16    8    4    2  col
   fGeomFile   = nullptr;   
   fGeomFile   = new TFile("XXXXGeom.root","recreate");
   TString NetworkDir 	= "Geom/";
   TString NetworkIdealDir	 = NetworkDir + "Ideal/";
   TString NetworkTargetDir      = NetworkDir + "Target/";
   TString NetworkCorrectedDir	 = NetworkDir + "Corrected/";
/*
      int sensor_Layer			= yGEOM->GetLayer(ichipID); 
      int sensor_HalfBarrel		= yGEOM->GetHalfBarrel(ichipID);  
      int sensor_Stave			= yGEOM->GetStave(ichipID); 
      int sensor_HalfStave		= yGEOM->GetHalfStave(ichipID);  
      int sensor_Module			= yGEOM->GetModule(ichipID);  
      int sensor_ChipIdInLayer		= yGEOM->GetChipIdInLayer(ichipID); 
      int sensor_ChipIdInStave		= yGEOM->GetChipIdInStave(ichipID); 
      int sensor_ChipIdInHalfStave	= yGEOM->GetChipIdInHalfStave(ichipID); 
      int sensor_ChipIdInModule		= yGEOM->GetChipIdInModule(ichipID); 
*/
   TNtuple *fIdealGeom       = new TNtuple("fIdealGeom",	"fIdealGeom",		"chipID:layer:halfbarrel:stave:halfstave:module:lchip:schip:hschip:mchip:gSx:gSy:gSr:gSphi:gSz:gS1:gS2:gS3");   
   TNtuple *fTargetGeom      = new TNtuple("fTargetGeom",	"fTargetGeom",		"chipID:layer:halfbarrel:stave:halfstave:module:lchip:schip:hschip:mchip:gSx:gSy:gSr:gSphi:gSz:gS1:gS2:gS3");   
   TNtuple *fCorrectedGeom   = new TNtuple("fCorrectedGeom",	"fCorrectedGeom",	"chipID:layer:halfbarrel:stave:halfstave:module:lchip:schip:hschip:mchip:gSx:gSy:gSr:gSphi:gSz:gS1:gS2:gS3");   

   TString SIdealGeom   = "gSxI:gSyI:gSrI:gSphiI:gSzI:gS1I:gS2I:gS3I";
   TString STargetGeom  = "gSxT:gSyT:gSrT:gSphiT:gSzT:gS1T:gS2T:gS3T";
   TString SCorrectGeom = "gSxC:gSyC:gSrC:gSphiC:gSzC:gS1C:gS2C:gS3C";

   TString STotGeom = "chipID:layer:halfbarrel:stave:halfstave:module:lchip:schip:hschip:mchip:" + SIdealGeom + ":" + STargetGeom + ":" + SCorrectGeom;

   TNtuple *fGeom            = new TNtuple("fGeom",	        "fGeom",         	STotGeom);   

   std::cout<<"Ideal Geom"<<std::endl;
   GetGeometry(NetworkIdealDir,	    fIdealGeom,    res_level);
   std::cout<<"Target Geom"<<std::endl;
   GetGeometry(NetworkTargetDir,    fTargetGeom,   res_level);
   std::cout<<"Corrected Geom"<<std::endl;
   GetGeometry(NetworkCorrectedDir, fCorrectedGeom,res_level);

   TotalizeGeometry(fIdealGeom,fTargetGeom,fCorrectedGeom,fGeom);

   fGeomFile->cd();
   fGeomFile->Write();  
   
   TString fExec_argument1 = "mv XXXXGeom.root ./Geom/XXXXGeom.root";

   gSystem->Exec(fExec_argument1);
}

////////////////////////////////////////////////////////////////////////////////
/// GetIdealGeometry

void YAlignment::GetGeometry(TString Network_dir, TNtuple* fNtuple, int res_level)
{
   
                                    //0  1    2    3    4     5     6      7
   int SensorBoundary[nLAYER + 1] = { 0, 108, 252, 432, 3120, 6480, 14712, 24120 };	

   int layer = 0;
   int nrow = std::pow(2,res_level);
   int prow =  512/nrow;
   int ncol = std::pow(2,res_level+1);
   int pcol = 1024/ncol;

   fMLPNetwork = new YMultiLayerPerceptron(fNetworkStructure,fInputTree,"(evno%10)>=0&&(evno%10)<6","(evno%10)>=6&&(evno%10)<8"); 
   fMLPNetwork->SetNpronged(nTrackMax);  
   fMLPNetwork->SetFitModel(FITMODEL);

   double** input_Max;
   double** input_Min;
   double** norm;
   
   input_Max = new double *[2];
   input_Min = new double *[2];  
   norm      = new double *[2];    
   for(int axis = 0; axis <2; axis++){
      input_Max[axis] = new double [nSensors];
      input_Min[axis] = new double [nSensors];  
      norm[axis]      = new double [nSensors];  
      for(int iID=0; iID<nSensors; iID++){  
         input_Max[axis][iID] = 0;
         input_Min[axis][iID] = 0; 
         norm[axis][iID]      = 0; 
      }    
   }

   for(int iID=0; iID<nSensors; iID++){
      double ip1 = yGEOM->GToS(iID,yGEOM->LToG(iID,0,0)(0),	
 				   yGEOM->LToG(iID,0,0)(1),
                        	   yGEOM->LToG(iID,0,0)(2))(0);
      double fp1 = yGEOM->GToS(iID,yGEOM->LToG(iID,511,1023)(0),
	                           yGEOM->LToG(iID,511,1023)(1),
		     	           yGEOM->LToG(iID,511,1023)(2))(0); 
      double ip2 = yGEOM->GToS(iID,yGEOM->LToG(iID,0,0)(0),
                                   yGEOM->LToG(iID,0,0)(1),
                                   yGEOM->LToG(iID,0,0)(2))(1);
      double fp2 = yGEOM->GToS(iID,yGEOM->LToG(iID,511,1023)(0),	
                                   yGEOM->LToG(iID,511,1023)(1),
                                   yGEOM->LToG(iID,511,1023)(2))(1);               
      input_Max[0][iID]=std::max(ip1,fp1);
      input_Min[0][iID]=std::min(ip1,fp1); 
      input_Max[1][iID]=std::max(ip2,fp2);
      input_Min[1][iID]=std::min(ip2,fp2);             
      norm[0][iID]=input_Max[0][iID]-input_Min[0][iID];          
      norm[1][iID]=input_Max[1][iID]-input_Min[1][iID];       
#ifdef YALIGNDEBUG
      std::cout<<"Sensor Boundary ChipID["<<iID<<"] "<<input_Max[0][iID]<<" "<<input_Min[0][iID]<<" "
                                                     <<input_Max[1][iID]<<" "<<input_Min[1][iID]<<" Norm (s1, s2) "<<norm[0][iID]<<" "<<norm[1][iID]<<std::endl;
#endif         
   }  
         
   TString prev_weight = Network_dir + "weights.txt";
   SetPrevWeight(prev_weight);
   fMLPNetwork->SetPrevWeight(fPrevWeights); 
   fMLPNetwork->Init_Randomize();
   fMLPNetwork->Init_RandomizeSensorCorrection();
   fMLPNetwork->LoadWeights(fPrevWeights);
   fMLPNetwork->PrintCurrentWeights();   

   for(int ichipID = 0; ichipID <nSensors; ichipID++ ){

      int sensor_layer = yGEOM->GetLayer(ichipID);
      int sensor_stave = yGEOM->GetStave(ichipID);

      int sensor_Layer			= yGEOM->GetLayer(ichipID); 
      int sensor_HalfBarrel		= yGEOM->GetHalfBarrel(ichipID);  
      int sensor_Stave			= yGEOM->GetStave(ichipID); 
      int sensor_HalfStave		= yGEOM->GetHalfStave(ichipID);  
      int sensor_Module			= yGEOM->GetModule(ichipID);  
      int sensor_ChipIdInLayer		= yGEOM->GetChipIdInLayer(ichipID); 
      int sensor_ChipIdInStave		= yGEOM->GetChipIdInStave(ichipID); 
      int sensor_ChipIdInHalfStave	= yGEOM->GetChipIdInHalfStave(ichipID); 
      int sensor_ChipIdInModule		= yGEOM->GetChipIdInModule(ichipID); 


      if(ichipID>=SensorBoundary[layer+1]) layer++;
      //if(layer>2) break;
      std::cout<<" * chipID "<<ichipID<<" layer "<<layer<<" [check] layer stave chip "<<sensor_Layer<<" "<<sensor_Stave<<" "<<sensor_ChipIdInStave<<std::endl;

      for(int srow = 0 ; srow < nrow; srow++){
         for(int scol = 0 ; scol < ncol; scol++){
            int irow = prow*(srow + 0.5);
            int icol = pcol*(scol + 0.5);
            //std::cout<<" ** irow "<<irow<<" icol "<<icol<<std::endl;
        
            float sgSx   = yGEOM->LToG(ichipID,irow,icol).X();
            float sgSy   = yGEOM->LToG(ichipID,irow,icol).Y();
            float sgSr   = TMath::Sqrt(sgSx*sgSx + sgSy*sgSy);
            float sgSphi = TMath::ATan2(sgSy,sgSx);
                  sgSphi = ( sgSphi >= 0 ) ? sgSphi : 2*TMath::ATan2(0,-1) + sgSphi;
            float sgSz   = yGEOM->LToG(ichipID,irow,icol).Z();         

            float sgS1   = yGEOM->GToS(ichipID,sgSx,sgSy,sgSz)(0);
            float sgS2   = yGEOM->GToS(ichipID,sgSx,sgSy,sgSz)(1);
            float sgS3   = 0;

            TVector3 sensorS = yGEOM->GToS(ichipID,sgSx,sgSy,sgSz);

            double mlpInput[2];

            mlpInput[0]   = (float)((sensorS(0)-input_Min[0][ichipID])/norm[0][ichipID] - fNorm_shift);
            mlpInput[1]	  = (float)((sensorS(1)-input_Min[1][ichipID])/norm[1][ichipID] - fNorm_shift); 

            double mlpOutput[3];
              
            mlpOutput[0]  = fMLPNetwork->Evaluate(0, mlpInput,ichipID);
            mlpOutput[1]  = fMLPNetwork->Evaluate(1, mlpInput,ichipID);
            mlpOutput[2]  = fMLPNetwork->Evaluate(2, mlpInput,ichipID);

            float fgS1   = sensorS(0) + mlpOutput[0];
            float fgS2   = sensorS(1) + mlpOutput[1];
            float fgS3   = mlpOutput[2];

            float fgSx   = yGEOM->SToG(ichipID,fgS1,fgS2,fgS3).X();
            float fgSy   = yGEOM->SToG(ichipID,fgS1,fgS2,fgS3).Y();
            float fgSr   = TMath::Sqrt(fgSx*fgSx + fgSy*fgSy);
            float fgSphi = TMath::ATan2(fgSy,fgSx);
                  fgSphi = ( fgSphi >= 0 ) ? fgSphi : 2*TMath::ATan2(0,-1) + fgSphi;

            float fgSz   = yGEOM->SToG(ichipID,fgS1,fgS2,fgS3).Z(); 

      int sensor_Layer			= yGEOM->GetLayer(ichipID); 
      int sensor_HalfBarrel		= yGEOM->GetHalfBarrel(ichipID);  
      int sensor_Stave			= yGEOM->GetStave(ichipID); 
      int sensor_HalfStave		= yGEOM->GetHalfStave(ichipID);  
      int sensor_Module			= yGEOM->GetModule(ichipID);  
      int sensor_ChipIdInLayer		= yGEOM->GetChipIdInLayer(ichipID); 
      int sensor_ChipIdInStave		= yGEOM->GetChipIdInStave(ichipID); 
      int sensor_ChipIdInHalfStave	= yGEOM->GetChipIdInHalfStave(ichipID); 
      int sensor_ChipIdInModule		= yGEOM->GetChipIdInModule(ichipID);

            //chipID:layer:halfbarrel:stave:halfstave:module:lchip:schip:hschip:mchip:gSx:gSy:gSr:gSphi:gSz:gS1:gS2:gS3
            Float_t fntuple[] = {(float)ichipID,
                                 (float)sensor_Layer,(float)sensor_HalfBarrel,
				 (float)sensor_Stave,(float)sensor_HalfStave,
				 (float)sensor_Module,
				 (float)sensor_ChipIdInLayer,(float)sensor_ChipIdInStave,(float)sensor_ChipIdInHalfStave,(float)sensor_ChipIdInModule,
                                 (float)fgSx, (float)fgSy,(float)fgSr,(float)fgSphi,(float)fgSz,
                                 (float)fgS1, (float)fgS2,(float)fgS3};
            fNtuple->Fill(fntuple);   
         }
      }
   }
}
////////////////////////////////////////////////////////////////////////////////
/// TotalizeGeometry

void YAlignment::TotalizeGeometry(TNtuple* fNtupleI, TNtuple* fNtupleT, TNtuple* fNtupleC, TNtuple* fNtuple)
{
   float b_chipID[3];

   float b_Layer[3];
   float b_HalfBarrel[3];
   float b_Stave[3];
   float b_HalfStave[3];
   float b_Module[3];
   float b_ChipIdInLayer[3];
   float b_ChipIdInStave[3];
   float b_ChipIdInHalfStave[3];
   float b_ChipIdInModule[3];

   float b_gSx[3];
   float b_gSy[3];
   float b_gSr[3];
   float b_gSphi[3];
   float b_gSz[3];
   float b_gS1[3];
   float b_gS2[3];
   float b_gS3[3];
   //chipID:layer:halfbarrel:stave:halfstave:module:lchip:schip:hschip:mchip:gSx:gSy:gSr:gSphi:gSz:gS1:gS2:gS3
   //Ideal
   fNtupleI->SetBranchAddress("chipID",   	&b_chipID[0]); 

   fNtupleI->SetBranchAddress("layer",    	&b_Layer[0]); 
   fNtupleI->SetBranchAddress("halfbarrel",   	&b_HalfBarrel[0]); 
   fNtupleI->SetBranchAddress("stave",    	&b_Stave[0]); 
   fNtupleI->SetBranchAddress("halfstave", 	&b_HalfStave[0]); 
   fNtupleI->SetBranchAddress("module",    	&b_Module[0]); 
   fNtupleI->SetBranchAddress("lchip",    	&b_ChipIdInLayer[0]); 
   fNtupleI->SetBranchAddress("schip",    	&b_ChipIdInStave[0]); 
   fNtupleI->SetBranchAddress("hschip",    	&b_ChipIdInHalfStave[0]); 
   fNtupleI->SetBranchAddress("mchip",    	&b_ChipIdInModule[0]); 

   fNtupleI->SetBranchAddress("gSx",      &b_gSx[0]); 
   fNtupleI->SetBranchAddress("gSy",      &b_gSy[0]); 
   fNtupleI->SetBranchAddress("gSr",      &b_gSr[0]); 
   fNtupleI->SetBranchAddress("gSphi",    &b_gSphi[0]); 
   fNtupleI->SetBranchAddress("gSz",      &b_gSz[0]); 
   fNtupleI->SetBranchAddress("gS1",      &b_gS1[0]); 
   fNtupleI->SetBranchAddress("gS2",      &b_gS2[0]); 
   fNtupleI->SetBranchAddress("gS3",      &b_gS3[0]); 
   //Target
   fNtupleT->SetBranchAddress("chipID",   &b_chipID[1]); 

   fNtupleT->SetBranchAddress("layer",    	&b_Layer[1]); 
   fNtupleT->SetBranchAddress("halfbarrel",   	&b_HalfBarrel[1]); 
   fNtupleT->SetBranchAddress("stave",    	&b_Stave[1]); 
   fNtupleT->SetBranchAddress("halfstave", 	&b_HalfStave[1]); 
   fNtupleT->SetBranchAddress("module",    	&b_Module[1]); 
   fNtupleT->SetBranchAddress("lchip",    	&b_ChipIdInLayer[1]); 
   fNtupleT->SetBranchAddress("schip",    	&b_ChipIdInStave[1]); 
   fNtupleT->SetBranchAddress("hschip",    	&b_ChipIdInHalfStave[1]); 
   fNtupleT->SetBranchAddress("mchip",    	&b_ChipIdInModule[1]); 

   fNtupleT->SetBranchAddress("gSx",      &b_gSx[1]); 
   fNtupleT->SetBranchAddress("gSy",      &b_gSy[1]); 
   fNtupleT->SetBranchAddress("gSr",      &b_gSr[1]); 
   fNtupleT->SetBranchAddress("gSphi",    &b_gSphi[1]); 
   fNtupleT->SetBranchAddress("gSz",      &b_gSz[1]); 
   fNtupleT->SetBranchAddress("gS1",      &b_gS1[1]); 
   fNtupleT->SetBranchAddress("gS2",      &b_gS2[1]); 
   fNtupleT->SetBranchAddress("gS3",      &b_gS3[1]); 
   //Correct
   fNtupleC->SetBranchAddress("chipID",   &b_chipID[2]); 

   fNtupleC->SetBranchAddress("layer",    	&b_Layer[2]); 
   fNtupleC->SetBranchAddress("halfbarrel",   	&b_HalfBarrel[2]); 
   fNtupleC->SetBranchAddress("stave",    	&b_Stave[2]); 
   fNtupleC->SetBranchAddress("halfstave", 	&b_HalfStave[2]); 
   fNtupleC->SetBranchAddress("module",    	&b_Module[2]); 
   fNtupleC->SetBranchAddress("lchip",    	&b_ChipIdInLayer[2]); 
   fNtupleC->SetBranchAddress("schip",    	&b_ChipIdInStave[2]); 
   fNtupleC->SetBranchAddress("hschip",    	&b_ChipIdInHalfStave[2]); 
   fNtupleC->SetBranchAddress("mchip",    	&b_ChipIdInModule[2]); 

   fNtupleC->SetBranchAddress("gSx",      &b_gSx[2]); 
   fNtupleC->SetBranchAddress("gSy",      &b_gSy[2]); 
   fNtupleC->SetBranchAddress("gSr",      &b_gSr[2]); 
   fNtupleC->SetBranchAddress("gSphi",    &b_gSphi[2]); 
   fNtupleC->SetBranchAddress("gSz",      &b_gSz[2]); 
   fNtupleC->SetBranchAddress("gS1",      &b_gS1[2]); 
   fNtupleC->SetBranchAddress("gS2",      &b_gS2[2]); 
   fNtupleC->SetBranchAddress("gS3",      &b_gS3[2]); 

   int NentriesI = fNtupleI->GetEntries();
   int NentriesT = fNtupleT->GetEntries();
   int NentriesC = fNtupleC->GetEntries();

   std::cout<<"TotalizeGeometry : "<<NentriesI<<" "<<NentriesT<<" "<<NentriesC<<std::endl;
  
   for(int i = 0; i<NentriesI; i++){

      fNtupleI->GetEntry(i);
      fNtupleT->GetEntry(i);
      fNtupleC->GetEntry(i);
      Float_t fntuple[] = {(float)b_chipID[0],
			   (float)b_Layer[0],(float)b_HalfBarrel[0],
			   (float)b_Stave[0],(float)b_HalfStave[0],
			   (float)b_Module[0],
			   (float)b_ChipIdInLayer[0],(float)b_ChipIdInStave[0],(float)b_ChipIdInHalfStave[0],(float)b_ChipIdInModule[0],
                           (float)b_gSx[0],(float)b_gSy[0],(float)b_gSr[0],(float)b_gSphi[0],(float)b_gSz[0],(float)b_gS1[0],(float)b_gS2[0],(float)b_gS3[0],
                           (float)b_gSx[1],(float)b_gSy[1],(float)b_gSr[1],(float)b_gSphi[1],(float)b_gSz[1],(float)b_gS1[1],(float)b_gS2[1],(float)b_gS3[1],
                           (float)b_gSx[2],(float)b_gSy[2],(float)b_gSr[2],(float)b_gSphi[2],(float)b_gSz[2],(float)b_gS1[2],(float)b_gS2[2],(float)b_gS3[2]};
      fNtuple->Fill(fntuple);   
   }
}

 
////////////////////////////////////////////////////////////////////////////////
/// AddSensorSet

void YAlignment::AddSensorSet(YSensorSet sensorset)
{

   fSensorset.push_back(sensorset);

   for(int j=0; j<sensorset.GetEntries(); j++){
      std::cout<<"AddSensorSet["<<fSensorset.size()-1<<"]: layer "<<fSensorset[fSensorset.size()-1].Getlayer(j)
                                                       <<" stave "<<fSensorset[fSensorset.size()-1].Getstave(j)
                                                       <<" chip "<<fSensorset[fSensorset.size()-1].Getchip(j)<<std::endl;                                     
   }


}
