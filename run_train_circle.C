#include "./Ymlp/src/YAlignment.cxx"
#include "YMLPParallel.h"
int run_train_circle(int step) {
   YMultiLayerPerceptron::ELearningMethod method = YMultiLayerPerceptron::kStochastic; //kSteepestDescent kStochastic kBatch
 
   YAlignment* yalign = new YAlignment();
   
   yalign->LoadNetworkUpdateList(false);
   //yalign->SetNetworkUpdateListLayerStave(NUlayer, NUstave);
   yalign->SetSourceDataName("./XXXXinput.root");
   yalign->SetSourceTreeName("DataInput"); //mlp_Input_S
   
   yalign->SetEpoch(nEPOCH);
   yalign->SetStep(step); //step

   vector<int> fhiddenlayer;
   fhiddenlayer.push_back(0); 
   yalign->SetHiddenLayer(fhiddenlayer);
   yalign->PrepareData(nDATA,jparallel);
   
   std::cout<<"Start TrainMLP !"<<std::endl;
   if(step==0){
      TString prevWeightSet = "";
      yalign->TrainMLP(method);
   }else{
      yalign->SetPrevUSL(Form("./MLPTrain_Step%d/UpdateSensorsList.txt",step-1)); 
      yalign->SetPrevWeight(Form("./MLPTrain_Step%d/weights/weights.txt",step-1)); 
      yalign->SetPrevWeightDU(Form("./MLPTrain_Step%d/weights/weightsDU.txt",step-1)); 
      yalign->TrainMLP(method);
   }
   gSystem->Exec(Form("mv UpdateSensorsList.txt MLPTrain/UpdateSensorsList.txt"));

   gSystem->Exec(Form("mkdir MLPTrain/Res_Monitor"));
   
   for(int l = 0; l < nLAYER; l++){
      gSystem->Exec(Form("mv ResMean_Layer%d_Epoch*.gif MLPTrain/Res_Monitor/",l));      
   }
   
   gSystem->Exec(Form("mkdir MLPTrain/Residual"));
   
   gSystem->Exec(Form("mv Residual_Epoch_At_*.root MLPTrain/Residual/"));
   gSystem->Exec(Form("mv Residual_Monitor_Epoch_At_*.root MLPTrain/Residual/"));
   gSystem->Exec(Form("mkdir MLPTrain/Track_Monitor"));
   
   gSystem->Exec(Form("mv Tracks*Layer*.gif MLPTrain/Track_Monitor/"));

   if(nEPOCH>0) {
      gSystem->Exec(Form("mkdir MLPTrain/Cost_Gradient"));

      gSystem->Exec(Form("mv CostGradient_Epoch_At_*.txt MLPTrain/Cost_Gradient/"));

      gSystem->Exec(Form("tar -zxvf TrendingNetwork.tgz"));

      // The trending macros load their own copy of YDetectorGeometry from inside
      // TrendingNetwork/, so they need the geometry the same way the job does. With the
      // cache backend that is the cache file, not o2sim_geometry.root; the latter is
      // still copied so a YGEOM_USE_O2 build keeps working.
      gSystem->Exec(Form("cp o2sim_geometry.root TrendingNetwork/o2sim_geometry.root"));
      gSystem->Exec(Form("mkdir -p TrendingNetwork/geometry"));
      gSystem->Exec(Form("cp geometry/its2_geom.root TrendingNetwork/geometry/its2_geom.root"));
      gSystem->Exec(Form("cp MLPTrain/weights/* TrendingNetwork/weights/"));
      gSystem->Exec(Form("cp MLPTrain/Cost_Gradient/* TrendingNetwork/Cost_Gradient/"));
   
      gSystem->Exec(Form("echo \"#NTracks by Sensor\" > NTracksBySensor_Epoch-1.txt"));
      gSystem->Exec(Form("sed 's/Chip//g' MLPTrain/Cost_Gradient/CostGradient_Epoch_At_0.txt | awk '{print $3\" \"$4}' >> NTracksBySensor_Epoch-1.txt"));
   
      gSystem->Exec(Form("mv NTracksBySensor_Epoch-1.txt TrendingNetwork/NTracks_Profile/NTracksBySensor_Epoch-1.txt"));
   
      //gSystem->Exec(Form("cd TrendingNetwork;root -l -q -b run_macros.C'(-1,%d)';cd ..",nEPOCH-1)); 
      gSystem->Exec(Form("cd TrendingNetwork;root -l -q -b checkCostGradients.C'(-1,%d)';cd ..",nEPOCH-1)); 
      gSystem->Exec(Form("cd TrendingNetwork;root -l -q -b checkWeightGradients_extended.C'(-1,%d)';cd ..",nEPOCH-1)); 
      
      gSystem->Exec(Form("rm TrendingNetwork/o2sim_geometry.root"));
      gSystem->Exec(Form("rm -rf TrendingNetwork/geometry"));
      gSystem->Exec(Form("rm TrendingNetwork/weights/*"));
      gSystem->Exec(Form("rm TrendingNetwork/Cost_Gradient/*"));

   }
   gSystem->Exec(Form("mv TrendingNetwork MLPTrain/"));
   //
   gSystem->Exec(Form("mv MLPTrain MLPTrain_Step%d",step));


   return 0;
}
