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
// YFitModel
//
//
///////////////////////////////////////////////////////////////////////////

#include "../inc/YFitModel.h"

static void at_exit_of_YFitModel() {
   if (FIT::Internal::yFITLocal)
      FIT::Internal::yFITLocal->~YFitModel();
}

// This local static object initializes the FIT system
namespace FIT {
namespace Internal {
   class YFitModelAllocator {

      char fHolder[sizeof(YFitModel)];
   public:
      YFitModelAllocator() {
         new(&(fHolder[0])) YFitModel();
      }

      ~YFitModelAllocator() {
         if (yFITLocal) {
            yFITLocal->~YFitModel();
         }
      }
   };

   extern YFitModel *yFITLocal;

   YFitModel *GetFIT1() {
      if (yFITLocal)
         return yFITLocal;
      static YFitModelAllocator alloc;
      return yFITLocal;
   }

   YFitModel *GetFIT2() {
      static Bool_t initInterpreter = kFALSE;
      if (!initInterpreter) {
         initInterpreter = kTRUE;
      }
      return yFITLocal;
   }
   typedef YFitModel *(*GetFITFun_t)();

   static GetFITFun_t yGetFIT = &GetFIT1;


} // end of Internal sub namespace
// back to FIT namespace

   YFitModel *GetFIT() {
      return (*Internal::yGetFIT)();
   }
}

YFitModel *FIT::Internal::yFITLocal = FIT::GetFIT();

ClassImp(YFitModel);

////////////////////////////////////////////////////////////////////////////////
/// Default Constructor YFitModel

YFitModel::YFitModel() {
   std::cout<<"Default Constructor YFitModel "<<std::endl;
   //These are supposed to be defined as constant header.
   fB     = 0;

   FIT::Internal::yFITLocal = this;
   FIT::Internal::yGetFIT = &FIT::Internal::GetFIT2;   
}

////////////////////////////////////////////////////////////////////////////////
/// Fit by selected model

void YFitModel::Fit(double* input, double* par, double &MSEvalue, int hitentries=3, YFitModel::EModel model = YFitModel::kLine){

   switch (model) {
      case YFitModel::kLine: {
         line3Dfit(input, par, MSEvalue, hitentries);
         break;
      }
      case YFitModel::kCircle: {
         //circle3Dfit(input, par, MSEvalue, hitentries);
         break;
      }      
      default:
      ;
   } 
} 
  
////////////////////////////////////////////////////////////////////////////////
/// Vertex Estimator

void YFitModel::EstimateVertex(double** input, double* kappa, double* vgz, int hitentries=3, int trackentries=2, YFitModel::EModel model = YFitModel::kLine){

   switch (model) {
      case YFitModel::kLine: {
         std::cout<<"Estimated Vertex by Straight Line Tracks"<<std::endl; 
         break;
      }
      case YFitModel::kCircle: {
         std::cout<<"Estimated Vertex by Circle(Helix) Tracks under B-Field"<<std::endl;            
         break;
      }      
      default:
      ;
   }
}
////////////////////////////////////////////////////////////////////////////////
/// DeltaZ

void YFitModel::DeltaZ(double* kappa, double* vgz, double* dvgz, int trackentries=2){

   for(int i =0; i < trackentries; i++){
      double vgzmean = 0;
      std::cout<<" YFitModel::DeltaZ "<<i<<" ";
      for(int j =0; j < trackentries; j++){
         if(i==j) continue;
         std::cout<<"["<<j<<"] "<<vgz[j]<<" ";
         vgzmean += (double)vgz[j]/(double)(trackentries-1);
      } 
      std::cout<<"vgz vgzmean "<<vgz[i]<<" "<<vgzmean<<" dvgz "<<dvgz[i]<<std::endl;
      dvgz[i] = vgz[i] - vgzmean;
   }  
}

////////////////////////////////////////////////////////////////////////////////
/// Clustering by DBSCAN

void YFitModel::Clustering(double* vgz, int* index, int trackentries=2, YFitModel::EClusterMethod method = YFitModel::kDBSCAN, double epsilon = 0.1, bool ascending =true){

   //std::cout<<"Clustering"<<std::endl;
   double npmin   = 1;
     
   int tag = 0;
   if(ascending==true){
      for(int i =0; i < trackentries; i++){
      //for(int i =trackentries-1; i >=0; i--){
         std::cout<<"vgz ["<<i<<"] "<<vgz[i]<<std::endl;
         if(index[i]!=-1) continue; 
         vector<int> itrack;
         itrack.push_back(i);
         index[i] = tag;
         for(int j =i+1; j < trackentries; j++){
         //for(int j =i-1; j >=0; j--){ 
            if(index[j]!=-1) continue;       
            itrack.push_back(j);
            double distance = std::abs(vgz[j] - vgz[i]);
            std::cout<<" >> "<<j<<" dist = "<<distance<<std::endl;
            if(distance<epsilon) {
               index[j] = tag;
            }
         }
         if(itrack.size()<npmin){
            for(int k =0; k < itrack.size(); k++){
               index[itrack[k]] = -1;
            } 
            continue;  
         }
         tag++;
      }  
   } else {
      //for(int i =0; i < trackentries; i++){
      for(int i =trackentries-1; i >=0; i--){
         std::cout<<"vgz ["<<i<<"] "<<vgz[i]<<std::endl;
         if(index[i]!=-1) continue; 
         vector<int> itrack;
         itrack.push_back(i);
         index[i] = tag;
         //for(int j =i+1; j < trackentries; j++){
         for(int j =i-1; j >=0; j--){ 
            if(index[j]!=-1) continue;       
            itrack.push_back(j);
            double distance = std::abs(vgz[j] - vgz[i]);
            std::cout<<" >> "<<j<<" dist = "<<distance<<std::endl;
            if(distance<epsilon) {
               index[j] = tag;
            }
         }
         if(itrack.size()<npmin){
            for(int k =0; k < itrack.size(); k++){
               index[itrack[k]] = -1;
            } 
            continue;  
         }
         tag++;
      }    
   }
   std::cout<<"index : ";
   for(int i =0; i < trackentries; i++){
      std::cout<<index[i]<<" ";
   }
   std::cout<<std::endl;
}



