// @(#)location
// Author: J.H.Kim

/*************************************************************************
 *   Yonsei Univ.                                                        *
 *                                                                       *
 *                                                                       *
 *                                                                       *
 *                                                                       *
 *************************************************************************/

#ifndef ROOT_YFitModel
#define ROOT_YFitModel

#include "LineFitError.h"
#include "CircleFitError.h"

#include "TObject.h"
#include "TString.h"
#include <vector>

//____________________________________________________________________
//
// YFitModel
//
// Description
//
//--------------------------------------------------------------------
class YFitModel;

namespace FIT {
namespace Internal {
   class YFitModelAllocator;

   YFitModel *GetFIT2();

} 
} // End ROOT::Internal


class YFitModel {

friend YFitModel *FIT::Internal::GetFIT2();

private:
   YFitModel(const YFitModel&);                   			//Not implemented
   YFitModel& operator=(const YFitModel&);        			//Not implemented

protected:
   YFitModel(const char *name, const char *title);              				  
   friend class ::FIT::Internal::YFitModelAllocator;

public:
   enum EModel { kLine, kCircle };
   enum EClusterMethod { kDBSCAN };
   
   YFitModel();
   virtual ~YFitModel() {};  
      
   void Fit(double* input, double* par, double &MSEvalue, int hitentries=3, YFitModel::EModel model = YFitModel::kLine);    	
   void EstimateVertex(double** input, double* kappa, double* vgz, int hitentries=3, int trackentries=2, YFitModel::EModel model = YFitModel::kLine);
   void DeltaZ(double* kappa, double* vgz, double* dvgz, int trackentries=2);
   void Clustering(double* vgz, int* index, int trackentries=2, YFitModel::EClusterMethod method = YFitModel::kDBSCAN, double epsilon = 0.1, bool ascending =true);

private:
   											// These are supposed to be defined in constant header.
   double 	fB; 								
   ClassDef(YFitModel,0)  								//Top level (or FIT) structure for all classes   
};

namespace FIT {
   YFitModel *GetFIT();
   namespace Internal {
   
      R__EXTERN YFitModel *yFITLocal;
   }
}

#define yFIT (FIT::GetFIT())

#endif
