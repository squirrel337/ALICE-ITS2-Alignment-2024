// @(#)location
// Author: J.H.Kim

/*************************************************************************
 *   Yonsei Univ.                                                        *
 *                                                                       *
 *                                                                       *
 *                                                                       *
 *                                                                       *
 *************************************************************************/

#ifndef ROOT_YSensorSet
#define ROOT_YSensorSet

#include "TObject.h"
#include "TString.h"

//____________________________________________________________________
//
// YSensorSet
//
// Description
//
//--------------------------------------------------------------------

class YSensorSet {

public:
   YSensorSet();
   YSensorSet(int index);
   virtual ~YSensorSet() {};  
      
   void AddSensor(int layer, int stave, int chip);
   void ResetSensorSet();
   int GetEntries(){ return flayer.size(); }
   int Getlayer(int i){ return flayer[i]; }
   int Getstave(int i){ return fstave[i]; }
   int Getchip(int i){ return fchip[i]; }     
   
private:
   int 			findex;				// SensorSet Index
   vector<int> 		flayer;				// Sensor layer index
   vector<int> 		fstave;				// Sensor stave index
   vector<int> 		fchip;				// Sensor chip index
   
};

#endif
