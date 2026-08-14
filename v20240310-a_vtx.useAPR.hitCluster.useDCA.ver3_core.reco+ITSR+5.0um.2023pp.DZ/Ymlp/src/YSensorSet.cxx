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
// YSensorSet
//
//
///////////////////////////////////////////////////////////////////////////

#include "../inc/YSensorSet.h"

////////////////////////////////////////////////////////////////////////////////
/// Default Constructor YSensorSet

YSensorSet::YSensorSet() {
   findex =0;
   std::cout<<"Default Constructor YSensorSet "<<findex<<std::endl;
   std::cout<<"Please Select Sensors"<<std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor Type 1 YSensorSet

YSensorSet::YSensorSet(int index) {
   findex = index;
   std::cout<<"Constructor YSensorSet "<<findex<<std::endl;
   std::cout<<"Please Select Sensors"<<std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/// AddSensor

void YSensorSet::AddSensor(int layer, int stave, int chip)
{
   flayer.push_back(layer);				
   fstave.push_back(stave);				
   fchip.push_back(chip);				
}

////////////////////////////////////////////////////////////////////////////////
/// ResetSensorSet

void YSensorSet::ResetSensorSet()
{
   flayer.clear();				
   fstave.clear();			
   fchip.clear();
}
