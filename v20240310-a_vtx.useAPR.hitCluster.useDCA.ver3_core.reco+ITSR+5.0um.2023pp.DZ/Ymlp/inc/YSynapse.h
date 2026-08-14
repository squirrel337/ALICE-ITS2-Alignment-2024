// @(#)root/mlp:$Id$
// Author: Christophe.Delaere@cern.ch   20/07/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_YSynapse
#define ROOT_YSynapse

#include "TObject.h"
#include "TSynapse.h"
#include "DetectorConstant.h"

class YNeuron;

//____________________________________________________________________
//
// YSynapse
//
// This is a simple weighted bidirectionnal connection between
// two neurons.
// A network is built connecting two neurons by a synapse.
// In addition to the value, the synapse can return the DcDw
//
//____________________________________________________________________

class YSynapse : public TSynapse {
 public:
   YSynapse();
   YSynapse(YNeuron*, YNeuron*, Double_t w = 1);
   virtual ~YSynapse() {}
   void SetPre(YNeuron* pre);
   void SetPost(YNeuron* post);
   inline YNeuron* GetPre()  const { return fpre; }
   inline YNeuron* GetPost() const { return fpost; }
   void SetWeight(Double_t w);
   inline Double_t GetWeight() const { return fweight; }
   //Double_t GetWeight() const { return fweight; }
   Double_t GetValue() const;
   Double_t GetDcDw() const;
   void SetDCDw(Double_t in);
   Double_t GetDCDw() const { return fDCDw; }
      
 private:
   YNeuron* fpre;         // the neuron before the synapse
   YNeuron* fpost;        // the neuron after the synapse
   Double_t fweight;      // the weight of the synapse
   Double_t fDCDw;        //! the derivative of the total error wrt the synapse weight   

   ClassDef(YSynapse, 1)  // simple weighted bidirectionnal connection between 2 neurons
};

#endif
