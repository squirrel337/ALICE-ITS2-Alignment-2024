// @(#)root/mlp:$Id$
// Author: Christophe.Delaere@cern.ch   21/08/2002

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////////////
//
// YSynapse
//
// This is a simple weighted bidirectionnal connection between
// two neurons.
// A network is built connecting two neurons by a synapse.
// In addition to the value, the synapse can return the DcDw
//
///////////////////////////////////////////////////////////////////////////

#include "../inc/YSynapse.h"
#include "../inc/YNeuron.h"
#include "Riostream.h"

ClassImp(YSynapse);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

YSynapse::YSynapse()
{
   fpre    = 0;
   fpost   = 0;
   fweight = 1;
   fDCDw   = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor that connects two neurons

YSynapse::YSynapse(YNeuron * pre, YNeuron * post, Double_t w)
{
   fpre    = pre;
   fpost   = post;
   fweight = w;
   fDCDw   = 0;
   pre->AddPost(this);
   post->AddPre(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the pre-neuron

void YSynapse::SetPre(YNeuron * pre)
{
   if (pre) {
      Error("SetPre","this synapse is already assigned to a pre-neuron.");
      return;
   }
   fpre = pre;
   pre->AddPost(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the post-neuron

void YSynapse::SetPost(YNeuron * post)
{
   if (post) {
      Error("SetPost","this synapse is already assigned to a post-neuron.");
      return;
   }
   fpost = post;
   post->AddPre(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the value: weithted input

Double_t YSynapse::GetValue() const
{
   if (fpre)
      return (fweight * fpre->GetValue());
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Computes the derivative of the error wrt the synapse weight.

Double_t YSynapse::GetDcDw() const
{
  
   if (!(fpre && fpost)) {
      return 0;
   }
   //std::cout<<"YSynapse::GetDcDw pre post : "<<fpre->GetValue()<<" "<<fpost->GetDcDw()<<std::endl;
   return (fpre->GetValue() * fpost->GetDcDw());
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the weight of the synapse.
/// This weight is the multiplying factor applied on the
/// output of a neuron in the linear combination given as input
/// of another neuron.

void YSynapse::SetWeight(Double_t w)
{
   fweight = w;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the derivative of the total error wrt the synapse weight

void YSynapse::SetDCDw(Double_t in)
{
   //std::cout<<"*YSynapse::SetDCDw "<<in<<std::endl;
   fDCDw = in;
}



