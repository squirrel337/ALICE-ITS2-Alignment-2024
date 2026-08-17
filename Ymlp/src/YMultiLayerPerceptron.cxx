// @(#)root/mlp:$Id$
// Author: Christophe.Delaere@cern.ch   20/07/03

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////////////
//
// YMultiLayerPerceptron
//
// This class describes a neural network.
// There are facilities to train the network and use the output.
//
// The input layer is made of inactive neurons (returning the
// optionaly normalized input) and output neurons are linear.
// The type of hidden neurons is free, the default being sigmoids.
// (One should still try to pass normalized inputs, e.g. between [0.,1])
//
// The basic input is a TTree and two (training and test) TEventLists.
// Input and output neurons are assigned a value computed for each event
// with the same possibilities as for TTree::Draw().
// Events may be weighted individualy or via TTree::SetWeight().
// 6 learning methods are available: kStochastic, kBatch,
// kSteepestDescent, kRibierePolak, kFletcherReeves and kBFGS.
//
// This implementation, written by C. Delaere, is *inspired* from
// the mlpfit package from J.Schwindling et al. with some extensions:
//   * the algorithms are globally the same
//   * in TMultilayerPerceptron, there is no limitation on the number of
//     layers/neurons, while MLPFIT was limited to 2 hidden layers
//   * TMultilayerPerceptron allows you to save the network in a root file, and
//     provides more export functionalities
//   * TMultilayerPerceptron gives more flexibility regarding the normalization of
//     inputs/outputs
//   * TMultilayerPerceptron provides, thanks to Andrea Bocci, the possibility to
//     use cross-entropy errors, which allows to train a network for pattern
//     classification based on Bayesian posterior probability.
//
///////////////////////////////////////////////////////////////////////////
//BEGIN_HTML <!--
/* -->
<UL>
        <LI><P><A NAME="intro"></A><FONT COLOR="#5c8526">
        <FONT SIZE=4 STYLE="font-size: 15pt">Introduction</FONT></FONT></P>
</UL>
<P>Neural Networks are more and more used in various fields for data
analysis and classification, both for research and commercial
institutions. Some randomly chosen examples are:</P>
<UL>
        <LI><P>image analysis</P>
        <LI><P>financial movements predictions and analysis</P>
        <LI><P>sales forecast and product shipping optimisation</P>
        <LI><P>in particles physics: mainly for classification tasks (signal
        over background discrimination)</P>
</UL>
<P>More than 50% of neural networks are multilayer perceptrons. This
implementation of multilayer perceptrons is inspired from the
<A HREF="http://schwind.home.cern.ch/schwind/MLPfit.html">MLPfit
package</A> originaly written by Jerome Schwindling. MLPfit remains
one of the fastest tool for neural networks studies, and this ROOT
add-on will not try to compete on that. A clear and flexible Object
Oriented implementation has been chosen over a faster but more
difficult to maintain code. Nevertheless, the time penalty does not
exceed a factor 2.</P>
<UL>
        <LI><P><A NAME="mlp"></A><FONT COLOR="#5c8526">
        <FONT SIZE=4 STYLE="font-size: 15pt">The
        MLP</FONT></FONT></P>
</UL>
<P>The multilayer perceptron is a simple feed-forward network with
the following structure:</P>
<P ALIGN=CENTER><IMG SRC="gif/mlp.png" NAME="MLP"
ALIGN=MIDDLE WIDTH=333 HEIGHT=358 BORDER=0>
</P>
<P>It is made of neurons characterized by a bias and weighted links
between them (let's call those links synapses). The input neurons
receive the inputs, normalize them and forward them to the first
hidden layer.
</P>
<P>Each neuron in any subsequent layer first computes a linear
combination of the outputs of the previous layer. The output of the
neuron is then function of that combination with <I>f</I> being
linear for output neurons or a sigmoid for hidden layers. This is
useful because of two theorems:</P>
<OL>
        <LI><P>A linear combination of sigmoids can approximate any
        continuous function.</P>
        <LI><P>Trained with output = 1 for the signal and 0 for the
        background, the approximated function of inputs X is the probability
        of signal, knowing X.</P>
</OL>
<UL>
        <LI><P><A NAME="lmet"></A><FONT COLOR="#5c8526">
        <FONT SIZE=4 STYLE="font-size: 15pt">Learning
        methods</FONT></FONT></P>
</UL>
<P>The aim of all learning methods is to minimize the total error on
a set of weighted examples. The error is defined as the sum in
quadrature, devided by two, of the error on each individual output
neuron.</P>
<P>In all methods implemented, one needs to compute
the first derivative of that error with respect to the weights.
Exploiting the well-known properties of the derivative, especialy the
derivative of compound functions, one can write:</P>
<UL>
        <LI><P>for a neuton: product of the local derivative with the
        weighted sum on the outputs of the derivatives.</P>
        <LI><P>for a synapse: product of the input with the local derivative
        of the output neuron.</P>
</UL>
<P>This computation is called back-propagation of the errors. A
loop over all examples is called an epoch.</P>
<P>Six learning methods are implemented.</P>
<P><FONT COLOR="#006b6b"><I>Stochastic minimization</I>:</FONT> This
is the most trivial learning method. This is the Robbins-Monro
stochastic approximation applied to multilayer perceptrons. The
weights are updated after each example according to the formula:</P>
<P ALIGN=CENTER>$w_{ij}(t+1) = w_{ij}(t) + \Delta w_{ij}(t)$
</P>
<P ALIGN=CENTER>with
</P>
<P ALIGN=CENTER>$\Delta w_{ij}(t) = - \eta(\d e_p / \d w_{ij} +
\delta) + \epsilon \Deltaw_{ij}(t-1)$</P>
<P>The parameters for this method are Eta, EtaDecay, Delta and
Epsilon.</P>
<P><FONT COLOR="#006b6b"><I>Steepest descent with fixed step size
(batch learning)</I>:</FONT> It is the same as the stochastic
minimization, but the weights are updated after considering all the
examples, with the total derivative dEdw. The parameters for this
method are Eta, EtaDecay, Delta and Epsilon.</P>
<P><FONT COLOR="#006b6b"><I>Steepest descent algorithm</I>: </FONT>Weights
are set to the minimum along the line defined by the gradient. The
only parameter for this method is Tau. Lower tau = higher precision =
slower search. A value Tau = 3 seems reasonable.</P>
<P><FONT COLOR="#006b6b"><I>Conjugate gradients with the
Polak-Ribiere updating formula</I>: </FONT>Weights are set to the
minimum along the line defined by the conjugate gradient. Parameters
are Tau and Reset, which defines the epochs where the direction is
reset to the steepes descent.</P>
<P><FONT COLOR="#006b6b"><I>Conjugate gradients with the
Fletcher-Reeves updating formula</I>: </FONT>Weights are set to the
minimum along the line defined by the conjugate gradient. Parameters
are Tau and Reset, which defines the epochs where the direction is
reset to the steepes descent.</P>
<P><FONT COLOR="#006b6b"><I>Broyden, Fletcher, Goldfarb, Shanno
(BFGS) method</I>:</FONT> Implies the computation of a NxN matrix
computation, but seems more powerful at least for less than 300
weights. Parameters are Tau and Reset, which defines the epochs where
the direction is reset to the steepes descent.</P>
<UL>
        <LI><P><A NAME="use"></A><FONT COLOR="#5c8526">
        <FONT SIZE=4 STYLE="font-size: 15pt">How
        to use it...</FONT></FONT></P></LI>
</UL>
<P><FONT SIZE=3>TMLP is build from 3 classes: YNeuron, YSynapse and
YMultiLayerPerceptron. Only YMultiLayerPerceptron should be used
explicitly by the user.</FONT></P>
<P><FONT SIZE=3>YMultiLayerPerceptron will take examples from a TTree
given in the constructor. The network is described by a simple
string: The input/output layers are defined by giving the expression for
each neuron, separated by comas. Hidden layers are just described
by the number of neurons. The layers are separated by colons.
In addition, input/output layer formulas can be preceded by '@' (e.g "@out")
if one wants to also normalize the data from the TTree.
Input and outputs are taken from the TTree given as second argument.
Expressions are evaluated as for TTree::Draw(), arrays are expended in
distinct neurons, one for each index.
This can only be done for fixed-size arrays.
If the formula ends with &quot;!&quot;, softmax functions are used for the output layer.
One defines the training and test datasets by TEventLists.</FONT></P>
<P STYLE="margin-left: 2cm"><FONT SIZE=3><SPAN STYLE="background: #e6e6e6">
<U><FONT COLOR="#ff0000">Example</FONT></U><SPAN STYLE="text-decoration: none">:
</SPAN>YMultiLayerPerceptron(&quot;x,y:10:5:f&quot;,inputTree);</SPAN></FONT></P>
<P><FONT SIZE=3>Both the TTree and the TEventLists can be defined in
the constructor, or later with the suited setter method. The lists
used for training and test can be defined either explicitly, or via
a string containing the formula to be used to define them, exactly as
for a TCut.</FONT></P>
<P><FONT SIZE=3>The learning method is defined using the
YMultiLayerPerceptron::SetLearningMethod() . Learning methods are :</FONT></P>
<P><FONT SIZE=3>YMultiLayerPerceptron::kStochastic, <BR>
YMultiLayerPerceptron::kBatch,<BR>
YMultiLayerPerceptron::kSteepestDescent,<BR>
YMultiLayerPerceptron::kRibierePolak,<BR>
YMultiLayerPerceptron::kFletcherReeves,<BR>
YMultiLayerPerceptron::kBFGS<BR></FONT></P>
<P>A weight can be assigned to events, either in the constructor, either
with YMultiLayerPerceptron::SetEventWeight(). In addition, the TTree weight
is taken into account.</P>
<P><FONT SIZE=3>Finally, one starts the training with
YMultiLayerPerceptron::Train(Int_t nepoch, Option_t* options). The
first argument is the number of epochs while option is a string that
can contain: &quot;text&quot; (simple text output) , &quot;graph&quot;
(evoluting graphical training curves), &quot;update=X&quot; (step for
the text/graph output update) or &quot;+&quot; (will skip the
randomisation and start from the previous values). All combinations
are available. </FONT></P>
<P STYLE="margin-left: 2cm"><FONT SIZE=3><SPAN STYLE="background: #e6e6e6">
<U><FONT COLOR="#ff0000">Example</FONT></U>:
net.Train(100,&quot;text, graph, update=10&quot;).</SPAN></FONT></P>
<P><FONT SIZE=3>When the neural net is trained, it can be used
directly ( YMultiLayerPerceptron::Evaluate() ) or exported to a
standalone C++ code ( YMultiLayerPerceptron::Export() ).</FONT></P>
<P><FONT SIZE=3>Finaly, note that even if this implementation is inspired from the mlpfit code,
the feature lists are not exactly matching:
<UL>
        <LI><P>mlpfit hybrid learning method is not implemented</P></LI>
        <LI><P>output neurons can be normalized, this is not the case for mlpfit</P></LI>
        <LI><P>the neural net is exported in C++, FORTRAN or PYTHON</P></LI>
        <LI><P>the drawResult() method allows a fast check of the learning procedure</P></LI>
</UL>
In addition, the paw version of mlpfit had additional limitations on the number of neurons, hidden layers and inputs/outputs that does not apply to YMultiLayerPerceptron.
<!-- */
// -->END_HTML

//#include "../inc/maincoordinate.h"
#include "../inc/YMultiLayerPerceptron.h"
#include "../inc/YSynapse.h"
#include "../inc/YNeuron.h"
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
#include "TTreeFormula.h"
#include "TTreeFormulaManager.h"
#include "TMarker.h"
#include "TLine.h"
#include "TText.h"
#include "TObjString.h"
#include <stdlib.h>
#define Sensor1 1.0
#define Sensor2 1.0
#define Sensor3 1.0

#include "../../YMLPBeamProfile.h"

//#define YMLPDEBUG0
//#define YMLPDEBUG
//#define YMLPComputeDCDwDEBUG
//#define YVTXFIT
//#define YSCNEURONDEBUG

double UpdateConstant  = 0.05;
double UpdateScale     = 1.0e-7;
double DETRES          = 1000.0;//um // Eta = UpdateConstant * UpdateScale * (DETRES*1e-4)^2
double ValidWindow     = 0.25; 
double RejectWindow    = 0.25;

//double UpdateConstant  = 4.0;
//double UpdateScale     = 1.0e-4;
//double DETRES          = 1000.0;//um // Eta = UpdateConstant * UpdateScale * (DETRES*1e-4)^2
//double ValidWindow = 0.25; 
//double RejectWindow = 0.25;

//double TrackRejection = 110.0;

double TARGET_R = 1e-10;
double TARGET_D = 1/TARGET_R;

ClassImp(YMultiLayerPerceptron);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

YMultiLayerPerceptron::YMultiLayerPerceptron()
{
   if(!TClass::GetClass("TTreePlayer")) gSystem->Load("libTreePlayer");
   fNetwork.SetOwner(true);
   fFirstLayer.SetOwner(false);
   fLastLayer.SetOwner(false);
   fSynapses.SetOwner(true);
   fAddition.SetOwner(true);     
   fData = 0;
   fCurrentTree = -1;
   fCurrentTreeWeight = 1;
   fStructure = "";
   fWeight = "1";
   fTraining = 0;
   fTrainingOwner = false;
   fTest = 0;
   fTestOwner = false;
   fEventWeight = 0;
   fManager = 0;
   fNpronged = 1; 
   fLearningMethod = YMultiLayerPerceptron::kBFGS;
   fEta = .1;
   fEtaDecay = 1;
   fDelta = 0;
   fEpsilon = 0;
   fTau = 3;
   fLastAlpha = 0;
   fReset = 50;
   fType = YNeuron::kSigmoid;
   fOutType =  YNeuron::kLinear;
   //fOutType = YNeuron::kLFETanh; 
   fextF = "";
   fextD = "";
   fFitModel = 1;					//[DetectorAlignment] kLineFit
   fRandomSeed =0;
   fLayerTrain =111; 
   fWeightName ="";
   fWeightStep =1;
   fPrevUSLName ="";
   fPrevWeightName ="";
   fPrevWeightDetectorUnitName ="";
   //fEventIndex    = EventIndex(0);   
   //fTrainingIndex = EventIndex(1); 
   //fTestIndex     = EventIndex(2);       
   std::cout<<"YMultiLayerPerceptron Default"<<std::endl;	//[DetectorAlignment] kLineFit
}

////////////////////////////////////////////////////////////////////////////////
/// The network is described by a simple string:
/// The input/output layers are defined by giving
/// the branch names separated by comas.
/// Hidden layers are just described by the number of neurons.
/// The layers are separated by colons.
/// Ex: "x,y:10:5:f"
/// The output can be prepended by '@' if the variable has to be
/// normalized.
/// The output can be followed by '!' to use Softmax neurons for the
/// output layer only.
/// Ex: "x,y:10:5:c1,c2,c3!"
/// Input and outputs are taken from the TTree given as second argument.
/// training and test are the two TEventLists defining events
/// to be used during the neural net training.
/// Both the TTree and the TEventLists  can be defined in the constructor,
/// or later with the suited setter method.

YMultiLayerPerceptron::YMultiLayerPerceptron(const char * layout, TTree * data,
                                             TEventList * training,
                                             TEventList * test,
                                             YNeuron::ENeuronType type,
                                             const char* extF, const char* extD)
{
   if(!TClass::GetClass("TTreePlayer")) gSystem->Load("libTreePlayer");
   fNetwork.SetOwner(true);
   fFirstLayer.SetOwner(false);
   fLastLayer.SetOwner(false);
   fSynapses.SetOwner(true);
   fAddition.SetOwner(true);       
   fStructure = layout;
   fData = data;
   fCurrentTree = -1;
   fCurrentTreeWeight = 1;
   fTraining = training;
   fTrainingOwner = false;
   fTest = test;
   fTestOwner = false;
   fWeight = "1";
   fType = type;
   fOutType =  YNeuron::kLinear;
   //fOutType = YNeuron::kLFETanh; 
   fextF = extF;
   fextD = extD;
   fEventWeight = 0;
   fManager = 0;
   fNpronged = 1;    
   if (data) {
      BuildNetwork();
      AttachData();
   }
   fLearningMethod = YMultiLayerPerceptron::kBFGS;
   fEta = .1;
   fEpsilon = 0;
   fDelta = 0;
   fEtaDecay = 1;
   fTau = 3;
   fLastAlpha = 0;
   fReset = 50;
   fFitModel = 1;					//[DetectorAlignment] kLineFit
   fRandomSeed =0;
   fLayerTrain =111;   
   fWeightName = "";
   fWeightStep =1;
   fPrevUSLName ="";   
   fPrevWeightName ="";
   fPrevWeightDetectorUnitName ="";
   //fEventIndex    = EventIndex(0);   
   //fTrainingIndex = EventIndex(1); 
   //fTestIndex     = EventIndex(2);   
   fSplitReferenceSensor = -1;
   std::cout<<"YMultiLayerPerceptron Type A"<<std::endl;	//[DetectorAlignment] kLineFit
}

////////////////////////////////////////////////////////////////////////////////
/// The network is described by a simple string:
/// The input/output layers are defined by giving
/// the branch names separated by comas.
/// Hidden layers are just described by the number of neurons.
/// The layers are separated by colons.
/// Ex: "x,y:10:5:f"
/// The output can be prepended by '@' if the variable has to be
/// normalized.
/// The output can be followed by '!' to use Softmax neurons for the
/// output layer only.
/// Ex: "x,y:10:5:c1,c2,c3!"
/// Input and outputs are taken from the TTree given as second argument.
/// training and test are the two TEventLists defining events
/// to be used during the neural net training.
/// Both the TTree and the TEventLists  can be defined in the constructor,
/// or later with the suited setter method.

YMultiLayerPerceptron::YMultiLayerPerceptron(const char * layout,
                                             const char * weight, TTree * data,
                                             TEventList * training,
                                             TEventList * test,
                                             YNeuron::ENeuronType type,
                                             const char* extF, const char* extD)
{
   if(!TClass::GetClass("TTreePlayer")) gSystem->Load("libTreePlayer");
   fNetwork.SetOwner(true);
   fFirstLayer.SetOwner(false);
   fLastLayer.SetOwner(false);
   fSynapses.SetOwner(true);
   fAddition.SetOwner(true);    
   fStructure = layout;
   fData = data;
   fCurrentTree = -1;
   fCurrentTreeWeight = 1;
   fTraining = training;
   fTrainingOwner = false;
   fTest = test;
   fTestOwner = false;
   fWeight = weight;
   fType = type;
   fOutType =  YNeuron::kLinear;
   //fOutType = YNeuron::kLFETanh; 
   fextF = extF;
   fextD = extD;
   fEventWeight = 0;
   fManager = 0;
   fNpronged = 1;    
   if (data) {
      BuildNetwork();
      AttachData();
   }
   fLearningMethod = YMultiLayerPerceptron::kBFGS;
   fEta = .1;
   fEtaDecay = 1;
   fDelta = 0;
   fEpsilon = 0;
   fTau = 3;
   fLastAlpha = 0;
   fReset = 50;
   fFitModel = 1;					//[DetectorAlignment] kLineFit
   fRandomSeed =0;
   fLayerTrain =111;  
   fWeightName = "";  
   fWeightStep =1;
   fPrevUSLName ="";   
   fPrevWeightName ="";   
   fPrevWeightDetectorUnitName ="";
   //fEventIndex    = EventIndex(0);   
   //fTrainingIndex = EventIndex(1); 
   //fTestIndex     = EventIndex(2); 
   fSplitReferenceSensor = -1;   
   std::cout<<"YMultiLayerPerceptron Type B"<<std::endl;	//[DetectorAlignment] kLineFit
}

////////////////////////////////////////////////////////////////////////////////
/// The network is described by a simple string:
/// The input/output layers are defined by giving
/// the branch names separated by comas.
/// Hidden layers are just described by the number of neurons.
/// The layers are separated by colons.
/// Ex: "x,y:10:5:f"
/// The output can be prepended by '@' if the variable has to be
/// normalized.
/// The output can be followed by '!' to use Softmax neurons for the
/// output layer only.
/// Ex: "x,y:10:5:c1,c2,c3!"
/// Input and outputs are taken from the TTree given as second argument.
/// training and test are two cuts (see TTreeFormula) defining events
/// to be used during the neural net training and testing.
/// Example: "Entry$%2", "(Entry$+1)%2".
/// Both the TTree and the cut can be defined in the constructor,
/// or later with the suited setter method.

YMultiLayerPerceptron::YMultiLayerPerceptron(const char * layout, TTree * data,
                                             const char * training,
                                             const char * test,
                                             YNeuron::ENeuronType type,
                                             const char* extF, const char* extD)
{
   if(!TClass::GetClass("TTreePlayer")) gSystem->Load("libTreePlayer");
   fNetwork.SetOwner(true);
   fFirstLayer.SetOwner(false);
   fLastLayer.SetOwner(false);
   fSynapses.SetOwner(true);
   fAddition.SetOwner(true);       
   fStructure = layout;
   fData = data;
   fDataClass = new EventData();  
   fData->SetBranchAddress("event",      &fDataClass); 
   
   fCurrentTree = -1;
   fCurrentTreeWeight = 1;
   fTraining = new TEventList(Form("fTrainingList_%lu",(ULong_t)this));
   fTrainingOwner = true;
   fTest = new TEventList(Form("fTestList_%lu",(ULong_t)this));
   fTestOwner = true;
   fWeight = "1";
   TString testcut = test;
   if(testcut=="") testcut = Form("!(%s)",training);
   fType = type;
   //fType = YNeuron::kTanh;
   fOutType = YNeuron::kLinear;
   //fOutType = YNeuron::kLFETanh;  
   fextF = extF;
   fextD = extD;
   fEventWeight = 0;
   fManager = 0;    
   fNpronged = 1;    
   if (data) {
      BuildNetwork();   
      PrepareChipsToNetwork();
      BuildSensorCorrectionNetwork();
      BuildDetectorUnitNetwork();
      //PrintDetectorUnitNetwork();
      InitDetectorUnitNetworkUpdateState();      
      SetDetectorUnitNetworkUpdateStateUser();
      std::cout<<"YMultiLayerPerceptron::Data Cut "<<training<<" "<<testcut<<std::endl;
      data->Print();
      data->Draw(Form(">>fTrainingList_%lu",(ULong_t)this),training,"goff");
      data->Draw(Form(">>fTestList_%lu",(ULong_t)this),(const char *)testcut,"goff");
      AttachData();    
   }
   else {
      Warning("YMultiLayerPerceptron::YMultiLayerPerceptron","Data not set. Cannot define datasets");
   }
   //fLearningMethod = YMultiLayerPerceptron::kBFGS;
   //fLearningMethod = YMultiLayerPerceptron::kSteepestDescent;
   fLearningMethod = YMultiLayerPerceptron::kStochastic;
   
   //fEta = .1;
   //fEta = 1.;
   fEta = UpdateConstant*UpdateScale*std::pow(DETRES*1e-4,2);
   fEtaDecay = 1;
   fDelta = 0;
   fEpsilon = 0;
   fTau = 3;
   fLastAlpha = 0;
   fReset = 50;
   fFitModel = 1;					//[DetectorAlignment] kLineFit
   fRandomSeed =0;
   fLayerTrain = LAYERTRAIN;   
   fWeightName = "";
   fWeightStep =1;
   fPrevUSLName ="";   
   fPrevWeightName ="";   
   fPrevWeightDetectorUnitName ="";
   fTotNEvents    = 0;
   fTotNTraining  = 0; 
   fTotNTest	  = 0;        
      
   fEventIndex    = EventIndex(0);   
   fTrainingIndex = EventIndex(1); 
   fTestIndex     = EventIndex(2); 
   fSplitReferenceSensor = -1;   
   fCostMonitor = new TH1D("hChi2TOT","Chi2 Distribution",150,0,150);
   int NbinsBeamVertex = 2000;
   double rangeBeamVertexXY = 1.0;
   double rangeBeamVertexZ  = 20.0;   
   fBeamXY        = new TH2D("hBeamXY","Beam Distribution XY",NbinsBeamVertex,-rangeBeamVertexXY,+rangeBeamVertexXY,NbinsBeamVertex,-rangeBeamVertexXY,+rangeBeamVertexXY);
   fBeamZR        = new TH2D("hBeamZR","Beam Distribution ZR",NbinsBeamVertex,-rangeBeamVertexZ,+rangeBeamVertexZ,NbinsBeamVertex,0,+rangeBeamVertexXY);
   fVertexFitXY   = new TH2D("hVertexFitXY","Vertex Fit Distribution From Beam XY",NbinsBeamVertex,-rangeBeamVertexXY,+rangeBeamVertexXY,NbinsBeamVertex,-rangeBeamVertexXY,+rangeBeamVertexXY);
   fVertexFitZR   = new TH2D("hVertexFitZR","Vertex Fit Distribution From Beam ZR",NbinsBeamVertex,-rangeBeamVertexZ,+rangeBeamVertexZ,NbinsBeamVertex,0,+rangeBeamVertexXY);
   fVertexXY   = new TH2D("hVertexXY","Vertex Distribution From Beam XY",NbinsBeamVertex,-rangeBeamVertexXY,+rangeBeamVertexXY,NbinsBeamVertex,-rangeBeamVertexXY,+rangeBeamVertexXY);
   fVertexZR   = new TH2D("hVertexZR","Vertex Distribution From Beam ZR",NbinsBeamVertex,-rangeBeamVertexZ,+rangeBeamVertexZ,NbinsBeamVertex,0,+rangeBeamVertexXY); 

#ifdef MONITORONLYUPDATES
   fUPDATESENSORS = new TH1D("hUpdateSensors","Update vs ChipID",ChipBoundary[nLAYER],0,ChipBoundary[nLAYER]);
   fUPDATETRACKS  = new TH1C("hUpdateTracks","N_{track} vs Track Tag (Update)",std::pow(2,nLAYER),0,std::pow(2,nLAYER));
#endif
   fCostChargeSym            = new TH2D("fCostChargeSym","pT vs N_{tracks} by Phi(PL2)",20,0,20,400,-20,20);
   fChargeSymMonitorPositive = new TH1D("fChargeSymMonitorPositive","N_{tracks} vs Phi(PL2)",20,0,20);
   fChargeSymMonitorNegative = new TH1D("fChargeSymMonitorNegative","N_{tracks} vs Phi(PL2)",20,0,20);

   fTotNEventsLoss   = -1;
   fTotNTrainingLoss = -1;
   fTotNTestLoss     = -1;

   std::cout<<"YMultiLayerPerceptron Type C with YSensorCorrection"<<std::endl;	

   fvertex_TRKF.SetXYZ(0,0,0);
   fvertex_track_TRKF = new TVector3[nTrackMax];

   std::cout<<"Beam Profile"<<std::endl;
   std::cout<<"BeamProfileXZ0 "<<BeamProfileXZ0<<std::endl;
   std::cout<<"BeamProfileXZ1 "<<BeamProfileXZ1<<std::endl;
   std::cout<<"BeamProfileYZ0 "<<BeamProfileYZ0<<std::endl;
   std::cout<<"BeamProfileYZ1 "<<BeamProfileYZ1<<std::endl;         

   fXY = new TF1("fXY","pol2");
   fZR = new TF1("fZR","pol1");

}

////////////////////////////////////////////////////////////////////////////////
/// The network is described by a simple string:
/// The input/output layers are defined by giving
/// the branch names separated by comas.
/// Hidden layers are just described by the number of neurons.
/// The layers are separated by colons.
/// Ex: "x,y:10:5:f"
/// The output can be prepended by '@' if the variable has to be
/// normalized.
/// The output can be followed by '!' to use Softmax neurons for the
/// output layer only.
/// Ex: "x,y:10:5:c1,c2,c3!"
/// Input and outputs are taken from the TTree given as second argument.
/// training and test are two cuts (see TTreeFormula) defining events
/// to be used during the neural net training and testing.
/// Example: "Entry$%2", "(Entry$+1)%2".
/// Both the TTree and the cut can be defined in the constructor,
/// or later with the suited setter method.

YMultiLayerPerceptron::YMultiLayerPerceptron(const char * layout,
                                             const char * weight, TTree * data,
                                             const char * training,
                                             const char * test,
                                             YNeuron::ENeuronType type,
                                             const char* extF, const char* extD)
{
   if(!TClass::GetClass("TTreePlayer")) gSystem->Load("libTreePlayer");
   fNetwork.SetOwner(true);
   fFirstLayer.SetOwner(false);
   fLastLayer.SetOwner(false);
   fSynapses.SetOwner(true);
   fAddition.SetOwner(true);    
   fStructure = layout;
   fData = data;
   fCurrentTree = -1;
   fCurrentTreeWeight = 1;
   fTraining = new TEventList(Form("fTrainingList_%lu",(ULong_t)this));
   fTrainingOwner = true;
   fTest = new TEventList(Form("fTestList_%lu",(ULong_t)this));
   fTestOwner = true;
   fWeight = weight;
   TString testcut = test;
   if(testcut=="") testcut = Form("!(%s)",training);
   fType = type;
   fOutType =  YNeuron::kLinear;
   //fOutType = YNeuron::kLFETanh; 
   fextF = extF;
   fextD = extD;
   fEventWeight = 0;
   fManager = 0;
   fNpronged = 1;    
   if (data) {
      BuildNetwork();
      data->Draw(Form(">>fTrainingList_%lu",(ULong_t)this),training,"goff");
      data->Draw(Form(">>fTestList_%lu",(ULong_t)this),(const char *)testcut,"goff");
      AttachData();
   }
   else {
      Warning("YMultiLayerPerceptron::YMultiLayerPerceptron","Data not set. Cannot define datasets");
   }
   fLearningMethod = YMultiLayerPerceptron::kBFGS;
   fEta = .1;
   fEtaDecay = 1;
   fDelta = 0;
   fEpsilon = 0;
   fTau = 3;
   fLastAlpha = 0;
   fReset = 50;
   fFitModel = 1;					//[DetectorAlignment] kLineFit
   fRandomSeed =0;
   fLayerTrain =111;   
   fWeightName = ""; 
   fWeightStep =1; 
   fPrevUSLName ="";   
   fPrevWeightName ="";   
   fPrevWeightDetectorUnitName ="";
   //fEventIndex    = EventIndex(0);   
   //fTrainingIndex = EventIndex(1); 
   //fTestIndex     = EventIndex(2);
   fSplitReferenceSensor = -1;   
   std::cout<<"YMultiLayerPerceptron Type D"<<std::endl;	//[DetectorAlignment] kLineFit
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

YMultiLayerPerceptron::~YMultiLayerPerceptron()
{
   if(fTraining && fTrainingOwner) delete fTraining;
   if(fTest && fTestOwner) delete fTest;

   for(int ichipID = 0; ichipID <nSensors; ichipID++){
      delete fSCNetwork[ichipID];
   }
   
   for(int l = 0; l <nLAYER; l++){
      delete fChi2Layer[l];
      delete fpTvsResLayer[l][0];
      delete fpTvsResLayer[l][1];       
      delete fpTvsChiLayer[l][0];
      delete fpTvsChiLayer[l][1];   
#ifdef MONITORHALFSTAVEUNIT                                            
      int nHalfBarrel = 2;
      for(int hb = 0; hb < nHalfBarrel; hb++){
         int nHalfStave = NSubStave[l]; 
         for(int hs = 0; hs < nHalfStave; hs++){
            delete fpTvsResLayerHBHS[l][hb][hs][0];
            delete fpTvsResLayerHBHS[l][hb][hs][1];
            delete fpTvsChiLayerHBHS[l][hb][hs][0];
            delete fpTvsChiLayerHBHS[l][hb][hs][1];
            delete fResidualsVsZLayerHBHS[l][hb][hs][0];
            delete fResidualsVsZLayerHBHS[l][hb][hs][1];
            delete fResidualsVsPhiLayerHBHS[l][hb][hs][0];
            delete fResidualsVsPhiLayerHBHS[l][hb][hs][1]; 
            delete fProfileVsZLayerHBHS[l][hb][hs][0];
            delete fProfileVsZLayerHBHS[l][hb][hs][1];
            delete fProfileVsPhiLayerHBHS[l][hb][hs][0];
            delete fProfileVsPhiLayerHBHS[l][hb][hs][1]; 
            delete fSensorCenterVsZLayerHBHS[l][hb][hs][0];
            delete fSensorCenterVsZLayerHBHS[l][hb][hs][1];
            delete fSensorCenterVsPhiLayerHBHS[l][hb][hs][0];
            delete fSensorCenterVsPhiLayerHBHS[l][hb][hs][1]; 
         }       
      }  
#endif       
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set the data source

void YMultiLayerPerceptron::SetData(TTree * data)
{
   if (fData) {
      std::cerr << "Error: data already defined." << std::endl;
      return;
   }
   fData = data;
   if (data) {
      BuildNetwork();
      AttachData();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set the event weight

void YMultiLayerPerceptron::SetEventWeight(const char * branch)
{
   fWeight=branch;
   if (fData) {
      if (fEventWeight) {
         fManager->Remove(fEventWeight);
         delete fEventWeight;
      }
      fManager->Add((fEventWeight = new TTreeFormula("NNweight",fWeight.Data(),fData)));
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the Training dataset.
/// Those events will be used for the minimization

void YMultiLayerPerceptron::SetTrainingDataSet(TEventList* train)
{
   if(fTraining && fTrainingOwner) delete fTraining;
   fTraining = train;
   fTrainingOwner = false;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the Test dataset.
/// Those events will not be used for the minimization but for control

void YMultiLayerPerceptron::SetTestDataSet(TEventList* test)
{
   if(fTest && fTestOwner) delete fTest;
   fTest = test;
   fTestOwner = false;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the Training dataset.
/// Those events will be used for the minimization.
/// Note that the tree must be already defined.

void YMultiLayerPerceptron::SetTrainingDataSet(const char * train)
{
   if(fTraining && fTrainingOwner) delete fTraining;
   fTraining = new TEventList(Form("fTrainingList_%lu",(ULong_t)this));
   fTrainingOwner = true;
   if (fData) {
      fData->Draw(Form(">>fTrainingList_%lu",(ULong_t)this),train,"goff");
   }
   else {
      Warning("YMultiLayerPerceptron::YMultiLayerPerceptron","Data not set. Cannot define datasets");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the Test dataset.
/// Those events will not be used for the minimization but for control.
/// Note that the tree must be already defined.

void YMultiLayerPerceptron::SetTestDataSet(const char * test)
{
   if(fTest && fTestOwner) {delete fTest; fTest=0;}
   if(fTest) if(strncmp(fTest->GetName(),Form("fTestList_%lu",(ULong_t)this),10)) delete fTest;
   fTest = new TEventList(Form("fTestList_%lu",(ULong_t)this));
   fTestOwner = true;
   if (fData) {
      fData->Draw(Form(">>fTestList_%lu",(ULong_t)this),test,"goff");
   }
   else {
      Warning("YMultiLayerPerceptron::YMultiLayerPerceptron","Data not set. Cannot define datasets");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the learning method.
/// Available methods are: kStochastic, kBatch,
/// kSteepestDescent, kRibierePolak, kFletcherReeves and kBFGS.
/// (look at the constructor for the complete description
/// of learning methods and parameters)

void YMultiLayerPerceptron::SetLearningMethod(YMultiLayerPerceptron::ELearningMethod method)
{
   fLearningMethod = method;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets Eta - used in stochastic minimisation
/// (look at the constructor for the complete description
/// of learning methods and parameters)

void YMultiLayerPerceptron::SetEta(Double_t eta)
{
   fEta = eta;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets Epsilon - used in stochastic minimisation
/// (look at the constructor for the complete description
/// of learning methods and parameters)

void YMultiLayerPerceptron::SetEpsilon(Double_t eps)
{
   fEpsilon = eps;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets Delta - used in stochastic minimisation
/// (look at the constructor for the complete description
/// of learning methods and parameters)

void YMultiLayerPerceptron::SetDelta(Double_t delta)
{
   fDelta = delta;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets EtaDecay - Eta *= EtaDecay at each epoch
/// (look at the constructor for the complete description
/// of learning methods and parameters)

void YMultiLayerPerceptron::SetEtaDecay(Double_t ed)
{
   fEtaDecay = ed;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets Tau - used in line search
/// (look at the constructor for the complete description
/// of learning methods and parameters)

void YMultiLayerPerceptron::SetTau(Double_t tau)
{
   fTau = tau;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets number of epochs between two resets of the
/// search direction to the steepest descent.
/// (look at the constructor for the complete description
/// of learning methods and parameters)

void YMultiLayerPerceptron::SetReset(Int_t reset)
{
   fReset = reset;
}

////////////////////////////////////////////////////////////////////////////////
/// Load an entry into the network

void YMultiLayerPerceptron::GetEntry(Int_t entry) const
{
   if (!fData) return;
   fData->GetEntry(entry);
   if (fData->GetTreeNumber() != fCurrentTree) {
      ((YMultiLayerPerceptron*)this)->fCurrentTree = fData->GetTreeNumber();
      fManager->Notify();
      ((YMultiLayerPerceptron*)this)->fCurrentTreeWeight = fData->GetWeight();
   }
   Int_t nentries = fNetwork.GetEntriesFast();
   for (Int_t i=0;i<nentries;i++) {
      YNeuron *neuron = (YNeuron *)fNetwork.At(i);
      neuron->SetNewEvent();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Train the network.
/// nEpoch is the number of iterations.
/// option can contain:
/// - "text" (simple text output)
/// - "graph" (evoluting graphical training curves)
/// - "update=X" (step for the text/graph output update)
/// - "+" will skip the randomisation and start from the previous values.
/// - "current" (draw in the current canvas)
/// - "minErrorTrain" (stop when NN error on the training sample gets below minE
/// - "minErrorTest" (stop when NN error on the test sample gets below minE
/// All combinations are available.

void YMultiLayerPerceptron::Train(Int_t nEpoch, Option_t * option, Double_t minE)
{
   fCurrentEpoch = -1;
   Int_t i;
   TString opt = option;
   opt.ToLower();
   // Decode options and prepare training.
   Int_t verbosity = 0;
   Bool_t newCanvas = true;
   Bool_t minE_Train = false;
   Bool_t minE_Test  = false;
   if (opt.Contains("text"))
      verbosity += 1;
   if (opt.Contains("graph"))
      verbosity += 2;
   Int_t displayStepping = 1;
   if (opt.Contains("update=")) {
      TRegexp reg("update=[0-9]*");
      TString out = opt(reg);
      displayStepping = atoi(out.Data() + 7);
   }
   if (opt.Contains("current"))
      newCanvas = false;
   if (opt.Contains("minerrortrain"))
      minE_Train = true;
   if (opt.Contains("minerrortest"))
      minE_Test = true;
   TVirtualPad *canvas = 0;
   TMultiGraph *residual_plot = 0;
   TGraph *train_residual_plot = 0;
   TGraph *test_residual_plot = 0;
   if ((!fData) || (!fTraining) || (!fTest)) {
      Error("Train","Training/Test samples still not defined. Cannot train the neural network");
      return;
   }
   Info("Train","Using %d train and %d test events.",
        (int)fTraining->GetN(), (int)fTest->GetN());     
   Info("Train","Using %d train and %d test tracks.",
        (int)fTotNTraining, (int)fTotNTest);            
   // Text and Graph outputs
   if (verbosity % 2)
      std::cout << "Training the Neural Network" << std::endl;
   if (verbosity / 2) {
      residual_plot = new TMultiGraph;
      if(newCanvas)
         canvas = new TCanvas("NNtraining", "Neural Net training");
      else {
         canvas = gPad;
         if(!canvas) canvas = new TCanvas("NNtraining", "Neural Net training");
      }
      train_residual_plot = new TGraph(nEpoch+1);
      test_residual_plot  = new TGraph(nEpoch+1);
      canvas->SetLeftMargin(0.14);
      train_residual_plot->SetLineColor(4);
      test_residual_plot->SetLineColor(2);
      residual_plot->Add(train_residual_plot);
      residual_plot->Add(test_residual_plot);
      residual_plot->Draw("LA");
      if (residual_plot->GetXaxis())  residual_plot->GetXaxis()->SetTitle("Epoch");
      if (residual_plot->GetYaxis())  residual_plot->GetYaxis()->SetTitle("Error");
   }
   std::cout<<"Npronged-Track  N = "<<fNpronged<<std::endl;
   //SetNetworkMatrix();
   std::cout<<"Init_Randomize Start!"<<std::endl;
   Init_Randomize();
   Init_RandomizeSensorCorrection();
   Init_RandomizeSensorCorrectionDetectorUnit();   
   std::cout<<"Init_Randomize Done!"<<std::endl;
   std::cout<<"Event Linking Check Start!"<<std::endl;
   //EventCheck();
   std::cout<<"Event Linking Check Done!"<<std::endl;
   //ComputeDCDw();
   
   if(fLearningMethod==YMultiLayerPerceptron::kOffsetTuneByMean){
      std::cout<<"OffsetTuneByMean Step 0"<<std::endl;   
      std::cout<<" weight "<<fPrevWeightName<<std::endl;   
      LoadWeights(fPrevWeightName);
      std::cout<<" weightDU "<<fPrevWeightDetectorUnitName<<std::endl;
      LoadWeightsDetectorUnit(fPrevWeightDetectorUnitName); 
      SetWeightsDetectorUnitUser();
      TString LoadedWeightName   = fWeightName + "_Epoch_At_" + TString::Itoa(-1,10) + ".txt";
      TString LoadedWeightNameDU = fWeightName + "DU_Epoch_At_" + TString::Itoa(-1,10) + ".txt";
      DumpWeights(LoadedWeightName);
      DumpWeightsDetectorUnit(LoadedWeightNameDU);   
      std::cout<<"OffsetTuneByMean Step 1"<<std::endl;
      for(int iID = 0; iID <nSensors; iID++){  
         fWUpdatebyMean.push_back(false);
      }   
      PrepareDumpResiduals();
      ComputeTrackProfile();
      std::cout<<"Load OffsetSlopeCorrectionParameters"<<std::endl;     
      LoadOffsetSlopeCorrectionParameters("OffsetSlopeCorrectionParams.txt");
      
      for(int ic = 0; ic < ChipBoundary[nLAYER]; ic++){
         std::cout<<"ChipID"<<ic<<" ";
         for (int ipar=0; ipar<17; ipar++) { 
            std::cout<<setprecision(10)<<fOffsetTuning[ic]->ftunePAR[ipar]<<" ";
         }
         std::cout<<std::endl;
      }  
      //return;
            
#ifdef MONITORONLYUPDATES
 #if MONITORONLYUPDATES_MODE==-1
      std::cout<<" MONITORONLYUPDATES STEP -1"<<std::endl;
      GetCost(YMultiLayerPerceptron::kTraining);
      InitEventLoss();   
      InitTrackLoss();      
      SetEventLoss(1);                  
      SetTrackLoss(1);       
 #elif MONITORONLYUPDATES_MODE==0
      std::cout<<" MONITORONLYUPDATES STEP 0"<<std::endl;
      GetCost(YMultiLayerPerceptron::kTraining);
      InitEventLoss();   
      InitTrackLoss();      
      SetEventLoss(1);                  
      SetTrackLoss(1);       
 #elif MONITORONLYUPDATES_MODE==1
      std::cout<<" MONITORONLYUPDATES STEP 1"<<std::endl; 
      std::cout<<" USL "<<fPrevUSLName<<std::endl;         
      LoadUpdateSensorList(fPrevUSLName);
      DumpUpdateSensorList("UpdateSensorsList.txt");                  
 #endif
#endif       

      double I_training_E = fTraining->GetN() > GetEventLoss(1) ? GetCost(YMultiLayerPerceptron::kTraining)/(fTraining->GetN() - GetEventLoss(1)) : 0;//fTraining->GetN();
      std::cout.precision(10); 
      std::cout << "(I)Epoch: " 
                << " learn=" << I_training_E;
      std::cout << " fTotN(E) "<< fTraining->GetN()
                << " fLoss(E) "<< GetEventLoss(1);
      std::cout << " fTotN(T) "<< fTotNTraining
                << " fLoss(T) "<< GetTrackLoss(1);
      std::cout << std::endl;   
      std::cout<<"OffsetTuneByMean Step 5"<<std::endl;
      DumpResiduals(-1); 

      DumpNTracksBySensor();               
      MonitorTracksBySensor();       
      MonitorTracksBySensor1D(); 
      MonitorTracksByHalfStave1D();   
      for(int iID = 0; iID <nSensors; iID++){  
         std::cout<<"UPDATE OFFSET & SLOPE CORRECTION SENSOR LIST Layer "<<yGEOM->GetLayer(iID)<<" "<<iID<<" "<<fWUpdatebyMean[iID]<<std::endl;
      }  
#ifdef MONITORONLYUPDATES
 #if MONITORONLYUPDATES_MODE==0
      std::cout<<" MONITORONLYUPDATES STEP 0"<<std::endl;
      DumpUpdateSensorList("UpdateSensorsList.txt");            
      return;
 #endif
#endif     
      for (Int_t iepoch = 0; iepoch < nEpoch; iepoch++) {

         std::cout<<"[OffsetTuneByMean] Epoch "<<iepoch<<std::endl;
#ifdef MONITORSENSORUNITprofile
         MLP_OffsetTuneByMean();            
#endif
         double training_E = GetCost(YMultiLayerPerceptron::kTraining) / (fTraining->GetN() - GetEventLoss(1));//fTraining->GetN();
         std::cout.precision(10); 
         std::cout << "(T)Epoch: " << iepoch
                   << " learn=" << training_E;
         std::cout << " fTotN(E) "<< fTraining->GetN()
                   << " fLoss(E) "<< GetEventLoss(1);
         std::cout << " fTotN(T) "<< fTotNTraining
                   << " fLoss(T) "<< GetTrackLoss(1);
         std::cout << std::endl;          
         
         DumpResiduals(iepoch);

         //## Weight Monitoring Option 2020 10 12
         if(fWeightName!=""&&iepoch>=0) {
            if(iepoch%(int(nEpoch/fWeightStep))==0){  
               TString fStepWeightName ="";
               TString fStepWeightNameDU ="";
               if(iepoch < nEpoch) {
                  fStepWeightName   = fWeightName + "_Epoch_At_" + TString::Itoa(iepoch,10) + ".txt";
                  fStepWeightNameDU = fWeightName + "DU_Epoch_At_" + TString::Itoa(iepoch,10) + ".txt";
               } else {
                  fStepWeightName   = fWeightName + "_Epoch_At_END.txt";   
                  fStepWeightNameDU = fWeightName + "_DU_Epoch_At_END.txt";   
               }
             
               DumpWeights(fStepWeightName);
               DumpWeightsDetectorUnit(fStepWeightNameDU);
               //std::clog<<"## Monitoring : Epoch "<<iepoch<<std::endl;  
               //std::clog<<gSystem->pwd()<<std::endl;            
               //PrintCurrentWeights();        
            }
         }      
      }
      
      return;    
   }

   std::cout<<"[TAG JH] GetCost For Training/Test Set (Init) START"<<std::endl;
   // If the option "+" is not set, one has to randomize the weights first  
   
   double Init_training_E = 0;
   double Init_test_E = 0;
   if(fPrevWeightName==""){
      PrepareDumpResiduals();   
      ComputeTrackProfile();
#ifdef MONITORONLYUPDATES
 #if MONITORONLYUPDATES_MODE==-1
      std::cout<<" MONITORONLYUPDATES STEP -1"<<std::endl;
      GetCost(YMultiLayerPerceptron::kTraining);
      InitEventLoss();   
      InitTrackLoss();      
      SetEventLoss(1);                  
      SetTrackLoss(1);       
 #elif MONITORONLYUPDATES_MODE==0
      std::cout<<" MONITORONLYUPDATES STEP 0"<<std::endl;
      GetCost(YMultiLayerPerceptron::kTraining);
      InitEventLoss();   
      InitTrackLoss();      
      SetEventLoss(1);                  
      SetTrackLoss(1);       
 #elif MONITORONLYUPDATES_MODE==1
      std::cout<<" MONITORONLYUPDATES STEP 1"<<std::endl; 
      std::cout<<" USL "<<fPrevUSLName<<std::endl;         
      LoadUpdateSensorList(fPrevUSLName);
      DumpUpdateSensorList("UpdateSensorsList.txt");                  
 #endif
#endif        

      Init_training_E = fTraining->GetN() > GetEventLoss(1) ? GetCost(YMultiLayerPerceptron::kTraining)/(fTraining->GetN() - GetEventLoss(1)) : 0;//fTraining->GetN();
      double Init_training_E_chargesym = 0;
      for(int pstv = 0; pstv < 20; pstv++) Init_training_E_chargesym += fCostChargeSymSum[pstv];
      Init_training_E_chargesym = std::abs(Init_training_E_chargesym / fTotNTraining);
      std::cout<<" COSTMONITOR[TRAINING] EPOCH"<<fCurrentEpoch<<" Fit + CHSYM = "<<Init_training_E<<" + "<<Init_training_E_chargesym<<" = ";
      Init_training_E += Init_training_E_chargesym;
      std::cout<<Init_training_E<<std::endl;        
      
      DumpResiduals(-1);

      DumpNTracksBySensor();               
      MonitorTracksBySensor();       
      MonitorTracksBySensor1D(); 
      MonitorTracksByHalfStave1D();  

#ifdef MONITORONLYUPDATES
 #if MONITORONLYUPDATES_MODE==0
      std::cout<<" MONITORONLYUPDATES STEP 0"<<std::endl;
      DumpUpdateSensorList("UpdateSensorsList.txt");            
      return;
 #endif
#endif 
      
      if(nEpoch==0) return;//exit(0);      
      Init_test_E     = fTest->GetN()     > GetEventLoss(2) ? GetCost(YMultiLayerPerceptron::kTest)/(fTest->GetN() - GetEventLoss(2)) : 0;//fTest->GetN();    
      
      double Init_test_E_chargesym = 0;
      for(int pstv = 0; pstv < 20; pstv++) Init_test_E_chargesym += fCostChargeSymSum[pstv];
      Init_test_E_chargesym = std::abs(Init_test_E_chargesym / fTotNTraining);
      std::cout<<" COSTMONITOR[TEST] EPOCH"<<fCurrentEpoch<<" Fit + CHSYM = "<<Init_test_E<<" + "<<Init_test_E_chargesym<<" = ";
      Init_test_E += Init_test_E_chargesym;
      std::cout<<Init_test_E<<std::endl;  
               
      //ComputeDCDw();      

      train_residual_plot->SetPoint(0, -1,Init_training_E);
      test_residual_plot->SetPoint(0, -1,Init_test_E);
    
      std::cout.precision(10); 
      std::cout << "(I)Epoch: " 
                << " learn=" << Init_training_E
                << " test=" << Init_test_E;
      std::cout << " fTotN(E) "<< fTraining->GetN()<<" "<<fTest->GetN()
                << " fLoss(E) "<< GetEventLoss(1)<<" "<<GetEventLoss(2);
      std::cout << " fTotN(T) "<< fTotNTraining<<" "<<fTotNTest
                << " fLoss(T) "<< GetTrackLoss(1)<<" "<<GetTrackLoss(2);
      std::cout << std::endl;    

      std::cout << "[ChargeSymmetry]"<<std::endl; 
      for(int pstv = 0; pstv < 20; pstv++){
         int ntr_positive = fChargeSymMonitorPositive->GetBinContent(pstv+1);
         int ntr_negative = fChargeSymMonitorNegative->GetBinContent(pstv+1);
         double ratio_positive = (ntr_positive)/( ntr_positive + ntr_negative );
         double ratio_negative = (ntr_negative)/( ntr_positive + ntr_negative );
         std::cout<<" PSTV#"<<pstv<<" (+) "<<ntr_positive<<" (-) "<<ntr_negative<<" Ratio "<<ratio_positive<<" "<<ratio_negative<<" |DIFF| "<<TMath::Abs(ratio_positive - ratio_negative)<<std::endl;
      }
   } else {
      std::cout<<" weight "<<fPrevWeightName<<std::endl;   
      LoadWeights(fPrevWeightName);
      std::cout<<" weightDU "<<fPrevWeightDetectorUnitName<<std::endl;
      LoadWeightsDetectorUnit(fPrevWeightDetectorUnitName); 
      SetWeightsDetectorUnitUser();
      TString LoadedWeightName   = fWeightName + "_Epoch_At_" + TString::Itoa(-1,10) + ".txt";
      TString LoadedWeightNameDU = fWeightName + "DU_Epoch_At_" + TString::Itoa(-1,10) + ".txt";
      DumpWeights(LoadedWeightName);
      DumpWeightsDetectorUnit(LoadedWeightNameDU);   
      PrepareDumpResiduals();              
      ComputeTrackProfile();   

#ifdef MONITORONLYUPDATES
 #if MONITORONLYUPDATES_MODE==-1
      std::cout<<" MONITORONLYUPDATES STEP -1"<<std::endl;
      GetCost(YMultiLayerPerceptron::kTraining);
      InitEventLoss();   
      InitTrackLoss();      
      SetEventLoss(1);                  
      SetTrackLoss(1);       
 #elif MONITORONLYUPDATES_MODE==0
      std::cout<<" MONITORONLYUPDATES STEP 0"<<std::endl;
      GetCost(YMultiLayerPerceptron::kTraining);
      InitEventLoss();   
      InitTrackLoss();      
      SetEventLoss(1);                  
      SetTrackLoss(1);       
 #elif MONITORONLYUPDATES_MODE==1
      std::cout<<" MONITORONLYUPDATES STEP 1"<<std::endl; 
      std::cout<<" USL "<<fPrevUSLName<<std::endl;         
      LoadUpdateSensorList(fPrevUSLName);
      DumpUpdateSensorList("UpdateSensorsList.txt");                  
 #endif   
#endif     
      std::cout<< fTraining->GetN()<<" "<<GetEventLoss(1)<<std::endl;
      Init_training_E = fTraining->GetN() > GetEventLoss(1) ? GetCost(YMultiLayerPerceptron::kTraining)/(fTraining->GetN() - GetEventLoss(1)) : 0;//fTraining->GetN();
      double Init_training_E_chargesym = 0;
      for(int pstv = 0; pstv < 20; pstv++) Init_training_E_chargesym += fCostChargeSymSum[pstv];
      Init_training_E_chargesym = std::abs(Init_training_E_chargesym / fTotNTraining);
      std::cout<<" COSTMONITOR[TRAINING] EPOCH"<<fCurrentEpoch<<" Fit + CHSYM = "<<Init_training_E<<" + "<<Init_training_E_chargesym<<" = ";
      Init_training_E += Init_training_E_chargesym;
      std::cout<<Init_training_E<<std::endl;           
      
      DumpResiduals(-1);

      DumpNTracksBySensor();     
      MonitorTracksBySensor();       
      MonitorTracksBySensor1D(); 
      MonitorTracksByHalfStave1D();    

#ifdef MONITORONLYUPDATES
 #if MONITORONLYUPDATES_MODE==0
      std::cout<<" MONITORONLYUPDATES STEP 0"<<std::endl;
      DumpUpdateSensorList("UpdateSensorsList.txt");            
      return;
 #endif
#endif 
            
      if(nEpoch==0) return;//exit(0);      
      Init_test_E     = fTest->GetN()     > GetEventLoss(2) ? GetCost(YMultiLayerPerceptron::kTest)/(fTest->GetN() - GetEventLoss(2)) : 0;//fTest->GetN();  
      double Init_test_E_chargesym = 0;        
      for(int pstv = 0; pstv < 20; pstv++) Init_test_E_chargesym += fCostChargeSymSum[pstv];
      Init_test_E_chargesym = std::abs(Init_test_E_chargesym / fTotNTraining);
      std::cout<<" COSTMONITOR[TEST] EPOCH"<<fCurrentEpoch<<" Fit + CHSYM = "<<Init_test_E<<" + "<<Init_test_E_chargesym<<" = ";
      Init_test_E += Init_test_E_chargesym;
      std::cout<<Init_test_E<<std::endl;        
      
      //ComputeDCDw();      

      train_residual_plot->SetPoint(0, -1,Init_training_E);
      test_residual_plot->SetPoint(0, -1,Init_test_E);
       
      std::cout.precision(10); 
      std::cout << "(C)Epoch: " 
                << " learn=" << Init_training_E
                << " test=" << Init_test_E;
      std::cout << " fTotN(E) "<< fTraining->GetN()<<" "<<fTest->GetN()
                << " fLoss(E) "<< GetEventLoss(1)<<" "<<GetEventLoss(2);
      std::cout << " fTotN(T) "<< fTotNTraining<<" "<<fTotNTest
                << " fLoss(T) "<< GetTrackLoss(1)<<" "<<GetTrackLoss(2);
      std::cout << std::endl; 
      
      std::cout << "[ChargeSymmetry]"<<std::endl; 
      for(int pstv = 0; pstv < 20; pstv++){
         int ntr_positive = fChargeSymMonitorPositive->GetBinContent(pstv+1);
         int ntr_negative = fChargeSymMonitorNegative->GetBinContent(pstv+1);
         double ratio_positive = (ntr_positive)/( ntr_positive + ntr_negative );
         double ratio_negative = (ntr_negative)/( ntr_positive + ntr_negative );
         std::cout<<" PSTV#"<<pstv<<" (+) "<<ntr_positive<<" (-) "<<ntr_negative<<" Ratio "<<ratio_positive<<" "<<ratio_negative<<" |DIFF| "<<TMath::Abs(ratio_positive - ratio_negative)<<std::endl;
      }        
   } 
   
   
   BuildDerivativesXY();
   if(nEpoch==0) return;//exit(0);
   
   // Initialisation
   fLastAlpha = 0;
   Int_t els = fSCNetwork[0]->GetNetwork().GetEntriesFast() + fSCNetwork[0]->GetSynapses().GetEntriesFast() + 6;

   std::cout<<"[TAG JH] buffer and dir Array Construction START"<<std::endl;
   Double_t **bufferArr;
   Double_t **dirArr;
   
   bufferArr = new Double_t *[nSensors];
   dirArr    = new Double_t *[nSensors];   
   for(int iID = 0; iID <nSensors; iID++){
      bufferArr[iID] = new Double_t [els];
      dirArr[iID]    = new Double_t [els];   
      for(int jk = 0; jk <els; jk++){
         bufferArr[iID][jk] = 0;
         dirArr[iID][jk]    = 0; 
      }                  
   }

   Double_t *bufferDetectorUnit0;
   Double_t *dirDetectorUnit0;     
   Double_t **bufferDetectorUnit1;
   Double_t **dirDetectorUnit1;     
   Double_t ***bufferDetectorUnit2;
   Double_t ***dirDetectorUnit2;   
   Double_t ****bufferDetectorUnit3;
   Double_t ****dirDetectorUnit3;
   Double_t *****bufferDetectorUnit4;
   Double_t *****dirDetectorUnit4;
   Double_t ******bufferDetectorUnit5;
   Double_t ******dirDetectorUnit5;
   Double_t *******bufferDetectorUnit6;
   Double_t *******dirDetectorUnit6;
   Double_t ********bufferDetectorUnit7;
   Double_t ********dirDetectorUnit7;
      
   bufferDetectorUnit0 = new Double_t [els];
   dirDetectorUnit0    = new Double_t [els]; 
   for(int jk = 0; jk <els; jk++){
      bufferDetectorUnit0[jk] = 0;
      dirDetectorUnit0[jk]    = 0; 
   }   

   bufferDetectorUnit1 = new Double_t *[3];
   dirDetectorUnit1    = new Double_t *[3]; 
   bufferDetectorUnit2 = new Double_t **[3];
   dirDetectorUnit2    = new Double_t **[3]; 
   bufferDetectorUnit3 = new Double_t ***[3];
   dirDetectorUnit3    = new Double_t ***[3]; 
   bufferDetectorUnit4 = new Double_t ****[3];
   dirDetectorUnit4    = new Double_t ****[3]; 
   bufferDetectorUnit5 = new Double_t *****[3];
   dirDetectorUnit5    = new Double_t *****[3];   
   bufferDetectorUnit6 = new Double_t ******[3];
   dirDetectorUnit6    = new Double_t ******[3]; 
   bufferDetectorUnit7 = new Double_t *******[3];
   dirDetectorUnit7    = new Double_t *******[3]; 
      
   bufferDetectorUnit1[0] = new Double_t [els];
   dirDetectorUnit1[0]    = new Double_t [els];  
   for(int jk = 0; jk <els; jk++){
      bufferDetectorUnit1[0][jk] = 0;
      dirDetectorUnit1[0][jk]    = 0; 
   }          
   bufferDetectorUnit2[0] = new Double_t *[nHalfStaveIB];
   dirDetectorUnit2[0]    = new Double_t *[nHalfStaveIB];   
   bufferDetectorUnit3[0] = new Double_t **[nHalfStaveIB];
   dirDetectorUnit3[0]    = new Double_t **[nHalfStaveIB];   
   bufferDetectorUnit4[0] = new Double_t ***[nHalfStaveIB];
   dirDetectorUnit4[0]    = new Double_t ***[nHalfStaveIB];   
   bufferDetectorUnit5[0] = new Double_t ****[nHalfStaveIB];
   dirDetectorUnit5[0]    = new Double_t ****[nHalfStaveIB];  
   bufferDetectorUnit6[0] = new Double_t *****[nHalfStaveIB];
   dirDetectorUnit6[0]    = new Double_t *****[nHalfStaveIB];  
   bufferDetectorUnit7[0] = new Double_t ******[nHalfStaveIB];
   dirDetectorUnit7[0]    = new Double_t ******[nHalfStaveIB];        
   for(int hs=0; hs<nHalfStaveIB; hs++){           
      bufferDetectorUnit2[0][hs] = new Double_t [els];
      dirDetectorUnit2[0][hs]    = new Double_t [els];   
      for(int jk = 0; jk <els; jk++){
         bufferDetectorUnit2[0][hs][jk] = 0;
         dirDetectorUnit2[0][hs][jk]    = 0; 
      }            

      bufferDetectorUnit3[0][hs] = new Double_t *[nChipsPerHicIB/nHalfStaveIB];
      dirDetectorUnit3[0][hs]    = new Double_t *[nChipsPerHicIB/nHalfStaveIB]; 
      bufferDetectorUnit4[0][hs] = new Double_t **[nChipsPerHicIB/nHalfStaveIB];
      dirDetectorUnit4[0][hs]    = new Double_t **[nChipsPerHicIB/nHalfStaveIB]; 
      bufferDetectorUnit5[0][hs] = new Double_t ***[nChipsPerHicIB/nHalfStaveIB];
      dirDetectorUnit5[0][hs]    = new Double_t ***[nChipsPerHicIB/nHalfStaveIB];  
      bufferDetectorUnit6[0][hs] = new Double_t ****[nChipsPerHicIB/nHalfStaveIB];
      dirDetectorUnit6[0][hs]    = new Double_t ****[nChipsPerHicIB/nHalfStaveIB];  
      bufferDetectorUnit7[0][hs] = new Double_t *****[nChipsPerHicIB/nHalfStaveIB];
      dirDetectorUnit7[0][hs]    = new Double_t *****[nChipsPerHicIB/nHalfStaveIB];              
      for(int cz=0; cz<nChipsPerHicIB/nHalfStaveIB; cz++){     
         bufferDetectorUnit3[0][hs][cz] = new Double_t [els];
         dirDetectorUnit3[0][hs][cz]    = new Double_t [els];  
         for(int jk = 0; jk <els; jk++){
            bufferDetectorUnit3[0][hs][cz][jk] = 0;
            dirDetectorUnit3[0][hs][cz][jk]    = 0; 
         }            

         bufferDetectorUnit4[0][hs][cz] = new Double_t *[nLAYERIB];
         dirDetectorUnit4[0][hs][cz]    = new Double_t *[nLAYERIB];                  
         bufferDetectorUnit5[0][hs][cz] = new Double_t **[nLAYERIB];
         dirDetectorUnit5[0][hs][cz]    = new Double_t **[nLAYERIB];  
         bufferDetectorUnit6[0][hs][cz] = new Double_t ***[nLAYERIB];
         dirDetectorUnit6[0][hs][cz]    = new Double_t ***[nLAYERIB]; 
         bufferDetectorUnit7[0][hs][cz] = new Double_t ****[nLAYERIB];
         dirDetectorUnit7[0][hs][cz]    = new Double_t ****[nLAYERIB];                   
         for(int l=0; l<nLAYERIB; l++){ 
            int lIB = l; 
            bufferDetectorUnit4[0][hs][cz][lIB] = new Double_t [els];
            dirDetectorUnit4[0][hs][cz][lIB]    = new Double_t [els];   
            for(int jk = 0; jk <els; jk++){
               bufferDetectorUnit4[0][hs][cz][lIB][jk] = 0;
               dirDetectorUnit4[0][hs][cz][lIB][jk]    = 0; 
            }           

            bufferDetectorUnit5[0][hs][cz][lIB] = new Double_t *[2];
            dirDetectorUnit5[0][hs][cz][lIB]    = new Double_t *[2];  
            bufferDetectorUnit6[0][hs][cz][lIB] = new Double_t **[2];
            dirDetectorUnit6[0][hs][cz][lIB]    = new Double_t **[2];  
            bufferDetectorUnit7[0][hs][cz][lIB] = new Double_t ***[2];
            dirDetectorUnit7[0][hs][cz][lIB]    = new Double_t ***[2];                          
            for(int hb=0; hb<2; hb++){ 
               bufferDetectorUnit5[0][hs][cz][lIB][hb] = new Double_t [els];   
               dirDetectorUnit5[0][hs][cz][lIB][hb]    = new Double_t [els];    
               for(int jk = 0; jk <els; jk++){
                  bufferDetectorUnit5[0][hs][cz][lIB][hb][jk] = 0;
                  dirDetectorUnit5[0][hs][cz][lIB][hb][jk]    = 0; 
               }  
               
               bufferDetectorUnit6[0][hs][cz][lIB][hb] = new Double_t *[NStaves[l]/2];
               dirDetectorUnit6[0][hs][cz][lIB][hb]    = new Double_t *[NStaves[l]/2];  
               bufferDetectorUnit7[0][hs][cz][lIB][hb] = new Double_t **[NStaves[l]/2];
               dirDetectorUnit7[0][hs][cz][lIB][hb]    = new Double_t **[NStaves[l]/2];   
               for(int st=0; st<NStaves[l]/2;st++){
                  bufferDetectorUnit6[0][hs][cz][lIB][hb][st] = new Double_t [els];   
                  dirDetectorUnit6[0][hs][cz][lIB][hb][st]    = new Double_t [els];    
                  for(int jk = 0; jk <els; jk++){
                     bufferDetectorUnit6[0][hs][cz][lIB][hb][st][jk] = 0;
                     dirDetectorUnit6[0][hs][cz][lIB][hb][st][jk]    = 0; 
                  }  
                       
                  bufferDetectorUnit7[0][hs][cz][lIB][hb][st] = new Double_t *[nHalfStaveIB];
                  dirDetectorUnit7[0][hs][cz][lIB][hb][st]    = new Double_t *[nHalfStaveIB];  
                  for(int bst=0; bst<nHalfStaveIB;bst++){    
                     bufferDetectorUnit7[0][hs][cz][lIB][hb][st][bst] = new Double_t [els];   
                     dirDetectorUnit7[0][hs][cz][lIB][hb][st][bst]    = new Double_t [els];    
                     for(int jk = 0; jk <els; jk++){
                        bufferDetectorUnit7[0][hs][cz][lIB][hb][st][bst][jk] = 0;
                        dirDetectorUnit7[0][hs][cz][lIB][hb][st][bst][jk]    = 0; 
                     }  
                  }                        
               }                              
            }       
         }
      }
   }

   bufferDetectorUnit1[1] = new Double_t [els];
   dirDetectorUnit1[1]    = new Double_t [els];  
   for(int jk = 0; jk <els; jk++){
      bufferDetectorUnit1[1][jk] = 0;
      dirDetectorUnit1[1][jk]    = 0; 
   }          
   
   bufferDetectorUnit2[1] = new Double_t *[nHalfStaveOB1];
   dirDetectorUnit2[1]    = new Double_t *[nHalfStaveOB1];   
   bufferDetectorUnit3[1] = new Double_t **[nHalfStaveOB1];
   dirDetectorUnit3[1]    = new Double_t **[nHalfStaveOB1];   
   bufferDetectorUnit4[1] = new Double_t ***[nHalfStaveOB1];
   dirDetectorUnit4[1]    = new Double_t ***[nHalfStaveOB1];   
   bufferDetectorUnit5[1] = new Double_t ****[nHalfStaveOB1];
   dirDetectorUnit5[1]    = new Double_t ****[nHalfStaveOB1];
   bufferDetectorUnit6[1] = new Double_t *****[nHalfStaveOB1];
   dirDetectorUnit6[1]    = new Double_t *****[nHalfStaveOB1];
   bufferDetectorUnit7[1] = new Double_t ******[nHalfStaveOB1];
   dirDetectorUnit7[1]    = new Double_t ******[nHalfStaveOB1];
          
   for(int hs=0; hs<nHalfStaveOB1; hs++){           
      bufferDetectorUnit2[1][hs] = new Double_t [els];
      dirDetectorUnit2[1][hs]    = new Double_t [els];   
      for(int jk = 0; jk <els; jk++){
         bufferDetectorUnit2[1][hs][jk] = 0;
         dirDetectorUnit2[1][hs][jk]    = 0; 
      }            

      bufferDetectorUnit3[1][hs] = new Double_t *[nChipsPerHicOB1/nHalfStaveOB1];
      dirDetectorUnit3[1][hs]    = new Double_t *[nChipsPerHicOB1/nHalfStaveOB1]; 
      bufferDetectorUnit4[1][hs] = new Double_t **[nChipsPerHicOB1/nHalfStaveOB1];
      dirDetectorUnit4[1][hs]    = new Double_t **[nChipsPerHicOB1/nHalfStaveOB1]; 
      bufferDetectorUnit5[1][hs] = new Double_t ***[nChipsPerHicOB1/nHalfStaveOB1];
      dirDetectorUnit5[1][hs]    = new Double_t ***[nChipsPerHicOB1/nHalfStaveOB1];  
      bufferDetectorUnit6[1][hs] = new Double_t ****[nChipsPerHicOB1/nHalfStaveOB1];
      dirDetectorUnit6[1][hs]    = new Double_t ****[nChipsPerHicOB1/nHalfStaveOB1]; 
      bufferDetectorUnit7[1][hs] = new Double_t *****[nChipsPerHicOB1/nHalfStaveOB1];
      dirDetectorUnit7[1][hs]    = new Double_t *****[nChipsPerHicOB1/nHalfStaveOB1];             
      for(int cz=0; cz<nChipsPerHicOB1/nHalfStaveOB1; cz++){     
         bufferDetectorUnit3[1][hs][cz] = new Double_t [els];
         dirDetectorUnit3[1][hs][cz]    = new Double_t [els];  
         for(int jk = 0; jk <els; jk++){
            bufferDetectorUnit3[1][hs][cz][jk] = 0;
            dirDetectorUnit3[1][hs][cz][jk]    = 0; 
         }          

         bufferDetectorUnit4[1][hs][cz] = new Double_t *[nLAYEROB1];
         dirDetectorUnit4[1][hs][cz]    = new Double_t *[nLAYEROB1];                  
         bufferDetectorUnit5[1][hs][cz] = new Double_t **[nLAYEROB1];
         dirDetectorUnit5[1][hs][cz]    = new Double_t **[nLAYEROB1];  
         bufferDetectorUnit6[1][hs][cz] = new Double_t ***[nLAYEROB1];
         dirDetectorUnit6[1][hs][cz]    = new Double_t ***[nLAYEROB1];  
         bufferDetectorUnit7[1][hs][cz] = new Double_t ****[nLAYEROB1];
         dirDetectorUnit7[1][hs][cz]    = new Double_t ****[nLAYEROB1];                    
         for(int l=nLAYERIB; l<nLAYERIB + nLAYEROB1; l++){ 
            int lOB = l - nLAYERIB;
            bufferDetectorUnit4[1][hs][cz][lOB] = new Double_t [els];
            dirDetectorUnit4[1][hs][cz][lOB]    = new Double_t [els];   
            for(int jk = 0; jk <els; jk++){
               bufferDetectorUnit4[1][hs][cz][lOB][jk] = 0;
               dirDetectorUnit4[1][hs][cz][lOB][jk]    = 0; 
            }                      
            bufferDetectorUnit5[1][hs][cz][lOB] = new Double_t *[2];
            dirDetectorUnit5[1][hs][cz][lOB]    = new Double_t *[2];  
            bufferDetectorUnit6[1][hs][cz][lOB] = new Double_t **[2];
            dirDetectorUnit6[1][hs][cz][lOB]    = new Double_t **[2];  
            bufferDetectorUnit7[1][hs][cz][lOB] = new Double_t ***[2];
            dirDetectorUnit7[1][hs][cz][lOB]    = new Double_t ***[2];                          
            for(int hb=0; hb<2; hb++){ 
               bufferDetectorUnit5[1][hs][cz][lOB][hb] = new Double_t [els];   
               dirDetectorUnit5[1][hs][cz][lOB][hb]    = new Double_t [els];    
               for(int jk = 0; jk <els; jk++){
                  bufferDetectorUnit5[1][hs][cz][lOB][hb][jk] = 0;
                  dirDetectorUnit5[1][hs][cz][lOB][hb][jk]    = 0; 
               }     
                                           
               bufferDetectorUnit6[1][hs][cz][lOB][hb] = new Double_t *[NStaves[l]/2];
               dirDetectorUnit6[1][hs][cz][lOB][hb]    = new Double_t *[NStaves[l]/2];  
               bufferDetectorUnit7[1][hs][cz][lOB][hb] = new Double_t **[NStaves[l]/2];
               dirDetectorUnit7[1][hs][cz][lOB][hb]    = new Double_t **[NStaves[l]/2];   
               for(int st=0; st<NStaves[l]/2;st++){
                  bufferDetectorUnit6[1][hs][cz][lOB][hb][st] = new Double_t [els];   
                  dirDetectorUnit6[1][hs][cz][lOB][hb][st]    = new Double_t [els];    
                  for(int jk = 0; jk <els; jk++){
                     bufferDetectorUnit6[1][hs][cz][lOB][hb][st][jk] = 0;
                     dirDetectorUnit6[1][hs][cz][lOB][hb][st][jk]    = 0; 
                  }  
                       
                  bufferDetectorUnit7[1][hs][cz][lOB][hb][st] = new Double_t *[nHalfStaveOB1];
                  dirDetectorUnit7[1][hs][cz][lOB][hb][st]    = new Double_t *[nHalfStaveOB1];  
                  for(int bst=0; bst<nHalfStaveOB1;bst++){    
                     bufferDetectorUnit7[1][hs][cz][lOB][hb][st][bst] = new Double_t [els];   
                     dirDetectorUnit7[1][hs][cz][lOB][hb][st][bst]    = new Double_t [els];    
                     for(int jk = 0; jk <els; jk++){
                        bufferDetectorUnit7[1][hs][cz][lOB][hb][st][bst][jk] = 0;
                        dirDetectorUnit7[1][hs][cz][lOB][hb][st][bst][jk]    = 0; 
                     }  
                  }                        
               }             
            }       
         }
      }
   }

   bufferDetectorUnit1[2] = new Double_t [els];
   dirDetectorUnit1[2]    = new Double_t [els];  
   for(int jk = 0; jk <els; jk++){
      bufferDetectorUnit1[2][jk] = 0;
      dirDetectorUnit1[2][jk]    = 0; 
   }          
   
   bufferDetectorUnit2[2] = new Double_t *[nHalfStaveOB2];
   dirDetectorUnit2[2]    = new Double_t *[nHalfStaveOB2];   
   bufferDetectorUnit3[2] = new Double_t **[nHalfStaveOB2];
   dirDetectorUnit3[2]    = new Double_t **[nHalfStaveOB2];   
   bufferDetectorUnit4[2] = new Double_t ***[nHalfStaveOB2];
   dirDetectorUnit4[2]    = new Double_t ***[nHalfStaveOB2];   
   bufferDetectorUnit5[2] = new Double_t ****[nHalfStaveOB2];
   dirDetectorUnit5[2]    = new Double_t ****[nHalfStaveOB2];  
   bufferDetectorUnit6[2] = new Double_t *****[nHalfStaveOB2];
   dirDetectorUnit6[2]    = new Double_t *****[nHalfStaveOB2];  
   bufferDetectorUnit7[2] = new Double_t ******[nHalfStaveOB2];
   dirDetectorUnit7[2]    = new Double_t ******[nHalfStaveOB2];  
      
   for(int hs=0; hs<nHalfStaveOB2; hs++){           
      bufferDetectorUnit2[2][hs] = new Double_t [els];
      dirDetectorUnit2[2][hs]    = new Double_t [els];   
      for(int jk = 0; jk <els; jk++){
         bufferDetectorUnit2[2][hs][jk] = 0;
         dirDetectorUnit2[2][hs][jk]    = 0; 
      }            
      bufferDetectorUnit3[2][hs] = new Double_t *[nChipsPerHicOB2/nHalfStaveOB2];
      dirDetectorUnit3[2][hs]    = new Double_t *[nChipsPerHicOB2/nHalfStaveOB2]; 
      bufferDetectorUnit4[2][hs] = new Double_t **[nChipsPerHicOB2/nHalfStaveOB2];
      dirDetectorUnit4[2][hs]    = new Double_t **[nChipsPerHicOB2/nHalfStaveOB2]; 
      bufferDetectorUnit5[2][hs] = new Double_t ***[nChipsPerHicOB2/nHalfStaveOB2];
      dirDetectorUnit5[2][hs]    = new Double_t ***[nChipsPerHicOB2/nHalfStaveOB2];  
      bufferDetectorUnit6[2][hs] = new Double_t ****[nChipsPerHicOB2/nHalfStaveOB2];
      dirDetectorUnit6[2][hs]    = new Double_t ****[nChipsPerHicOB2/nHalfStaveOB2];  
      bufferDetectorUnit7[2][hs] = new Double_t *****[nChipsPerHicOB2/nHalfStaveOB2];
      dirDetectorUnit7[2][hs]    = new Double_t *****[nChipsPerHicOB2/nHalfStaveOB2];              
      for(int cz=0; cz<nChipsPerHicOB2/nHalfStaveOB2; cz++){     
         bufferDetectorUnit3[2][hs][cz] = new Double_t [els];
         dirDetectorUnit3[2][hs][cz]    = new Double_t [els];  
         for(int jk = 0; jk <els; jk++){
            bufferDetectorUnit3[2][hs][cz][jk] = 0;
            dirDetectorUnit3[2][hs][cz][jk]    = 0; 
         }            
         bufferDetectorUnit4[2][hs][cz] = new Double_t *[nLAYEROB2];
         dirDetectorUnit4[2][hs][cz]    = new Double_t *[nLAYEROB2];                  
         bufferDetectorUnit5[2][hs][cz] = new Double_t **[nLAYEROB2];
         dirDetectorUnit5[2][hs][cz]    = new Double_t **[nLAYEROB2];  
         bufferDetectorUnit6[2][hs][cz] = new Double_t ***[nLAYEROB2];
         dirDetectorUnit6[2][hs][cz]    = new Double_t ***[nLAYEROB2];  
         bufferDetectorUnit7[2][hs][cz] = new Double_t ****[nLAYEROB2];
         dirDetectorUnit7[2][hs][cz]    = new Double_t ****[nLAYEROB2];                    
         for(int l=nLAYERIB + nLAYEROB1; l<nLAYER; l++){ 
            int lOB = l - (nLAYERIB + nLAYEROB1);
            bufferDetectorUnit4[2][hs][cz][lOB] = new Double_t [els];
            dirDetectorUnit4[2][hs][cz][lOB]    = new Double_t [els];   
            for(int jk = 0; jk <els; jk++){
               bufferDetectorUnit4[2][hs][cz][lOB][jk] = 0;
               dirDetectorUnit4[2][hs][cz][lOB][jk]    = 0; 
            }                      
            bufferDetectorUnit5[2][hs][cz][lOB] = new Double_t *[2];
            dirDetectorUnit5[2][hs][cz][lOB]    = new Double_t *[2];  
            bufferDetectorUnit6[2][hs][cz][lOB] = new Double_t **[2];
            dirDetectorUnit6[2][hs][cz][lOB]    = new Double_t **[2]; 
            bufferDetectorUnit7[2][hs][cz][lOB] = new Double_t ***[2];
            dirDetectorUnit7[2][hs][cz][lOB]    = new Double_t ***[2];                         
            for(int hb=0; hb<2; hb++){ 
               bufferDetectorUnit5[2][hs][cz][lOB][hb] = new Double_t [els];   
               dirDetectorUnit5[2][hs][cz][lOB][hb]    = new Double_t [els];    
               for(int jk = 0; jk <els; jk++){
                  bufferDetectorUnit5[2][hs][cz][lOB][hb][jk] = 0;
                  dirDetectorUnit5[2][hs][cz][lOB][hb][jk]    = 0; 
               }

               bufferDetectorUnit6[2][hs][cz][lOB][hb] = new Double_t *[NStaves[l]/2];
               dirDetectorUnit6[2][hs][cz][lOB][hb]    = new Double_t *[NStaves[l]/2];  
               bufferDetectorUnit7[2][hs][cz][lOB][hb] = new Double_t **[NStaves[l]/2];
               dirDetectorUnit7[2][hs][cz][lOB][hb]    = new Double_t **[NStaves[l]/2];   
               for(int st=0; st<NStaves[l]/2;st++){
                  bufferDetectorUnit6[2][hs][cz][lOB][hb][st] = new Double_t [els];   
                  dirDetectorUnit6[2][hs][cz][lOB][hb][st]    = new Double_t [els];    
                  for(int jk = 0; jk <els; jk++){
                     bufferDetectorUnit6[2][hs][cz][lOB][hb][st][jk] = 0;
                     dirDetectorUnit6[2][hs][cz][lOB][hb][st][jk]    = 0; 
                  }  
                       
                  bufferDetectorUnit7[2][hs][cz][lOB][hb][st] = new Double_t *[nHalfStaveOB2];
                  dirDetectorUnit7[2][hs][cz][lOB][hb][st]    = new Double_t *[nHalfStaveOB2];  
                  for(int bst=0; bst<nHalfStaveOB2;bst++){    
                     bufferDetectorUnit7[2][hs][cz][lOB][hb][st][bst] = new Double_t [els];   
                     dirDetectorUnit7[2][hs][cz][lOB][hb][st][bst]    = new Double_t [els];    
                     for(int jk = 0; jk <els; jk++){
                        bufferDetectorUnit7[2][hs][cz][lOB][hb][st][bst][jk] = 0;
                        dirDetectorUnit7[2][hs][cz][lOB][hb][st][bst][jk]    = 0; 
                     }  
                  }                        
               }                                                   
            }       
         }
      }
   }
   
   std::cout<<"[TAG JH] buffer and dir Array Construction END"<<std::endl;

   Int_t matrix_size = fLearningMethod==YMultiLayerPerceptron::kBFGS ? els : 1;
   TMatrixD bfgsh(matrix_size, matrix_size);
   TMatrixD gamma(matrix_size, 1);
   TMatrixD delta(matrix_size, 1);
   // Epoch loop. Here is the training itself.
   Double_t training_E = 1e10;
   Double_t test_E = 1e10;
   
   Double_t training_Epast = 1e10;
   Double_t test_Epast = 1e10; 
   
   int nEtaOptTrial = 0;
   std::cout<<"Start Training"<<std::endl;
   for (Int_t iepoch = 0; (iepoch < nEpoch) && (!minE_Train || training_E>minE) && (!minE_Test || test_E>minE) ; iepoch++) {
      std::cerr<<" Training in progress, Current Epoch "<<iepoch<<" with Eta "<<fEta<<std::endl;         
      fCurrentEpoch = iepoch;   
      switch (fLearningMethod) {
      case YMultiLayerPerceptron::kStochastic:
         {
            MLP_StochasticArr(bufferArr);   
            DumpCostGradients(fCurrentEpoch, bufferArr, -1);
            DumpCostGradients(fCurrentEpoch, bufferArr, 2);                     
            break;
         }
      case YMultiLayerPerceptron::kBatch:
         {
            ComputeDCDw();
            MLP_BatchArr(bufferArr);  
            DumpCostGradients(fCurrentEpoch, bufferArr, -1);
            DumpCostGradients(fCurrentEpoch, bufferArr, 2);                 
            break;
         }   
      case YMultiLayerPerceptron::kBatchDetectorUnitUser:
         {
            ComputeDCDwDetectorUnit();
            MLP_BatchDetectorUnit(bufferArr, bufferDetectorUnit0,
                                             bufferDetectorUnit1,
                                             bufferDetectorUnit2,
                                             bufferDetectorUnit3,
                                             bufferDetectorUnit4,
                                             bufferDetectorUnit5,
                                             bufferDetectorUnit6,
                                             bufferDetectorUnit7);  
            DumpCostGradients(fCurrentEpoch, bufferArr, -1);
            DumpCostGradients(fCurrentEpoch, bufferArr, 2);            
            break;
         }               
      case YMultiLayerPerceptron::kSteepestDescent:
         {
            std::cout<<"ComputeDCDw epoch "<<iepoch<<std::endl;
            ComputeDCDw();
            std::cout<<"SteepestDirArr epoch "<<iepoch<<std::endl;
            SteepestDirArr(dirArr);           
            std::cout<<"LineSearchArr epoch "<<iepoch<<std::endl;            
            if (LineSearchArr(dirArr,bufferArr))
               MLP_BatchArr(bufferArr);               
            break;
         }
      case YMultiLayerPerceptron::kRibierePolak:
         {
            break;
         }
      case YMultiLayerPerceptron::kFletcherReeves:
         {
            break;
         }
      case YMultiLayerPerceptron::kBFGS:
         {
            break;
         }     
      default:
         break;    
      }
      //return; //QUICKLOOP_ONLY_FEATURING
      // Security: would the learning lead to non real numbers,
      // the learning should stop now.
      //if (TMath::IsNaN(GetCost(YMultiLayerPerceptron::kTraining))) {
      //   Error("YMultiLayerPerceptron::Train()","Stop.");
      //   iepoch = nEpoch;
      //}      
      // Process other ROOT events.  Time penalty is less than
      // 1/1000 sec/evt on a mobile AMD Athlon(tm) XP 1500+
      gSystem->ProcessEvents();
      training_E = fTraining->GetN() > GetEventLoss(1) ? GetCost(YMultiLayerPerceptron::kTraining)/(fTraining->GetN() - GetEventLoss(1)) : 0;//fTraining->GetN();
      double training_E_chargesym = 0;
      for(int pstv = 0; pstv < 20; pstv++) training_E_chargesym += fCostChargeSymSum[pstv];
      training_E_chargesym = std::abs(training_E_chargesym / fTotNTraining);
      std::cout<<" COSTMONITOR[TRAINING] EPOCH"<<fCurrentEpoch<<" Fit + CHSYM = "<<training_E<<" + "<<training_E_chargesym<<" = ";
      training_E += training_E_chargesym;
      std::cout<<training_E<<std::endl;      
      
      if (TMath::IsNaN(training_E)) {
         Error("YMultiLayerPerceptron::Train() Training Set Cost NAN","Stop.");
         iepoch = nEpoch;
      }  
      if ((training_Epast*1.005<training_E) || (Init_training_E*1.02<training_E)){
         nEtaOptTrial++;      
         std::cerr<<"Eta Optimization Step "<<nEtaOptTrial<<std::endl;
         iepoch = iepoch-1;
         LoadWeights(Form("weights_Epoch_At_%d.txt",iepoch));
         LoadWeightsDetectorUnit(Form("weightsDU_Epoch_At_%d.txt",iepoch)); 
         fEta /= std::sqrt(10);
         
         if(nEtaOptTrial<4) {
            continue;
         } else {
            iepoch = nEpoch;
         }         
      }    
      DumpResiduals(iepoch);
      //if(iepoch%5==4) DumpResiduals(iepoch);
      test_E     = fTest->GetN()     > GetEventLoss(2) ? GetCost(YMultiLayerPerceptron::kTest)/(fTest->GetN() - GetEventLoss(2)) : 0;//fTest->GetN();    
      double test_E_chargesym = 0;
      for(int pstv = 0; pstv < 20; pstv++) test_E_chargesym += fCostChargeSymSum[pstv];
      test_E_chargesym = std::abs(test_E_chargesym / fTotNTraining);
      std::cout<<" COSTMONITOR[TEST] EPOCH"<<fCurrentEpoch<<" Fit + CHSYM = "<<test_E<<" + "<<test_E_chargesym<<" = ";
      test_E += test_E_chargesym;
      std::cout<<test_E<<std::endl;       
                
      if (TMath::IsNaN(test_E)) {
         Error("YMultiLayerPerceptron::Train() Test Set Cost NAN","Stop.");
         iepoch = nEpoch;
      } 
     
      training_Epast = training_E;
      test_Epast     = test_E;
        
      if ((verbosity % 2) && ((!(iepoch % displayStepping)) || (iepoch == nEpoch - 1))) {
         if(iepoch < nEpoch) std::cout << "(Ave)Epoch: " << iepoch;
         else std::cout << "(Ave)Epoch: END" ;
         std::cout.precision(10);
         std::cout<< " learn=" << training_E
                  << " test=" << test_E;
         std::cout << " fTotN(E) "<< fTraining->GetN()<<" "<<fTest->GetN()
                   << " fLoss(E) "<< GetEventLoss(1)<<" "<<GetEventLoss(2);  
         std::cout << " fTotN(T) "<< fTotNTraining<<" "<<fTotNTest
                   << " fLoss(T) "<< GetTrackLoss(1)<<" "<<GetTrackLoss(2);        
         std::cout << std::endl;

         std::cout << "[ChargeSymmetry]"<<std::endl; 
         for(int pstv = 0; pstv < 20; pstv++){
            int ntr_positive = fChargeSymMonitorPositive->GetBinContent(pstv+1);
            int ntr_negative = fChargeSymMonitorNegative->GetBinContent(pstv+1);
            double ratio_positive = (ntr_positive)/( ntr_positive + ntr_negative );
            double ratio_negative = (ntr_negative)/( ntr_positive + ntr_negative );
            std::cout<<" PSTV#"<<pstv<<" (+) "<<ntr_positive<<" (-) "<<ntr_negative<<" Ratio "<<ratio_positive<<" "<<ratio_negative<<" |DIFF| "<<TMath::Abs(ratio_positive - ratio_negative)<<std::endl;
         }
        
      }
      if (verbosity / 2) {
         train_residual_plot->SetPoint(iepoch+1, iepoch,training_E);
         test_residual_plot->SetPoint(iepoch+1, iepoch,test_E);
         if (!iepoch) {
            Double_t trp = train_residual_plot->GetY()[0];
            Double_t tep = test_residual_plot->GetY()[0];
            for (i = 1; i < nEpoch; i++) {
               train_residual_plot->SetPoint(i+1, i, trp);
               test_residual_plot->SetPoint(i+1, i, tep);
            }
         }
         if ((!(iepoch % displayStepping)) || (iepoch == nEpoch - 1)) {
            if (residual_plot->GetYaxis()) {
               residual_plot->GetYaxis()->UnZoom();
               residual_plot->GetYaxis()->SetTitleOffset(1.4);
               residual_plot->GetYaxis()->SetDecimals();
            }
            canvas->Modified();
            canvas->Update();
         }
      }
      //## Weight Monitoring Option 2020 10 12
      if(fWeightName!=""&&iepoch>=0) {
         if(iepoch%(int(nEpoch/fWeightStep))==0){  
            TString fStepWeightName ="";
            TString fStepWeightNameDU ="";
            if(iepoch < nEpoch) {
               fStepWeightName   = fWeightName + "_Epoch_At_" + TString::Itoa(iepoch,10) + ".txt";
               fStepWeightNameDU = fWeightName + "DU_Epoch_At_" + TString::Itoa(iepoch,10) + ".txt";
            } else {
               fStepWeightName   = fWeightName + "_Epoch_At_END.txt";   
               fStepWeightNameDU = fWeightName + "_DU_Epoch_At_END.txt";   
            }
             
            DumpWeights(fStepWeightName);
            DumpWeightsDetectorUnit(fStepWeightNameDU);
            //std::clog<<"## Monitoring : Epoch "<<iepoch<<std::endl;  
            //std::clog<<gSystem->pwd()<<std::endl;            
            //PrintCurrentWeights();        
         }
      }   
   }

   std::cout << "EpochLast: learn=" << training_E
                        << " test=" << test_E
                                    << std::endl;   

   std::cout << "[ChargeSymmetry]"<<std::endl; 
   for(int pstv = 0; pstv < 20; pstv++){
      int ntr_positive = fChargeSymMonitorPositive->GetBinContent(pstv+1);
      int ntr_negative = fChargeSymMonitorNegative->GetBinContent(pstv+1);
      double ratio_positive = (double)(ntr_positive)/(double)( ntr_positive + ntr_negative );
      double ratio_negative = (double)(ntr_negative)/(double)( ntr_positive + ntr_negative );
      std::cout<<" PSTV#"<<pstv<<" (+) "<<ntr_positive<<" (-) "<<ntr_negative<<" Ratio "<<ratio_positive<<" "<<ratio_negative<<" |DIFF| "<<TMath::Abs(ratio_positive - ratio_negative)<<std::endl;
   }
   
   // Cleaning

   for(int iID = 0; iID <nSensors; iID++){     
      delete [] bufferArr[iID];   
      delete [] dirArr[iID];            
   } 
   delete [] bufferArr;      
   delete [] dirArr;      


   for(int hs=0; hs<nHalfStaveIB; hs++){           
      for(int cz=0; cz<nChipsPerHicIB/nHalfStaveIB; cz++){
         for(int l=0; l<nLAYERIB; l++){ 
            int lIB = l;
            for(int hb=0; hb<2; hb++){ 
               for(int st=0; st<NStaves[l]/2; st++){
                  for(int bst=0; bst<nHalfStaveIB; bst++){
                     delete [] bufferDetectorUnit7[0][hs][cz][lIB][hb][st][bst];
                     delete [] dirDetectorUnit7[0][hs][cz][lIB][hb][st][bst]; 
                  }
                  delete [] bufferDetectorUnit6[0][hs][cz][lIB][hb][st];
                  delete [] dirDetectorUnit6[0][hs][cz][lIB][hb][st];    
                  delete [] bufferDetectorUnit7[0][hs][cz][lIB][hb][st];
                  delete [] dirDetectorUnit7[0][hs][cz][lIB][hb][st];                            
               }
               delete [] bufferDetectorUnit5[0][hs][cz][lIB][hb];
               delete [] dirDetectorUnit5[0][hs][cz][lIB][hb];   
               delete [] bufferDetectorUnit6[0][hs][cz][lIB][hb];
               delete [] dirDetectorUnit6[0][hs][cz][lIB][hb];    
               delete [] bufferDetectorUnit7[0][hs][cz][lIB][hb];
               delete [] dirDetectorUnit7[0][hs][cz][lIB][hb];                   
            }            
            delete [] bufferDetectorUnit4[0][hs][cz][lIB];
            delete [] dirDetectorUnit4[0][hs][cz][lIB];             
            delete [] bufferDetectorUnit5[0][hs][cz][lIB];
            delete [] dirDetectorUnit5[0][hs][cz][lIB];   
            delete [] bufferDetectorUnit6[0][hs][cz][lIB];
            delete [] dirDetectorUnit6[0][hs][cz][lIB];    
            delete [] bufferDetectorUnit7[0][hs][cz][lIB];
            delete [] dirDetectorUnit7[0][hs][cz][lIB];                
         }
         delete [] bufferDetectorUnit3[0][hs][cz];
         delete [] dirDetectorUnit3[0][hs][cz];       
         delete [] bufferDetectorUnit4[0][hs][cz];
         delete [] dirDetectorUnit4[0][hs][cz];          
         delete [] bufferDetectorUnit5[0][hs][cz];
         delete [] dirDetectorUnit5[0][hs][cz];
         delete [] bufferDetectorUnit6[0][hs][cz];
         delete [] dirDetectorUnit6[0][hs][cz];    
         delete [] bufferDetectorUnit7[0][hs][cz];
         delete [] dirDetectorUnit7[0][hs][cz];           
      }
      delete [] bufferDetectorUnit2[0][hs];
      delete [] dirDetectorUnit2[0][hs];      
      delete [] bufferDetectorUnit3[0][hs];
      delete [] dirDetectorUnit3[0][hs];       
      delete [] bufferDetectorUnit4[0][hs];
      delete [] dirDetectorUnit4[0][hs];     
      delete [] bufferDetectorUnit5[0][hs];
      delete [] dirDetectorUnit5[0][hs];
      delete [] bufferDetectorUnit6[0][hs];      
      delete [] dirDetectorUnit6[0][hs];    
      delete [] bufferDetectorUnit7[0][hs];
      delete [] dirDetectorUnit7[0][hs];       
   } 

   for(int hs=0; hs<nHalfStaveOB1; hs++){           
      for(int cz=0; cz<nChipsPerHicOB1/nHalfStaveOB1; cz++){
         for(int l=nLAYERIB; l<nLAYERIB + nLAYEROB1; l++){ 
            int lOB = l - nLAYERIB;
            for(int hb=0; hb<2; hb++){ 
               for(int st=0; st<NStaves[l]/2; st++){
                  for(int bst=0; bst<nHalfStaveOB1; bst++){
                     delete [] bufferDetectorUnit7[1][hs][cz][lOB][hb][st][bst];
                     delete [] dirDetectorUnit7[1][hs][cz][lOB][hb][st][bst]; 
                  }
                  delete [] bufferDetectorUnit6[1][hs][cz][lOB][hb][st];
                  delete [] dirDetectorUnit6[1][hs][cz][lOB][hb][st];    
                  delete [] bufferDetectorUnit7[1][hs][cz][lOB][hb][st];
                  delete [] dirDetectorUnit7[1][hs][cz][lOB][hb][st];                            
               }            
               delete [] bufferDetectorUnit5[1][hs][cz][lOB][hb];
               delete [] dirDetectorUnit5[1][hs][cz][lOB][hb];    
               delete [] bufferDetectorUnit6[1][hs][cz][lOB][hb];
               delete [] dirDetectorUnit6[1][hs][cz][lOB][hb];    
               delete [] bufferDetectorUnit7[1][hs][cz][lOB][hb];
               delete [] dirDetectorUnit7[1][hs][cz][lOB][hb];                 
            }            
            delete [] bufferDetectorUnit4[1][hs][cz][lOB];
            delete [] dirDetectorUnit4[1][hs][cz][lOB];             
            delete [] bufferDetectorUnit5[1][hs][cz][lOB];
            delete [] dirDetectorUnit5[1][hs][cz][lOB];     
            delete [] bufferDetectorUnit6[1][hs][cz][lOB];
            delete [] dirDetectorUnit6[1][hs][cz][lOB];    
            delete [] bufferDetectorUnit7[1][hs][cz][lOB];
            delete [] dirDetectorUnit7[1][hs][cz][lOB];             
         }
         delete [] bufferDetectorUnit3[1][hs][cz];
         delete [] dirDetectorUnit3[1][hs][cz];       
         delete [] bufferDetectorUnit4[1][hs][cz];
         delete [] dirDetectorUnit4[1][hs][cz];          
         delete [] bufferDetectorUnit5[1][hs][cz];
         delete [] dirDetectorUnit5[1][hs][cz];
         delete [] bufferDetectorUnit6[1][hs][cz];
         delete [] dirDetectorUnit6[1][hs][cz];    
         delete [] bufferDetectorUnit7[1][hs][cz];
         delete [] dirDetectorUnit7[1][hs][cz];           
      }
      delete [] bufferDetectorUnit2[1][hs];
      delete [] dirDetectorUnit2[1][hs];      
      delete [] bufferDetectorUnit3[1][hs];
      delete [] dirDetectorUnit3[1][hs];       
      delete [] bufferDetectorUnit4[1][hs];
      delete [] dirDetectorUnit4[1][hs];     
      delete [] bufferDetectorUnit5[1][hs];
      delete [] dirDetectorUnit5[1][hs];
      delete [] bufferDetectorUnit6[1][hs];
      delete [] dirDetectorUnit6[1][hs];    
      delete [] bufferDetectorUnit7[1][hs];
      delete [] dirDetectorUnit7[1][hs];       
   } 
   
   for(int hs=0; hs<nHalfStaveOB2; hs++){           
      for(int cz=0; cz<nChipsPerHicOB2/nHalfStaveOB2; cz++){
         for(int l=nLAYERIB + nLAYEROB1; l<nLAYER; l++){ 
            int lOB = l - (nLAYERIB + nLAYEROB1);
            for(int hb=0; hb<2; hb++){
               for(int st=0; st<NStaves[l]/2; st++){
                  for(int bst=0; bst<nHalfStaveOB2; bst++){
                     delete [] bufferDetectorUnit7[2][hs][cz][lOB][hb][st][bst];
                     delete [] dirDetectorUnit7[2][hs][cz][lOB][hb][st][bst]; 
                  }
                  delete [] bufferDetectorUnit6[2][hs][cz][lOB][hb][st];
                  delete [] dirDetectorUnit6[2][hs][cz][lOB][hb][st];    
                  delete [] bufferDetectorUnit7[2][hs][cz][lOB][hb][st];
                  delete [] dirDetectorUnit7[2][hs][cz][lOB][hb][st];                            
               }             
               delete [] bufferDetectorUnit5[2][hs][cz][lOB][hb];
               delete [] dirDetectorUnit5[2][hs][cz][lOB][hb];     
               delete [] bufferDetectorUnit6[2][hs][cz][lOB][hb];
               delete [] dirDetectorUnit6[2][hs][cz][lOB][hb];    
               delete [] bufferDetectorUnit7[2][hs][cz][lOB][hb];
               delete [] dirDetectorUnit7[2][hs][cz][lOB][hb];                
            }            
            delete [] bufferDetectorUnit4[2][hs][cz][lOB];
            delete [] dirDetectorUnit4[2][hs][cz][lOB];             
            delete [] bufferDetectorUnit5[2][hs][cz][lOB];
            delete [] dirDetectorUnit5[2][hs][cz][lOB];     
            delete [] bufferDetectorUnit6[2][hs][cz][lOB];
            delete [] dirDetectorUnit6[2][hs][cz][lOB];    
            delete [] bufferDetectorUnit7[2][hs][cz][lOB];
            delete [] dirDetectorUnit7[2][hs][cz][lOB];              
         }
         delete [] bufferDetectorUnit3[2][hs][cz];
         delete [] dirDetectorUnit3[2][hs][cz];       
         delete [] bufferDetectorUnit4[2][hs][cz];
         delete [] dirDetectorUnit4[2][hs][cz];          
         delete [] bufferDetectorUnit5[2][hs][cz];
         delete [] dirDetectorUnit5[2][hs][cz];
         delete [] bufferDetectorUnit6[2][hs][cz];
         delete [] dirDetectorUnit6[2][hs][cz];    
         delete [] bufferDetectorUnit7[2][hs][cz];
         delete [] dirDetectorUnit7[2][hs][cz];                
      }
      delete [] bufferDetectorUnit2[2][hs];
      delete [] dirDetectorUnit2[2][hs];      
      delete [] bufferDetectorUnit3[2][hs];
      delete [] dirDetectorUnit3[2][hs];       
      delete [] bufferDetectorUnit4[2][hs];
      delete [] dirDetectorUnit4[2][hs];     
      delete [] bufferDetectorUnit5[2][hs];
      delete [] dirDetectorUnit5[2][hs];
      delete [] bufferDetectorUnit6[2][hs];
      delete [] dirDetectorUnit6[2][hs];    
      delete [] bufferDetectorUnit7[2][hs];
      delete [] dirDetectorUnit7[2][hs];       
   } 
      
   
   delete [] bufferDetectorUnit1[0];
   delete [] dirDetectorUnit1[0];   
   delete [] bufferDetectorUnit2[0];
   delete [] dirDetectorUnit2[0];   
   delete [] bufferDetectorUnit3[0];
   delete [] dirDetectorUnit3[0];   
   delete [] bufferDetectorUnit4[0];
   delete [] dirDetectorUnit4[0];   
   delete [] bufferDetectorUnit5[0];
   delete [] dirDetectorUnit5[0];    
   delete [] bufferDetectorUnit6[0];
   delete [] dirDetectorUnit6[0];   
   delete [] bufferDetectorUnit7[0];
   delete [] dirDetectorUnit7[0];     
   delete [] bufferDetectorUnit1[1];
   delete [] dirDetectorUnit1[1];   
   delete [] bufferDetectorUnit2[1];
   delete [] dirDetectorUnit2[1];   
   delete [] bufferDetectorUnit3[1];
   delete [] dirDetectorUnit3[1];   
   delete [] bufferDetectorUnit4[1];
   delete [] dirDetectorUnit4[1];   
   delete [] bufferDetectorUnit5[1];
   delete [] dirDetectorUnit5[1];
   delete [] bufferDetectorUnit6[1];
   delete [] dirDetectorUnit6[1];   
   delete [] bufferDetectorUnit7[1];
   delete [] dirDetectorUnit7[1];   
   delete [] bufferDetectorUnit1[2];
   delete [] dirDetectorUnit1[2];   
   delete [] bufferDetectorUnit2[2];
   delete [] dirDetectorUnit2[2];   
   delete [] bufferDetectorUnit3[2];
   delete [] dirDetectorUnit3[2];   
   delete [] bufferDetectorUnit4[2];
   delete [] dirDetectorUnit4[2];   
   delete [] bufferDetectorUnit5[2];
   delete [] dirDetectorUnit5[2];
   delete [] bufferDetectorUnit6[2];
   delete [] dirDetectorUnit6[2];   
   delete [] bufferDetectorUnit7[2];
   delete [] dirDetectorUnit7[2];   
   
   delete [] bufferDetectorUnit0;
   delete [] dirDetectorUnit0;
   delete [] bufferDetectorUnit1;
   delete [] dirDetectorUnit1;
   delete [] bufferDetectorUnit2;
   delete [] dirDetectorUnit2;
   delete [] bufferDetectorUnit3;
   delete [] dirDetectorUnit3;
   delete [] bufferDetectorUnit4;
   delete [] dirDetectorUnit4;
   delete [] bufferDetectorUnit5;
   delete [] dirDetectorUnit5;
   delete [] bufferDetectorUnit6;
   delete [] dirDetectorUnit6;
   delete [] bufferDetectorUnit7;
   delete [] dirDetectorUnit7;
   // Final Text and Graph outputs
   if (verbosity % 2)
      std::cout << "Training done." << std::endl;
   if (verbosity / 2) {
      TLegend *legend = new TLegend(.75, .80, .95, .95);
      legend->AddEntry(residual_plot->GetListOfGraphs()->At(0),
                       "Training sample", "L");
      legend->AddEntry(residual_plot->GetListOfGraphs()->At(1),
                       "Test sample", "L");
      legend->Draw();
      canvas->SaveAs("LossCurve.gif","gif");
      delete canvas;
      delete legend;       
   }

}

void YMultiLayerPerceptron::MonitorTracksBySensor1D()
{
   // x : chipID in stave n = nHicPerStave[l]*nChipsPerHic[l]
   // y : stave
   std::cout<<"YMultiLayerPerceptron::MonitorTracksBySensor1D Start"<<std::endl;   
   TCanvas *canvas = new TCanvas("MonitorTracksBySensor1D", "Alignment MonitorTracksBySensor1D",1600,1200);   
   canvas->SetLogy();
   TH1D*   htrbysensor1D[nLAYER];
   TH1D*   htrbysensor1Dratio[nLAYER];   
   //gStyle->SetPaintTextFormat("1.0f");
   gStyle->SetOptStat(0);     
   //gStyle->SetPalette(55);  //kRainBow 
   std::cout<<"YMultiLayerPerceptron::MonitorTracksBySensor1D Step 1"<<std::endl;  
   int nbinsX = 10;
   int nbinsY = 9;
   for(int l = 0; l < nLAYER; l++){
      TString hname1D  = "hL" + TString::Itoa(l,10) + "TrbySensor";
      TString htitle1D = "N_{track}/Sensor Distribution Layer " + TString::Itoa(l,10);       
      htrbysensor1D[l] = new TH1D(hname1D,htitle1D,nbinsX*nbinsY+1,-1,nbinsX*nbinsY);// 0 1e+0 1e+1 1e+2 1e+3 1e+4 1e+5 1e+6 1e+7 1e+8 1e+9
      htrbysensor1D[l]->GetXaxis()->SetBinLabel(1,Form("Empty"));   
      for (int nb = 0; nb < nbinsX; nb++) {
         htrbysensor1D[l]->GetXaxis()->SetBinLabel(nbinsY*nb + (4+1) + 1,Form("10^{%d}",nb));
         htrbysensor1D[l]->GetXaxis()->LabelsOption("h");         
      }      
      htrbysensor1D[l]->SetXTitle("N_{track}/Sensor");
      htrbysensor1D[l]->SetYTitle("count");
      htrbysensor1D[l]->SetMinimum(0.2);    
      //htrbysensor1D[l]->SetMaximum(maxIB[l]);             
      htrbysensor1D[l]->GetXaxis()->SetNdivisions(115);
      htrbysensor1D[l]->GetXaxis()->SetLabelSize(0.06);            
      htrbysensor1D[l]->GetYaxis()->SetNdivisions(110);
      htrbysensor1D[l]->GetXaxis()->SetTickLength(0.01);
      htrbysensor1D[l]->GetYaxis()->SetTickLength(0.01);         
      htrbysensor1D[l]->GetZaxis()->SetLabelSize(0.02);
      htrbysensor1D[l]->SetLineWidth(4);
      //htrbysensor1D[l]->GetZaxis()->SetLabelOffset(1);  

      TString hname1Dratio  = "hL" + TString::Itoa(l,10) + "TrbySensorRatio";
      TString htitle1Dratio = "Ratio(Acc) plot";        
      htrbysensor1Dratio[l] = new TH1D(hname1Dratio,htitle1Dratio,nbinsX*nbinsY+1,-1,nbinsX*nbinsY);// 0 1e+0 1e+1 1e+2 1e+3 1e+4 1e+5 1e+6 1e+7 1e+8 1e+9
      htrbysensor1Dratio[l]->GetXaxis()->SetBinLabel(1,Form("Empty"));   
      for (int nb = 0; nb < nbinsX; nb++) {
         htrbysensor1Dratio[l]->GetXaxis()->SetBinLabel(nbinsY*nb + (4+1) + 1,Form("10^{%d}",nb));
         htrbysensor1Dratio[l]->GetXaxis()->LabelsOption("h");         
      }      
      htrbysensor1Dratio[l]->SetXTitle("N_{track}/Sensor");
      htrbysensor1Dratio[l]->SetYTitle("count");
      htrbysensor1Dratio[l]->SetMinimum(0);    
      htrbysensor1Dratio[l]->SetMaximum(1.05);             
      htrbysensor1Dratio[l]->GetXaxis()->SetNdivisions(115);
      htrbysensor1Dratio[l]->GetXaxis()->SetLabelSize(0.06);      
      htrbysensor1Dratio[l]->GetYaxis()->SetNdivisions(111);
      htrbysensor1Dratio[l]->GetXaxis()->SetTickLength(0.01);
      htrbysensor1Dratio[l]->GetYaxis()->SetTickLength(0.01);         
      htrbysensor1Dratio[l]->GetZaxis()->SetLabelSize(0.02);
      htrbysensor1Dratio[l]->SetLineWidth(4);
      //htrbysensor1Dratio[l]->GetZaxis()->SetLabelOffset(1);       
      
   }

   std::cout<<"YMultiLayerPerceptron::MonitorTracksBySensor1D Step 2"<<std::endl; 
   int nhits_less100[] = {0, 0, 0, 0, 0, 0, 0};
   int nhits_less25[] = {0, 0, 0, 0, 0, 0, 0};
   int nhits_less16[] = {0, 0, 0, 0, 0, 0, 0};
   int nhits_less9[] = {0, 0, 0, 0, 0, 0, 0};       
        
   for(int ichipID = 0; ichipID < ChipBoundary[nLAYER]; ichipID++){
      int nhits = fSCNetwork[NetworkChips[ichipID]]->GetnEvents();  
      int layer       = yGEOM->GetLayer(ichipID);
      int staveID     = yGEOM->GetStave(ichipID);
      int chipIDstave = yGEOM->GetChipIdInStave(ichipID);  
      
      int xlevel = nhits>0 ? floor(TMath::Log10(nhits)) : -1;
      int ylevel = nhits>0 ? nhits/(int)std::pow(10,floor(TMath::Log10(nhits))) : 1;
      
      int fbin = nhits>0 ? nbinsY*xlevel + ylevel - 1 : -1;
      if(fSCNetwork[NetworkChips[ichipID]]->GetUpdateState()==true) htrbysensor1D[layer]->Fill(fbin);   

      if(nhits < 100) nhits_less100[layer]++;
      if(nhits < 25) nhits_less25[layer]++;
      if(nhits < 16) nhits_less16[layer]++;            
      if(nhits < 9) nhits_less9[layer]++;   
                 
   }
   
   double SectionR[nLAYER][2];
   int    SectionCnt[nLAYER][2];   
   for(int l = 0; l < nLAYER; l++){
      
      for(int nb =0; nb< htrbysensor1D[l]->GetNbinsX(); nb++){
         htrbysensor1Dratio[l]->SetBinContent(nb+1, htrbysensor1D[l]->Integral(0, nb+1)/htrbysensor1D[l]->GetEntries());
      }
      SectionR[l][0] = htrbysensor1D[l]->Integral(0, 1)/htrbysensor1D[l]->GetEntries();
      SectionR[l][1] = htrbysensor1D[l]->Integral(0, nbinsY*3 + 1)/htrbysensor1D[l]->GetEntries();
      SectionCnt[l][0] = htrbysensor1D[l]->Integral(0, 1);
      SectionCnt[l][1] = htrbysensor1D[l]->Integral(0, nbinsY*3 + 1);
   }

   std::cout<<"YMultiLayerPerceptron::MonitorTracksBySensor1D Step 3"<<std::endl;      
   for(int l = 0; l < nLAYER; l++){
      canvas->Clear();
      canvas->Divide(1,2,0.001,0.001);
      //canvas->SetTickx(1);
      //canvas->SetTicky(1);  
      canvas->cd(1);  
      canvas->cd(1)->SetLogy();      
      double maxBD = std::pow(10,floor(TMath::Log10(htrbysensor1D[l]->GetBinContent(htrbysensor1D[l]->GetMaximumBin())*1.05*5.00))+1);
      htrbysensor1D[l]->SetMaximum(maxBD*1.1);                    
      htrbysensor1D[l]->Draw("");
      htrbysensor1D[l]->Draw("text90same");      
      TLine* bdlevel[10];
      std::cout<<"L"<<l<<" Y max : "<<gPad->GetUymax()<<" "<<maxBD<<std::endl;            
      for (int xpoint = 0; xpoint < 10; xpoint++) {
         bdlevel[xpoint] = new TLine(nbinsY*xpoint,0,nbinsY*xpoint,maxBD*1.1);
         if(xpoint==3) {
            bdlevel[xpoint]->SetLineColor(2);
            bdlevel[xpoint]->SetLineWidth(2);
         }            
         bdlevel[xpoint]->Draw("same");
      }  

      TLatex *t10 = new TLatex(nbinsY*6.2, maxBD*0.5,Form("nSensors : %d",ChipBoundary[l+1]-ChipBoundary[l]));      
      t10->SetTextSize(0.04);
      t10->SetTextColor(2); 
      t10->Draw("same");       
      canvas->cd(2);  
      canvas->cd(2)->SetGridy();
      htrbysensor1Dratio[l]->Draw("");
      //htrbysensor1Dratio[l]->Draw("text90same");      
      TLine* bdlevelratio[10];
      for (int xpoint = 0; xpoint < 10; xpoint++) {
         bdlevelratio[xpoint] = new TLine(nbinsY*xpoint,0,nbinsY*xpoint,1.05); 
         if(xpoint==3) {
            bdlevelratio[xpoint]->SetLineColor(2);
            bdlevelratio[xpoint]->SetLineWidth(2);
         }      
         bdlevelratio[xpoint]->Draw("same");
                  
      }  
      TLatex *t20 = new TLatex(nbinsY*6.2, 0.85, Form("Empty Sensors : %d (%2.1f %%)",SectionCnt[l][0],SectionR[l][0]*100));         
      t20->SetTextSize(0.04);
      t20->SetTextColor(2);
      TLatex *t21 = new TLatex(nbinsY*6.2, 0.75, Form("Non Empty Sensors (N_{track}<10^{3}) : %d (%2.1f %%)",SectionCnt[l][1]-SectionCnt[l][0],(SectionR[l][1]-SectionR[l][0])*100));
      t21->SetTextSize(0.04);
      t21->SetTextColor(2); 

      double SectionR_less100 = nhits_less100[l]/htrbysensor1D[l]->GetEntries();
      double SectionR_less25 = nhits_less25[l]/htrbysensor1D[l]->GetEntries();
      double SectionR_less16 = nhits_less16[l]/htrbysensor1D[l]->GetEntries();
      double SectionR_less9 = nhits_less9[l]/htrbysensor1D[l]->GetEntries();                  

      TLatex *t211 = new TLatex(nbinsY*6.2, 0.65, Form("Non Empty Sensors (N_{track}<10^{2}) : %d (%2.1f %%)",nhits_less100[l]-SectionCnt[l][0],(SectionR_less100-SectionR[l][0])*100));
      t211->SetTextSize(0.04);
      t211->SetTextColor(2); 
      TLatex *t212 = new TLatex(nbinsY*6.2, 0.55, Form("Non Empty Sensors (N_{track}<25) : %d (%2.1f %%)",nhits_less25[l]-SectionCnt[l][0],(SectionR_less25-SectionR[l][0])*100));
      t212->SetTextSize(0.04);
      t212->SetTextColor(2); 
      TLatex *t213 = new TLatex(nbinsY*6.2, 0.45, Form("Non Empty Sensors (N_{track}<16) : %d (%2.1f %%)",nhits_less16[l]-SectionCnt[l][0],(SectionR_less16-SectionR[l][0])*100));
      t213->SetTextSize(0.04);
      t213->SetTextColor(2);             
      TLatex *t214 = new TLatex(nbinsY*6.2, 0.35, Form("Non Empty Sensors (N_{track}<9) : %d (%2.1f %%)",nhits_less9[l]-SectionCnt[l][0],(SectionR_less9-SectionR[l][0])*100));
      t214->SetTextSize(0.04);
      t214->SetTextColor(2);         
      TLatex *t22 = new TLatex(nbinsY*6.2, 0.25, Form("nEvents (Training, Test) : (%d, %d)",(int)fTraining->GetN(),(int)fTest->GetN()));
      t22->SetTextSize(0.04);
      t22->SetTextColor(2);
      TLatex *t23 = new TLatex(nbinsY*6.2, 0.15, Form("nTracks (Training, Test) : (%d, %d)",(int)fTotNTraining, (int)fTotNTest));
      t23->SetTextSize(0.04);
      t23->SetTextColor(2);
            
      t20->Draw("same"); 
      t21->Draw("same");  
      t211->Draw("same");  
      t212->Draw("same");  
      t213->Draw("same");  
      t214->Draw("same");                            
      t22->Draw("same"); 
      t23->Draw("same");       
      TString htrbysensor1D  = "TracksBySensor1D_Layer" + TString::Itoa(l,10) + ".gif"; 
      canvas->SaveAs(htrbysensor1D,"gif");      
  
      //htrbysensor1D[l]->Reset();
      //delete htrbysensor1D[l];                            
                                   
   }
   std::cout<<"[TRACKSBYSENSOR] Layer nEvents nTracks, nSensors : Total Ntr<1000 Ntr<100 Ntr<25 Ntr<16 Ntr<9 Empty"<<std::endl; 
   for(int l = 0; l < nLAYER; l++){
      std::cout<<"[TRACKSBYSENSOR] "<<l<<
                                 " "<<(int)fTraining->GetN()<<
                                 " "<<(int)fTotNTraining<<
                                 " "<<ChipBoundary[l+1]-ChipBoundary[l]<<
                                 " "<<SectionCnt[l][1]-SectionCnt[l][0]<<
                                 " "<<nhits_less100[l]-SectionCnt[l][0]<<
                                 " "<<nhits_less25[l]-SectionCnt[l][0]<<
                                 " "<<nhits_less16[l]-SectionCnt[l][0]<<
                                 " "<<nhits_less9[l]-SectionCnt[l][0]<<
                                 " "<<SectionCnt[l][0]<<std::endl;  
   }

   std::cout<<"YMultiLayerPerceptron::MonitorTracksBySensor1D End"<<std::endl;      
}

void YMultiLayerPerceptron::MonitorTracksByHalfStave1D()
{
   // x : chipID in stave n = nHicPerStave[l]*nChipsPerHic[l]
   // y : stave
   std::cout<<"YMultiLayerPerceptron::MonitorTracksByHalfStave1D Start"<<std::endl;   
   TCanvas *canvas = new TCanvas("MonitorTracksByHalfStave1D", "Alignment MonitorTracksByHalfStave1D",1600,1200);   
   canvas->SetLogy();
   TH1D*   htrbyhalfstave1D[nLAYER];
   TH1D*   htrbyhalfstave1Dratio[nLAYER];   
   //gStyle->SetPaintTextFormat("1.0f");
   gStyle->SetOptStat(0);     
   //gStyle->SetPalette(55);  //kRainBow 
   std::cout<<"YMultiLayerPerceptron::MonitorTracksByHalfStave1D Step 1"<<std::endl;  
   int nbinsX = 10;
   int nbinsY = 9;
   for(int l = 0; l < nLAYER; l++){
      TString hname1D  = "hL" + TString::Itoa(l,10) + "TrbyHalfStave";
      TString htitle1D = "N_{track}/HalfStave(Sensor) Distribution Layer " + TString::Itoa(l,10);       
      htrbyhalfstave1D[l] = new TH1D(hname1D,htitle1D,nbinsX*nbinsY+1,-1,nbinsX*nbinsY);// 0 1e+0 1e+1 1e+2 1e+3 1e+4 1e+5 1e+6 1e+7 1e+8 1e+9
      htrbyhalfstave1D[l]->GetXaxis()->SetBinLabel(1,Form("Empty"));   
      for (int nb = 0; nb < nbinsX; nb++) {
         htrbyhalfstave1D[l]->GetXaxis()->SetBinLabel(nbinsY*nb + (4+1) + 1,Form("10^{%d}",nb));
         htrbyhalfstave1D[l]->GetXaxis()->LabelsOption("h");         
      }      
      htrbyhalfstave1D[l]->SetXTitle("N_{track}/HalfStave(Sensor)");
      htrbyhalfstave1D[l]->SetYTitle("count");
      htrbyhalfstave1D[l]->SetMinimum(0.2);    
      //htrbyhalfstave1D[l]->SetMaximum(maxIB[l]);             
      htrbyhalfstave1D[l]->GetXaxis()->SetNdivisions(115);
      htrbyhalfstave1D[l]->GetXaxis()->SetLabelSize(0.06);            
      htrbyhalfstave1D[l]->GetYaxis()->SetNdivisions(110);
      htrbyhalfstave1D[l]->GetXaxis()->SetTickLength(0.01);
      htrbyhalfstave1D[l]->GetYaxis()->SetTickLength(0.01);         
      htrbyhalfstave1D[l]->GetZaxis()->SetLabelSize(0.02);
      htrbyhalfstave1D[l]->SetLineWidth(4);
      //htrbyhalfstave1D[l]->GetZaxis()->SetLabelOffset(1);  

      TString hname1Dratio  = "hL" + TString::Itoa(l,10) + "TrbyHalfStaveRatio";
      TString htitle1Dratio = "Ratio(Acc) plot";        
      htrbyhalfstave1Dratio[l] = new TH1D(hname1Dratio,htitle1Dratio,nbinsX*nbinsY+1,-1,nbinsX*nbinsY);// 0 1e+0 1e+1 1e+2 1e+3 1e+4 1e+5 1e+6 1e+7 1e+8 1e+9
      htrbyhalfstave1Dratio[l]->GetXaxis()->SetBinLabel(1,Form("Empty"));   
      for (int nb = 0; nb < nbinsX; nb++) {
         htrbyhalfstave1Dratio[l]->GetXaxis()->SetBinLabel(nbinsY*nb + (4+1) + 1,Form("10^{%d}",nb));
         htrbyhalfstave1Dratio[l]->GetXaxis()->LabelsOption("h");         
      }      
      htrbyhalfstave1Dratio[l]->SetXTitle("N_{track}/HalfStave(Sensor)");
      htrbyhalfstave1Dratio[l]->SetYTitle("count");
      htrbyhalfstave1Dratio[l]->SetMinimum(0);    
      htrbyhalfstave1Dratio[l]->SetMaximum(1.05);             
      htrbyhalfstave1Dratio[l]->GetXaxis()->SetNdivisions(115);
      htrbyhalfstave1Dratio[l]->GetXaxis()->SetLabelSize(0.06);      
      htrbyhalfstave1Dratio[l]->GetYaxis()->SetNdivisions(111);
      htrbyhalfstave1Dratio[l]->GetXaxis()->SetTickLength(0.01);
      htrbyhalfstave1Dratio[l]->GetYaxis()->SetTickLength(0.01);         
      htrbyhalfstave1Dratio[l]->GetZaxis()->SetLabelSize(0.02);
      htrbyhalfstave1Dratio[l]->SetLineWidth(4);
      //htrbyhalfstave1Dratio[l]->GetZaxis()->SetLabelOffset(1);       
      
   }

   std::cout<<"YMultiLayerPerceptron::MonitorTracksByHalfStave1D Step 2"<<std::endl;  
   int** nhits = new int *[nLAYER];
   for(int l = 0; l < nLAYER; l++){
      nhits[l] = new int [nHicPerStave[l]*nChipsPerHic[l]];
      for(int s = 0; s < nHicPerStave[l]*nChipsPerHic[l]; s++){
         nhits[l][s] = 0;
      }
   }
       
   for(int ichipID = 0; ichipID < ChipBoundary[nLAYER]; ichipID++){
      //int nhits = fSCNetwork[NetworkChips[ichipID]]->GetnEvents();
      int layer       = yGEOM->GetLayer(ichipID);
      int staveID     = yGEOM->GetStave(ichipID);
      int hs          = yGEOM->GetHalfStave(ichipID); 
      int chipIDstave = yGEOM->GetChipIdInStave(ichipID);  
      
      nhits[layer][chipIDstave] += fSCNetwork[NetworkChips[ichipID]]->GetnEvents();             
   }
   
   for(int l = 0; l < nLAYER; l++){ 
      for(int s = 0; s < nHicPerStave[l]*nChipsPerHic[l]; s++){
         int xlevel = nhits[l][s]>0 ? floor(TMath::Log10(nhits[l][s])) : -1;
         int ylevel = nhits[l][s]>0 ? nhits[l][s]/(int)std::pow(10,floor(TMath::Log10(nhits[l][s]))) : 1;
      
         int fbin = nhits[l][s]>0 ? nbinsY*xlevel + ylevel - 1 : -1;
         htrbyhalfstave1D[l]->Fill(fbin); 
      }
   }
   
   double SectionR[nLAYER][2];
   int    SectionCnt[nLAYER][2];   
   for(int l = 0; l < nLAYER; l++){
      
      for(int nb =0; nb< htrbyhalfstave1D[l]->GetNbinsX(); nb++){
         htrbyhalfstave1Dratio[l]->SetBinContent(nb+1, htrbyhalfstave1D[l]->Integral(0, nb+1)/htrbyhalfstave1D[l]->GetEntries());
      }
      SectionR[l][0] = htrbyhalfstave1D[l]->Integral(0, 1)/htrbyhalfstave1D[l]->GetEntries();
      SectionR[l][1] = htrbyhalfstave1D[l]->Integral(0, nbinsY*3 + 1)/htrbyhalfstave1D[l]->GetEntries();
      SectionCnt[l][0] = htrbyhalfstave1D[l]->Integral(0, 1);
      SectionCnt[l][1] = htrbyhalfstave1D[l]->Integral(0, nbinsY*3 + 1);
   }

   std::cout<<"YMultiLayerPerceptron::MonitorTracksByHalfStave1D Step 3"<<std::endl;      
   for(int l = 0; l < nLAYER; l++){
      canvas->Clear();
      canvas->Divide(1,2,0.001,0.001);
      //canvas->SetTickx(1);
      //canvas->SetTicky(1);  
      canvas->cd(1);  
      canvas->cd(1)->SetLogy();      
      double maxBD = std::pow(10,floor(TMath::Log10(htrbyhalfstave1D[l]->GetBinContent(htrbyhalfstave1D[l]->GetMaximumBin())*1.05*5.00))+1);
      htrbyhalfstave1D[l]->SetMaximum(maxBD*1.1);                    
      htrbyhalfstave1D[l]->Draw("");
      htrbyhalfstave1D[l]->Draw("text90same");      
      TLine* bdlevel[10];
      std::cout<<"L"<<l<<" Y max : "<<gPad->GetUymax()<<" "<<maxBD<<std::endl;            
      for (int xpoint = 0; xpoint < 10; xpoint++) {
         bdlevel[xpoint] = new TLine(nbinsY*xpoint,0,nbinsY*xpoint,maxBD*1.1);
         if(xpoint==3) {
            bdlevel[xpoint]->SetLineColor(2);
            bdlevel[xpoint]->SetLineWidth(2);
         }            
         bdlevel[xpoint]->Draw("same");
      }  

      TLatex *t10 = new TLatex(nbinsY*6.2, maxBD*0.5, Form("#splitline{nSensors/HalfStave : %d}{nStave (nHalfStave) : %d (%d)}",nHicPerStave[l]*nChipsPerHic[l],NStaves[l],NSubStave[l]));
      t10->SetTextSize(0.04);
      t10->SetTextColor(2); 
      t10->Draw("same");       
      canvas->cd(2);  
      canvas->cd(2)->SetGridy();
      htrbyhalfstave1Dratio[l]->Draw("");
      //htrbyhalfstave1Dratio[l]->Draw("text90same");      
      TLine* bdlevelratio[10];
      for (int xpoint = 0; xpoint < 10; xpoint++) {
         bdlevelratio[xpoint] = new TLine(nbinsY*xpoint,0,nbinsY*xpoint,1.05); 
         if(xpoint==3) {
            bdlevelratio[xpoint]->SetLineColor(2);
            bdlevelratio[xpoint]->SetLineWidth(2);
         }      
         bdlevelratio[xpoint]->Draw("same");
                  
      }  
      TLatex *t20 = new TLatex(nbinsY*6.2, 0.85, Form("Empty Sensors : %d (%2.1f %%)",SectionCnt[l][0],SectionR[l][0]*100));         
      t20->SetTextSize(0.04);
      t20->SetTextColor(2);
      TLatex *t21 = new TLatex(nbinsY*6.2, 0.75, Form("Non Empty Sensors (N_{track}<10^{3}) : %d (%2.1f %%)",SectionCnt[l][1]-SectionCnt[l][0],(SectionR[l][1]-SectionR[l][0])*100));
      t21->SetTextSize(0.04);
      t21->SetTextColor(2); 
      TLatex *t22 = new TLatex(nbinsY*6.2, 0.65, Form("nEvents (Training, Test) : (%d, %d)",(int)fTraining->GetN(),(int)fTest->GetN()));
      t22->SetTextSize(0.04);
      t22->SetTextColor(2);
      TLatex *t23 = new TLatex(nbinsY*6.2, 0.55, Form("nTracks (Training, Test) : (%d, %d)",(int)fTotNTraining, (int)fTotNTest));
      t23->SetTextSize(0.04);
      t23->SetTextColor(2);
      
      t20->Draw("same"); 
      t21->Draw("same"); 
      t22->Draw("same"); 
      t23->Draw("same");        
      TString htrbyhalfstave1D  = "TracksByHalfStave1D_Layer" + TString::Itoa(l,10) + ".gif"; 
      canvas->SaveAs(htrbyhalfstave1D,"gif");      
  
      //htrbyhalfstave1D[l]->Reset();
      //delete htrbyhalfstave1D[l];
   }
   std::cout<<"YMultiLayerPerceptron::MonitorTracksByHalfStave1D End"<<std::endl;      
}

void YMultiLayerPerceptron::MonitorTracksBySensor()
{

   // x : chipID in stave n = nHicPerStave[l]*nChipsPerHic[l]
   // y : stave
   std::cout<<"YMultiLayerPerceptron::MonitorTracksBySensor Start"<<std::endl;   
   TVirtualPad *canvas = new TCanvas("MonitorTracksBySensor", "Alignment MonitorTracksBySensor",3000,1200);   
   
   TH2I*   htrbysensor[nLAYER];
   gStyle->SetPaintTextFormat("1.0f");
   std::cout<<"YMultiLayerPerceptron::MonitorTracksBySensor Step 1"<<std::endl;   
   for(int l = 0; l < nLAYERIB; l++){
      int    nbinsX = nHicPerStave[l]*nChipsPerHic[l];
      int    nbinsY = NStaves[l];    

      TString hname  = "hL" + TString::Itoa(l,10) + "TrbySensor";
      TString htitle = "N_{track}/Sensor Distribution Layer " + TString::Itoa(l,10);        
      htrbysensor[l] = new TH2I(hname,htitle,nbinsX,-0.5,nbinsX-0.5,nbinsY,-0.5,nbinsY-0.5);
      htrbysensor[l]->SetXTitle("ChipID in Stave");
      htrbysensor[l]->SetYTitle("StaveID");
      //htrbysensor[l]->SetMaximum(trmaximum);
      htrbysensor[l]->SetMinimum(0);
      htrbysensor[l]->GetXaxis()->SetNdivisions(110);
      htrbysensor[l]->GetYaxis()->SetNdivisions(120);
      htrbysensor[l]->GetXaxis()->SetTickLength(0.01);
      htrbysensor[l]->GetYaxis()->SetTickLength(0.01);         
      htrbysensor[l]->GetZaxis()->SetLabelSize(0.02);
      
      for (int ypoint = 0; ypoint < (nbinsY/2); ypoint++) {
         htrbysensor[l]->GetYaxis()->SetBinLabel(ypoint+1+(nbinsY/2),Form("%d",ypoint));
      }         
      for (int ypoint = (nbinsY/2); ypoint < nbinsY; ypoint++) {
         htrbysensor[l]->GetYaxis()->SetBinLabel(ypoint+1-(nbinsY/2),Form("%d",ypoint));
      } 

   }
   for(int l = nLAYERIB; l < nLAYER; l++){
      int    nbinsX = (nHicPerStave[l]/2)*(nChipsPerHic[l]/2);
      int    nbinsY = NStaves[l]*NSubStave[l]*NSubStave2[l];    

      TString hname  = "hL" + TString::Itoa(l,10) + "TrbySensor";
      TString htitle = "N_{track}/Sensor Distribution Layer " + TString::Itoa(l,10);        
      htrbysensor[l] = new TH2I(hname,htitle,nbinsX,-0.5,nbinsX-0.5,nbinsY,-0.5,nbinsY-0.5);
      htrbysensor[l]->SetXTitle("ModuleID in Half-Stave");
      htrbysensor[l]->SetYTitle("StaveID");
      htrbysensor[l]->GetYaxis()->SetTitleOffset(1.25);
      //htrbysensor[l]->SetMaximum(trmaximum);
      htrbysensor[l]->SetMinimum(0);
      htrbysensor[l]->GetXaxis()->SetNdivisions(110);
      htrbysensor[l]->GetYaxis()->SetNdivisions(100);
      htrbysensor[l]->GetXaxis()->SetTickLength(0.01);
      htrbysensor[l]->GetYaxis()->SetTickLength(0.01);         
      htrbysensor[l]->GetZaxis()->SetLabelSize(0.02);

      for (int xpoint = 0; xpoint < (nHicPerStave[l]/2); xpoint++) {
         htrbysensor[l]->GetXaxis()->SetBinLabel((nChipsPerHic[l]/2)*xpoint+4,Form("M%d",xpoint));
         htrbysensor[l]->GetXaxis()->LabelsOption("h");
      }
      //for (int ypoint = 0; ypoint < (nbinsY/4); ypoint++) {
      //   htrbysensor[l]->GetYaxis()->SetBinLabel(NSubStave[l]*NSubStave2[l]*ypoint+2,Form("%d",ypoint));
      //}      
      for (int ypoint = 0; ypoint < (nbinsY/8); ypoint++) {
         htrbysensor[l]->GetYaxis()->SetBinLabel(NSubStave[l]*NSubStave2[l]*(ypoint+(nbinsY/8))+2,Form("%d",ypoint));
      }         
      for (int ypoint = (nbinsY/8); ypoint < (nbinsY/4); ypoint++) {
         htrbysensor[l]->GetYaxis()->SetBinLabel(NSubStave[l]*NSubStave2[l]*(ypoint-(nbinsY/8))+2,Form("%d",ypoint));
      }        
   }

   std::cout<<"YMultiLayerPerceptron::MonitorTracksBySensor Step 2"<<std::endl;      
   for(int ichipID = 0; ichipID < nSensors; ichipID++){
      int nhits = fSCNetwork[NetworkChips[ichipID]]->GetnEvents();
      
      int layer       = yGEOM->GetLayer(NetworkChips[ichipID]);
      int staveID     = yGEOM->GetStave(NetworkChips[ichipID]);
      int chipIDstave = yGEOM->GetChipIdInStave(NetworkChips[ichipID]);  
      if(layer<nLAYERIB){
         int xbin, ybin;
         int hb		  = yGEOM->GetHalfBarrel(NetworkChips[ichipID]);     
         ybin = staveID;
         if(hb==0) ybin = ybin + (NStaves[layer]/2);
         if(hb==1) ybin = ybin - (NStaves[layer]/2);
            
         xbin = chipIDstave;
         htrbysensor[layer]->SetBinContent(xbin+1, ybin+1, nhits);        
         //htrbysensor[layer]->SetBinContent(chipIDstave+1, staveID+1, nhits);
         //std::cout<<"layer staveID chipIDstave nhits "<<layer<<" "<<staveID<<" "<<chipIDstave<<" "<<nhits<<std::endl;
      } else {
         int xbin, ybin;
         int hb		  = yGEOM->GetHalfBarrel(NetworkChips[ichipID]); 
         int hs           = yGEOM->GetHalfStave(NetworkChips[ichipID]); 
         int ybinInStave  = chipIDstave%nChipsPerHic[layer] < nChipsPerHic[layer]/2 ? 2*hs+1 : 2*hs; 
         
         ybin = NSubStave[layer]*NSubStave2[layer]*staveID + ybinInStave;
         if(hb==0) ybin = ybin + NSubStave[layer]*NSubStave2[layer]*(NStaves[layer]/2);
         if(hb==1) ybin = ybin - NSubStave[layer]*NSubStave2[layer]*(NStaves[layer]/2);
         
         int md	= yGEOM->GetModule(NetworkChips[ichipID]);            
         int xbinInModule = chipIDstave%nChipsPerHic[layer] < nChipsPerHic[layer]/2 ? (chipIDstave%7) 
                                                                                    : std::abs((chipIDstave%7)-6); 
         xbin = 7*md + xbinInModule;     
         htrbysensor[layer]->SetBinContent(xbin+1, ybin+1, nhits);                        
      }
   }
   std::cout<<"YMultiLayerPerceptron::MonitorTracksBySensor Step 3"<<std::endl;      
   for(int l = 0; l < nLAYER; l++){
      canvas->Clear();
      canvas->Divide(3,1,0.001,0.001);      
      canvas->SetTickx(1);
      canvas->SetTicky(1);            
      if(l<3) {
         canvas->cd(1);
         htrbysensor[l]->Draw("colztext");
         canvas->cd(2);
         TH1D* htrbysensorXproj = (TH1D*) htrbysensor[l]->ProjectionX();
         TString htitleXproj = "Projection z; ChipID in Stave; Count";                 
         htrbysensorXproj->SetTitle(htitleXproj);
         htrbysensorXproj->GetYaxis()->SetNdivisions(110);
         htrbysensorXproj->GetYaxis()->SetLabelSize(0.02);
         htrbysensorXproj->Draw();
         canvas->cd(3);
         TH1D* htrbysensorYproj = (TH1D*) htrbysensor[l]->ProjectionY();
         TString htitleYproj = "Projection #phi; StaveID; Count";                 
         htrbysensorYproj->SetTitle(htitleYproj); 
         for (int ypoint = 0; ypoint < NStaves[l]; ypoint++) {
            htrbysensorYproj->GetXaxis()->SetBinLabel(ypoint+1,"");
         }            
         for (int ypoint = 0; ypoint < (NStaves[l]/2); ypoint++) {
            htrbysensorYproj->GetXaxis()->SetBinLabel(ypoint+(NStaves[l]/2)+1,Form("%d",ypoint));
         }   
         for (int ypoint = (NStaves[l]/2); ypoint < NStaves[l]; ypoint++) {
            htrbysensorYproj->GetXaxis()->SetBinLabel(ypoint-(NStaves[l]/2)+1,Form("%d",ypoint));
         }                    
         htrbysensorYproj->GetYaxis()->SetNdivisions(110);
         htrbysensorYproj->GetYaxis()->SetLabelSize(0.02);    
         htrbysensorYproj->GetXaxis()->SetRangeUser(-0.5,NStaves[l]-0.5);  
         htrbysensorYproj->SetMinimum(0);                    
         htrbysensorYproj->Draw();        
      } else {
         canvas->cd(1);      
         htrbysensor[l]->Draw("colz");      
         int    nbinsX = (nHicPerStave[l]/2)*(nChipsPerHic[l]/2);
         int    nbinsY = NStaves[l]*NSubStave[l]*NSubStave2[l];         
         TLine* bdmodule[nHicPerStave[l]/2];
         for (int xpoint = 0; xpoint < (nHicPerStave[l]/2); xpoint++) {
            bdmodule[xpoint] = new TLine((nChipsPerHic[l]/2)*xpoint-0.5,-0.5,(nChipsPerHic[l]/2)*xpoint-0.5,nbinsY-0.5);
            bdmodule[xpoint]->Draw("same");
         }  
         TLine* bdstave[NStaves[l]];
         for (int ypoint = 0; ypoint < NStaves[l]; ypoint++) {
            bdstave[ypoint] = new TLine(-0.5,NSubStave[l]*NSubStave2[l]*ypoint-0.5,nbinsX-0.5,NSubStave[l]*NSubStave2[l]*ypoint-0.5);
            bdstave[ypoint]->Draw("same");            
         }        
         
         canvas->cd(2);
         TH1D* htrbysensorXproj = (TH1D*) htrbysensor[l]->ProjectionX()->Clone();
         TString htitleXproj = "Projection z; ModuleID in Half-Stave; Count";                 
         htrbysensorXproj->SetTitle(htitleXproj);
         for (int xpoint = 0; xpoint < htrbysensorXproj->GetXaxis()->GetNbins(); xpoint++) {
            htrbysensorXproj->GetXaxis()->SetBinLabel(xpoint+1,"");
         }   
         for (int xpoint = 0; xpoint < (nHicPerStave[l]/2); xpoint++) {
            htrbysensorXproj->GetXaxis()->SetBinLabel((nChipsPerHic[l]/2)*xpoint+4,Form("M%d",xpoint));
            htrbysensorXproj->GetXaxis()->LabelsOption("h");
         }           
         htrbysensorXproj->GetXaxis()->SetTitleOffset(1.0);
         htrbysensorXproj->GetXaxis()->SetNdivisions(100);         
         htrbysensorXproj->GetYaxis()->SetNdivisions(110);
         htrbysensorXproj->GetYaxis()->SetLabelSize(0.02);         
         htrbysensorXproj->Draw();
         std::cout<<"L"<<l<<" X proj max : "<<gPad->GetUymax()<<" "<<htrbysensorXproj->GetBinContent(htrbysensorXproj->GetMaximumBin())*1.05<<std::endl;         
         TLine* bdmoduleProjX[nHicPerStave[l]/2];         
         for (int xpoint = 0; xpoint < (nHicPerStave[l]/2); xpoint++) {
            bdmoduleProjX[xpoint] = new TLine((nChipsPerHic[l]/2)*xpoint-0.5,0,(nChipsPerHic[l]/2)*xpoint-0.5,htrbysensorXproj->GetBinContent(htrbysensorXproj->GetMaximumBin())*1.05);         
            //bdmoduleProjX[xpoint]->Draw();       
         }     
         canvas->cd(3);
         TH1D* htrbysensorYproj = (TH1D*) htrbysensor[l]->ProjectionY()->Clone();
         TString htitleYproj = "Projection #phi; StaveID; Count"; 
         htrbysensorYproj->SetTitle(htitleYproj);          
         for (int ypoint = 0; ypoint < htrbysensorYproj->GetXaxis()->GetNbins(); ypoint++) {
            htrbysensorYproj->GetXaxis()->SetBinLabel(ypoint+1,"");
         }            
         //for (int ypoint = 0; ypoint < (nbinsY/4); ypoint++) {
         //   htrbysensorYproj->GetXaxis()->SetBinLabel(NSubStave[l]*NSubStave2[l]*ypoint+2,Form("%d",ypoint));
         //}               
         for (int ypoint = 0; ypoint < (nbinsY/8); ypoint++) {
            htrbysensorYproj->GetXaxis()->SetBinLabel(NSubStave[l]*NSubStave2[l]*(ypoint+(nbinsY/8))+2,Form("%d",ypoint));
         }   
         for (int ypoint = (nbinsY/8); ypoint < (nbinsY/4); ypoint++) {
            htrbysensorYproj->GetXaxis()->SetBinLabel(NSubStave[l]*NSubStave2[l]*(ypoint-(nbinsY/8))+2,Form("%d",ypoint));
         }                           
         htrbysensorYproj->GetXaxis()->SetTitleOffset(1.0);  
         htrbysensorYproj->GetXaxis()->SetNdivisions(100);                              
         htrbysensorYproj->GetYaxis()->SetNdivisions(110);
         htrbysensorYproj->GetYaxis()->SetLabelSize(0.02);            
         htrbysensorYproj->Draw();
         std::cout<<"L"<<l<<" Y proj max : "<<gPad->GetUymax()<<" "<<htrbysensorYproj->GetBinContent(htrbysensorYproj->GetMaximumBin())*1.05<<std::endl;
         TLine* bdstaveProjY[NStaves[l]];
         for (int ypoint = 0; ypoint < NStaves[l]; ypoint++) {
            bdstaveProjY[ypoint] = new TLine(NSubStave[l]*NSubStave2[l]*ypoint-0.5,0,NSubStave[l]*NSubStave2[l]*ypoint-0.5,htrbysensorYproj->GetBinContent(htrbysensorYproj->GetMaximumBin())*1.05);
            //bdstaveProjY[ypoint]->Draw();            
         }          
      }

      TString htrbysensorname  = "TracksBySensor_Layer" + TString::Itoa(l,10) + ".gif"; 
      canvas->SaveAs(htrbysensorname,"gif");      
  
      htrbysensor[l]->Reset();
      delete htrbysensor[l];
   }
   std::cout<<"YMultiLayerPerceptron::MonitorTracksBySensor End"<<std::endl;      
}

bool YMultiLayerPerceptron::DumpNTracksBySensor()
{
   std::cout<<"YMultiLayerPerceptron::DumpNTracksBySensor Start"<<std::endl;      

   TString filen = "NTracksBySensor.txt";
   std::ostream * output;
   if (filen == "") {
      Error("YMultiLayerPerceptron::DumpNTracksBySensor()","Invalid file name");
      return kFALSE;
   }
   if (filen == "-")
      output = &std::cout;
   else
      output = new std::ofstream(filen.Data());

   *output << "#NTracks by Sensor" << std::endl;
   for(int ic = 0; ic < nSensors; ic++){
      int nhits = fSCNetwork[NetworkChips[ic]]->GetnEvents();   
      *output << NetworkChips[ic] <<" "<< nhits <<std::endl;                                        
   } 
   if (filen != "-") {
      ((std::ofstream *) output)->close();
      delete output;
   }
   std::cout<<"YMultiLayerPerceptron::DumpNTracksBySensor End"<<std::endl;            
   return kTRUE;
   

}

void YMultiLayerPerceptron::EvaluateCost(int step, int core)
{
   std::cout<<"YMultiLayerPerceptron::EvaluateCost PrevWeightName : "<<fPrevWeightName<<std::endl;
   Info("Train","Using %d train and %d test events.",
        (int)fTraining->GetN(), (int)fTest->GetN());     
   Info("Train","Using %d train and %d test tracks.",
        (int)fTotNTraining, (int)fTotNTest); 
   std::cout<<"Npronged-Track  N = "<<fNpronged<<std::endl;
   //SetNetworkMatrix();
   std::cout<<"Init_Randomize Start!"<<std::endl;
   Init_Randomize();
   Init_RandomizeSensorCorrection();
   Init_RandomizeSensorCorrectionDetectorUnit();   
   std::cout<<"Init_Randomize Done!"<<std::endl;
   LoadWeights(fPrevWeightName);
   double C_training_E = fTraining->GetN() > GetEventLoss(1) ? GetCost(YMultiLayerPerceptron::kTraining) / (fTraining->GetN() - GetEventLoss(1)) : 0;//fTraining->GetN();
   double C_test_E     = fTest->GetN()     > GetEventLoss(2) ? GetCost(YMultiLayerPerceptron::kTest) / (fTest->GetN() - GetEventLoss(2)) : 0;//fTest->GetN();   
   std::cout.precision(10); 

   std::cout << "Step "<<step<<" Core "<<core<<" EpochLast: " 
             << " learn=" << C_training_E
             << " test=" << C_test_E     
             << " nEvents Training "<<fTraining->GetN()<<" "<<GetEventLoss(1)<<" "
             << " nEvents Test "    <<fTest->GetN()    <<" "<<GetEventLoss(2)<<" "                     
             << " nTracks Training "<<fTotNTraining<<" "<<GetTrackLoss(1)<<" "
             << " nTracks Test "    <<fTotNTest    <<" "<<GetTrackLoss(2)<<" "
             << std::endl;     
}   

////////////////////////////////////////////////////////////////////////////////
/// Cost on the output for a given event

Double_t YMultiLayerPerceptron::GetCost(Int_t ntrack) //const
{
   double cost = 0; 

   // look at 1st output neruon to determine type and cost_sensor function
   Int_t nEntries = fLastLayer.GetEntriesFast();
   if (nEntries == 0) return 0.0;
   if(ntrack==1){
      switch (fOutType) {
         case (YNeuron::kLinear):{
            cost = GetCost_Sensor(fFitModel);
            break;
         }
         case (YNeuron::kLFETanh):{
            cost = GetCost_Sensor(fFitModel);
            break;
         }
         default:
            cost = 0;
      }
     
   } else if(ntrack>1){
      cost = GetCost_Beam(fFitModel,ntrack);
   }      
   cost *= fEventWeight->EvalInstance();
   cost *= fCurrentTreeWeight;       
   
   return cost;
}

////////////////////////////////////////////////////////////////////////////////
/// Cost on the whole dataset

Double_t YMultiLayerPerceptron::GetCost(YMultiLayerPerceptron::EDataSet set) //const
{
   fCostMonitor->Reset();   
   fBeamXY->Reset();   
   fBeamZR->Reset(); 
   fVertexFitXY->Reset(); 
   fVertexFitZR->Reset(); 
   fVertexXY->Reset(); 
   fVertexZR->Reset();     
   
   for(int pstv = 0; pstv < 20; pstv++) { 
      fCostChargeSymSum[pstv] = 0;
      fCostChargeSymNtr[pstv] = 0;
   }
   fCostChargeSym->Reset();
   fChargeSymMonitorPositive->Reset();
   fChargeSymMonitorNegative->Reset();   
#ifdef MONITORONLYUPDATES   
   //fUPDATESENSORS->Reset();   
   fUPDATETRACKS->Reset();
#endif   
   ResidualMonitor = new TFile(Form("Residual_Monitor_Epoch_At_%d.root",fCurrentEpoch),"recreate");

   fResidualMonitor = new TTree("ResMonitor","ResMonitor");   
   b_resmonitor = new YResidualMonitor();
   fResidualMonitor->Branch("monitor",      &b_resmonitor);

   for(int l = 0; l <nLAYER; l++){
      fChi2Layer[l]->Reset();  
      fpTvsResLayer[l][0]->Reset();  
      fpTvsResLayer[l][1]->Reset();        
      fpTvsChiLayer[l][0]->Reset();  
      fpTvsChiLayer[l][1]->Reset();  
#ifdef MONITORHALFSTAVEUNIT                                            
      int nHalfBarrel = 2;
      for(int hb = 0; hb < nHalfBarrel; hb++){
         int nHalfStave = NSubStave[l]; 
         for(int hs = 0; hs < nHalfStave; hs++){
            fpTvsResLayerHBHS[l][hb][hs][0]->Reset(); 
            fpTvsResLayerHBHS[l][hb][hs][1]->Reset(); 
            fpTvsChiLayerHBHS[l][hb][hs][0]->Reset(); 
            fpTvsChiLayerHBHS[l][hb][hs][1]->Reset(); 
            fResidualsVsZLayerHBHS[l][hb][hs][0]->Reset();
            fResidualsVsZLayerHBHS[l][hb][hs][1]->Reset();
            fResidualsVsPhiLayerHBHS[l][hb][hs][0]->Reset();
            fResidualsVsPhiLayerHBHS[l][hb][hs][1]->Reset();    
            fProfileVsZLayerHBHS[l][hb][hs][0]->Reset();
            fProfileVsZLayerHBHS[l][hb][hs][1]->Reset();
            fProfileVsPhiLayerHBHS[l][hb][hs][0]->Reset();
            fProfileVsPhiLayerHBHS[l][hb][hs][1]->Reset();
            fSensorCenterVsZLayerHBHS[l][hb][hs][0]->Reset();
            fSensorCenterVsZLayerHBHS[l][hb][hs][1]->Reset();
            fSensorCenterVsPhiLayerHBHS[l][hb][hs][0]->Reset();
            fSensorCenterVsPhiLayerHBHS[l][hb][hs][1]->Reset();
         }       
      }  
#endif          
   }   
 
   for(int ichipID = 0; ichipID < nSensors; ichipID++){   
      fSCNetwork[NetworkChips[ichipID]]->InitResProfile();
#ifdef MONITORSENSORUNITpT         
      fSCNetwork[NetworkChips[ichipID]]->InitpTvsRes();                       
      fSCNetwork[NetworkChips[ichipID]]->InitpTvsChi();                       
      fSCNetwork[NetworkChips[ichipID]]->InitChi2();      
#endif          
   }
   TEventList *list =
       ((set == YMultiLayerPerceptron::kTraining) ? fTraining : fTest);
   Double_t cost = 0;
   InitEventLoss();   
   InitTrackLoss();
   Int_t i;
   if (list) {
      Int_t nEvents = 0; // list->GetN();
      if(list == fTraining) {
         for(int ichipID = 0; ichipID < nSensors; ichipID++){      
            fSCNetwork[NetworkChips[ichipID]]->SetnEvents(0);    
      
            if(fLearningMethod==YMultiLayerPerceptron::kBatchDetectorUnitUser || fLearningMethod==YMultiLayerPerceptron::kOffsetTuneByMean) {
      
               int SensorID 	= ichipID;
                           
               int layer         = yGEOM->GetLayer(SensorID);
               int det           = layer < nLAYERIB ? 0 : (layer < nLAYERIB + nLAYEROB1 ? 1 : 2); 
               int detIdx        = layer < nLAYERIB ? layer : (layer < nLAYERIB + nLAYEROB1 ? layer - nLAYERIB : layer - (nLAYERIB + nLAYEROB1));
               int hb	 	 = yGEOM->GetHalfBarrel(SensorID);  		//HalfBarrel
               int hs	 	 = yGEOM->GetHalfStave(SensorID);  		//HalfStave               
               int stave	 = yGEOM->GetStave(SensorID); 
               int staveIdx      = stave%(NStaves[layer]/2);      
               int ChipIdInStave = yGEOM->GetChipIdInStave(SensorID);
               int chipZ         = ChipIdInStave%(nHicPerStave[layer]*nChipsPerHic[layer]/NSubStave[layer]);  
      
               // det hs chipZ detIdx hb    
               fDetectorUnit->fSCNetwork->SetnEvents(0);
               fDetectorUnit->SubUnit[det]->fSCNetwork->SetnEvents(0);
               fDetectorUnit->SubUnit[det]->SubUnit[hs]->fSCNetwork->SetnEvents(0);
               fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->fSCNetwork->SetnEvents(0);
               fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->fSCNetwork->SetnEvents(0);
               fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->fSCNetwork->SetnEvents(0);
               fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->fSCNetwork->SetnEvents(0);
               fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->SubUnit[hs]->fSCNetwork->SetnEvents(0);
            }
         }
         
         nEvents = fTraining->GetN();
         //std::cout<<"GetCost Training Set :: nEvents "<<nEvents<<std::endl;
         for (i = 0; i < nEvents; i++) {
            GetEntry(fTraining->GetEntry(i));
            double cost_train = GetCost(fTrainingIndex[i][1]);
            //std::cout<<"Epoch"<<fCurrentEpoch<<" Cost(Train Set) :: Event"<<i<<" "<<cost_train<<std::endl;
            cost += cost_train;
         }
         SetEventLoss(1);         
         SetTrackLoss(1);
                  
#ifdef MONITORONLYUPDATES         
 #if MONITORONLYUPDATES_MODE==-1
         fUPDATESENSORS->Reset();
         fUPDATESENSORS->SetEntries(0);
 #elif MONITORONLYUPDATES_MODE==0
         if(fCurrentEpoch==-1){
            fUPDATESENSORS->Reset();
            fUPDATESENSORS->SetEntries(0);
         }         
 #endif
#endif         
         for(int ichipID = 0; ichipID < nSensors; ichipID++){   
            int nTracks = fSCNetwork[NetworkChips[ichipID]]->hNtracksByRejection->GetBinContent(1+6);   
            fSCNetwork[NetworkChips[ichipID]]->SetnEvents(nTracks);    
#ifdef MONITORONLYUPDATES
 #if MONITORONLYUPDATES_MODE==-1
            if(fSCNetwork[NetworkChips[ichipID]]->GetnEvents()>=Min_Cluster_by_Sensor) fUPDATESENSORS->SetBinContent(1+NetworkChips[ichipID],1);
            else fUPDATESENSORS->SetBinContent(1+NetworkChips[ichipID],0);
 #elif MONITORONLYUPDATES_MODE==0
            if(fCurrentEpoch==-1){
               if(fSCNetwork[NetworkChips[ichipID]]->GetnEvents()>=Min_Cluster_by_Sensor) fUPDATESENSORS->SetBinContent(1+NetworkChips[ichipID],1));
               else fUPDATESENSORS->SetBinContent(1+NetworkChips[ichipID],0);   
            }
 #endif
#endif            
            if(fLearningMethod==YMultiLayerPerceptron::kBatchDetectorUnitUser || fLearningMethod==YMultiLayerPerceptron::kOffsetTuneByMean) {
      
               int SensorID 	= ichipID;
                           
               int layer         = yGEOM->GetLayer(SensorID);
               int det           = layer < nLAYERIB ? 0 : (layer < nLAYERIB + nLAYEROB1 ? 1 : 2); 
               int detIdx        = layer < nLAYERIB ? layer : (layer < nLAYERIB + nLAYEROB1 ? layer - nLAYERIB : layer - (nLAYERIB + nLAYEROB1));
               int hb		= yGEOM->GetHalfBarrel(SensorID);  		//HalfBarrel
               int hs		= yGEOM->GetHalfStave(SensorID);  		//HalfStave               
               int stave		= yGEOM->GetStave(SensorID); 
               int staveIdx      = stave%(NStaves[layer]/2);      
               int ChipIdInStave = yGEOM->GetChipIdInStave(SensorID);
               int chipZ         = ChipIdInStave%(nHicPerStave[layer]*nChipsPerHic[layer]/NSubStave[layer]);  
      
               // det hs chipZ detIdx hb    
               fDetectorUnit->fSCNetwork->SetnEvents(nTracks +
               fDetectorUnit->fSCNetwork->GetnEvents());
               fDetectorUnit->SubUnit[det]->fSCNetwork->SetnEvents(nTracks +
               fDetectorUnit->SubUnit[det]->fSCNetwork->GetnEvents());
               fDetectorUnit->SubUnit[det]->SubUnit[hs]->fSCNetwork->SetnEvents(nTracks+
               fDetectorUnit->SubUnit[det]->SubUnit[hs]->fSCNetwork->GetnEvents());
               fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->fSCNetwork->SetnEvents(nTracks+
               fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->fSCNetwork->GetnEvents());
               fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->fSCNetwork->SetnEvents(nTracks+
               fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->fSCNetwork->GetnEvents());
               fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->fSCNetwork->SetnEvents(nTracks+
               fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->fSCNetwork->GetnEvents());
               fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->fSCNetwork->SetnEvents(nTracks+
               fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->fSCNetwork->GetnEvents());
               fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->SubUnit[hs]->fSCNetwork->SetnEvents(nTracks+
               fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->SubUnit[hs]->fSCNetwork->GetnEvents());
            }
         }              
                 
      } else {
         nEvents = fTest->GetN();
         //std::cout<<"GetCost Test Set     :: nEvents "<<nEvents<<std::endl;                
         for (i = 0; i < nEvents; i++) {
            GetEntry(fTest->GetEntry(i));     
            double cost_test = GetCost(fTestIndex[i][1]);
            //std::cout<<"Epoch"<<fCurrentEpoch<<" Cost(Test Set) :: Event"<<i<<" "<<cost_test<<std::endl;                    
            cost += cost_test;
         }
         SetEventLoss(2);                  
         SetTrackLoss(2);            
      }

   } else if (fData) {
      Int_t nEvents = fData->GetEntriesFast();   
      //std::cout<<"GetCost Data Set     :: nEvents "<<nEvents<<std::endl;                         
      for (i = 0; i < nEvents; i++) {
         fData->GetEntry(i);
         double cost_data = GetCost(fEventIndex[i][1]);
         //std::cout<<"Epoch"<<fCurrentEpoch<<" Cost(Data Set) :: Event"<<i<<" "<<cost_data<<std::endl;                    
         cost += cost_data;
      }
      SetEventLoss(0);               
      SetTrackLoss(0);   
   }
   fResidualMonitor->Write();

   delete fResidualMonitor;
   delete ResidualMonitor; 

   //std::cout<<"GetCost by Set"<<cost<<std::endl;
   return cost;
}


////////////////////////////////////////////////////////////////////////////////
/// Line Fit cost for LineFit output neurons, for a given event


Double_t YMultiLayerPerceptron::GetCost_Sensor(int fit) //const
{
  double cost = 0;
   switch (fit) {
      case 1: {
         cost = GetCost_Sensor_LineFit();
         return cost;
         break;
      }   
      case 2: {
         cost = GetCost_Sensor_CircleFit();
         return cost;
         break;
      }      
      default: {
         cost = GetCost_Sensor_LineFit();
         return cost;
      }      
   }
}

Double_t YMultiLayerPerceptron::GetCost_Beam(int fit, int ntrack) //const
{
   //std::cout<<"YMultiLayerPerceptron::GetCost_Beam( "<< fit <<" , "<< ntrack <<" )"<<std::endl; 
   double cost = 0;
   int ntrack_loss = 0;
   switch (fit) {
      case 1: {
         InitVertex_LineFit();
         for(int t=0; t<ntrack; t++){
            double cost_track = GetCost_Beam_LineFit(t);
            cost += cost_track;
            if(cost_track==0) {
               ntrack_loss++;
            }
         }
         double cost_vtx = GetCost_Vertex_LineFit();               
         cost += cost_vtx;
         int ndf = (2*(nLAYER) - 4)*(ntrack-ntrack_loss) + 1;
         if(ntrack>ntrack_loss) { 
            cost = cost/ndf;
         } else {
            cost = 0;
            AddEventLoss();            
         }
         return cost;
         break;
      }   
      case 2: {
         InitVertex_CircleFit();
         //UpdateVertexByAlignment();
         for(int t=0; t<ntrack; t++){
            double cost_track = GetCost_Beam_CircleFit(t);
            cost += cost_track;
            if(cost_track==0) {
               ntrack_loss++;
            }
         }      
         //std::cout<<"[monitor] track cost = "<<cost<<std::endl;
         int ndf = (2*(nLAYER + 1) - 4)*(ntrack-ntrack_loss) + 1;
         if(ntrack>ntrack_loss) { 
            cost = cost/ndf;
         } else {
            cost = 0;
            AddEventLoss();            
         }         
         return cost;
         break;
      }      
      default: {
         InitVertex_LineFit();
         for(int t=0; t<ntrack; t++){
            double cost_track = GetCost_Beam_LineFit(t);
            cost += GetCost_Beam_LineFit(t);
            if(cost_track==0) {
               ntrack_loss++;
            }
         }      
         double cost_vtx = GetCost_Vertex_LineFit();               
         cost += cost_vtx;      
         int ndf = (2*(nLAYER) - 4)*(ntrack-ntrack_loss) + 1;
         if(ntrack>ntrack_loss) { 
            cost = cost/ndf;
         } else {
            cost = 0;
            AddEventLoss();            
         }         
         return cost;
      }      
   }
}

void YMultiLayerPerceptron::BetaLinearization(double* beta, TVector3* dirXc, std::vector<bool> hitUpdate)
{
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
        
   for(int ld = 0; ld < lin_dum; ld++){   
      double linear_beta_arr[5];
      double linear_beta_dev = 2*std::atan2(0,-1);
      for(int lc = 0; lc < 5; lc++){
         linear_beta_arr[lc] = 2*std::atan2(0,-1)*(lc-2) + std::atan2(dirXc[dum_index[ld]].Y(), dirXc[dum_index[ld]].X());
         if(linear_beta_dev > TMath::Abs(linear_beta_arr[lc] - beta_dum[(ld+lin_dum)%(lin_dum+1)])) {
            beta[dum_index[ld]] = linear_beta_arr[lc];
            beta_dum[ld] = linear_beta_arr[lc];
            linear_beta_dev = TMath::Abs(linear_beta_arr[lc] - beta_dum[(ld+lin_dum)%(lin_dum+1)]);
         }
      }              
   }   
}

void YMultiLayerPerceptron::GetProjectionPoints(TVector3 vecCircle_center, double RecRadius, TVector3* vecSensorNorm, TVector3* vecXc_meas, TVector3* vecXc_proj, TVector3* vecXc_norm)
{


   double pAp[nLAYER];
   double pBp[nLAYER];
   double pCp[nLAYER];

   double pAn[nLAYER];
   double pBn[nLAYER];
   double pCn[nLAYER];
   
   double xproj[nLAYER];
   double yproj[nLAYER];
   
   double xnorm[nLAYER];
   double ynorm[nLAYER];   

   //std::cout<<" Circle gX "<<vecCircle_center.X()<<" gY "<<vecCircle_center.Y()<<" gR "<<  RecRadius<<std::endl;
   //for(int l = 0; l < nLAYER+1; l++){    
   //   std::cout<<" Layer "<<l<<" "<<vecXc_meas[l].X()<<" "<<vecXc_meas[l].Y()<<" "<<vecXc_meas[l].Z()<<std::endl;
   //}   
   for(int l = 0; l < nLAYER; l++){    
   
      pAp[l] = 0;
      pBp[l] = 0;
      pCp[l] = 0;
      
      pAn[l] = 0;
      pBn[l] = 0;
      pCn[l] = 0;      
#ifdef YSCNEURONDEBUG  
      std::cout<<"Sensor Plane "<<vecSensorNorm[l].X()<<" "<<vecSensorNorm[l].Y()<<" "<<vecSensorNorm[l].Z()<<std::endl;
#endif    
      double xf[2], yf[2];
      double xno[2], yno[2];            
      if(std::abs(vecSensorNorm[l].X())>1e-8) {
         if(std::abs(vecSensorNorm[l].Y())>1e-8){ //nx != 0, ny != 0
#ifdef YSCNEURONDEBUG
            std::cout<<" L"<<l<<" Type nx!=0 ny!=0 "<<std::endl;   
            std::cout<<" L"<<l<<" Sensor Norm Vector "<< vecSensorNorm[l].X()<<" "<<vecSensorNorm[l].Y()<<" "<<vecSensorNorm[l].Z()<<std::endl;
            std::cout<<" L"<<l<<" Measured Position  "<< vecXc_meas[l].X()   <<" "<<vecXc_meas[l].Y()   <<" "<<vecXc_meas[l].Z()   <<std::endl;
#endif
            pAp[l] = 1 + std::pow(vecSensorNorm[l].X()/vecSensorNorm[l].Y(),2);
            pBp[l] = 2 * (-vecCircle_center.X() - (vecSensorNorm[l].X()/vecSensorNorm[l].Y())*((vecSensorNorm[l].X()/vecSensorNorm[l].Y())*vecXc_meas[l].X() + vecXc_meas[l].Y() - vecCircle_center.Y()));
            pCp[l] = std::pow(vecCircle_center.X(),2) + std::pow(((vecSensorNorm[l].X()/vecSensorNorm[l].Y())*vecXc_meas[l].X() + vecXc_meas[l].Y() - vecCircle_center.Y()),2) - std::pow(RecRadius,2);  
            
            pAn[l] = 1 + std::pow(vecSensorNorm[l].Y()/vecSensorNorm[l].X(),2);
            pBn[l] = 2 * (-vecCircle_center.X() + (vecSensorNorm[l].Y()/vecSensorNorm[l].X())*(-(vecSensorNorm[l].Y()/vecSensorNorm[l].X())*vecXc_meas[l].X() + vecXc_meas[l].Y() - vecCircle_center.Y()));
            pCn[l] = std::pow(vecCircle_center.X(),2) + std::pow((-(vecSensorNorm[l].Y()/vecSensorNorm[l].X())*vecXc_meas[l].X() + vecXc_meas[l].Y() - vecCircle_center.Y()),2) - std::pow(RecRadius,2);
#ifdef YSCNEURONDEBUG 
            std::cout<<" Type Normal :: Coefficient A "<<pAp[l]<<" B "<<pBp[l]<<" C "<<pCp[l]<<std::endl;    
#endif     
            double det_fit = pBp[l]*pBp[l]-4*pAp[l]*pCp[l];
            if(det_fit >= 0) {            
               xf[0] = (- pBp[l] + std::sqrt(pBp[l]*pBp[l] - 4*pAp[l]*pCp[l]))/(2*pAp[l]);
               yf[0] = - (vecSensorNorm[l].X()/vecSensorNorm[l].Y())*(xf[0] - vecXc_meas[l].X()) + vecXc_meas[l].Y();
               xf[1] = (- pBp[l] - std::sqrt(pBp[l]*pBp[l] - 4*pAp[l]*pCp[l]))/(2*pAp[l]);
               yf[1] = - (vecSensorNorm[l].X()/vecSensorNorm[l].Y())*(xf[1] - vecXc_meas[l].X()) + vecXc_meas[l].Y();  
            } else {
#ifdef YSCNEURONDEBUG
               std::cout<<"Abnormal case report I"<<std::endl;
#endif
               xf[0] = vecXc_meas[l].X();
               yf[0] = vecXc_meas[l].Y();
               xf[1] = vecXc_meas[l].X();
               yf[1] = vecXc_meas[l].Y();                                             
            }
            
            double det_no = pBn[l]*pBn[l]-4*pAn[l]*pCn[l];
            if(det_no >= 0) {
               xno[0] = (- pBn[l] + std::sqrt(pBn[l]*pBn[l] - 4*pAn[l]*pCn[l]))/(2*pAn[l]);
               yno[0] = + (vecSensorNorm[l].Y()/vecSensorNorm[l].X())*(xno[0] - vecXc_meas[l].X()) + vecXc_meas[l].Y();
               xno[1] = (- pBn[l] - std::sqrt(pBn[l]*pBn[l] - 4*pAn[l]*pCn[l]))/(2*pAn[l]);
               yno[1] = + (vecSensorNorm[l].Y()/vecSensorNorm[l].X())*(xno[1] - vecXc_meas[l].X()) + vecXc_meas[l].Y();            
            } else {
               xno[0] = xf[0];
               yno[0] = yf[0];
               xno[1] = xf[1];
               yno[1] = yf[1];                                
            }                       
         } else {                // nx != 0 , ny =  0
#ifdef YSCNEURONDEBUG
            std::cout<<" Type nx!=0 ny=0 "<<std::endl;    
#endif         
            xf[0] = vecXc_meas[l].X();
            yf[0] = vecCircle_center.Y() + std::sqrt(std::pow(RecRadius,2) - std::pow(xf[0]-vecCircle_center.X(),2));
            xf[1] = vecXc_meas[l].X();
            yf[1] = vecCircle_center.Y() - std::sqrt(std::pow(RecRadius,2) - std::pow(xf[0]-vecCircle_center.X(),2));  
 
            pAn[l] = 1;
            pBn[l] = 2 * (-vecCircle_center.X());
            pCn[l] = std::pow(vecCircle_center.X(),2) + std::pow((vecXc_meas[l].Y() - vecCircle_center.Y()),2) - std::pow(RecRadius,2);              
         
            double det_no = pBn[l]*pBn[l]-4*pAn[l]*pCn[l];
#ifdef YSCNEURONDEBUG
            std::cout<<"det_no : "<<det_no<<std::endl;
#endif
            if(det_no >= 0) {
               xno[0] = (- pBn[l] + std::sqrt(pBn[l]*pBn[l] - 4*pAn[l]*pCn[l]))/(2*pAn[l]);
               yno[0] = + (vecSensorNorm[l].Y()/vecSensorNorm[l].X())*(xno[0] - vecXc_meas[l].X()) + vecXc_meas[l].Y();
               xno[1] = (- pBn[l] - std::sqrt(pBn[l]*pBn[l] - 4*pAn[l]*pCn[l]))/(2*pAn[l]);
               yno[1] = + (vecSensorNorm[l].Y()/vecSensorNorm[l].X())*(xno[1] - vecXc_meas[l].X()) + vecXc_meas[l].Y();            
            } else {
               xno[0] = xf[0];
               yno[0] = yf[0];
               xno[1] = xf[1];
               yno[1] = yf[1];                                
            }            
#ifdef YSCNEURONDEBUG
            std::cout<<"DEBUG Xc Yc yno sqrt "<<vecCircle_center.X()<<" "<<vecCircle_center.Y()<<" "<<yno[0]<<std::sqrt(std::pow(RecRadius,2) - std::pow(yno[0]-vecCircle_center.Y(),2))<<std::endl;
#endif
         }
      } else {
         if(std::abs(vecSensorNorm[l].Y())>1e-8){ // nx =  0 , ny != 0
#ifdef YSCNEURONDEBUG
            std::cout<<" Type nx=0 ny!=0 "<<std::endl;   
#endif
            pAp[l] = 1;
            pBp[l] = 2 * (-vecCircle_center.X());
            pCp[l] = std::pow(vecCircle_center.X(),2) + std::pow((vecXc_meas[l].Y() - vecCircle_center.Y()),2) - std::pow(RecRadius,2);   
                         
            double det_fit = pBp[l]*pBp[l]-4*pAp[l]*pCp[l];
#ifdef YSCNEURONDEBUG
            std::cout<<"det_fit : "<<det_fit<<std::endl;
#endif
            if(det_fit >= 0) {            
               xf[0] = (- pBp[l] + std::sqrt(pBp[l]*pBp[l] - 4*pAp[l]*pCp[l]))/(2*pAp[l]);
               yf[0] = - (vecSensorNorm[l].X()/vecSensorNorm[l].Y())*(xf[0] - vecXc_meas[l].X()) + vecXc_meas[l].Y();
               xf[1] = (- pBp[l] - std::sqrt(pBp[l]*pBp[l] - 4*pAp[l]*pCp[l]))/(2*pAp[l]);
               yf[1] = - (vecSensorNorm[l].X()/vecSensorNorm[l].Y())*(xf[1] - vecXc_meas[l].X()) + vecXc_meas[l].Y();  
            } else {
#ifdef YSCNEURONDEBUG
               std::cout<<"Abnormal case report II"<<std::endl;
#endif
               xf[0] = vecXc_meas[l].X();
               yf[0] = vecXc_meas[l].Y();
               xf[1] = vecXc_meas[l].X();
               yf[1] = vecXc_meas[l].Y();                                             
            }     
            yno[0] = vecXc_meas[l].Y();
            xno[0] = vecCircle_center.X() + std::sqrt(std::pow(RecRadius,2) - std::pow(yno[0]-vecCircle_center.Y(),2));
            yno[1] = vecXc_meas[l].Y();
            xno[1] = vecCircle_center.X() - std::sqrt(std::pow(RecRadius,2) - std::pow(yno[0]-vecCircle_center.Y(),2));   
         } else {
#ifdef YSCNEURONDEBUG
            std::cout<<"Abnormal case report III"<<std::endl;
#endif
         }
      }
#ifdef YSCNEURONDEBUG
      std::cout<< "(N)Sensor at "<< l<<" Fit X MEAS Proj1 Proj2 : "<< vecXc_meas[l].X()<<" "<<xf[0] <<" "<<xf[1] <<std::endl;
      std::cout<< "(N)Sensor at "<< l<<" Fit Y MEAS Proj1 Proj2 : "<< vecXc_meas[l].Y()<<" "<<yf[0] <<" "<<yf[1] <<std::endl;
#endif
      double df0 = std::pow(vecXc_meas[l].X()- xf[0],2) + std::pow(vecXc_meas[l].Y()- yf[0],2); 
      double df1 = std::pow(vecXc_meas[l].X()- xf[1],2) + std::pow(vecXc_meas[l].Y()- yf[1],2);     
      xproj[l] = (df0 < df1) ? xf[0] : xf[1];
      yproj[l] = (df0 < df1) ? yf[0] : yf[1];   
      xproj[l] = (double)std::round(xproj[l]*TARGET_D)/TARGET_D;
      yproj[l] = (double)std::round(yproj[l]*TARGET_D)/TARGET_D;
#ifdef YSCNEURONDEBUG
      std::cout<< "(N)Sensor at "<< l<<" Fit X MEAS Proj : "<< vecXc_meas[l].X()<<" "<<xproj[l] <<std::endl;
      std::cout<< "(N)Sensor at "<< l<<" Fit Y MEAS Proj : "<< vecXc_meas[l].Y()<<" "<<yproj[l] <<std::endl;     
#endif
      vecXc_proj[l].SetXYZ(xproj[l], yproj[l], 0);   

#ifdef YSCNEURONDEBUG
      std::cout<< "(N)Normal at "<< l<<" Fit X MEAS norm1 norm2 : "<< vecXc_meas[l].X()<<" "<<xno[0] <<" "<<xno[1] <<std::endl;
      std::cout<< "(N)Normal at "<< l<<" Fit Y MEAS norm1 norm2 : "<< vecXc_meas[l].Y()<<" "<<yno[0] <<" "<<yno[1] <<std::endl;  
#endif 
      double dno0 = std::pow(vecXc_meas[l].X()- xno[0],2) + std::pow(vecXc_meas[l].Y()- yno[0],2); 
      double dno1 = std::pow(vecXc_meas[l].X()- xno[1],2) + std::pow(vecXc_meas[l].Y()- yno[1],2); 
      xnorm[l] = (dno0 < dno1) ? xno[0] : xno[1];
      ynorm[l] = (dno0 < dno1) ? yno[0] : yno[1];
      xnorm[l] = (double)std::round(xnorm[l]*TARGET_D)/TARGET_D;
      ynorm[l] = (double)std::round(ynorm[l]*TARGET_D)/TARGET_D;    
#ifdef YSCNEURONDEBUG
      std::cout<< "(N)Sensor at "<< l<<" Fit X MEAS norm : "<< vecXc_meas[l].X()<<" "<<xnorm[l] <<std::endl;
      std::cout<< "(N)Sensor at "<< l<<" Fit Y MEAS norm : "<< vecXc_meas[l].Y()<<" "<<ynorm[l] <<std::endl;    
#endif
      vecXc_norm[l].SetXYZ(xnorm[l], ynorm[l], 0);            
   }   
   vecXc_proj[nLAYER] = vecXc_meas[nLAYER];
   vecXc_norm[nLAYER] = vecXc_meas[nLAYER];
}

void YMultiLayerPerceptron::BuildDerivativesXY(){

/*
 Case 0 Layer 0 Delta 9 |dR|   vs |R|   p0, p1, p2 4.23141, 0.00168967, 6.47847e-06 d|Phi| vs 1/|R| p0, p1 7.76894e-05, 0.143749
 Case 0 Layer 1 Delta 9 |dR|   vs |R|   p0, p1, p2 4.84812, 0.00379778, 8.06859e-06 d|Phi| vs 1/|R| p0, p1 0.000103511, 0.19574
 Case 0 Layer 2 Delta 9 |dR|   vs |R|   p0, p1, p2 5.55983, 0.00294879, 1.15231e-05 d|Phi| vs 1/|R| p0, p1 0.000244778, 0.118684
 Case 0 Layer 3 Delta 9 |dR|   vs |R|   p0, p1, p2 -7.59608, 0.00799639, 2.10203e-05 d|Phi| vs 1/|R| p0, p1 0.000497616, -0.0870282
 Case 0 Layer 4 Delta 9 |dR|   vs |R|   p0, p1, p2 -5.11968, 0.00451787, 1.75176e-05 d|Phi| vs 1/|R| p0, p1 0.000421157, -0.0747484
 Case 0 Layer 5 Delta 9 |dR|   vs |R|   p0, p1, p2 -0.123571, 0.000427501, 4.06065e-06 d|Phi| vs 1/|R| p0, p1 2.5131e-05, -0.0037394
 Case 0 Layer 6 Delta 9 |dR|   vs |R|   p0, p1, p2 2.03526, -0.00685859, 2.69771e-05 d|Phi| vs 1/|R| p0, p1 0.000300189, -0.0477041

 Case 1 Layer 0 Delta 9 |dR|   vs |R|   p0, p1, p2 0.928085, 0.00101524, 1.99747e-06 d|Phi| vs 1/|R| p0, p1 2.63014e-05, 0.0425397
 Case 1 Layer 1 Delta 9 |dR|   vs |R|   p0, p1, p2 1.24324, 0.00139651, 2.58144e-06 d|Phi| vs 1/|R| p0, p1 3.39974e-05, 0.05713
 Case 1 Layer 2 Delta 9 |dR|   vs |R|   p0, p1, p2 1.48702, 0.0017379, 3.10563e-06 d|Phi| vs 1/|R| p0, p1 4.09293e-05, 0.0696507
 Case 1 Layer 3 Delta 9 |dR|   vs |R|   p0, p1, p2 -0.880183, -0.000245531, 7.68909e-06 d|Phi| vs 1/|R| p0, p1 0.000148576, -0.0260885
 Case 1 Layer 4 Delta 9 |dR|   vs |R|   p0, p1, p2 -0.560366, -0.000466389, 6.13899e-06 d|Phi| vs 1/|R| p0, p1 0.000125231, -0.0225048
 Case 1 Layer 5 Delta 9 |dR|   vs |R|   p0, p1, p2 -0.265622, 0.000513289, 1.17346e-06 d|Phi| vs 1/|R| p0, p1 5.75903e-06, -0.0010002
 Case 1 Layer 6 Delta 9 |dR|   vs |R|   p0, p1, p2 -0.976909, 0.000829599, 6.61853e-06 d|Phi| vs 1/|R| p0, p1 9.01096e-05, -0.014199
*/

   paramRparr = new double* [nLAYER];
   paramRperp = new double* [nLAYER];
   for(int ln = 0; ln<nLAYER; ln++){   
      paramRparr[ln] = new double [4];
      paramRperp[ln] = new double [4];               
      for(int np = 0; np < 4; np++){
         paramRparr[ln][np] = 0;
         paramRperp[ln][np] = 0;                  
      }
   }
   paramRparr[0][0] = -1; paramRparr[0][1] = 4.23141;   paramRparr[0][2] = 0.00168967;   paramRparr[0][3] = 6.47847e-06;
   paramRparr[1][0] = -1; paramRparr[1][1] = 4.84812;   paramRparr[1][2] = 0.00379778;   paramRparr[1][3] = 8.06859e-06;
   paramRparr[2][0] = -1; paramRparr[2][1] = 5.55983;   paramRparr[2][2] = 0.00294879;   paramRparr[2][3] = 1.15231e-05;
   paramRparr[3][0] = -1; paramRparr[3][1] = -7.59608;  paramRparr[3][2] = 0.00799639;   paramRparr[3][3] = 2.10203e-05;
   paramRparr[4][0] = -1; paramRparr[4][1] = -5.11968;  paramRparr[4][2] = 0.00451787;   paramRparr[4][3] = 1.75176e-05;
   paramRparr[5][0] = +1; paramRparr[5][1] = -0.123571; paramRparr[5][2] = 0.000427501;  paramRparr[5][3] = 4.06065e-06;
   paramRparr[6][0] = +1; paramRparr[6][1] = 2.03526;   paramRparr[6][2] = -0.00685859;  paramRparr[6][3] = 2.69771e-05;

   paramRperp[0][0] = +1; paramRperp[0][1] = 0.928085;  paramRperp[0][2] = 0.00101524;   paramRperp[0][3] = 1.99747e-06;
   paramRperp[1][0] = +1; paramRperp[1][1] = 1.24324;   paramRperp[1][2] = 0.00139651;   paramRperp[1][3] = 2.58144e-06;
   paramRperp[2][0] = +1; paramRperp[2][1] = 1.48702;   paramRperp[2][2] = 0.0017379;    paramRperp[2][3] = 3.10563e-06;
   paramRperp[3][0] = +1; paramRperp[3][1] = -0.880183; paramRperp[3][2] = -0.000245531; paramRperp[3][3] = 7.68909e-06;
   paramRperp[4][0] = +1; paramRperp[4][1] = -0.560366; paramRperp[4][2] = -0.000466389; paramRperp[4][3] = 6.13899e-06;
   paramRperp[5][0] = -1; paramRperp[5][1] = -0.265622; paramRperp[5][2] = 0.000513289;  paramRperp[5][3] = 1.17346e-06;
   paramRperp[6][0] = -1; paramRperp[6][1] = -0.976909; paramRperp[6][2] = 0.000829599;  paramRperp[6][3] = 6.61853e-06;

   paramQparr = new double* [nLAYER];
   paramQperp = new double* [nLAYER];
   for(int ln = 0; ln<nLAYER; ln++){   
      paramQparr[ln] = new double [3];
      paramQperp[ln] = new double [3];               
      for(int np = 0; np < 3; np++){
         paramQparr[ln][np] = 0;
         paramQperp[ln][np] = 0;                  
      }
   }
   paramQparr[0][0] = -1; paramQparr[0][1] = 7.768940e-5; paramQparr[0][2] = 0.143749;
   paramQparr[1][0] = -1; paramQparr[1][1] = 0.000103511; paramQparr[1][2] = 0.195740;
   paramQparr[2][0] = -1; paramQparr[2][1] = 0.000244778; paramQparr[2][2] = 0.118684;
   paramQparr[3][0] = -1; paramQparr[3][1] = 0.000497616; paramQparr[3][2] = -0.0870282;
   paramQparr[4][0] = -1; paramQparr[4][1] = 0.000421157; paramQparr[4][2] = -0.0747484;
   paramQparr[5][0] = -1; paramQparr[5][1] = 2.5131e-5;   paramQparr[5][2] = -0.0037394;
   paramQparr[6][0] = +1; paramQparr[6][1] = 0.000300189; paramQparr[6][2] = -0.0477041;

   paramQperp[0][0] = +1; paramQperp[0][1] = 2.63014e-5; paramQperp[0][2] = 0.0425397;
   paramQperp[1][0] = +1; paramQperp[1][1] = 3.39974e-5; paramQperp[1][2] = 0.0571300;
   paramQperp[2][0] = +1; paramQperp[2][1] = 4.09293e-5; paramQperp[2][2] = 0.0696507;
   paramQperp[3][0] = +1; paramQperp[3][1] = 0.000148576; paramQperp[3][2] = -0.0260885;
   paramQperp[4][0] = +1; paramQperp[4][1] = 0.000125231; paramQperp[4][2] = -0.0225048;
   paramQperp[5][0] = +1; paramQperp[5][1] = 5.75903e-6; paramQperp[5][2] = -0.0010002;
   paramQperp[6][0] = -1; paramQperp[6][1] = 9.01096e-5; paramQperp[6][2] = -0.014199;            
}


Double_t YMultiLayerPerceptron::GetCost_Sensor_LineFit() //const
{
   return 0;
}

Double_t YMultiLayerPerceptron::GetCost_Beam_LineFit(int track) // const
{
   return 0;
}

Double_t YMultiLayerPerceptron::GetCost_Vertex_LineFit()
{
   return 0;
}     

void YMultiLayerPerceptron::InitVertex_LineFit()
{
   
}

void YMultiLayerPerceptron::AddVertex_LineFit(double* z, double* beta)
{

}

Double_t YMultiLayerPerceptron::GetCost_Sensor_CircleFit() //const
{

   double Cost = 0;

   return Cost;
}

void YMultiLayerPerceptron::UpdateVertexByAlignment()
{
   fvertex_TRKF.SetXYZ(0,0,0);
      
   float norm_shift = 0.5;
   
   int NFirstLayer = fFirstLayer.GetEntriesFast();
   int NLastLayer = fLastLayer.GetEntriesFast();
#ifdef YMLPDEBUG
   std::cout<<" NFirstLayer : "<<NFirstLayer<<" NLastLayer : "<<NLastLayer<<std::endl;
#endif
   int NBeamLayer = 3; 
   int Nfitparam = 6; 
   
   double Cost = 0;
   double Cost_Fit = 0;   
   double Cost_Beam = 0;
      
   double Fitpar[nTrackMax][Nfitparam];      
        
   double inputS_C[nTrackMax][NLastLayer]; 
   double inputS_0[nTrackMax][NLastLayer];   
   double inputG_C[nTrackMax][NLastLayer + NBeamLayer];
   double inputG_0[nTrackMax][NLastLayer + NBeamLayer]; 

   double inputS[nTrackMax][nLAYER][2];   
   double outputS[nTrackMax][NLastLayer];
   double extended[nTrackMax][NLastLayer];
   double addition[nTrackMax][8];
   
   double input_Max[nTrackMax][NLastLayer];
   double input_Min[nTrackMax][NLastLayer];     

   int chipID[nTrackMax][nLAYER];

   double BeamPos[nTrackMax][3];
   double BeamMom[nTrackMax][3];

   std::vector<bool> hitUpdate[nTrackMax];

   for(int track = 0; track < nTrackMax; track++){
      for(int k = 0; k < 2*nLAYER; k++){   
         YNeuron *neuron_in = (YNeuron *) fFirstLayer.At(k);  // 6 input -> 2+1 ; 2+1 ; 2+1 build dummy
         neuron_in->SetNewEvent();   
         neuron_in->SetNeuronIndex(track);   
         int l1 = (int)(k/2);
         int l2 = (int)(k%2);  
         inputS[track][l1][l2] = neuron_in->GetValue();
      }
      
      for(int k = 0; k < NLastLayer; k++) {
         YNeuron *neuron_out = (YNeuron *) fLastLayer.At(k);
         neuron_out->SetNewEvent();               
         neuron_out->SetNeuronIndex(track);  
         extended[track][k] = neuron_out->GetBranch();
      }   

      for(int ln = 0; ln<nLAYER; ln++){             
         chipID[track][ln]     = (int)extended[track][(3*ln)+2];  
         bool layUpdate = chipID[track][ln] < 0 ? false : true;
         hitUpdate[track].push_back(layUpdate);
      }    

      for(int k = 0; k < 8; k++) {
         YNeuron *neuron_addition = (YNeuron *) fAddition.At(0);
         neuron_addition->SetNeuronIndex(1000 + k);  
         neuron_addition->SetNewEvent();         
         addition[track][k] = neuron_addition->GetBranchAddition();
      } 

      BeamPos[track][0] = addition[track][0];
      BeamPos[track][1] = addition[track][1];
      BeamPos[track][2] = addition[track][2];   
      BeamMom[track][0] = addition[track][3];
      BeamMom[track][1] = addition[track][4];
      BeamMom[track][2] = addition[track][5];        
      
      for(int k = 0; k<fNetwork.GetEntriesFast();k++) {
         YNeuron *neuron = (YNeuron *)fNetwork.At(k);
         neuron->SetNeuronIndex(track);      
         neuron->SetNewEvent();
      }   

      for(int ln = 0; ln < nLAYER; ln ++){   
         if(chipID[track][ln]<0) continue;   
         for(int k=0; k< fSCNetwork[chipID[track][ln]]->GetNetwork().GetEntriesFast(); k++){
            YNeuron *neuron = (YNeuron *) fSCNetwork[chipID[track][ln]]->GetNetwork().At(k);
            neuron->SetNeuronIndex(track);  
            neuron->SetNewEvent();            
         }     
         for(int k=0; k< fSCNetwork[chipID[track][ln]]->GetLastLayer().GetEntriesFast(); k++){
            YNeuron *neuron_out = (YNeuron *) fSCNetwork[chipID[track][ln]]->GetLastLayer().At(k);         
            outputS[track][3*ln + k] = Evaluate(k, inputS[track][ln], chipID[track][ln]); 
         }
      }
    
      for(int ln = 0; ln<nLAYER; ln++){    
         if(chipID[track][ln]<0) continue;     
         for(int iaxis = 0; iaxis<3; iaxis++){
            double ip, fp;   
            ip = yGEOM->GToS(chipID[track][ln],yGEOM->LToG(chipID[track][ln],0,0)(0),	
                                               yGEOM->LToG(chipID[track][ln],0,0)(1),
                                               yGEOM->LToG(chipID[track][ln],0,0)(2))(iaxis);
            fp = yGEOM->GToS(chipID[track][ln],yGEOM->LToG(chipID[track][ln],511,1023)(0),
                                               yGEOM->LToG(chipID[track][ln],511,1023)(1),
                                               yGEOM->LToG(chipID[track][ln],511,1023)(2))(iaxis);       
            int index = (3*ln)+iaxis;
            input_Max[track][index] = std::max(ip,fp);
            input_Min[track][index] = std::min(ip,fp); 

            if(iaxis==2){
               inputS_0[track][index]  = 0;
               inputS_C[track][index]  = outputS[track][index]; //d = exp - pos -> pos + d = exp
            } else {    
               inputS_0[track][index]  = (inputS[track][ln][iaxis] + norm_shift)*(input_Max[track][index]-input_Min[track][index])+input_Min[track][index];     			
               inputS_C[track][index]  = (inputS[track][ln][iaxis] + norm_shift)*(input_Max[track][index]-input_Min[track][index])+input_Min[track][index] + outputS[track][index];     
            }         
         } 
      }             

      for(int ln = 0; ln<nLAYER; ln++){   
         if(chipID[track][ln]<0) continue;         
         for(int iaxis = 0; iaxis<3; iaxis++){
            int index = (3*ln)+iaxis;      
            inputG_C[track][index]  = yGEOM->SToG(chipID[track][ln],inputS_C[track][(3*ln)+0],inputS_C[track][(3*ln)+1],inputS_C[track][(3*ln)+2])(iaxis); 
            inputG_0[track][index]  = yGEOM->SToG(chipID[track][ln],inputS_0[track][(3*ln)+0],inputS_0[track][(3*ln)+1],inputS_0[track][(3*ln)+2])(iaxis);  
         }
      }      
   }

   TVector3 TRKF_fXFit[nTrackMax][nLAYER];
   double TRKF_fdXY[nTrackMax][nLAYER];
   double TRKF_fdZR[nTrackMax][nLAYER];
   double TRKF_fparXY[nTrackMax][3]; // lvtx 0, 1 and R
   double TRKF_fparZR[nTrackMax][2]; // lvtx 0, 1
   double TRKF_mPAR[nTrackMax][5];
   for(int track = 0; track < nTrackMax; track++){
      //for(int ln = 0; ln<nLAYER; ln++){   
      //   std::cout<<"Layer(TRKFG) "<<ln<<" "<<chipID[track][ln]<<" "<<inputG_C[track][(3*ln)+0]<<" "<<inputG_C[track][(3*ln)+1]<<" "<<inputG_C[track][(3*ln)+2]<<std::endl;
      //}
      TrackerFit(inputG_C[track], hitUpdate[track], TRKF_fXFit[track], TRKF_fdXY[track], TRKF_fdZR[track], TRKF_fparXY[track], TRKF_fparZR[track], TRKF_mPAR[track]);
      /*
      std::cout<<"[TRACKER FIT RESULT] track"<<track<<std::endl;
      std::cout<<"fvertex : "<<BeamPos[track][0]<<" "<<BeamPos[track][1]<<" "<<BeamPos[track][2]<<std::endl;
      std::cout<<"fdXY "<<TRKF_fdXY[track][0]<<" "
                        <<TRKF_fdXY[track][1]<<" "
                        <<TRKF_fdXY[track][2]<<" "
                        <<TRKF_fdXY[track][3]<<" "
                        <<TRKF_fdXY[track][4]<<" "
                        <<TRKF_fdXY[track][5]<<" "
                        <<TRKF_fdXY[track][6]<<std::endl;
      std::cout<<"fdZR "<<TRKF_fdZR[track][0]<<" "
                        <<TRKF_fdZR[track][1]<<" "
                        <<TRKF_fdZR[track][2]<<" "
                        <<TRKF_fdZR[track][3]<<" "
                        <<TRKF_fdZR[track][4]<<" "
                        <<TRKF_fdZR[track][5]<<" "
                        <<TRKF_fdZR[track][6]<<std::endl;     
      std::cout<<"fparXY "<<TRKF_fparXY[track][0]<<" "<<TRKF_fparXY[track][1]<<" "<<TRKF_fparXY[track][2]<<std::endl;
      std::cout<<"fparZR "<<TRKF_fparZR[track][0]<<" "<<TRKF_fparZR[track][1]<<std::endl;
      std::cout<<"mpar "<<TRKF_mPAR[track][0]<<" "
                        <<TRKF_mPAR[track][1]<<" "
                        <<TRKF_mPAR[track][2]<<" "
                        <<TRKF_mPAR[track][3]<<" "
                        <<TRKF_mPAR[track][4]<<std::endl;
      */
   }

   //Step 1. Obtain ParP and ParQ
   std::vector<TVector3> parP;
   std::vector<TVector3> parQ;
   for(int track = 0; track < nTrackMax; track++){
      double x1, x2;
      double y1, y2;
      double z1, z2;
      
      //std::cout<<"[Vertex Estimation] track"<<track<<std::endl;
      
      double innerMostFitX = 0;
      double innerMostFitY = 0;
      for(int ln = 0; ln<nLAYER; ln++){   
         if(hitUpdate[track][ln]==true){
            innerMostFitX = TRKF_fXFit[track][ln].X();
            innerMostFitY = TRKF_fXFit[track][ln].Y();
            break;
         }
      }

      double clsx_near0 = 0;
      double clsy_near0 = 0;      
      double phi_near0  = std::atan2(clsy_near0-TRKF_fparXY[track][1], clsx_near0-TRKF_fparXY[track][0])>0 ? 
                      std::atan2(clsy_near0-TRKF_fparXY[track][1], clsx_near0-TRKF_fparXY[track][0]) : 
                      2*std::atan2(0,-1) + std::atan2(clsy_near0-TRKF_fparXY[track][1],clsx_near0-TRKF_fparXY[track][0]);
        
      double innerMostFitX_near0 = TMath::Abs(TRKF_fparXY[track][2])*TMath::Cos(phi_near0) + TRKF_fparXY[track][0];              
      double innerMostFitY_near0 = TMath::Abs(TRKF_fparXY[track][2])*TMath::Sin(phi_near0) + TRKF_fparXY[track][1]; 
      
      innerMostFitX += innerMostFitX_near0;
      innerMostFitY += innerMostFitY_near0;
      innerMostFitX /= 2.0;
      innerMostFitY /= 2.0;
      
      double parXYA = 1 + std::pow(TRKF_fparXY[track][1]/TRKF_fparXY[track][0],2);
      double parXYB = TRKF_fparXY[track][0] + TRKF_fparXY[track][1]*TRKF_fparXY[track][1]/TRKF_fparXY[track][0];
      double parXYC = std::pow(TRKF_fparXY[track][0],2) + std::pow(TRKF_fparXY[track][1],2) - std::pow(TRKF_fparXY[track][2],2);
      
      double candX1 = (parXYB + std::sqrt(parXYB*parXYB - parXYA*parXYC))/parXYA;
      double candY1 = TRKF_fparXY[track][1]/TRKF_fparXY[track][0]*candX1;
      double candX2 = (parXYB - std::sqrt(parXYB*parXYB - parXYA*parXYC))/parXYA;
      double candY2 = TRKF_fparXY[track][1]/TRKF_fparXY[track][0]*candX2;    
      
      double dist1 = std::sqrt(candX1*candX1 + candY1*candY1);
      double dist2 = std::sqrt(candX2*candX2 + candY2*candY2);
      
      double optX = ( dist1 < dist2 ) ? candX1 : candX2;
      double optY = ( dist1 < dist2 ) ? candY1 : candY2;
      
      //std::cout<<"parXY A B C : "<<parXYA<<" "<<parXYB<<" "<<parXYC<<" cand (X Y) [1] : "<<candX1<<" "<<candY1<<" [2] : "<<candX2<<" "<<candY2<<std::endl;
      //std::cout<<"dist [1] : "<<dist1<<" [2] : "<<dist2<<" >> opt (X Y) : "<<optX<<" "<<optY<<std::endl;
      
      // y = p1 * x + p0      
      x1 = innerMostFitX - 0.01*innerMostFitX/std::abs(innerMostFitX);
      y1 = - (innerMostFitX - TRKF_fparXY[track][0]) / (innerMostFitY - TRKF_fparXY[track][1]) * (x1 - innerMostFitX) + innerMostFitY;
      x2 = innerMostFitX + 0.01*innerMostFitX/std::abs(innerMostFitX);
      y2 = - (innerMostFitX - TRKF_fparXY[track][0]) / (innerMostFitY - TRKF_fparXY[track][1]) * (x2 - innerMostFitX) + innerMostFitY;

      // z = p1 * r + p0 
      // r = sqrt(x*x+y*y);
      
      z1 = TRKF_fparZR[track][1]*std::sqrt(x1*x1 + y1*y1) + TRKF_fparZR[track][0];
      z2 = TRKF_fparZR[track][1]*std::sqrt(x2*x2 + y2*y2) + TRKF_fparZR[track][0];
      
      //std::cout<<"X1(|r|=0) : "<<x1<<" "<<y1<<" "<<z1<<" X2(|r|=1) : "<<x2<<" "<<y2<<" "<<z2<<std::endl;      
      
      parP.push_back(TVector3(x1,y1,z1));
      parQ.push_back(TVector3(x2-x1,y2-y1,z2-z1));      
      fvertex_track_TRKF[track].SetXYZ(optX, optY, TRKF_fparZR[track][0]);
   }
    
   //Step 2. Obtain t'_j -> Vj and t'k -> Vk
    
   TVector3 V_mean(0,0,0);
   int cnt_Vpairs = 0;
   bool isSingleTrack = false;
   for(int j=0; j<nTrackMax; j++){
      for(int k=j+1; k<nTrackMax; k++){

         // TYPE 1
         TVector3 Pkj = parP[k] - parP[j];
         if((parQ[k].Cross(parQ[j])).Mag()<1e-6) {
            //std::cout<<"* V["<<j<<"]["<<k<<"] : Qj "<<parQ[j].X()<<" "<<parQ[j].Y()<<" "<<parQ[j].Z()<<" Qk "<<parQ[k].X()<<" "<<parQ[k].Y()<<" "<<parQ[k].Z()<<" >> j // k"<<std::endl;
            continue;
         }

         double Qjj = parQ[j].Dot(parQ[j]);
         double Qjk = parQ[j].Dot(parQ[k]);
         double Qkj = parQ[k].Dot(parQ[j]);               
         double Qkk = parQ[k].Dot(parQ[k]);               
         double cQ  = 1/(-(Qkj)*(Qjk) + (Qjj)*(Qkk));
        
         double tk = cQ*((Qjk)*(Pkj.Dot(parQ[j])) - (Qjj)*(Pkj.Dot(parQ[k])));
         double tj = cQ*((Qkk)*(Pkj.Dot(parQ[j])) - (Qkj)*(Pkj.Dot(parQ[k])));
    
         TVector3 Vj = parP[j] + tj*parQ[j];
         TVector3 Vk = parP[k] + tk*parQ[k];
          
         double wjk = std::abs(TRKF_fparXY[j][2]) + std::abs(TRKF_fparXY[k][2]);
         double wj  = std::abs(TRKF_fparXY[j][2]);
         double wk  = std::abs(TRKF_fparXY[k][2]);
                    
         //TVector3 Vjk_mean = (wj*Vj + wk*Vk)/wjk;
         TVector3 Vjk_mean((wj*Vj.X() + wk*Vk.X())/wjk,(wj*Vj.Y() + wk*Vk.Y())/wjk,(wj*Vj.Z() + wk*Vk.Z())/wjk);
         //std::cout<<"V["<<j<<"]["<<k<<"] : "<<Vj.X()<<" "<<Vj.Y()<<" "<<Vj.Z()<<" and "<<Vk.X()<<" "<<Vk.Y()<<" "<<Vk.Z()<<" >> "<<Vjk_mean.X()<<" "<<Vjk_mean.Y()<<" "<<Vjk_mean.Z()<<std::endl;
         /*
         // TYPE 2  
         TVector3 Vj = fvertex_track_TRKF[j];
         TVector3 Vk = fvertex_track_TRKF[k];

         double wjk = std::abs(TRKF_fparXY[j][2]) + std::abs(TRKF_fparXY[k][2]);
         double wj  = std::abs(TRKF_fparXY[j][2]);
         double wk  = std::abs(TRKF_fparXY[k][2]);

         TVector3 Vjk_mean((wj*Vj.X() + wk*Vk.X())/wjk,(wj*Vj.Y() + wk*Vk.Y())/wjk,(wj*Vj.Z() + wk*Vk.Z())/wjk);
         */
         V_mean.SetX(V_mean.X() + Vjk_mean.X());
         V_mean.SetY(V_mean.Y() + Vjk_mean.Y());
         V_mean.SetZ(V_mean.Z() + Vjk_mean.Z());                
         cnt_Vpairs++;
      }   
   }
    
   V_mean.SetX(V_mean.X()/cnt_Vpairs);
   V_mean.SetY(V_mean.Y()/cnt_Vpairs);
   V_mean.SetZ(V_mean.Z()/cnt_Vpairs); 

   fvertex_TRKF.SetXYZ(V_mean.X(), V_mean.Y(), V_mean.Z());
   //std::cout<<"updated vertex : "<<fvertex_TRKF.X()<<" "<<fvertex_TRKF.Y()<<" "<<fvertex_TRKF.Z()<<std::endl; 
   //for (int t=0; t<nTrackMax; t++ ) {
   //   std::cout<<"updated vertex(track["<<t<<"]) : "<<fvertex_track_TRKF[t].X()<<" "<<fvertex_track_TRKF[t].Y()<<" "<<fvertex_track_TRKF[t].Z()<<std::endl;       
   //}   
   //return 0;
}

bool YMultiLayerPerceptron::TrackerFit(double* input, std::vector<bool> hitUpdate, TVector3* fXFit, double* fdXY, double* fdZR, double* fparXY, double* fparZR, double* mPAR)
{

  //SetChi2XY(chi2XY); 		mPAR[0]
  //SetNdfXY(nlayerfit-3); 	mPAR[1]
  //SetChi2ZR(chi2ZR);    	mPAR[2]
  //SetNdfZR(nlayerfit-2); 	mPAR[3]
  //SetChi2Tot(chi2XY+chi2ZR);  mPAR[4]

  bool UseHit[7] = {false, false, false, false, false, false, false};
  
  int nlayerfit = 0;
  for(int a=0; a<hitUpdate.size(); a++){
     if(hitUpdate[a]==true) nlayerfit++;
  }  
  
  if(nlayerfit==0) {
    return false;
  }


  double hX[nLAYER];
  double hY[nLAYER];
  double hZ[nLAYER];
    
  double LX[nlayerfit];
  double LY[nlayerfit];
  double LZ[nlayerfit];
  double mR[nlayerfit];
  double LR[nlayerfit];
  double theta[nlayerfit];
  int layer=0;
  
  double LXIB(0),  LYIB(0);
  double LXOBa(0), LYOBa(0);
  double LXOBb(0), LYOBb(0);
  int 	 nIB(0), nOBa(0), nOBb(0);
  
  //i[fa]=input[(3*i)+0]
  //j[fa]=input[(3*i)+1]
  //k[fa]=input[(3*i)+2]
  
  vector<int> layerIB;
  for(int i=0;i<7;i++){
    if(hitUpdate[i]==true){
      hX[i]=input[(3*i)+0];
      hY[i]=input[(3*i)+1];
      hZ[i]=input[(3*i)+2];      
    
      mR[layer]=sqrt(input[(3*i)+0]*input[(3*i)+0] + input[(3*i)+1]*input[(3*i)+1]);    

      if(i==0||i==1||i==2) {
        LXIB += input[(3*i)+0];
        LYIB += input[(3*i)+1];
        nIB++;
      } else if(i==3||i==4) {
        LXOBa += input[(3*i)+0];
        LYOBa += input[(3*i)+1];
        nOBa++;  
      } else if(i==5||i==6) {
        LXOBb += input[(3*i)+0];
        LYOBb += input[(3*i)+1];
        nOBb++;    
      }      
      LX[layer]=input[(3*i)+0];
      LY[layer]=input[(3*i)+1];
      LZ[layer]=input[(3*i)+2];
      LR[layer]=sqrt(input[(3*i)+0]*input[(3*i)+0] + input[(3*i)+1]*input[(3*i)+1]);
      if(i==0||i==1||i==2) layerIB.push_back(i);
      layer++;
      UseHit[i]=true;

    } else {
      mR[layer]=0;     
    }
  }
  int nNotUselayer = nlayerfit - layer;
  nlayerfit = layer;
  
  if(layer<3) return false;

  if(nIB>0 && nOBa>0 && nOBb>0){
    LXIB/=nIB;		
    LYIB/=nIB;  
    LXOBa/=nOBa;	
    LYOBa/=nOBa;
    LXOBb/=nOBb;	
    LYOBb/=nOBb;     
  } else {
    LXIB  = LX[0];		
    LYIB  = LY[0];    
    LXOBa = LX[nlayerfit-2];     
    LYOBa = LY[nlayerfit-2];
    LXOBb = LX[nlayerfit-1];     
    LYOBb = LY[nlayerfit-1];    
  }

  double thetaR = std::atan2(LY[0],LX[0]);
  //frotXY=thetaR;
  
  double LXR[nlayerfit];
  double LYR[nlayerfit];  
  TMatrixD RotR(2,2);
  RotR[0] = { TMath::Cos(thetaR),	TMath::Sin(thetaR)};
  RotR[1] = {-TMath::Sin(thetaR),	TMath::Cos(thetaR)};  
  TMatrixD Xp[layer];
  TMatrixD XpR[layer];  
  for(int l=0; l<layer;l++){
    Xp[l].ResizeTo(1,2);
    Xp[l][0] = { LX[l], LY[l]};
    Xp[l].T();
    XpR[l].ResizeTo(2,1);
    XpR[l] = RotR * Xp[l];
    XpR[l].T();
    LXR[l] = XpR[l][0][0];
    LYR[l] = XpR[l][0][1];    
  } 
  
  double d10=(LYOBa-LYIB) /(LXOBa-LXIB);
  double d21=(LYOBb-LYOBa)/(LXOBb-LXOBa);  
  double n10=-1/d10;
  double n21=-1/d21;
  
  double MXa = 0.5*(LXIB + LXOBa);
  double MYa = 0.5*(LYIB + LYOBa); 
  double MXb = 0.5*(LXOBa + LXOBb);
  double MYb = 0.5*(LYOBa + LYOBb);
   
  double initXc = ((n10*MXa + MYa) - (n21*MXb + MYb))/(n10 - n21);
  double initYc = n10*(initXc - MXa) + MYa;
  double initRc = std::sqrt((LXIB-initXc)*(LXIB-initXc) + (LYIB-initYc)*(LYIB-initYc));
  TGraph grXYR(nlayerfit,LXR,LYR);
  TGraph grXY(nlayerfit,LX,LY);
  TGraph grZR(nlayerfit,LZ,LR);
  TGraph grRZ(nlayerfit,LR,LZ);
  
  double slopeXY=0;
  double maxX=0;
  double minX=0;
  double maxXR=0;
  double minXR=0;
  double slopeZR=0;  
  double maxZ=0;
  double minZ=0;
  double maxR=0;
  double minR=0;
  double chi2XY =0;
  double chi2ZR =0;
  
  if(LX[nlayerfit-1]-LX[0]>0){
    maxX=LX[nlayerfit-1]+0.6*(LX[nlayerfit-1]-LX[0]);
    minX=LX[0]-0.6*(LX[nlayerfit-1]-LX[0]);
  } else {
    minX=LX[nlayerfit-1]+0.6*(LX[nlayerfit-1]-LX[0]);
    maxX=LX[0]-0.6*(LX[nlayerfit-1]-LX[0]);
  }

  if(LXR[nlayerfit-1]-LXR[0]>0){
    maxX=LXR[nlayerfit-1]+0.6*(LXR[nlayerfit-1]-LXR[0]);
    minX=LXR[0]-0.6*(LXR[nlayerfit-1]-LXR[0]);
  } else {
    minX=LXR[nlayerfit-1]+0.6*(LXR[nlayerfit-1]-LXR[0]);
    maxX=LXR[0]-0.6*(LXR[nlayerfit-1]-LXR[0]);
  }

  if(LZ[nlayerfit-1]-LZ[0]>0){
    maxZ=LZ[nlayerfit-1]+0.6*(LZ[nlayerfit-1]-LZ[0]);
    minZ=LZ[0]-0.6*(LZ[nlayerfit-1]-LZ[0]);
  }else{
    minZ=LZ[nlayerfit-1]+0.6*(LZ[nlayerfit-1]-LZ[0]);
    maxZ=LZ[0]-0.6*(LZ[nlayerfit-1]-LZ[0]);
  }
  
  if(LR[nlayerfit-1]-LR[0]>0){
    maxR=LR[nlayerfit-1]+0.6*(LR[nlayerfit-1]-LR[0]);
    minR=LR[0]-0.6*(LR[nlayerfit-1]-LR[0]);
  }else{
    minR=LR[nlayerfit-1]+0.6*(LR[nlayerfit-1]-LR[0]);
    maxR=LR[0]-0.6*(LR[nlayerfit-1]-LR[0]);
  }

  double pXY0 =0;
  double pXY1 =0;
  bool  doLineFit = false; 
  if(doLineFit==true){
    //int sign = LX[nlayerfit-1]-LX[0] > 0 ? +1 : -1;
    //slopeXY=sign*100;
    //pXY0 = -slopeXY*(0.5*(LX[0]+LX[nlayerfit-1])) + 0.5*(LY[0]+LY[nlayerfit-1]);
    //pXY1 = slopeXY; 
    //F_LINEAR(fXY, minX, maxX, pXY0, pXY1);  
    ////grXY.Fit(fXY,"rQ");    
    //chi2XY=0;
  } else {
    slopeXY=(LY[nlayerfit-1]-LY[0])/(LX[nlayerfit-1]-LX[0]);
    pXY0 = -slopeXY*(0.5*(LX[0]+LX[nlayerfit-1])) + 0.5*(LY[0]+LY[nlayerfit-1]);
    pXY1 = slopeXY;  
    
    // Two Step : [1] R estimation by pol2
    
    double LYE = 0.5*(LXR[0]+LXR[nlayerfit-1]);
    double LYM = LXR[1];
    int Approx_sign = (LYE-LYM)>0 ? +1 : -1;
    
    double pXYPpar2 = -(1/(2*initRc*Approx_sign));        
        
    F_POL2(fXY, minXR, maxXR, 0, 0, 0);
    grXYR.Fit(fXY,"rQ");         
    chi2XY=fXY->GetChisquare();

    for(int i=0;i<7;i++){
      if(hitUpdate[i]==true && std::abs(hX[i]) < 500 && std::abs(hY[i]) < 500){
        TMatrixD Xt, Xtr;
        Xt.ResizeTo(1,2);
        Xt[0] = { hX[i], hY[i]};
        Xt.T();
        Xtr.ResizeTo(2,1);
        Xtr = RotR * Xt;
        Xtr.T();
        double rtX = Xtr[0][0];
        double rtY = Xtr[0][1];    
        double fitY = fXY->GetParameter(0) + rtX*fXY->GetParameter(1) + rtX*rtX*fXY->GetParameter(2);
        fdXY[i] = fitY - rtY;
      } else {
        fdXY[i] = -9999;
      }
    }    
    double pcr = -(1/(2*fXY->GetParameter(2)));
    double pcx = fXY->GetParameter(1)*pcr;
    double pcy = fXY->GetParameter(0) - pcr + (pcx*pcx)/(2*pcr);

    double fcr = std::abs(pcr);

    RotR[0] = { TMath::Cos(-thetaR),	TMath::Sin(-thetaR)};
    RotR[1] = {-TMath::Sin(-thetaR),	TMath::Cos(-thetaR)};  
    TMatrixD Xpc[2];
  
    Xpc[0].ResizeTo(1,2);
    Xpc[0][0] = { pcx, pcy};
    Xpc[0].T();
    Xpc[1].ResizeTo(2,1);
    Xpc[1] = RotR * Xpc[0];
    Xpc[1].T();
    double fcx = Xpc[1][0][0];
    double fcy = Xpc[1][0][1];    

    //std::clog<<"FitMonitor : "<<initXc<<" "<<initYc<<" "<<initRc<<" >> "<<fcx<<" "<<fcy<<" "<<fcr<<std::endl;
    
    //Drawing arc range setting
    double phiI = std::atan2(0-fcy,0-fcx)>0 ? 
                     std::atan2(0-fcy,0-fcx) : 
                     2*std::atan2(0,-1) + std::atan2(0-fcy,0-fcx);
    double phiF = std::atan2(LY[nlayerfit-1]-fcy,LX[nlayerfit-1]-fcx)>0 ? 
                     std::atan2(LY[nlayerfit-1]-fcy,LX[nlayerfit-1]-fcx) : 
                     2*std::atan2(0,-1) + std::atan2(LY[nlayerfit-1]-fcy,LX[nlayerfit-1]-fcx);
    
    double cfA = 2*fcx;
    double cfB = 2*fcy;
    double cfC = fcr*fcr - 45*45 - fcx*fcx - fcy*fcy;
     
    double cfa = 1 + (cfA/cfB)*(cfA/cfB);
    double cfb = 2*cfA*cfC/(cfB*cfB);
    double cfc = (cfC/cfB)*(cfC/cfB) - 45*45;
         
    double fx1 = (-cfb + std::sqrt(cfb*cfb - 4*cfa*cfc))/(2*cfa);
    double fx2 = (-cfb - std::sqrt(cfb*cfb - 4*cfa*cfc))/(2*cfa);
    double fy1 = (-cfA/cfB)*fx1 + (-cfC/cfB);
    double fy2 = (-cfA/cfB)*fx2 + (-cfC/cfB);
    
    double phiF1 = std::atan2(fy1-fcy,fx1-fcx)>0 ? std::atan2(fy1-fcy,fx1-fcx) : 2*std::atan2(0,-1) + std::atan2(fy1-fcy,fx1-fcx);  
    double phiF2 = std::atan2(fy2-fcy,fx2-fcx)>0 ? std::atan2(fy2-fcy,fx2-fcx) : 2*std::atan2(0,-1) + std::atan2(fy2-fcy,fx2-fcx);   
    double dphiF1 = std::abs(phiF1 - phiF) < std::atan2(0,-1) ? std::abs(phiF1 - phiF) : 2*std::atan2(0,-1) - std::abs(phiF1 - phiF);
    double dphiF2 = std::abs(phiF2 - phiF) < std::atan2(0,-1) ? std::abs(phiF2 - phiF) : 2*std::atan2(0,-1) - std::abs(phiF2 - phiF);
    double phiF_Opt = dphiF1 < dphiF2 ? phiF1 : phiF2;

    if(std::abs(phiI-phiF_Opt)>std::atan2(0,-1)) {
      if(phiI>phiF_Opt) phiI = phiI - 2*std::atan2(0,-1);
      else phiF_Opt = phiF_Opt - 2*std::atan2(0,-1);
    
    }

    fparXY[0] = fcx;
    fparXY[1] = fcy;
    fparXY[2] = pcr;
    //fsign = phiI - phiF_Opt > 0 ? +1 : -1;

    for(int i=0;i<7;i++){

      if(hitUpdate[i]==true && std::abs(hX[i]) < 500 && std::abs(hY[i]) < 500){
        double clsx = hX[i];
        double clsy = hY[i];
        double phi  = std::atan2(clsy-fcy,clsx-fcx)>0 ? 
                      std::atan2(clsy-fcy,clsx-fcx) : 
                      2*std::atan2(0,-1) + std::atan2(clsy-fcy,clsx-fcx);
        
        double fitx = fcr*TMath::Cos(phi) + fcx;              
        double fity = fcr*TMath::Sin(phi) + fcy;   
        fXFit[i].SetX(fitx);     
        fXFit[i].SetY(fity);        
      } else {
        fXFit[i].SetX(-9999);     
        fXFit[i].SetY(-9999); 
      }
    } 
  }

  double pZR0 =0;
  double pZR1 =0;

  slopeZR=(LZ[nlayerfit-1]-LZ[0])/(LR[nlayerfit-1]-LR[0]);
  pZR0 = -slopeZR*(0.5*(LR[0]+LR[nlayerfit-1])) + 0.5*(LZ[0]+LZ[nlayerfit-1]);
  pZR1 = slopeZR; 
  F_LINEAR(fZR, minR, maxR, pZR0, pZR1);    
  grRZ.Fit(fZR,"rQ");     
  chi2ZR=fZR->GetChisquare(); 
  double rp0 = fZR->GetParameter(0);
  double rp1 = fZR->GetParameter(1);

  double p0 = -fZR->GetParameter(0)/fZR->GetParameter(1);
  double p1 = 1/fZR->GetParameter(1);
  double atan_theta = TMath::ATan(p1);
  double atan2_theta = (atan_theta>0) ? atan_theta : TMath::Pi()+atan_theta; 
  //SetTheta(atan2_theta);
    
  //fZR->SetParameter(0,-rp0/rp1); 	
  //fZR->SetParameter(1,1/rp1);	
  //SetV(TVector3(0,0,-(-rp0/rp1)/(1/rp1)));  

  fparZR[0] = fZR->GetParameter(0);
  fparZR[1] = fZR->GetParameter(1);
  
  for(int i=0;i<7;i++){
    if(hitUpdate[i]==true && std::abs(hX[i]) < 500 && std::abs(hY[i]) < 500){
      double tZ = hZ[i];
      double tR = sqrt(hX[i]*hX[i] + hY[i]*hY[i]); 
      double fitZ = fZR->GetParameter(0) + tR*fZR->GetParameter(1);
      fdZR[i] = fitZ - tZ;
      fXFit[i].SetZ(fitZ);
    } else {
      fXFit[i].SetZ(-9999);    
      fdZR[i] = -9999;
    }
  } 

/*
  for(int i=0;i<7;i++){
    std::cout<<"Layer "<<i<<" ";
    if(UseHit[i]==true){
      std::cout<<" fdXY = "<<fdXY[i]
               <<" fdZR = "<<fdZR[i]<<std::endl;        
    } else {
      std::cout<<" Not Used "<<std::endl;
    }
  }
*/
  mPAR[0] = chi2XY;
  mPAR[1] = nlayerfit-3; 	
  mPAR[2] = chi2ZR;    	
  mPAR[3] = nlayerfit-2; 	
  mPAR[4] = chi2XY+chi2ZR;  
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


Double_t YMultiLayerPerceptron::GetCost_Beam_CircleFit(int track) //const
{

#ifdef YMLPDEBUG0
   std::cout<<"GetCost_Beam_CircleFit "<<track<<std::endl;
#endif

   float norm_shift = 0.5;
   
   int NFirstLayer = fFirstLayer.GetEntriesFast();
   int NLastLayer = fLastLayer.GetEntriesFast();
#ifdef YMLPDEBUG
   std::cout<<" NFirstLayer : "<<NFirstLayer<<" NLastLayer : "<<NLastLayer<<std::endl;
#endif
   int NBeamLayer = 3; 
   int Nfitparam = 6; 
   
   double Cost = 0;
   double Cost_Fit = 0;   
   double Cost_Beam = 0;
      
   double Fitpar[Nfitparam];      
        
   double inputS_C[NLastLayer]; 
   double inputS_0[NLastLayer];   
   double inputG_C[NLastLayer + NBeamLayer];
   double inputG_0[NLastLayer + NBeamLayer]; 

   double inputS[nLAYER][2];   
   double outputS[NLastLayer];
   double extended[NLastLayer];
   double addition[8];
   
   TVector3 vecXc_meas[nLAYER+1];
   TVector3 vecXc_proj[nLAYER+1];    
   TVector3 vecXc_norm[nLAYER+1];
   
   double input_Max[NLastLayer];
   double input_Min[NLastLayer];     

   int staveIndex[nLAYER], chipIndex[nLAYER], chipID[nLAYER];
   TVector3 vecSensorNorm[nLAYER];

   double BeamPos[3];
   double BeamMom[3];

   for(int k = 0; k < 2*nLAYER; k++){   
      YNeuron *neuron_in = (YNeuron *) fFirstLayer.At(k);  // 6 input -> 2+1 ; 2+1 ; 2+1 build dummy
      neuron_in->SetNewEvent();   
      neuron_in->SetNeuronIndex(track);   
      int l1 = (int)(k/2);
      int l2 = (int)(k%2);  
      inputS[l1][l2] = neuron_in->GetValue();
#ifdef YMLPDEBUG
      std::cout<<"track "<<track<<" "<<"inputS["<<l1<<"]["<<l2<<"] "<<inputS[l1][l2]<<" "<<std::endl;
#endif
   }

   for(int k = 0; k < NLastLayer; k++) {
      YNeuron *neuron_out = (YNeuron *) fLastLayer.At(k);
      neuron_out->SetNewEvent();               
      neuron_out->SetNeuronIndex(track);  
      extended[k] = neuron_out->GetBranch();
   }   

   bool IsAccessorial = (fSplitReferenceSensor==-1) ? false : true;
   std::vector<bool> hitUpdate;
   std::vector<bool> hitUpdate_Z;          
   int nVALIDLAYERS = 0;
#ifdef MONITORONLYUPDATES
   short IsCorrection = 0;
#endif   
   bool untrained = false;
   for(int ln = 0; ln<nLAYER; ln++){             
      staveIndex[ln] = (int)extended[(3*ln)+0];   
      chipIndex[ln]  = (int)extended[(3*ln)+1];  
      chipID[ln]     = (int)extended[(3*ln)+2];  
      bool layUpdate = chipID[ln] < 0 ? false : true;
      if(chipID[ln] >= 0) fSCNetwork[chipID[ln]]->hNtracksByRejection->Fill(0);
      if(chipID[ln] >= 0 && fSplitReferenceSensor == chipID[ln]) IsAccessorial=false;
      hitUpdate.push_back(layUpdate);
      hitUpdate_Z.push_back(layUpdate);
      if(chipID[ln] >= 0) nVALIDLAYERS++;      
      //vecSensorNorm[ln] = yGEOM->NormalVector(chipID[ln]);
      vecSensorNorm[ln] = GetCorrectedNormalVector(chipID[ln]);            
#ifdef MONITORONLYUPDATES     
      IsCorrection <<= 1;
      if(chipID[ln] >= 0) {
         short us_value =  fUPDATESENSORS->GetBinContent(1+chipID[ln]) >= 1 ? 1 : 0;
         //if(fUPDATESENSORS->GetBinContent(1+chipID[ln])==2) untrained = true;
         IsCorrection += us_value;
      } else IsCorrection += 1; // no hit case, count (virtually) for use
#endif
   }    
#ifdef MONITORONLYUPDATES 
   fUPDATETRACKS->Fill(IsCorrection);
#endif    
   hitUpdate.push_back(true);  
   hitUpdate_Z.push_back(true);  
   if(IsAccessorial==true) {
      AddTrackLoss();   
      return 0;
   }
   for(int ln = 0; ln<nLAYER; ln++){             
      if(chipID[ln] >= 0) fSCNetwork[chipID[ln]]->hNtracksByRejection->Fill(1);
   }
   for(int k = 0; k < 8; k++) {
      YNeuron *neuron_addition = (YNeuron *) fAddition.At(0);
      neuron_addition->SetNeuronIndex(1000 + k);  
      neuron_addition->SetNewEvent();         
      addition[k] = neuron_addition->GetBranchAddition();
   } 

   for(int k = 0; k<fNetwork.GetEntriesFast();k++) {
      YNeuron *neuron = (YNeuron *)fNetwork.At(k);
      neuron->SetNeuronIndex(track);      
      neuron->SetNewEvent();
   }   

   BeamPos[0] = addition[0];
   BeamPos[1] = addition[1];
   BeamPos[2] = addition[2];   
   BeamMom[0] = addition[3];
   BeamMom[1] = addition[4];
   BeamMom[2] = addition[5];        
      
   for(int ln = 0; ln < nLAYER; ln ++){   
      if(chipID[ln]<0) continue;   
      for(int k=0; k< fSCNetwork[chipID[ln]]->GetNetwork().GetEntriesFast(); k++){
         YNeuron *neuron = (YNeuron *) fSCNetwork[chipID[ln]]->GetNetwork().At(k);
         neuron->SetNeuronIndex(track);  
         neuron->SetNewEvent();            
      }     
      for(int k=0; k< fSCNetwork[chipID[ln]]->GetLastLayer().GetEntriesFast(); k++){
         YNeuron *neuron_out = (YNeuron *) fSCNetwork[chipID[ln]]->GetLastLayer().At(k);         
         outputS[3*ln + k] = Evaluate(k, inputS[ln], chipID[ln]); 
      }
   }
    
   for(int ln = 0; ln<nLAYER; ln++){    
      if(chipID[ln]<0) continue;     
      for(int iaxis = 0; iaxis<3; iaxis++){
         double ip, fp;   
         ip = yGEOM->GToS(chipID[ln],yGEOM->LToG(chipID[ln],0,0)(0),	
                                     yGEOM->LToG(chipID[ln],0,0)(1),
                                     yGEOM->LToG(chipID[ln],0,0)(2))(iaxis);
         fp = yGEOM->GToS(chipID[ln],yGEOM->LToG(chipID[ln],511,1023)(0),
                                     yGEOM->LToG(chipID[ln],511,1023)(1),
                                     yGEOM->LToG(chipID[ln],511,1023)(2))(iaxis);       
         int index = (3*ln)+iaxis;
         input_Max[index] = std::max(ip,fp);
         input_Min[index] = std::min(ip,fp); 

         if(iaxis==2){
            inputS_0[index]  = 0;
            inputS_C[index]  = outputS[index]; 													//d = exp - pos -> pos + d = exp
         } else {    
            inputS_0[index]  = (inputS[ln][iaxis] + norm_shift)*(input_Max[index]-input_Min[index])+input_Min[index];     			
            inputS_C[index]  = (inputS[ln][iaxis] + norm_shift)*(input_Max[index]-input_Min[index])+input_Min[index] + outputS[index];     	//d = exp - pos -> pos + d = exp 
         }         
      } 
#ifdef YMLPDEBUG0
      std::cout<<"Layer(MLP S) "<<ln<<" "<<chipID[ln]<<" "<<inputS_C[(3*ln)+0]<<" "<<inputS_C[(3*ln)+1]<<" "<<inputS_C[(3*ln)+2]<<std::endl;
#endif
   }             

   for(int ln = 0; ln<nLAYER; ln++){   
      if(chipID[ln]<0) continue;         
      for(int iaxis = 0; iaxis<3; iaxis++){
         int index = (3*ln)+iaxis;      
         inputG_C[index]  =yGEOM->SToG(chipID[ln],inputS_C[(3*ln)+0],inputS_C[(3*ln)+1],inputS_C[(3*ln)+2])(iaxis); 
         inputG_0[index]  =yGEOM->SToG(chipID[ln],inputS_0[(3*ln)+0],inputS_0[(3*ln)+1],inputS_0[(3*ln)+2])(iaxis);  
      }
      vecXc_meas[ln].SetXYZ(inputG_C[(3*ln)+0], inputG_C[(3*ln)+1], inputG_C[(3*ln)+2]);      
      double row = yGEOM->SToL(chipID[ln],inputS_C[(3*ln)+0],inputS_C[(3*ln)+1],inputS_C[(3*ln)+2])(0); 
      double col = yGEOM->SToL(chipID[ln],inputS_C[(3*ln)+0],inputS_C[(3*ln)+1],inputS_C[(3*ln)+2])(1); 
#ifdef YMLPDEBUG0
      std::cout<<"Layer(MLP G) "<<ln<<" "<<chipID[ln]<<" "<<inputG_C[(3*ln)+0]<<" "<<inputG_C[(3*ln)+1]<<" "<<inputG_C[(3*ln)+2]<<" row col : "<<row<<" "<<col<<std::endl;
#endif
   }
   vector<TVector3> vecBeam[2]; // 0 : track, 1 : beam

   //TVector3 vertexZ(0.0002*fvertex.Z() - 0.045, -0.0004*fvertex.Z() - 0.0721, fvertex.Z());   
   //TVector3 BeamCenter(-7.29228e-06*BeamPos[2] - 0.0533001, 1.58528e-05*BeamPos[2] - 0.082321, BeamPos[2]);   //BeamPos[0],BeamPos[1],BeamPos[2]   505645  
   //TVector3 BeamCenter(BeamProfileXZ1*BeamPos[2] + BeamProfileXZ0, BeamProfileYZ1*BeamPos[2] + BeamProfileYZ0, BeamPos[2]);   //BeamPos[0],BeamPos[1],BeamPos[2]
   TVector3 BeamCenter(BeamPos[0], BeamPos[1], BeamPos[2]);   //BeamPos[0],BeamPos[1],BeamPos[2]
   
   //bool UseUpdatedVertexByAlignment = true;
   //if(UseUpdatedVertexByAlignment==true) BeamCenter = fvertex_TRKF;

   //TVector3 BeamCenter(0.0,0.0,BeamPos[2]);   //BeamPos[0],BeamPos[1],BeamPos[2]
#ifdef YMLPDEBUG0
   std::cout<<"YMLP::Beam(C) track :: "<<BeamCenter(0)<<" "<<BeamCenter(1)<<" "<<BeamCenter(2)<<std::endl;    
#endif
   
   inputG_C[(3*nLAYER)+0] = BeamCenter(0); 
   inputG_C[(3*nLAYER)+1] = BeamCenter(1);
   inputG_C[(3*nLAYER)+2] = BeamCenter(2);  
   inputG_0[(3*nLAYER)+0] = BeamCenter(0); 
   inputG_0[(3*nLAYER)+1] = BeamCenter(1);
   inputG_0[(3*nLAYER)+2] = BeamCenter(2);    
 
   vecXc_meas[nLAYER].SetXYZ(inputG_C[(3*nLAYER)+0], inputG_C[(3*nLAYER)+1], inputG_C[(3*nLAYER)+2]);     
                          
   for(int k = 0; k < Nfitparam; k++){
      Fitpar[k]=0.0;
   }
   Cost_Fit = 0;    

   circle3Dfit(inputG_C, Fitpar, Cost_Fit, hitUpdate, 0);

   double min_Cost_Fit_Scale = ((int)Fitpar[5])%10==0 ? 1e+10 : Cost_Fit;
   double Cost_FitD(0), FitparD[6];
   int search_strategy[] = {-2, +2, +4};
   if(((int)Fitpar[5])%10>=0 || Cost_Fit>1.0e-4) {
      for(int isch = 0; isch < 3; isch++){
         for(int j = 0; j < 6; j++){
            FitparD[j]=0.0;
         }
   
         circle3Dfit(inputG_C, FitparD, Cost_FitD, hitUpdate, search_strategy[isch]);

         if(min_Cost_Fit_Scale>Cost_FitD && ((int)FitparD[5])%10==1){
            min_Cost_Fit_Scale = Cost_FitD;
            Cost_Fit = Cost_FitD;
            for(int j = 0; j < 6; j++){
               Fitpar[j]=FitparD[j];
            }
         }
      }
   }

   if(((int)Fitpar[5])%10==0){
      //((YNeuron*)this)->fMSE = 0;
      return 0;
   } 
   // Found vertex cut : 20240314
   if(std::sqrt((BeamCenter(0) + Fitpar[2])*(BeamCenter(0) + Fitpar[2]) + (BeamCenter(1) + Fitpar[3])*(BeamCenter(1) + Fitpar[3]))>0.1) return 0;

   for(int ln = 0; ln<nLAYER; ln++){             
      if(chipID[ln] >= 0) fSCNetwork[chipID[ln]]->hNtracksByRejection->Fill(2);
   }
      
   double RecRadius = std::abs(1/Fitpar[0]);
   double CircleXc  = Fitpar[0]>0 ? RecRadius*std::cos(Fitpar[1]+Fitpar[4] + 0.5*TMath::Pi()) : RecRadius*std::cos(Fitpar[1]+Fitpar[4] - 0.5*TMath::Pi());
   double CircleYc  = Fitpar[0]>0 ? RecRadius*std::sin(Fitpar[1]+Fitpar[4] + 0.5*TMath::Pi()) : RecRadius*std::sin(Fitpar[1]+Fitpar[4] - 0.5*TMath::Pi()); 
   
   //std::cout<<" Cost(XY) "<<Cost_Fit<<" "<<RecRadius<<" "<<CircleXc<<" "<<CircleYc<<" "<<Fitpar[2]<<" "<<Fitpar[3];
   CircleXc = CircleXc + BeamCenter(0) + Fitpar[2];
   CircleYc = CircleYc + BeamCenter(1) + Fitpar[3];

   double RecMomentumT = 0.3*DET_MAG*RecRadius*1.0e-2;  //r[m] -> r[cm]
   //std::cout<<" pT(GeV) "<<RecMomentumT;
   //std::cout<<" Chip ";
   for(int ln = 0; ln<nLAYER; ln++){             
      //std::cout<<chipID[ln]<<" ";
   }   
   if(RecMomentumT<Update_pTmin || RecMomentumT>Update_pTmax) {
      AddTrackLoss();
      //std::cout<<std::endl;
      return 0;
   }
   for(int ln = 0; ln<nLAYER; ln++){             
      if(chipID[ln] >= 0) fSCNetwork[chipID[ln]]->hNtracksByRejection->Fill(3);
   }
#ifdef YMLPDEBUG
   std::cout<<"Radius Xc Yc "<<RecRadius<<" "<<CircleXc<<" "<<CircleYc<<std::endl;     
#endif     

   TVector3 vecCircle_center(CircleXc, CircleYc, 0);
   TVector3 dirXc_meas[nLAYER+1];
   TVector3 dirXc_proj[nLAYER+1];
   for(int a=0; a<nLAYER+1;a++){
      dirXc_meas[a] = vecXc_meas[a] - vecCircle_center;
   }

   double beta[nLAYER+1];
   for(int l = 0; l < nLAYER+1; l++){    
      beta[l] = std::atan2(dirXc_meas[l].Y(), dirXc_meas[l].X());// > 0 ? std::atan2(dirXc_meas[l].Y(), dirXc_meas[l].X()) : 2*std::atan2(0,-1) + std::atan2(dirXc_meas[l].Y(), dirXc_meas[l].X());
   }     

   //beta linearization
   BetaLinearization(beta, dirXc_meas, hitUpdate);
   GetProjectionPoints(vecCircle_center, RecRadius, vecSensorNorm, vecXc_meas, vecXc_proj, vecXc_norm);

#ifdef YMLPDEBUG
   std::cout<<"YMultiLayerPerceptron Meas Proj(FIT)"<<std::endl;
   for(int l = 0; l < nLAYER+1; l++){   
      std::cout<<"Layer "<<l<<" X "<<vecXc_meas[l].X()<<" "<<vecXc_proj[l].X()<<std::endl;
      std::cout<<"Layer "<<l<<" Y "<<vecXc_meas[l].Y()<<" "<<vecXc_proj[l].Y()<<std::endl;
      std::cout<<"Layer "<<l<<" Z "<<vecXc_meas[l].Z()<<" "<<vecXc_proj[l].Z()<<std::endl;
   } 
#endif
   for(int a=0; a<nLAYER+1;a++){
      dirXc_proj[a] = vecXc_proj[a] - vecCircle_center;
   }
   
   double pbeta_proj[nLAYER+1]; 

   for(int l = 0; l < nLAYER+1; l++){    
      pbeta_proj[l] = std::atan2(dirXc_proj[l].Y(), dirXc_proj[l].X());// > 0 ? std::atan2(dirXc_proj[l].Y(), dirXc_proj[l].X()) : 2*std::atan2(0,-1) + std::atan2(dirXc_proj[l].Y(), dirXc_proj[l].X());
   }    

   //beta linearization
   BetaLinearization(pbeta_proj, dirXc_proj, hitUpdate);
   for(int l = 0; l < nLAYER+1; l++){    
       beta[l]     = pbeta_proj[l]; 
   } 

   double parz[2];
   double z_meas[nLAYER+1];
#ifdef YMLPDEBUG0
   std::cout<<"YMultiLayerPerceptron zmeas ";
#endif
   for(int a=0; a<nLAYER+1; a++){
      z_meas[a] = vecXc_meas[a].Z();
#ifdef YMLPDEBUG0
      std::cout<<z_meas[a]<<" ";
#endif
   }
#ifdef YMLPDEBUG0
   std::cout<<std::endl;  
#endif  
   circle3Dfit_Z(z_meas, beta, parz, RecRadius, VERTEXFIT, hitUpdate_Z); //Not Including Vtx
   AddVertex_CircleFit(track, z_meas, beta, parz, RecRadius, true);
#ifdef YMLPDEBUG0
   std::cout<<"YMultiLayerPerceptron parz "<<parz[0]<<" "<<parz[1]<<std::endl;                  
#endif      
   Cost_Beam=0;
   double b_Cost_Beam[nLAYER+1];

   double meas_GXc[nLAYER+1], meas_GYc[nLAYER+1], meas_GZc[nLAYER+1];
   double meas_S1c[nLAYER+1], meas_S2c[nLAYER+1], meas_S3c[nLAYER+1];
          
   double proj_GXc[nLAYER+1], proj_GYc[nLAYER+1], proj_GZc[nLAYER+1];
   double proj_S1c[nLAYER+1], proj_S2c[nLAYER+1], proj_S3c[nLAYER+1];

   int nValidHits = 0;
   int nBadFits   = 0;
   
   double Residual_s1[nLAYER+1], Residual_s2[nLAYER+1];
   double Chi_s1[nLAYER+1], Chi_s2[nLAYER+1];
   double Chi2TOT[nLAYER+1];
   //std::cout<<"Chi Fit Meas ";
   for(int ln = 0; ln < nLAYER+1; ln++){   
      if(hitUpdate[ln]==false) continue;
#ifdef YMLPDEBUG
      std::cout<<" YMultiLayerPerceptron :: layer ["<<ln<<"] beta = "<<beta[ln]<<std::endl;
      std::cout<<" YMultiLayerPerceptron :: layer ["<<ln<<"] pbeta_proj = "<<pbeta_proj[ln]<<std::endl;      
      std::cout<<" YMultiLayerPerceptron :: layer ["<<ln<<"] dirXc_meas "<<dirXc_meas[ln](0)<<" "<<dirXc_meas[ln](1)<<" "<<dirXc_meas[ln](2)<<std::endl;     
#endif      
      meas_GXc[ln] = inputG_C[(3*ln)+0]; //alpha
      meas_GYc[ln] = inputG_C[(3*ln)+1]; //beta
      meas_GZc[ln] = inputG_C[(3*ln)+2]; //gamma                                  
      proj_GXc[ln] = RecRadius*std::cos(beta[ln]) + CircleXc;
      proj_GYc[ln] = RecRadius*std::sin(beta[ln]) + CircleYc;
      proj_GZc[ln] = (parz[0])*(beta[ln]) + (parz[1]); 
#ifdef YMLPDEBUG
      std::cout<<setprecision(9)<<" YMultiLayerPerceptron :: Glayer ["<<ln<<"]  pos1 est1 "<<meas_GXc[ln]<<" "<<proj_GXc[ln]<<std::endl;
      std::cout<<setprecision(9)<<" YMultiLayerPerceptron :: Glayer ["<<ln<<"]  pos2 est2 "<<meas_GYc[ln]<<" "<<proj_GYc[ln]<<std::endl;
      std::cout<<setprecision(9)<<" YMultiLayerPerceptron :: Glayer ["<<ln<<"]  pos3 est3 "<<meas_GZc[ln]<<" "<<proj_GZc[ln]<<std::endl; 
#endif
      //Cost_Beam += std::pow(meas_GXc[ln]-proj_GXc[ln],2) + std::pow(meas_GYc[ln]-proj_GYc[ln],2) + std::pow(meas_GZc[ln]-proj_GZc[ln],2);  
      //b_Cost_Beam[ln] = std::pow(meas_GXc[ln]-proj_GXc[ln],2) + std::pow(meas_GYc[ln]-proj_GYc[ln],2) + std::pow(meas_GZc[ln]-proj_GZc[ln],2);
      if(ln<nLAYER) {            
         meas_S1c[ln] = yGEOM->GToS(chipID[ln],meas_GXc[ln],meas_GYc[ln],meas_GZc[ln])(0);
         meas_S2c[ln] = yGEOM->GToS(chipID[ln],meas_GXc[ln],meas_GYc[ln],meas_GZc[ln])(1);
         meas_S3c[ln] = yGEOM->GToS(chipID[ln],meas_GXc[ln],meas_GYc[ln],meas_GZc[ln])(2);
         proj_S1c[ln] = yGEOM->GToS(chipID[ln],proj_GXc[ln],proj_GYc[ln],proj_GZc[ln])(0);
         proj_S2c[ln] = yGEOM->GToS(chipID[ln],proj_GXc[ln],proj_GYc[ln],proj_GZc[ln])(1);
         proj_S3c[ln] = yGEOM->GToS(chipID[ln],proj_GXc[ln],proj_GYc[ln],proj_GZc[ln])(2);
#ifdef YMLPDEBUG0      
         if(TMath::Abs(meas_S1c[ln]-proj_S1c[ln])>0.5){
            std::cout<<"YMLP ZFIT ERROR row "<<yGEOM->GToL(chipID[ln],proj_GXc[ln],proj_GYc[ln],proj_GZc[ln])(0)<<
                                      " col "<<yGEOM->GToL(chipID[ln],proj_GXc[ln],proj_GYc[ln],proj_GZc[ln])(1)<<std::endl;         
         
         }
         std::cout<<setprecision(9)<<" YMultiLayerPerceptron :: Slayer ["<<ln<<"]  pos1 est1 "<<meas_S1c[ln]<<" "<<proj_S1c[ln]<<std::endl;
         std::cout<<setprecision(9)<<" YMultiLayerPerceptron :: Slayer ["<<ln<<"]  pos2 est2 "<<meas_S2c[ln]<<" "<<proj_S2c[ln]<<std::endl;
         std::cout<<setprecision(9)<<" YMultiLayerPerceptron :: Slayer ["<<ln<<"]  pos3 est3 "<<meas_S3c[ln]<<" "<<proj_S3c[ln]<<std::endl;     
#endif
         nValidHits++;

         Residual_s1[ln] = proj_S1c[ln]-meas_S1c[ln];
         Residual_s2[ln] = proj_S2c[ln]-meas_S2c[ln];
         Chi_s1[ln]      = Residual_s1[ln]/GetSigma(RecRadius, ln, DET_MAG, 0);
         Chi_s2[ln]      = Residual_s2[ln]/GetSigma(RecRadius, ln, DET_MAG, 1);
         Chi2TOT[ln]     = std::pow(Residual_s1[ln],2)/std::pow(GetSigma(RecRadius, ln, DET_MAG, 0),2) 
                         + std::pow(Residual_s2[ln],2)/std::pow(GetSigma(RecRadius, ln, DET_MAG, 1),2);

         b_Cost_Beam[ln] = std::pow(Residual_s1[ln],2) + std::pow(Residual_s2[ln],2);

         Cost_Beam      += std::pow(Residual_s1[ln],2)/std::pow(GetSigma(RecRadius, ln, DET_MAG, 0),2) 
                         + std::pow(Residual_s2[ln],2)/std::pow(GetSigma(RecRadius, ln, DET_MAG, 1),2);
          
         if(ln<nLAYERIB){
            if(std::abs(Chi_s1[ln])>RANGE_CHI_IB || std::abs(Chi_s2[ln])>RANGE_CHI_IB) nBadFits++;
         } else {
            if(std::abs(Chi_s1[ln])>RANGE_CHI_OB || std::abs(Chi_s2[ln])>RANGE_CHI_OB) nBadFits++;
         }
         
         //std::cout<<" ["<<ln<<"] "<<Chi_s1[ln]<<" "<<proj_S1c[ln]<<" "<<meas_S1c[ln]<<" "<<GetSigma(RecRadius, ln, DET_MAG, 0)<<" "
         //                         <<Chi_s2[ln]<<" "<<proj_S2c[ln]<<" "<<meas_S2c[ln]<<" "<<GetSigma(RecRadius, ln, DET_MAG, 1);
      } else {
         Residual_s1[ln] = proj_GZc[ln]-meas_GZc[ln];  // Z
         
         double meas_gRc  = std::sqrt(std::pow(meas_GXc[ln]-CircleXc,2) + std::pow(meas_GYc[ln]-CircleYc,2));
         double sign_vtxR = RecRadius > meas_gRc ? +1 : -1;   
          
         Residual_s2[ln] = sign_vtxR*std::sqrt(std::pow(proj_GXc[ln]-meas_GXc[ln],2) + std::pow(proj_GYc[ln]-meas_GYc[ln],2));
         Chi_s1[ln]      = Residual_s1[ln]/GetSigma(RecRadius, ln, DET_MAG, 0);
         Chi_s2[ln]      = Residual_s2[ln]/GetSigma(RecRadius, ln, DET_MAG, 1);
         Chi2TOT[ln]     = std::pow(Residual_s1[ln],2)/std::pow(GetSigma(RecRadius, ln, DET_MAG, 0),2) 
                         + std::pow(Residual_s2[ln],2)/std::pow(GetSigma(RecRadius, ln, DET_MAG, 1),2);

         b_Cost_Beam[ln] = std::pow(Residual_s1[ln],2) + std::pow(Residual_s2[ln],2);

         Cost_Beam      += std::pow(Residual_s1[ln],2)/std::pow(GetSigma(RecRadius, ln, DET_MAG, 0),2) 
                         + std::pow(Residual_s2[ln],2)/std::pow(GetSigma(RecRadius, ln, DET_MAG, 1),2);
   
         //double vtx_residualXY2 = std::pow(Fitpar[2],2) + std::pow(Fitpar[3],2);
         //Cost_Beam += vtx_residualXY2 / std::pow(GetSigma(RecRadius, ln, DET_MAG, 1),2);
      }
   }   

   if(nValidHits>=5) {
      if(Cost_Beam>TrackRejection || nBadFits>0) {
         AddTrackLoss();      
         Cost = 0;
      } else {
         Cost = Cost_Beam;
      }
   } else {
      AddTrackLoss();
      Cost = 0;
   }

   b_resmonitor->clear();
   b_resmonitor->registerCurvature(1/Fitpar[0]);    
   b_resmonitor->registerVertex(BeamCenter(0), BeamCenter(1), BeamCenter(2));
   b_resmonitor->registerVertexEvent(BeamPos[0],BeamPos[1],BeamPos[2]);
   b_resmonitor->registerVertexFit(Fitpar[2], Fitpar[3], (parz[0])*(beta[nLAYER]) + (parz[1]));
   
   double CircleXf = CircleXc - BeamCenter(0);
   double CircleYf = CircleYc - BeamCenter(1);   
   double param_dcaA = 1 + std::pow(CircleYf/CircleXf,2);
   double param_dcaB = -(CircleXf + CircleYf * CircleYf / CircleXf);
   double param_dcaC = std::pow(CircleXf,2) + std::pow(CircleYf,2) - std::pow(RecRadius,2);
   
   double x_dca_1 = (-param_dcaB - std::sqrt(param_dcaB*param_dcaB - param_dcaA*param_dcaC))/(param_dcaA);
   double y_dca_1 = CircleYf / CircleXf * x_dca_1;
   double x_dca_2 = (-param_dcaB + std::sqrt(param_dcaB*param_dcaB - param_dcaA*param_dcaC))/(param_dcaA);
   double y_dca_2 = CircleYf / CircleXf * x_dca_2;
   
   double x_dca = std::sqrt(std::pow(x_dca_1,2) + std::pow(y_dca_1,2))
                < std::sqrt(std::pow(x_dca_2,2) + std::pow(y_dca_2,2)) 
                ? x_dca_1 : x_dca_2;
   double y_dca = std::sqrt(std::pow(x_dca_1,2) + std::pow(y_dca_1,2))
                < std::sqrt(std::pow(x_dca_2,2) + std::pow(y_dca_2,2)) 
                ? y_dca_1 : y_dca_2;
               
   double Check_Rdca = std::sqrt(std::pow(x_dca - CircleXf,2) + std::pow(y_dca - CircleYf,2));
   //std::cout<<" VTXCHECK20221124 : V(dca) "<<x_dca<<" "<<y_dca<<" R(dca) "<<Check_Rdca<<" R(fit) "<<RecRadius<<std::endl;  
   double Check_Rfit = std::sqrt(std::pow(Fitpar[2] - CircleXf,2) + std::pow(Fitpar[3] - CircleYf,2));
   //std::cout<<" VTXCHECK20221124 : V(fit) "<<Fitpar[2]<<" "<<Fitpar[3]<<" R(fit) "<<Check_Rfit<<" R(fit) "<<RecRadius<<std::endl;        
   double beta_dca = std::atan2(y_dca - CircleYf, x_dca - CircleXf);
   double z_dca = (parz[0])*(beta_dca) + (parz[1]);
   b_resmonitor->registerDCA(x_dca,y_dca,z_dca);   

   TMatrix RotPhi(2,2);
   double angleRotPhi = Fitpar[0] > 0 ? -90*TMath::DegToRad() : +90*TMath::DegToRad();
   TMatrixD matRotPhi(2,2);
   matRotPhi[0] = { +std::cos(angleRotPhi), -std::sin(angleRotPhi)};
   matRotPhi[1] = { +std::sin(angleRotPhi), +std::cos(angleRotPhi)};
   TMatrixD matPhi[2];
   matPhi[0].ResizeTo(1,2);
   matPhi[0][0] = { CircleXf - Fitpar[2], CircleYf - Fitpar[3]};
   matPhi[0].T();
   matPhi[1].ResizeTo(2,1);
   matPhi[1] = matRotPhi * matPhi[0];  
   matPhi[1].T();      
   double vtx_phi   = std::atan2(matPhi[1][0][1], matPhi[1][0][0]);

   double gX_VERTEX    = Fitpar[2];      
   double gY_VERTEX    = Fitpar[3];      
   double gZ_VERTEX    = (parz[0])*(beta[nLAYER]) + (parz[1]);
   double gX_OUTERMOST = 0;      
   double gY_OUTERMOST = 0;      
   double gZ_OUTERMOST = 0;
   for(int lay = 0; lay<nLAYER; lay++){
      if(chipID[(nLAYER-1)-lay]>=0) {
         gX_OUTERMOST = proj_GXc[(nLAYER-1)-lay];
         gY_OUTERMOST = proj_GYc[(nLAYER-1)-lay];                     
         gZ_OUTERMOST = proj_GZc[(nLAYER-1)-lay];
        break;
      }
   }
   double deltaZ = gZ_OUTERMOST - gZ_VERTEX;
   double deltaR = std::sqrt(std::pow(gX_OUTERMOST - gX_VERTEX, 2) + std::pow(gY_OUTERMOST - gY_VERTEX, 2));   
   double vtx_theta = std::atan2(deltaR,deltaZ);
   b_resmonitor->registerMomentum(RecMomentumT, vtx_phi, vtx_theta);
      
// Track parametrization by Layer0.    
   double vxyz[] = {proj_GXc[0], proj_GYc[0], proj_GZc[0]};
   
   double FitFrameGx = proj_GXc[0] - BeamCenter(0);
   double FitFrameGy = proj_GYc[0] - BeamCenter(1);
   
   TMatrixD matPhiL0[2];
   matPhiL0[0].ResizeTo(1,2);
   matPhiL0[0][0] = { CircleXf - FitFrameGx, CircleYf - FitFrameGy};
   matPhiL0[0].T();
   matPhiL0[1].ResizeTo(2,1);
   matPhiL0[1] = matRotPhi * matPhiL0[0];  
   matPhiL0[1].T(); 
   
   double pT_L0     = RecMomentumT;
   double phi_L0    = std::atan2(matPhiL0[1][0][1], matPhiL0[1][0][0]);
   double theta_L0  = vtx_theta; //invariant for all layers
   double eta_L0    = -std::log(std::tan(theta_L0/2.));
   double p_L0      = pT_L0*std::cosh(eta_L0);
   double pz_L0     = pT_L0*std::sinh(eta_L0);
   double px_L0     = pT_L0*std::cos(phi_L0);
   double py_L0     = pT_L0*std::sin(phi_L0); 
   
   double pxyz[] = {px_L0, py_L0, pz_L0};
   int track_charge = Fitpar[0]>0 ? -1 : +1;
   YImpactParameter b_IP;
   
   b_IP.TrackParametrization(vxyz, pxyz, track_charge);
   // 1T = 10kG;
   double det_mag_in_kG = -10*DET_MAG;
   b_IP.getImpactParams(BeamCenter(0), BeamCenter(1), BeamCenter(2),det_mag_in_kG);
   //b_IP.getImpactParams(fvertex_TRKF.X(), fvertex_TRKF.Y(), fvertex_TRKF.Z(),det_mag_in_kG);
   //b_IP.getImpactParams(BeamProfileXZ1*BeamPos[2] + BeamProfileXZ0, BeamProfileYZ1*BeamPos[2] + BeamProfileYZ0, BeamPos[2], det_mag_in_kG);
   if(TMath::Abs(b_IP.ip[0])>RANGE_IMPACTPARAMS_R || TMath::Abs(b_IP.ip[1])>RANGE_IMPACTPARAMS_Z) untrained = true;
   //b_IP.Print();
   b_resmonitor->registerImpactParameters(b_IP);
   
   // vertex part
   int tot_nEvents_trk = 0;
   int tot_uSensor_trk = 0;
   for(int ln = 0; ln < nLAYER; ln++){   
      if(hitUpdate[ln]==false) continue;   
      tot_nEvents_trk += fSCNetwork[chipID[ln]]->GetnEvents();
      tot_uSensor_trk += fUPDATESENSORS->GetBinContent(1+chipID[ln]);
   
   }
   b_resmonitor->registerChipUpdateStatus(nLAYER, tot_nEvents_trk, tot_uSensor_trk);    

   b_resmonitor->registerResidual(nLAYER, 
                                  0, 0, 
                                  Residual_s1[nLAYER], Residual_s2[nLAYER], 							
                                  inputG_0[(3*nLAYER)+0], inputG_0[(3*nLAYER)+1], inputG_0[(3*nLAYER)+2], 
                                  std::sqrt(inputG_0[(3*nLAYER)+0]*inputG_0[(3*nLAYER)+0] + inputG_0[(3*nLAYER)+1]*inputG_0[(3*nLAYER)+1]),
                                  std::atan2(inputG_0[(3*nLAYER)+1],inputG_0[(3*nLAYER)+0]));

   b_resmonitor->registerCorrectionFunction(nLAYER, 0, 0, 0,
                                                    0, 0, 0);

   b_resmonitor->registerchipInfo(nLAYER, -1, -1, -1, -1, -1, -1, -1, -1, -1);
   
   // track part
   
   
   int PatchPhi = -1;   
   for(int ln = 0; ln < nLAYER; ln++){   
      if(hitUpdate[ln]==false) continue;
               if(chipID[ln] >= 0) fSCNetwork[chipID[ln]]->hNtracksByRejection->Fill(4);
      if(nValidHits<5) continue;      
               if(chipID[ln] >= 0) fSCNetwork[chipID[ln]]->hNtracksByRejection->Fill(5);
      if(nBadFits>0) continue;
               if(chipID[ln] >= 0) fSCNetwork[chipID[ln]]->hNtracksByRejection->Fill(6);       

      //if(Fitpar[0]<0) continue; // +q 
      //if(Fitpar[0]>0) continue; // -q
      //         if(chipID[ln] >= 0) fSCNetwork[chipID[ln]]->hNtracksByRejection->Fill(7);     

#ifdef MONITORONLYUPDATES
      if(IsCorrection!=(short)(std::pow(2,nLAYER)-1)) continue;
               if(chipID[ln] >= 0) fSCNetwork[chipID[ln]]->hNtracksByRejection->Fill(7);       
#endif 

      fpTvsResLayer[ln][0]->Fill(Residual_s1[ln],RecMomentumT);
      fpTvsResLayer[ln][1]->Fill(Residual_s2[ln],RecMomentumT);
      fpTvsChiLayer[ln][0]->Fill(Chi_s1[ln],RecMomentumT);
      fpTvsChiLayer[ln][1]->Fill(Chi_s2[ln],RecMomentumT);
      fChi2Layer[ln]->Fill(Chi2TOT[ln]);
                
#ifdef MONITORHALFSTAVEUNIT 
      int hb	= yGEOM->GetHalfBarrel(chipID[ln]);  		//HalfBarrel
      int stv	= yGEOM->GetStave(chipID[ln]);  		//Stave        
      int hs	= yGEOM->GetHalfStave(chipID[ln]);  		//HalfStave  
      int md    = yGEOM->GetModule(chipID[ln]);  
      int lchip = yGEOM->GetChipIdInLayer(chipID[ln]);    /// Get chip number within layer, from 0
      int schip = yGEOM->GetChipIdInStave(chipID[ln]);    /// Get chip number within stave, from 0
      int hschip= yGEOM->GetChipIdInHalfStave(chipID[ln]); /// Get chip number within stave, from 0
      int mchip = yGEOM->GetChipIdInModule(chipID[ln]);    /// Get chip number within module, from 0      	

      if(ln==2) PatchPhi = stv;
      if(untrained==true){
         b_resmonitor->registerChipUpdateStatus(ln, fSCNetwork[chipID[ln]]->GetnEvents(), -1);    
      } else {
         b_resmonitor->registerChipUpdateStatus(ln, fSCNetwork[chipID[ln]]->GetnEvents(), fUPDATESENSORS->GetBinContent(1+chipID[ln]));    
      }
      b_resmonitor->registerResidual(ln, 
                                     inputS[ln][0], inputS[ln][1], 
                                     Residual_s1[ln], Residual_s2[ln], 
                                     inputG_0[(3*ln)+0], inputG_0[(3*ln)+1], inputG_0[(3*ln)+2], 
                                     std::sqrt(inputG_0[(3*ln)+0]*inputG_0[(3*ln)+0] + inputG_0[(3*ln)+1]*inputG_0[(3*ln)+1]),
                                     std::atan2(inputG_0[(3*ln)+1],inputG_0[(3*ln)+0]));

      b_resmonitor->registerCorrectionFunction(ln, inputG_C[(3*ln)+0] - inputG_0[(3*ln)+0], inputG_C[(3*ln)+1] - inputG_0[(3*ln)+1], inputG_C[(3*ln)+2] - inputG_0[(3*ln)+2],
                                                   inputS_C[(3*ln)+0] - inputS_0[(3*ln)+0], inputS_C[(3*ln)+1] - inputS_0[(3*ln)+1], inputS_C[(3*ln)+2] - inputS_0[(3*ln)+2]);

      b_resmonitor->registerchipInfo(ln, chipID[ln], hb, stv, hs, md, lchip, schip, hschip, mchip);

      fpTvsResLayerHBHS[ln][hb][hs][0]->Fill(Residual_s1[ln],RecMomentumT);
      fpTvsResLayerHBHS[ln][hb][hs][1]->Fill(Residual_s2[ln],RecMomentumT);
      fpTvsChiLayerHBHS[ln][hb][hs][0]->Fill(Chi_s1[ln],RecMomentumT);
      fpTvsChiLayerHBHS[ln][hb][hs][1]->Fill(Chi_s2[ln],RecMomentumT);
      fResidualsVsZLayerHBHS[ln][hb][hs][0]->Fill(inputG_0[(3*ln)+2],Residual_s1[ln]);
      fResidualsVsZLayerHBHS[ln][hb][hs][1]->Fill(inputG_0[(3*ln)+2],Residual_s2[ln]);
      fResidualsVsPhiLayerHBHS[ln][hb][hs][0]->Fill(std::atan2(inputG_0[(3*ln)+1],inputG_0[(3*ln)+0]),Residual_s1[ln]);
      fResidualsVsPhiLayerHBHS[ln][hb][hs][1]->Fill(std::atan2(inputG_0[(3*ln)+1],inputG_0[(3*ln)+0]),Residual_s2[ln]);
      fProfileVsZLayerHBHS[ln][hb][hs][0]->Fill(inputG_0[(3*ln)+2],Residual_s1[ln]);
      fProfileVsZLayerHBHS[ln][hb][hs][1]->Fill(inputG_0[(3*ln)+2],Residual_s2[ln]);
      fProfileVsPhiLayerHBHS[ln][hb][hs][0]->Fill(std::atan2(inputG_0[(3*ln)+1],inputG_0[(3*ln)+0]),Residual_s1[ln]);
      fProfileVsPhiLayerHBHS[ln][hb][hs][1]->Fill(std::atan2(inputG_0[(3*ln)+1],inputG_0[(3*ln)+0]),Residual_s2[ln]);

      double SensorCenterX = yGEOM->LToG(chipID[ln],255,511).X();
      double SensorCenterY = yGEOM->LToG(chipID[ln],255,511).Y();
      double SensorCenterZ = yGEOM->LToG(chipID[ln],255,511).Z();

      if(std::abs(SensorCenterZ - inputG_0[(3*ln)+2])<0.15) {
         fSensorCenterVsZLayerHBHS[ln][hb][hs][0]->Fill(inputG_0[(3*ln)+2],Residual_s1[ln]);
         fSensorCenterVsZLayerHBHS[ln][hb][hs][1]->Fill(inputG_0[(3*ln)+2],Residual_s2[ln]);
      }
      if(std::sqrt(SensorCenterX*SensorCenterX + SensorCenterY*SensorCenterY)*std::abs(std::atan2(SensorCenterY,SensorCenterX) - std::atan2(inputG_0[(3*ln)+1],inputG_0[(3*ln)+0]))<0.15) {
         fSensorCenterVsPhiLayerHBHS[ln][hb][hs][0]->Fill(std::atan2(inputG_0[(3*ln)+1],inputG_0[(3*ln)+0]),Residual_s1[ln]);
         fSensorCenterVsPhiLayerHBHS[ln][hb][hs][1]->Fill(std::atan2(inputG_0[(3*ln)+1],inputG_0[(3*ln)+0]),Residual_s2[ln]);
      }
#endif    

#ifdef MONITORSENSORUNITprofile
      fSCNetwork[chipID[ln]]->hpds1_s1->Fill(inputS[ln][0],Residual_s1[ln]);
      fSCNetwork[chipID[ln]]->hpds1_s2->Fill(inputS[ln][1],Residual_s1[ln]);                             
      fSCNetwork[chipID[ln]]->hpds2_s1->Fill(inputS[ln][0],Residual_s2[ln]);
      fSCNetwork[chipID[ln]]->hpds2_s2->Fill(inputS[ln][1],Residual_s2[ln]);
#endif

#ifdef MONITORSENSORUNITpT         
      fSCNetwork[chipID[ln]]->FillpTvsRes(0, RecMomentumT, Residual_s1[ln]);
      fSCNetwork[chipID[ln]]->FillpTvsRes(1, RecMomentumT, Residual_s2[ln]);       
      fSCNetwork[chipID[ln]]->FillpTvsChi(0, RecMomentumT, Chi_s1[ln]);
      fSCNetwork[chipID[ln]]->FillpTvsChi(1, RecMomentumT, Chi_s2[ln]);      
      fSCNetwork[chipID[ln]]->FillChi2(Chi2TOT[ln]);
#endif      
      
   }

   fResidualMonitor->Fill();

#ifdef MONITORONLYUPDATES
   if(IsCorrection!=(short)(std::pow(2,nLAYER)-1)) {
      Cost = 0;
   } else if (untrained == true){
      Cost = 0;
   } else {
      fCostMonitor->Fill(Cost_Beam);

      fBeamXY->Fill(BeamCenter(0),BeamCenter(1));   
      fBeamZR->Fill(BeamPos[2],std::sqrt(BeamCenter(0)*BeamCenter(0) + BeamCenter(1)*BeamCenter(1))); 
      fVertexFitXY->Fill(Fitpar[2],Fitpar[3]); 
      fVertexFitZR->Fill(BeamPos[2],std::sqrt(Fitpar[2]*Fitpar[2] + Fitpar[3]*Fitpar[3])); 
      fVertexXY->Fill(BeamCenter(0) + Fitpar[2], BeamCenter(1) + Fitpar[3]); 
      fVertexZR->Fill(BeamPos[2], std::sqrt((BeamCenter(0) + Fitpar[2])*(BeamCenter(0) + Fitpar[2]) + (BeamCenter(1) + Fitpar[3])*(BeamCenter(1) + Fitpar[3])));
      
      if(PatchPhi!=-1){
         fCostChargeSymSum[PatchPhi] += track_charge*RecMomentumT;
         fCostChargeSymNtr[PatchPhi]++;
         fCostChargeSym->Fill(PatchPhi, track_charge*RecMomentumT);
         if(track_charge==+1) fChargeSymMonitorPositive->Fill(PatchPhi);
         else if(track_charge==-1) fChargeSymMonitorNegative->Fill(PatchPhi);     
      }           
   }
   //std::cout<<std::endl;   
   return Cost;
#endif

   fCostMonitor->Fill(Cost_Beam);

   fBeamXY->Fill(BeamCenter(0),BeamCenter(1));   
   fBeamZR->Fill(BeamPos[2],std::sqrt(BeamCenter(0)*BeamCenter(0) + BeamCenter(1)*BeamCenter(1))); 
   fVertexFitXY->Fill(Fitpar[2],Fitpar[3]); 
   fVertexFitZR->Fill(BeamPos[2],std::sqrt(Fitpar[2]*Fitpar[2] + Fitpar[3]*Fitpar[3])); 
   fVertexXY->Fill(BeamCenter(0) + Fitpar[2], BeamCenter(1) + Fitpar[3]); 
   fVertexZR->Fill(BeamPos[2], std::sqrt((BeamCenter(0) + Fitpar[2])*(BeamCenter(0) + Fitpar[2]) + (BeamCenter(1) + Fitpar[3])*(BeamCenter(1) + Fitpar[3])));      

   fChargeSymMonitorPositive->Fill(-1);
   fChargeSymMonitorNegative->Fill(-1);

#ifdef YMLPDEBUG0
   for(int ln = 0; ln < nLAYER; ln++){ 
      std::cout<<setprecision(9)<<"[MLP] :: Layer "<<ln<<" "<<chipID[ln]<<" Cost "<<b_Cost_Beam[ln]<<
                                                        " Sigma "<<GetSigma(RecRadius, ln, DET_MAG, 0)<<" "
                                                                 <<GetSigma(RecRadius, ln, DET_MAG, 1)<<" "
                                                        " Sigma2 "<<std::pow(GetSigma(RecRadius, ln, DET_MAG, 0),2)<<" "   
                                                                  <<std::pow(GetSigma(RecRadius, ln, DET_MAG, 1),2)<<std::endl;               
   }   
   std::cout<<setprecision(9)<<" (R, Xc, Yc) = ("<<RecRadius<<", "<<CircleXc<<", "<<CircleYc<<") Cost_Beam "<<Cost_Beam<<" nValidHits "<<nValidHits<<" Cost "<<Cost<<std::endl; 
   std::cout<<"DERIVCHECK REF Radius Xc Yc thetaB "<<RecRadius<<" "<<CircleXc<<" "<<CircleYc<<" "<<Fitpar[1]+Fitpar[4]<<std::endl;          
#endif  
   //std::cout<<std::endl;
   return Cost;

}       

void YMultiLayerPerceptron::CalculateEventDcdw(int ntracks)
{
   //UpdateVertexByAlignment();         
   float norm_shift = 0.5;
   
   int NFirstLayer = fFirstLayer.GetEntriesFast();
   int NLastLayer = fLastLayer.GetEntriesFast();
#ifdef YSCNEURONDEBUG
   std::cout<<" NFirstLayer : "<<NFirstLayer<<" NLastLayer : "<<NLastLayer<<std::endl;
#endif
   int NBeamLayer = 3;   
   int Nfitparam = 6; 

   int ROutputLayerTrain[NLastLayer];
   for(int ln=0; ln<nLAYER; ln++){
      ROutputLayerTrain[3*ln + 0] = (int)(fLayerTrain%(int)(std::pow(10,(nLAYER)-ln)))/(int)(std::pow(10,(nLAYER-1)-ln));
      ROutputLayerTrain[3*ln + 1] = (int)(fLayerTrain%(int)(std::pow(10,(nLAYER)-ln)))/(int)(std::pow(10,(nLAYER-1)-ln));
      ROutputLayerTrain[3*ln + 2] = (int)(fLayerTrain%(int)(std::pow(10,(nLAYER)-ln)))/(int)(std::pow(10,(nLAYER-1)-ln));
   } 

   double input_nS1[nTrackMax][nLAYER], input_nS2[nTrackMax][nLAYER];

   double meas_GXc[nTrackMax][nLAYER+1], meas_GYc[nTrackMax][nLAYER+1], meas_GZc[nTrackMax][nLAYER+1];
   double meas_S1c[nTrackMax][nLAYER+1], meas_S2c[nTrackMax][nLAYER+1], meas_S3c[nTrackMax][nLAYER+1];
          
   double proj_GXc[nTrackMax][nLAYER+1], proj_GYc[nTrackMax][nLAYER+1], proj_GZc[nTrackMax][nLAYER+1];
   double proj_S1c[nTrackMax][nLAYER+1], proj_S2c[nTrackMax][nLAYER+1], proj_S3c[nTrackMax][nLAYER+1];

   YVertexFitParameter vtxfit[nTrackMax];
   std::vector<bool> hitUpdateTrack[nTrackMax];
#ifdef MONITORONLYUPDATES
   short IsCorrection[nTrackMax];
#endif       
   bool IsAccessorialTrack[nTrackMax];
   double Cost_Beam[nTrackMax];
   int    nBadFits[nTrackMax];
   int    NVALIDHITS[nTrackMax];   
   for(int track = 0; track < ntracks; track++){
      IsAccessorialTrack[track] = (fSplitReferenceSensor==-1) ? false : true;
#ifdef MONITORONLYUPDATES      
      IsCorrection[track] = 0;
#endif      
   } 
   for(int track = 0; track < ntracks; track++){

      double DCost = 0;
   
      double Cost = 0;
      double Cost_Fit = 0;   

      double Fitpar[Nfitparam];      
        
      double inputS_C[NLastLayer]; 
      double inputS_0[NLastLayer];   
      double inputG_C[NLastLayer + NBeamLayer];
      double inputG_0[NLastLayer + NBeamLayer]; 

      double inputS[nLAYER][2];   
      double outputS[NLastLayer];
      double extended[NLastLayer];
      double addition[8];
      
      TVector3 vecXc_meas[nLAYER+1];
      TVector3 vecXc_proj[nLAYER+1];    
      TVector3 vecXc_norm[nLAYER+1];                 
        
      double input_Max[NLastLayer];
      double input_Min[NLastLayer];     

      int staveIndex[nLAYER], chipIndex[nLAYER], chipID[nLAYER];
      TVector3 vecSensorNorm[nLAYER];
   
      double BeamPos[3];
      double BeamMom[3];

      for(int ln = 0; ln < nLAYER+1; ln++){    
         vtxfit[track].z_meas[ln] = 0;
         vtxfit[track].beta[ln]   = 0;
      }
      
      vtxfit[track].parz[0]=0;
      vtxfit[track].parz[1]=0;
      vtxfit[track].Radius =0;
      vtxfit[track].valid  =false;

      for(int k = 0; k < 2*nLAYER; k++){   
         YNeuron *neuron_in = (YNeuron *) fFirstLayer.At(k);  // 6 input -> 2+1 ; 2+1 ; 2+1 build dummy
         neuron_in->SetNewEvent();   
         neuron_in->SetNeuronIndex(track);   
         int l1 = (int)(k/2);
         int l2 = (int)(k%2);  
         inputS[l1][l2] = neuron_in->GetValue();
         if(l2==0) input_nS1[track][l1] = inputS[l1][l2];
         else if(l2==1) input_nS2[track][l1] = inputS[l1][l2];
#ifdef YSCNEURONDEBUG
         std::cout<<"track "<<track<<" "<<"inputS["<<l1<<"]["<<l2<<"] "<<inputS[l1][l2]<<" "<<std::endl;
#endif
      }
         
      for(int k = 0; k < NLastLayer; k++) {
         YNeuron *neuron_out = (YNeuron *) fLastLayer.At(k);
         neuron_out->SetNewEvent();               
         neuron_out->SetNeuronIndex(track);  
         extended[k] = neuron_out->GetBranch();
      }   

      std::vector<bool> hitUpdate;
      std::vector<bool> hitUpdate_Z; 
   
      for(int ln = 0; ln<nLAYER; ln++){             
         staveIndex[ln] = (int)extended[(3*ln)+0];   
         chipIndex[ln]  = (int)extended[(3*ln)+1];  
         chipID[ln]     = (int)extended[(3*ln)+2];  
         bool layUpdate = chipID[ln] < 0 ? false : true;
         if(chipID[ln] >= 0 && fSplitReferenceSensor == chipID[ln]) IsAccessorialTrack[track]=false;         
         hitUpdateTrack[track].push_back(layUpdate);
         hitUpdate.push_back(layUpdate);
         hitUpdate_Z.push_back(layUpdate);          
         //vecSensorNorm[ln] = yGEOM->NormalVector(chipID[ln]);
         vecSensorNorm[ln] = GetCorrectedNormalVector(chipID[ln]);    
#ifdef MONITORONLYUPDATES     
         IsCorrection[track] <<= 1;
         if(chipID[ln] >= 0) {
            short us_value =  fUPDATESENSORS->GetBinContent(1+chipID[ln]) >= 1 ? 1 : 0;
            IsCorrection[track] += us_value;
         } else IsCorrection[track] += 1; // no hit case, count (virtually) for use
#endif                    
#ifdef YSCNEURONDEBUG
         std::cout<<" layer stave chip ID "<<ln<<" "<<staveIndex[ln]<<" "<<chipIndex[ln]<<" "<<chipID[ln]<<" "<<std::endl;
#endif
      }     
      hitUpdateTrack[track].push_back(true);      
      hitUpdate.push_back(true);  
      hitUpdate_Z.push_back(true);  
#ifdef MONITORONLYUPDATES
      if(IsCorrection[track]!=std::pow(2,nLAYER)-1){
         for(int ln = 0; ln<nLAYER; ln++){         
            for(int iaxis = 0; iaxis<3; iaxis++){
               int index = (3*ln)+iaxis;
               SetSCNeuronDcdw(index,track,0.0);
            }
         }      
         continue;
      }
#endif 

      for(int k = 0; k < 8; k++) {
         YNeuron *neuron_addition = (YNeuron *) fAddition.At(0);
         neuron_addition->SetNeuronIndex(1000 + k);  
         neuron_addition->SetNewEvent();         
         addition[k] = neuron_addition->GetBranchAddition();
      } 

      for(int k = 0; k<fNetwork.GetEntriesFast();k++) {
         YNeuron *neuron = (YNeuron *)fNetwork.At(k);
         neuron->SetNeuronIndex(track);      
         neuron->SetNewEvent();
      }   

      BeamPos[0] = addition[0];
      BeamPos[1] = addition[1];
      BeamPos[2] = addition[2];   
      BeamMom[0] = addition[3];
      BeamMom[1] = addition[4];
      BeamMom[2] = addition[5];        
      
      for(int ln = 0; ln < nLAYER; ln ++){      
         if(chipID[ln]<0) continue;         
         for(int k=0; k< fSCNetwork[chipID[ln]]->GetNetwork().GetEntriesFast(); k++){
            YNeuron *neuron = (YNeuron *) fSCNetwork[chipID[ln]]->GetNetwork().At(k);
            neuron->SetNeuronIndex(track);  
            neuron->SetNewEvent();            
         }     
         for(int k=0; k< fSCNetwork[chipID[ln]]->GetLastLayer().GetEntriesFast(); k++){
            YNeuron *neuron_out = (YNeuron *) fSCNetwork[chipID[ln]]->GetLastLayer().At(k);         
            outputS[3*ln + k] = Evaluate(k, inputS[ln], chipID[ln]); 
         }
      }
    
      for(int ln = 0; ln<nLAYER; ln++){         
         for(int iaxis = 0; iaxis<3; iaxis++){
            double ip, fp;   
            ip = yGEOM->GToS(chipID[ln],yGEOM->LToG(chipID[ln],0,0)(0),	
                                        yGEOM->LToG(chipID[ln],0,0)(1),
                                        yGEOM->LToG(chipID[ln],0,0)(2))(iaxis);
            fp = yGEOM->GToS(chipID[ln],yGEOM->LToG(chipID[ln],511,1023)(0),
                                        yGEOM->LToG(chipID[ln],511,1023)(1),
                                        yGEOM->LToG(chipID[ln],511,1023)(2))(iaxis);       
            int index = (3*ln)+iaxis;
            input_Max[index] = std::max(ip,fp);
            input_Min[index] = std::min(ip,fp); 
   
            if(iaxis==2){
               inputS_0[index]  = 0;
               inputS_C[index]  = outputS[index]; 										//d = exp - pos -> pos + d = exp
            } else {    
               inputS_0[index]  = (inputS[ln][iaxis] + norm_shift)*(input_Max[index]-input_Min[index])+input_Min[index];     			
               inputS_C[index]  = (inputS[ln][iaxis] + norm_shift)*(input_Max[index]-input_Min[index])+input_Min[index] + outputS[index];     	//d = exp - pos -> pos + d = exp 
            }         
         } 
#ifdef YSCNEURONDEBUG
         std::cout<<"Layer(Neuron S) "<<ln<<" "<<chipID[ln]<<" "<<inputS_C[(3*ln)+0]<<" "<<inputS_C[(3*ln)+1]<<" "<<inputS_C[(3*ln)+2]<<std::endl;
#endif
      }             

      for(int ln = 0; ln<nLAYER; ln++){         
         for(int iaxis = 0; iaxis<3; iaxis++){
            int index = (3*ln)+iaxis;      
            inputG_C[index]  =yGEOM->SToG(chipID[ln],inputS_C[(3*ln)+0],inputS_C[(3*ln)+1],inputS_C[(3*ln)+2])(iaxis); 
            inputG_0[index]  =yGEOM->SToG(chipID[ln],inputS_0[(3*ln)+0],inputS_0[(3*ln)+1],inputS_0[(3*ln)+2])(iaxis);  
         }
         vecXc_meas[ln].SetXYZ(inputG_C[(3*ln)+0], inputG_C[(3*ln)+1], inputG_C[(3*ln)+2]);      
         double row = yGEOM->SToL(chipID[ln],inputS_C[(3*ln)+0],inputS_C[(3*ln)+1],inputS_C[(3*ln)+2])(0); 
         double col = yGEOM->SToL(chipID[ln],inputS_C[(3*ln)+0],inputS_C[(3*ln)+1],inputS_C[(3*ln)+2])(1); 
#ifdef YSCNEURONDEBUG
         std::cout<<"Layer(Neuron G) "<<ln<<" "<<chipID[ln]<<" "<<inputG_C[(3*ln)+0]<<" "<<inputG_C[(3*ln)+1]<<" "<<inputG_C[(3*ln)+2]<<" row col : "<<row<<" "<<col<<std::endl;
#endif
      }
      vector<TVector3> vecBeam[2]; // 0 : track, 1 : beam

      //TVector3 BeamCenter(BeamProfileXZ1*BeamPos[2] + BeamProfileXZ0, BeamProfileYZ1*BeamPos[2] + BeamProfileYZ0, BeamPos[2]);   //BeamPos[0],BeamPos[1],BeamPos[2] 
      TVector3 BeamCenter(BeamPos[0], BeamPos[1], BeamPos[2]);   //BeamPos[0],BeamPos[1],BeamPos[2]
         
      //bool UseUpdatedVertexByAlignment = true;
      //if(UseUpdatedVertexByAlignment==true) BeamCenter = fvertex_TRKF; 
  
      //TVector3 BeamCenter(0.0,0.0,BeamPos[2]);   //BeamPos[0],BeamPos[1],BeamPos[2]
#ifdef YSCNEURONDEBUG
      std::cout<<"YNeuron::Beam(C) track :: "<<BeamCenter(0)<<" "<<BeamCenter(1)<<" "<<BeamCenter(2)<<std::endl;    
#endif
   
      inputG_C[(3*nLAYER)+0] = BeamCenter(0); 
      inputG_C[(3*nLAYER)+1] = BeamCenter(1);
      inputG_C[(3*nLAYER)+2] = BeamCenter(2);  
      inputG_0[(3*nLAYER)+0] = BeamCenter(0); 
      inputG_0[(3*nLAYER)+1] = BeamCenter(1);
      inputG_0[(3*nLAYER)+2] = BeamCenter(2);    
    
      vecXc_meas[nLAYER].SetXYZ(inputG_C[(3*nLAYER)+0], inputG_C[(3*nLAYER)+1], inputG_C[(3*nLAYER)+2]);     
                          
      for(int k = 0; k < Nfitparam; k++){
         Fitpar[k]=0.0;
      }

      Cost_Fit = 0;    

      circle3Dfit(inputG_C, Fitpar, Cost_Fit, hitUpdate, 0);

      double min_Cost_Fit_Scale = ((int)Fitpar[5])%10==0 ? 1e+10 : Cost_Fit;
      double Cost_FitD(0), FitparD[6];
      int search_strategy[] = {-2, +2, +4};
      if(((int)Fitpar[5])%10>=0 || Cost_Fit>1.0e-4) {
         for(int isch = 0; isch < 3; isch++){
            for(int j = 0; j < 6; j++){
               FitparD[j]=0.0;
            }
   
            circle3Dfit(inputG_C, FitparD, Cost_FitD, hitUpdate, search_strategy[isch]);

            if(min_Cost_Fit_Scale>Cost_FitD && ((int)FitparD[5])%10==1){
               min_Cost_Fit_Scale = Cost_FitD;
               Cost_Fit = Cost_FitD;
               for(int j = 0; j < 6; j++){
                  Fitpar[j]=FitparD[j];
               }
            }
         }
      }
  
      if(((int)Fitpar[5])%10==0){
         for(int ln = 0; ln<nLAYER; ln++){         
            for(int iaxis = 0; iaxis<3; iaxis++){
               int index = (3*ln)+iaxis;
               SetSCNeuronDcdw(index,track,0.0);
            }
         }      
         continue;
      }  

      // Found vertex cut : 20240314
      if(std::sqrt((BeamCenter(0) + Fitpar[2])*(BeamCenter(0) + Fitpar[2]) + (BeamCenter(1) + Fitpar[3])*(BeamCenter(1) + Fitpar[3]))>0.1) continue;
      
      double FitRadius = 1/Fitpar[0];
      double RecRadius = std::abs(1/Fitpar[0]);
      double CircleXc  = Fitpar[0]>0 ? RecRadius*std::cos(Fitpar[1]+Fitpar[4] + 0.5*TMath::Pi()) : RecRadius*std::cos(Fitpar[1]+Fitpar[4] - 0.5*TMath::Pi());
      double CircleYc  = Fitpar[0]>0 ? RecRadius*std::sin(Fitpar[1]+Fitpar[4] + 0.5*TMath::Pi()) : RecRadius*std::sin(Fitpar[1]+Fitpar[4] - 0.5*TMath::Pi()); 

      CircleXc = CircleXc + BeamCenter(0) + Fitpar[2];
      CircleYc = CircleYc + BeamCenter(1) + Fitpar[3];
   
      double RecMomentumT = 0.3*DET_MAG*RecRadius*1.0e-2;  //r[m] -> r[cm]
      if(RecMomentumT<Update_pTmin || RecMomentumT>Update_pTmax) {
         //std::cout<<"(Neuron) momentum range cut"<<std::endl;
         for(int ln = 0; ln<nLAYER; ln++){         
            for(int iaxis = 0; iaxis<3; iaxis++){
               int index = (3*ln)+iaxis;
               SetSCNeuronDcdw(index,track,0.0);
            }
         }              
         continue;
      }
  
#ifdef YSCNEURONDEBUG
      std::cout<<"Radius Xc Yc "<<RecRadius<<" "<<CircleXc<<" "<<CircleYc<<std::endl;     
#endif                  
      TVector3 vecCircle_center(CircleXc, CircleYc, 0);     
      TVector3 dirXc_meas[nLAYER+1];
      TVector3 dirXc_proj[nLAYER+1];   
      TVector3 dirXc_norm[nLAYER+1];    
           
      for(int a=0; a<nLAYER+1;a++){
         dirXc_meas[a] = vecXc_meas[a] - vecCircle_center;
      }

      double beta[nLAYER+1];
      for(int l = 0; l < nLAYER+1; l++){    
         beta[l] = std::atan2(dirXc_meas[l].Y(), dirXc_meas[l].X());// > 0 ? std::atan2(dirXc_meas[l].Y(), dirXc_meas[l].X()) : 2*std::atan2(0,-1) + std::atan2(dirXc_meas[l].Y(), dirXc_meas[l].X());
      }     

      //beta linearization
      BetaLinearization(beta, dirXc_meas, hitUpdate);      
      GetProjectionPoints(vecCircle_center, RecRadius, vecSensorNorm, vecXc_meas, vecXc_proj, vecXc_norm);

      for(int a=0; a<nLAYER+1;a++){
         dirXc_proj[a] = vecXc_proj[a] - vecCircle_center;
         dirXc_norm[a] = vecXc_norm[a] - vecCircle_center;
      }
   
      double pbeta_proj[nLAYER+1]; 
      double pbeta_norm[nLAYER+1];

      for(int l = 0; l < nLAYER+1; l++){    
         pbeta_proj[l] = std::atan2(dirXc_proj[l].Y(), dirXc_proj[l].X());// > 0 ? std::atan2(dirXc_proj[l].Y(), dirXc_proj[l].X()) : 2*std::atan2(0,-1) + std::atan2(dirXc_proj[l].Y(), dirXc_proj[l].X());
         pbeta_norm[l] = std::atan2(dirXc_norm[l].Y(), dirXc_norm[l].X());// > 0 ? std::atan2(dirXc_norm[l].Y(), dirXc_norm[l].X()) : 2*std::atan2(0,-1) + std::atan2(dirXc_norm[l].Y(), dirXc_norm[l].X()); 
      } 

      //beta linearization
      BetaLinearization(pbeta_proj, dirXc_proj, hitUpdate);      
      BetaLinearization(pbeta_norm, dirXc_norm, hitUpdate);      
      for(int l = 0; l < nLAYER+1; l++){    
         beta[l]     = pbeta_proj[l]; 
      }        
   
      double parz[2];
      double z_meas[nLAYER+1];
#ifdef YSCNEURONDEBUG
      std::cout<<"YNeuron zmeas ";
#endif
      for(int a=0; a<nLAYER+1; a++){
         z_meas[a] = vecXc_meas[a].Z();
#ifdef YSCNEURONDEBUG
         std::cout<<z_meas[a]<<" ";
#endif
      }
#ifdef YSCNEURONDEBUG
      std::cout<<std::endl;  
#endif   
   
      circle3Dfit_Z(z_meas, beta, parz, RecRadius, VERTEXFIT, hitUpdate_Z);
      for(int ln = 0; ln < nLAYER+1; ln++){    
         vtxfit[track].chipID[ln] = ln < nLAYER ? chipID[ln] : -9999;         
         vtxfit[track].z_meas[ln] = z_meas[ln];
         vtxfit[track].beta[ln]   = beta[ln];
      }
      vtxfit[track].parz[0]=parz[0];
      vtxfit[track].parz[1]=parz[1];
      vtxfit[track].Radius =RecRadius;
      vtxfit[track].valid  =true;

// Track angle
      double    IncS1[nLAYER];
   
      for(int ln = 0; ln < nLAYER; ln++){
         double angle_rs3 = std::abs(parz[0]/RecRadius);
         if(hitUpdate[ln]==true) IncS1[ln] = 1/std::cos(std::atan(angle_rs3));
         else IncS1[ln] = 0;
         //std::cout<<"(Incident Angle) Layer "<<ln<<" cos(lambda)=dr/dcs3 : "<<IncS1[ln]<<" "<<parz[0]<<" "<<std::atan(angle_rs3)<<" "<<std::cos(std::atan(angle_rs3))<<std::endl;
         
      }

#ifdef YSCNEURONDEBUG
      std::cout<<"YNeuron parz "<<parz[0]<<" "<<parz[1]<<std::endl;
#endif
      double znorm[nLAYER];   
      for(int a=0; a<nLAYER;a++){
         znorm[a] = (parz[0])*(pbeta_norm[a]) + (parz[1]);
      }               
      
      Cost_Beam[track]=0;
      nBadFits[track]=0;          
      NVALIDHITS[track]=0;          
      double b_Cost_Beam[nLAYER+1];

      int nValidHits = 0;
      for(int ln = 0; ln < nLAYER+1; ln++){
         if(hitUpdate[ln]==false) continue;
#ifdef YSCNEURONDEBUG
         std::cout<<" YNeuron :: layer ["<<ln<<"] beta = "<<beta[ln]<<std::endl;
         std::cout<<" YNeuron :: layer ["<<ln<<"] dirXc_meas "<<dirXc_meas[ln](0)<<" "<<dirXc_meas[ln](1)<<" "<<dirXc_meas[ln](2)<<std::endl;   
#endif            
         meas_GXc[track][ln] = (double)std::round(inputG_C[(3*ln)+0]*TARGET_D)/TARGET_D; //alpha
         meas_GYc[track][ln] = (double)std::round(inputG_C[(3*ln)+1]*TARGET_D)/TARGET_D; //beta
         meas_GZc[track][ln] = (double)std::round(inputG_C[(3*ln)+2]*TARGET_D)/TARGET_D; //gamma      
                             
         proj_GXc[track][ln] = (double)std::round((RecRadius*std::cos(beta[ln]) + CircleXc)*TARGET_D)/TARGET_D;
         proj_GYc[track][ln] = (double)std::round((RecRadius*std::sin(beta[ln]) + CircleYc)*TARGET_D)/TARGET_D;
         proj_GZc[track][ln] = (double)std::round(((parz[0])*(beta[ln]) + (parz[1]))*TARGET_D)/TARGET_D;
#ifdef YSCNEURONDEBUG
         std::cout<<" YNeuron :: Glayer pos1 est1 "<<meas_GXc[track][ln]<<" "<<proj_GXc[track][ln]<<std::endl;
         std::cout<<" YNeuron :: Glayer pos2 est2 "<<meas_GYc[track][ln]<<" "<<proj_GYc[track][ln]<<std::endl;
         std::cout<<" YNeuron :: Glayer pos3 est3 "<<meas_GZc[track][ln]<<" "<<proj_GZc[track][ln]<<std::endl;    
#endif
         if(ln>=nLAYER) continue;                                
         meas_S1c[track][ln] = yGEOM->GToS(chipID[ln],meas_GXc[track][ln],meas_GYc[track][ln],meas_GZc[track][ln])(0);
         meas_S2c[track][ln] = yGEOM->GToS(chipID[ln],meas_GXc[track][ln],meas_GYc[track][ln],meas_GZc[track][ln])(1);
         meas_S3c[track][ln] = yGEOM->GToS(chipID[ln],meas_GXc[track][ln],meas_GYc[track][ln],meas_GZc[track][ln])(2);

         proj_S1c[track][ln] = yGEOM->GToS(chipID[ln],proj_GXc[track][ln],proj_GYc[track][ln],proj_GZc[track][ln])(0);
         proj_S2c[track][ln] = yGEOM->GToS(chipID[ln],proj_GXc[track][ln],proj_GYc[track][ln],proj_GZc[track][ln])(1);
         proj_S3c[track][ln] = yGEOM->GToS(chipID[ln],proj_GXc[track][ln],proj_GYc[track][ln],proj_GZc[track][ln])(2);
#ifdef YSCNEURONDEBUG      
         if(TMath::Abs(meas_S1c[track][ln]-proj_S1c[track][ln])>0.5) {
            std::cout<<"YNeuron ZFIT ERROR"<<std::endl;         
         }
         std::cout<<" YNeuron :: Slayer pos1 est1 "<<meas_S1c[track][ln]<<" "<<proj_S1c[track][ln]<<std::endl;
         std::cout<<" YNeuron :: Slayer pos2 est2 "<<meas_S2c[track][ln]<<" "<<proj_S2c[track][ln]<<std::endl;
         std::cout<<" YNeuron :: Slayer pos3 est3 "<<meas_S3c[track][ln]<<" "<<proj_S3c[track][ln]<<std::endl;      
#endif                 
         b_Cost_Beam[ln]   = std::pow(meas_S1c[track][ln]-proj_S1c[track][ln],2)
                           + std::pow(meas_S2c[track][ln]-proj_S2c[track][ln],2);// + std::pow(meas_S3c[ln]-proj_S3c[ln],2);     
         Cost_Beam[track] += std::pow(meas_S1c[track][ln]-proj_S1c[track][ln],2)/std::pow(GetSigma(RecRadius, ln, DET_MAG, 0),2) 
                           + std::pow(meas_S2c[track][ln]-proj_S2c[track][ln],2)/std::pow(GetSigma(RecRadius, ln, DET_MAG, 1),2);
         NVALIDHITS[track]++;        

         double chi_s1 = (proj_S1c[track][ln]-meas_S1c[track][ln])/GetSigma(RecRadius, ln, DET_MAG, 0);
         double chi_s2 = (proj_S2c[track][ln]-meas_S2c[track][ln])/GetSigma(RecRadius, ln, DET_MAG, 1); 
         if(ln<nLAYERIB){
            if(std::abs(chi_s1)>RANGE_CHI_IB_TRAINING || std::abs(chi_s2)>RANGE_CHI_IB_TRAINING) nBadFits[track]++;
         } else {
            if(std::abs(chi_s1)>RANGE_CHI_OB_TRAINING || std::abs(chi_s2)>RANGE_CHI_OB_TRAINING) nBadFits[track]++;
         }
      }  
      // TOTAL CHI REJECTION
      if(Cost_Beam[track]>TrackRejection) continue;
      // INDIVIDUAL SENSOR CHECK
      if(nBadFits[track]>0) continue;
      // IP REJECTION

      double CircleXf = CircleXc - BeamCenter(0);
      double CircleYf = CircleYc - BeamCenter(1);

      TMatrix RotPhi(2,2);
      double angleRotPhi = Fitpar[0] > 0 ? -90*TMath::DegToRad() : +90*TMath::DegToRad();
      TMatrixD matRotPhi(2,2);
      matRotPhi[0] = { +std::cos(angleRotPhi), -std::sin(angleRotPhi)};
      matRotPhi[1] = { +std::sin(angleRotPhi), +std::cos(angleRotPhi)};
      TMatrixD matPhi[2];
      matPhi[0].ResizeTo(1,2);
      matPhi[0][0] = { CircleXf - Fitpar[2], CircleYf - Fitpar[3]};
      matPhi[0].T();
      matPhi[1].ResizeTo(2,1);
      matPhi[1] = matRotPhi * matPhi[0];  
      matPhi[1].T();      
      double vtx_phi   = std::atan2(matPhi[1][0][1], matPhi[1][0][0]);

      double gX_VERTEX    = Fitpar[2];      
      double gY_VERTEX    = Fitpar[3];      
      double gZ_VERTEX    = (parz[0])*(beta[nLAYER]) + (parz[1]);
      double gX_OUTERMOST = 0;      
      double gY_OUTERMOST = 0;      
      double gZ_OUTERMOST = 0;
      for(int lay = 0; lay<nLAYER; lay++){
         if(chipID[(nLAYER-1)-lay]>=0) {
            gX_OUTERMOST = proj_GXc[track][(nLAYER-1)-lay];
            gY_OUTERMOST = proj_GYc[track][(nLAYER-1)-lay];                     
            gZ_OUTERMOST = proj_GZc[track][(nLAYER-1)-lay];
           break;
         }
      }
      double deltaZ = gZ_OUTERMOST - gZ_VERTEX;
      double deltaR = std::sqrt(std::pow(gX_OUTERMOST - gX_VERTEX, 2) + std::pow(gY_OUTERMOST - gY_VERTEX, 2));   
      double vtx_theta = std::atan2(deltaR,deltaZ);
      
      // Track parametrization by Layer0.    
      double vxyz[] = {proj_GXc[track][0], proj_GYc[track][0], proj_GZc[track][0]};
   
      double FitFrameGx = proj_GXc[track][0] - BeamCenter(0);
      double FitFrameGy = proj_GYc[track][0] - BeamCenter(1);
   
      TMatrixD matPhiL0[2];
      matPhiL0[0].ResizeTo(1,2);
      matPhiL0[0][0] = { CircleXf - FitFrameGx, CircleYf - FitFrameGy};
      matPhiL0[0].T();
      matPhiL0[1].ResizeTo(2,1);
      matPhiL0[1] = matRotPhi * matPhiL0[0];  
      matPhiL0[1].T(); 
   
      double pT_L0     = RecMomentumT;
      double phi_L0    = std::atan2(matPhiL0[1][0][1], matPhiL0[1][0][0]);
      double theta_L0  = vtx_theta; //invariant for all layers
      double eta_L0    = -std::log(std::tan(theta_L0/2.));
      double p_L0      = pT_L0*std::cosh(eta_L0);
      double pz_L0     = pT_L0*std::sinh(eta_L0);
      double px_L0     = pT_L0*std::cos(phi_L0);
      double py_L0     = pT_L0*std::sin(phi_L0); 
   
      double pxyz[] = {px_L0, py_L0, pz_L0};
      int track_charge = Fitpar[0]>0 ? -1 : +1;
      YImpactParameter b_IP;
   
      b_IP.TrackParametrization(vxyz, pxyz, track_charge);
      // 1T = 10kG;
      double det_mag_in_kG = -10*DET_MAG;
      b_IP.getImpactParams(BeamCenter(0), BeamCenter(1), BeamCenter(2),det_mag_in_kG);
      //b_IP.getImpactParams(fvertex_TRKF.X(), fvertex_TRKF.Y(), fvertex_TRKF.Z(),det_mag_in_kG);
      //b_IP.getImpactParams(BeamProfileXZ1*BeamPos[2] + BeamProfileXZ0, BeamProfileYZ1*BeamPos[2] + BeamProfileYZ0, BeamPos[2],det_mag_in_kG);
      if(TMath::Abs(b_IP.ip[0])>RANGE_IMPACTPARAMS_R || TMath::Abs(b_IP.ip[1])>RANGE_IMPACTPARAMS_Z) continue;      
          
//
      // Numarical Calculation part // slope p0 p1

      // ds1_cs3
      int valid_nLAYER = 0;
      for(int Layer = 0; Layer < nLAYER + 1; Layer++){
         if(hitUpdateTrack[track][Layer]==true) valid_nLAYER++;      
      }      
      
      double z_meas_valid[valid_nLAYER];
      double beta_valid[valid_nLAYER];
      int sensorID_valid[valid_nLAYER];      
      int idxLayer_valid[valid_nLAYER];
      
      int lay=0;
      for(int Layer = 0; Layer < nLAYER + 1; Layer++){
         if(hitUpdateTrack[track][Layer]==true){ 
            sensorID_valid[lay]  = Layer < nLAYER ? vtxfit[track].chipID[Layer] : -9999;         
            z_meas_valid[lay]    = vtxfit[track].z_meas[Layer];
            beta_valid[lay]      = vtxfit[track].beta[Layer];
            idxLayer_valid[lay]  = Layer;
            lay++;
         }
      }

      // ds2_cs2 ds2_cs3
      double R_ref = 1/Fitpar[0];
      double Q_ref = Fitpar[1]+Fitpar[4];
            
      double pds2_FIT[nLAYER];
      double pds3_FIT[nLAYER];
      double pds2_FIT_parr[nLAYER][nLAYER];
      double pds3_FIT_parr[nLAYER][nLAYER];   
      double pds2_FIT_perp[nLAYER][nLAYER];
      double pds3_FIT_perp[nLAYER][nLAYER];                        
      for(int lnD = 0; lnD<nLAYER; lnD++){ 
         pds2_FIT[lnD] = 0.0;
         pds3_FIT[lnD] = 0.0;
         for(int lnT = 0; lnT<nLAYER; lnT++){ 
            pds2_FIT_parr[lnT][lnD] = 0.0;
            pds3_FIT_parr[lnT][lnD] = 0.0;   
            pds2_FIT_perp[lnT][lnD] = 0.0;
            pds3_FIT_perp[lnT][lnD] = 0.0;                                    
         }               
      }         
            
      for(int lnD = 0; lnD<nLAYER; lnD++){   
         double sign = R_ref > 0 ? +1 : -1;
               
         double dR_parr = paramRparr[lnD][0]*sign*( paramRparr[lnD][1] + paramRparr[lnD][2]*std::abs(R_ref)     + paramRparr[lnD][3]*std::abs(R_ref*R_ref) );
         double R_parr = R_ref + dR_parr;
         double dQ_parr = paramQparr[lnD][0]*sign*( paramQparr[lnD][1] + paramQparr[lnD][2]*(1/std::abs(R_ref)) + paramQparr[lnD][3]*(1/std::abs(R_ref*R_ref)) );
         double Q_parr = Q_ref + dQ_parr;              
               
         double dR_perp = paramRperp[lnD][0]*sign*( paramRperp[lnD][1] + paramRperp[lnD][2]*std::abs(R_ref)     + paramRperp[lnD][3]*std::abs(R_ref*R_ref) );
         double R_perp = R_ref + dR_perp;
         double dQ_perp = paramQperp[lnD][0]*sign*( paramQperp[lnD][1] + paramQperp[lnD][2]*(1/std::abs(R_ref)) + paramQperp[lnD][3]*(1/std::abs(R_ref*R_ref)) );
         double Q_perp = Q_ref + dQ_perp;    

      // parr component
         double RecR_parr      = std::abs(R_parr);
         double CircleXc_parr  = R_parr>0 ? RecR_parr*std::cos(Q_parr + 0.5*TMath::Pi()) : RecR_parr*std::cos(Q_parr - 0.5*TMath::Pi());   
         double CircleYc_parr  = R_parr>0 ? RecR_parr*std::sin(Q_parr + 0.5*TMath::Pi()) : RecR_parr*std::sin(Q_parr - 0.5*TMath::Pi()); 
         CircleXc_parr = CircleXc_parr + BeamCenter(0) + Fitpar[2];
         CircleYc_parr = CircleYc_parr + BeamCenter(1) + Fitpar[3];       

         TVector3 vecCircle_center_parr(CircleXc_parr, CircleYc_parr, 0);
         //use vecXc_meas, 
         TVector3 dirXc_meas_parr[nLAYER+1];
         TVector3 dirXc_proj_parr[nLAYER+1];
         for(int a=0; a<nLAYER+1;a++){
            dirXc_meas_parr[a] = vecXc_meas[a] - vecCircle_center_parr;
         }

         double beta_parr[nLAYER+1];
         for(int l = 0; l < nLAYER+1; l++){    
            beta_parr[l] = std::atan2(dirXc_meas_parr[l].Y(), dirXc_meas_parr[l].X());
         }     

         //beta linearization
         BetaLinearization(beta_parr, dirXc_meas_parr, hitUpdate);  
         TVector3 vecXc_proj_parr[nLAYER+1];            
         TVector3 vecXc_norm_parr[nLAYER+1];                  
         GetProjectionPoints(vecCircle_center_parr, RecR_parr, vecSensorNorm, vecXc_meas, vecXc_proj_parr, vecXc_norm_parr);

       // perp component
         double RecR_perp      = std::abs(R_perp);
         double CircleXc_perp  = R_perp>0 ? RecR_perp*std::cos(Q_perp + 0.5*TMath::Pi()) : RecR_perp*std::cos(Q_perp - 0.5*TMath::Pi());   
         double CircleYc_perp  = R_perp>0 ? RecR_perp*std::sin(Q_perp + 0.5*TMath::Pi()) : RecR_perp*std::sin(Q_perp - 0.5*TMath::Pi()); 
         CircleXc_perp = CircleXc_perp + BeamCenter(0) + Fitpar[2];
         CircleYc_perp = CircleYc_perp + BeamCenter(1) + Fitpar[3];

         TVector3 vecCircle_center_perp(CircleXc_perp, CircleYc_perp, 0);
         //use vecXc_meas, 
         TVector3 dirXc_meas_perp[nLAYER+1];
         TVector3 dirXc_proj_perp[nLAYER+1];
         for(int a=0; a<nLAYER+1;a++){
            dirXc_meas_perp[a] = vecXc_meas[a] - vecCircle_center_perp;
         }

         double beta_perp[nLAYER+1];
         for(int l = 0; l < nLAYER+1; l++){    
            beta_perp[l] = std::atan2(dirXc_meas_perp[l].Y(), dirXc_meas_perp[l].X());
         }     

         //beta linearization
         BetaLinearization(beta_perp, dirXc_meas_perp, hitUpdate);  
         TVector3 vecXc_proj_perp[nLAYER+1];            
         TVector3 vecXc_norm_perp[nLAYER+1];                  
         GetProjectionPoints(vecCircle_center_perp, RecR_perp, vecSensorNorm, vecXc_meas, vecXc_proj_perp, vecXc_norm_perp);  

         //std::cout<<"DERIVCHECK Case"<<ics<<" L"<<lay<<" |Delta"<<id<<"| = "<<delta[id]<<" NumericalDerivatives(parr) R Xc Yc -> Rd Xcd Ycd : "
         //<<R_ref<<" "<<CircleXc<<" "<<CircleYc<<" : "<<RecR_parr<<" "<<CircleXc_parr<<" "<<CircleYc_parr<<std::endl;
         //std::cout<<"DERIVCHECK Case"<<ics<<" L"<<lay<<" |Delta"<<id<<"| = "<<delta[id]<<" NumericalDerivatives(perp) R Xc Yc -> Rd Xcd Ycd : "
         //<<R_ref<<" "<<CircleXc<<" "<<CircleYc<<" : "<<RecR_perp<<" "<<CircleXc_perp<<" "<<CircleYc_perp<<std::endl;               
                                                                                   
         double proj_GXc_parr[nLAYER+1], proj_GYc_parr[nLAYER+1], proj_GZc_parr[nLAYER+1];
         double proj_S1c_parr[nLAYER+1], proj_S2c_parr[nLAYER+1], proj_S3c_parr[nLAYER+1];                 
         double proj_GXc_perp[nLAYER+1], proj_GYc_perp[nLAYER+1], proj_GZc_perp[nLAYER+1];
         double proj_S1c_perp[nLAYER+1], proj_S2c_perp[nLAYER+1], proj_S3c_perp[nLAYER+1];

         for(int lnT = 0; lnT < nLAYER+1; lnT++){   
               
            proj_GXc_parr[lnT] = RecR_parr*std::cos(beta_parr[lnT]) + CircleXc_parr;
            proj_GYc_parr[lnT] = RecR_parr*std::sin(beta_parr[lnT]) + CircleYc_parr;
            proj_GZc_parr[lnT] = proj_GZc[track][lnT]; 
               
            proj_GXc_perp[lnT] = RecR_perp*std::cos(beta_perp[lnT]) + CircleXc_perp;
            proj_GYc_perp[lnT] = RecR_perp*std::sin(beta_perp[lnT]) + CircleYc_perp;
            proj_GZc_perp[lnT] = proj_GZc[track][lnT]; 

            if(lnT>=nLAYER) continue;
            proj_S1c_parr[lnT] = yGEOM->GToS(chipID[lnT],proj_GXc_parr[lnT],proj_GYc_parr[lnT],proj_GZc_parr[lnT])(0);
            proj_S2c_parr[lnT] = yGEOM->GToS(chipID[lnT],proj_GXc_parr[lnT],proj_GYc_parr[lnT],proj_GZc_parr[lnT])(1);
            proj_S3c_parr[lnT] = yGEOM->GToS(chipID[lnT],proj_GXc_parr[lnT],proj_GYc_parr[lnT],proj_GZc_parr[lnT])(2);

            proj_S1c_perp[lnT] = yGEOM->GToS(chipID[lnT],proj_GXc_perp[lnT],proj_GYc_perp[lnT],proj_GZc_perp[lnT])(0);
            proj_S2c_perp[lnT] = yGEOM->GToS(chipID[lnT],proj_GXc_perp[lnT],proj_GYc_perp[lnT],proj_GZc_perp[lnT])(1);
            proj_S3c_perp[lnT] = yGEOM->GToS(chipID[lnT],proj_GXc_perp[lnT],proj_GYc_perp[lnT],proj_GZc_perp[lnT])(2);

            TVector3 dirN  = vecSensorNorm[lnT];
            TVector3 dirU  = GetCorrectedS2Vector(chipID[lnT]);
            TVector3 dirD  = dirXc_meas[lnT];
            double sigmaR      = std::acos((dirU.Dot(dirD))/(dirU.Mag()*dirD.Mag()));
            double sign_sigmaR = dirU.Cross(dirD).Z() > 0 ? +1 : -1;
            double rhoR        = std::acos((dirN.Dot(dirD))/(dirN.Mag()*dirD.Mag()));
            double sign_rhoR   = 0<=rhoR && rhoR<TMath::Pi()/2. ? +1 : -1;        
            
            double ParrToPerp = R_ref > 0 ? +90*TMath::DegToRad() : -90*TMath::DegToRad();
            TMatrixD MatParrToPerpR(2,2);
            MatParrToPerpR[0] = { +std::cos(ParrToPerp), -std::sin(ParrToPerp)};
            MatParrToPerpR[1] = { +std::sin(ParrToPerp), +std::cos(ParrToPerp)};
            
            // slope - Circle center correction
            TMatrixD Mat_dirDparr(2,1);
            Mat_dirDparr[0] = {dirD(0)};
            Mat_dirDparr[1] = {dirD(1)};
            TMatrixD Mat_dirDperp(2,1);   
            Mat_dirDperp = MatParrToPerpR * Mat_dirDparr;                             
            TVector3 dirDn(Mat_dirDperp[0][0], Mat_dirDperp[1][0],0);
            
            double sigmaP      = std::acos((dirU.Dot(dirDn))/(dirU.Mag()*dirDn.Mag()));
            double sign_sigmaP = dirU.Cross(dirDn).Z() > 0 ? +1 : -1;
            double rhoP        = std::acos((dirN.Dot(dirDn))/(dirN.Mag()*dirDn.Mag()));
            double sign_rhoP   = 0<=rhoP && rhoP<TMath::Pi()/2. ? +1 : -1;          
            
            pds2_FIT_parr[lnT][lnD] = (proj_S2c_parr[lnT] - proj_S2c[track][lnT])*(std::cos(sign_sigmaR*sign_rhoR*sigmaR)); 
            pds3_FIT_parr[lnT][lnD] = (proj_S3c_parr[lnT] - proj_S3c[track][lnT])*(std::sin(sign_sigmaR*sign_rhoR*sigmaR)); 
            pds2_FIT_perp[lnT][lnD] = (proj_S2c_perp[lnT] - proj_S2c[track][lnT])*(std::cos(sign_sigmaP*sign_rhoP*sigmaP)); 
            pds3_FIT_perp[lnT][lnD] = (proj_S3c_perp[lnT] - proj_S3c[track][lnT])*(std::sin(sign_sigmaP*sign_rhoP*sigmaP));    
         }                                
      }     
 
      for(int lnT = 0; lnT<nLAYER; lnT++){ 
         for(int lnD = 0; lnD<nLAYER; lnD++){ 
            pds2_FIT[lnT] += (pds2_FIT_parr[lnT][lnD] + pds2_FIT_perp[lnT][lnD]);
            pds3_FIT[lnT] += (pds3_FIT_parr[lnT][lnD] + pds3_FIT_perp[lnT][lnD]); 
         }               
      } 

//
      double derivatives_ds1cs1[valid_nLAYER][valid_nLAYER];                 
      double derivatives_ds1cs3[valid_nLAYER][valid_nLAYER];                 
      for(int layT = 0; layT < valid_nLAYER; layT++){
         for(int layD = 0; layD < valid_nLAYER; layD++){
            derivatives_ds1cs1[layT][layD] = 0.0;
            derivatives_ds1cs3[layT][layD] = 0.0;
         }      
         Getds1FITdcs1(layT, z_meas_valid, beta_valid, RecRadius, valid_nLAYER, derivatives_ds1cs1[layT]);               
         Getds1FITdcs3(layT, z_meas_valid, beta_valid, RecRadius, valid_nLAYER, derivatives_ds1cs3[layT]);      
      }                 

      for(int Layer = 0; Layer < nLAYER; Layer++){    
         if(chipID[Layer]<0) continue;
         for(int Axis = 0; Axis<3; Axis++){
            if(ROutputLayerTrain[(3*Layer)+Axis]==0){
               SetSCNeuronDcdw((3*Layer)+Axis,track,0.0);                      
               continue;
            }          
            switch (Axis) {
               case 0 : {
                  if(chipID[Layer]<0){
                     DCost = 0.0;
                     break;
                  }      
                  
                  int lnD = Layer;
                  double Derivatives = 0;   
                  
                  for(int lnT = 0; lnT < valid_nLAYER; lnT++){
                  
                     int lnT_valid = idxLayer_valid[lnT];
                     double CorrCostC1 = (1/std::pow(GetSigma(RecRadius, lnT_valid, DET_MAG, 0),2)); 
//                            CorrCostC1 = (CorrCostC1<1) ? CorrCostC1: 1;                      
                     //std::cout<<"ds1FIT/dcs1(Layer "<<lnT_valid<<") = "<<derivatives_ds1cs1[lnT][lnD]<<" "<<CorrCostC1<<std::endl;
                     double sign_ds1 = (proj_S1c[track][lnT_valid]-meas_S1c[track][lnT_valid])>0 ? +1 : -1;
                     double del_ds1  = TMath::Abs(proj_S1c[track][lnT_valid]-meas_S1c[track][lnT_valid])>ValidWindow ? sign_ds1*ValidWindow : proj_S1c[track][lnT_valid]-meas_S1c[track][lnT_valid];     
                                              
                     if(idxLayer_valid[lnD]==lnT_valid) {
                        Derivatives += CorrCostC1*del_ds1*(derivatives_ds1cs1[lnT][lnD]-1);
                     } else {
                        Derivatives += CorrCostC1*del_ds1*(derivatives_ds1cs1[lnT][lnD]);
                     }
                  }        
                                           
                  DCost    = Derivatives;                   
                  //std::cout<<"[DCost] C"<<chipID[Layer]<<" Layer Axis "<<Layer<<" "<<Axis<<" "
                  //<<proj_S1c[track][Layer]<<" "<<meas_S1c[track][Layer]<<" Sigma : "<<GetSigma(RecRadius, Layer, DET_MAG, 0)<<std::endl;
                  break;
               }
               case 1 : { 
                  if(chipID[Layer]<0){
                     DCost = 0.0;
                     break;
                  }
                  
                  int lnD = Layer;
                  double Derivatives = 0;

                  for(int lnT = 0; lnT < nLAYER; lnT++){
                     if(hitUpdateTrack[track][lnT]==true){ 
                        double CorrCostC2 = (1/std::pow(GetSigma(RecRadius, lnT, DET_MAG, 1),2)); 
//                               CorrCostC2 = (CorrCostC2<1) ? CorrCostC2: 1;                       
                        double sign_ds2 = (proj_S2c[track][lnT]-meas_S2c[track][lnT])>0 ? +1 : -1;
                        double del_ds2  = TMath::Abs(proj_S2c[track][lnT]-meas_S2c[track][lnT])>ValidWindow ? sign_ds2*ValidWindow : proj_S2c[track][lnT]-meas_S2c[track][lnT];                     
                        if(lnD==lnT)   Derivatives += CorrCostC2*del_ds2*(pds2_FIT_parr[lnT][lnD] + pds2_FIT_perp[lnT][lnD] - 1);
                        else           Derivatives += CorrCostC2*del_ds2*(pds2_FIT_parr[lnT][lnD] + pds2_FIT_perp[lnT][lnD]);
                     }
                  }                  
                  DCost    = Derivatives;
                  //std::cout<<"[DCost] C"<<chipID[Layer]<<" Layer Axis "<<Layer<<" "<<Axis<<" "
                  //<<proj_S2c[track][Layer]<<" "<<meas_S2c[track][Layer]<<" Sigma : "<<GetSigma(RecRadius, Layer, DET_MAG, 1)<<" Derivatives : "<<DCost<<std::endl;
                  break;
               }
               case 2 : {
                  if(chipID[Layer]<0){
                     DCost = 0.0;
                     break;
                  }

                  int lnD = Layer;
                  double Derivatives1 = 0;
       
                  for(int lnT = 0; lnT < valid_nLAYER; lnT++){
                  
                     int lnT_valid = idxLayer_valid[lnT];
                     double CorrCostC1 = (1/std::pow(GetSigma(RecRadius, lnT_valid, DET_MAG, 0),2)); 
//                            CorrCostC1 = (CorrCostC1<1) ? CorrCostC1: 1;                      
                     //std::cout<<"ds1FIT/dcs3(Layer "<<lnT_valid<<") = "<<derivatives_ds1cs3[lnT][lnD]<<" "<<CorrCostC1<<std::endl;
                     double sign_ds1 = (proj_S1c[track][lnT_valid]-meas_S1c[track][lnT_valid])>0 ? +1 : -1;
                     double del_ds1  = TMath::Abs(proj_S1c[track][lnT_valid]-meas_S1c[track][lnT_valid])>ValidWindow ? sign_ds1*ValidWindow : proj_S1c[track][lnT_valid]-meas_S1c[track][lnT_valid];     
                                              
                     if(idxLayer_valid[lnD]==lnT_valid) {
                        Derivatives1 += CorrCostC1*del_ds1*(derivatives_ds1cs3[lnT][lnD]*IncS1[lnT]);
                     } else {
                        Derivatives1 += CorrCostC1*del_ds1*(derivatives_ds1cs3[lnT][lnD]*IncS1[lnT]);
                     }
                  }  

                  double Derivatives2 = 0;
                  for(int lnT = 0; lnT < nLAYER; lnT++){
                     if(hitUpdateTrack[track][lnT]==true){ 
                        double CorrCostC2 = (1/std::pow(GetSigma(RecRadius, lnT, DET_MAG, 1),2)); 
//                               CorrCostC2 = (CorrCostC2<1) ? CorrCostC2: 1;       
                        //std::cout<<"ds2FIT/dcs3(Layer "<<lnT<<") = "<<pds3_FIT_parr[lnT][lnD] + pds3_FIT_perp[lnT][lnD]<<" "<<CorrCostC2<<std::endl;
                        double sign_ds2 = (proj_S2c[track][lnT]-meas_S2c[track][lnT])>0 ? +1 : -1;
                        double del_ds2  = TMath::Abs(proj_S2c[track][lnT]-meas_S2c[track][lnT])>ValidWindow ? sign_ds2*ValidWindow : proj_S2c[track][lnT]-meas_S2c[track][lnT];                     
                        if(lnD==lnT)   Derivatives2 += CorrCostC2*del_ds2*(pds3_FIT_parr[lnT][lnD] + pds3_FIT_perp[lnT][lnD]);
                        else           Derivatives2 += CorrCostC2*del_ds2*(pds3_FIT_parr[lnT][lnD] + pds3_FIT_perp[lnT][lnD]);
                     }
                  }                 

                  DCost  = - (Derivatives1 + Derivatives2); 
                  
                  //std::cout<<"[DCost] C"<<chipID[Layer]<<" Layer Axis "<<Layer<<" "<<Axis<<" "<<proj_S3c[track][Layer]<<" "<<meas_S3c[track][Layer]<<std::endl;
                  break;
               }
               default : {
                  DCost = 0; 
               }
            }
            if(IsAccessorialTrack[track]==true) {
               SetSCNeuronDcdw((3*Layer)+Axis,track,0);   
            } else {
               SetSCNeuronDcdw((3*Layer)+Axis,track,DCost);   
            }         
         }
      }     
   }

      
   /*
   for(int track = 0; track < ntracks; track++){
      if(IsAccessorialTrack[track]==true) continue;
      if(NVALIDHITS[track]<nLAYER) continue;      
      if(vtxfit[track].chipID[0]!=0) continue;
      std::cout<<"[COSTMONITOR] TRACK "<<track<<" COST "<<Cost_Beam[track]<<" ";
      for(int lay =0; lay < nLAYER; lay++){
         std::cout<<GetSCNeuronDcdw(3*lay+0,track)<<" "<<GetSCNeuronDcdw(3*lay+1,track)<<" "<<GetSCNeuronDcdw(3*lay+2,track)<<" ";
      }
      std::cout<<"[POSMONITOR]";
      for(int lay =0; lay < nLAYER; lay++){
         std::cout<<" I "<<input_nS1[track][lay]<<" "<<input_nS2[track][lay]<<
                    " M "<<meas_S1c[track][lay]<<" "<<meas_S2c[track][lay]<<" "<<meas_S3c[track][lay]<<
                    " F "<<proj_S1c[track][lay]<<" "<<proj_S2c[track][lay]<<" "<<proj_S3c[track][lay]<<" ";
      }     
      std::cout<<"[RESOLUTIONMONITOR]";
      for(int lay =0; lay < nLAYER; lay++){
         std::cout<<" "<<GetSigma(vtxfit[track].Radius, lay, DET_MAG, 0)<<" "<<GetSigma(vtxfit[track].Radius, lay, DET_MAG, 1)<<" ";
      }      
      std::cout<<"[RESRADIUSMONITOR]";      
      std::cout<<" "<<vtxfit[track].Radius<<" ";
      std::cout<<std::endl;
   }   
   */
}

void YMultiLayerPerceptron::SetSCNeuronDcdw(int i, int j, double x)
{
   fSCNeuronDcdw[i][j] = x;                             
}

 
Double_t YMultiLayerPerceptron::GetCost_Vertex_CircleFit()
{
   double Cost=0;
   
   double Cost_vertex_track = 0; // track <-> track_mean(weighted)
   double Cost_vertex_event = 0; // event <-> track_mean(weighted);  
   
   int    nhits = 0;
#ifdef YMLPDEBUG0   
   std::cout<<"GetCost_Vertex_CircleFit"<<std::endl;
#endif  
   double z_event = 0;
   bool   valid_event = false;
   
   double z_track[nTrackMax];
   double z_track_mean = 0;
   double z_track_weight = 0;
   for(int t = 0; t< nTrackMax; t++){
      z_track[t]=0;   
      if(fvertex[t].valid==false) continue;
      
      if(valid_event==false){      
         z_event = fvertex_TRKF.Z();//fvertex[t].z_meas[nLAYER];
         valid_event = true;
      }
      z_track[t] = (fvertex[t].parz[0])*(fvertex[t].beta[nLAYER]) + (fvertex[t].parz[1]);       
#ifdef YMLPDEBUG0         
      std::cout<<"Track "<<t<<std::endl;
         
      std::cout<<" z :: "; 

      for(int l = 0; l< nLAYER; l++){
         std::cout<< fvertex[t].z_meas[l] <<" ";
      }
      std::cout<<std::endl;
      std::cout<<" beta :: ";
      for(int l = 0; l< nLAYER; l++){
         std::cout<< fvertex[t].beta[l] <<" ";
      }
      std::cout<<std::endl;
 
      std::cout<<" parz :: "<<fvertex[t].parz[0]<<" "<<fvertex[t].parz[1]<<std::endl;
      std::cout<<" R :: "<<fvertex[t].Radius<<" "<<fvertex[t].valid<<std::endl;
      
      std::cout<< " z_event "<<z_event<<std::endl;

      std::cout<< " z_track "<<z_track[t]<<std::endl;
#endif      
      z_track_mean   += z_track[t]/std::pow(GetSigma(fvertex[t].Radius, nLAYER, DET_MAG, 0),2);
      z_track_weight += 1/std::pow(GetSigma(fvertex[t].Radius, nLAYER, DET_MAG, 0),2);
      nhits++;
   }  
   z_track_mean = z_track_mean/z_track_weight;
#ifdef YMLPDEBUG0         
   std::cout<<"z_track(mean) "<<z_track_mean<<std::endl;
#endif  
   for(int t = 0; t< nTrackMax; t++){
      if(fvertex[t].valid==false) continue;
      Cost_vertex_track += std::pow(z_track[t] - z_track_mean,2)/std::pow(Sigma_MEAS[0][nLAYER]*1e-4,2);
   }
   
   Cost_vertex_event = std::pow(z_event - z_track_mean,2)/std::pow(Sigma_MEAS[0][nLAYER]*1e-4,2);
   
   Cost = Cost_vertex_track + Cost_vertex_event;
   //std::cout<<"[monitor] vertex cost = track + event = "<<Cost_vertex_track<<" "<<Cost_vertex_event<<" = "<<Cost<<std::endl;
   return Cost;
}           

void YMultiLayerPerceptron::InitVertex_CircleFit()
{
   for(int t = 0; t< nTrackMax; t++){
      for(int l = 0; l< nLAYER+1; l++){
         fvertex[t].z_meas[l] = 0.0;
         fvertex[t].beta[l]   = 0.0;
      }
               
      fvertex[t].parz[0] = 0.0;
      fvertex[t].parz[1] = 0.0;      
      fvertex[t].Radius  = 0.0;
      fvertex[t].valid   = false;
   }
}             
void YMultiLayerPerceptron::AddVertex_CircleFit(int track, double* z, double* beta, double* parz, double Radius, bool valid)
{
   for(int l = 0; l< nLAYER+1; l++){
      fvertex[track].z_meas[l] = z[l];
      fvertex[track].beta[l]   = beta[l];
   }
     
   fvertex[track].parz[0] = parz[0];
   fvertex[track].parz[1] = parz[1]; 
   fvertex[track].Radius  = Radius;
   fvertex[track].valid   = valid;   
}

void YMultiLayerPerceptron::EventCheck()
{
   std::cout<<"[YMLP] EventCheck"<<std::endl;
   int nEvents = fData->GetEntriesFast();   
   int NLastLayer = fLastLayer.GetEntriesFast();
   for (int i = 0; i < nEvents; i++) {
      fData->GetEntry(i);
      std::cout<<"[YMLP] Event : "<<i<<" nTracks : "<<fEventIndex[i][1]<<std::endl;
      for(int j =0; j < fEventIndex[i][1]; j++){
         std::cout<<"[YMLP]  Track : "<<j<<std::endl;
         double extended[NLastLayer];
         for(int k = 0; k < NLastLayer; k++) {
            YNeuron *neuron_out = (YNeuron *) fLastLayer.At(k);
            neuron_out->SetNewEvent();               
            neuron_out->SetNeuronIndex(j);  
            extended[k] = neuron_out->GetBranch();
         }   
         int staveIndex[nLAYER], chipIndex[nLAYER], chipID[nLAYER];
         for(int ln = 0; ln<nLAYER; ln++){             
            staveIndex[ln] = (int)extended[(3*ln)+0];   
            chipIndex[ln]  = (int)extended[(3*ln)+1];  
            chipID[ln]     = (int)extended[(3*ln)+2];  
            std::cout<<"[YMLP]  layer stave chip ID "<<ln<<" "<<staveIndex[ln]<<" "<<chipIndex[ln]<<" "<<chipID[ln]<<" "<<std::endl;
         }  
      }
   } 
}

void YMultiLayerPerceptron::SetFitModel(int x)
{
   fFitModel = x;
}

void YMultiLayerPerceptron::SetRandomSeed(int x){
   fRandomSeed = x;
}
   
void YMultiLayerPerceptron::SetLayerTrain(int x){
   fLayerTrain = x;
} 

void YMultiLayerPerceptron::ComputeTrackProfile()
{
#ifdef YMLPComputeDCDwDEBUG
   std::cout<<"YMultiLayerPerceptron::ComputeTrackProfile"<<std::endl;
#endif

   Int_t i,j;
   Int_t nentries;
   
   YSynapse *synapse;
   YNeuron *neuron;

   YNeuron *neuron_O; 

   fCostMonitor->Reset();   
   fBeamXY->Reset();   
   fBeamZR->Reset(); 
   fVertexFitXY->Reset(); 
   fVertexFitZR->Reset(); 
   fVertexXY->Reset(); 
   fVertexZR->Reset();  
   
   for(int pstv = 0; pstv < 20; pstv++) {
      fCostChargeSymSum[pstv] = 0;   
      fCostChargeSymNtr[pstv] = 0;  
   }
   fCostChargeSym->Reset();
   fChargeSymMonitorPositive->Reset();
   fChargeSymMonitorNegative->Reset();      
#ifdef MONITORONLYUPDATES   
   fUPDATESENSORS->Reset();   
   fUPDATETRACKS->Reset();
#endif

   for(int l = 0; l < nLAYER; l++){    
      fChi2Layer[l]->Reset();  
      fpTvsResLayer[l][0]->Reset();  
      fpTvsResLayer[l][1]->Reset();        
      fpTvsChiLayer[l][0]->Reset();  
      fpTvsChiLayer[l][1]->Reset();     
#ifdef MONITORHALFSTAVEUNIT             
      int nHalfBarrel = 2;
      for(int hb = 0; hb < nHalfBarrel; hb++){
         int nHalfStave = NSubStave[l]; 
         for(int hs = 0; hs < nHalfStave; hs++){
            fpTvsResLayerHBHS[l][hb][hs][0]->Reset(); 
            fpTvsResLayerHBHS[l][hb][hs][1]->Reset(); 
            fpTvsChiLayerHBHS[l][hb][hs][0]->Reset(); 
            fpTvsChiLayerHBHS[l][hb][hs][1]->Reset();          
            fResidualsVsZLayerHBHS[l][hb][hs][0]->Reset();
            fResidualsVsZLayerHBHS[l][hb][hs][1]->Reset();
            fResidualsVsPhiLayerHBHS[l][hb][hs][0]->Reset();
            fResidualsVsPhiLayerHBHS[l][hb][hs][1]->Reset();
            fProfileVsZLayerHBHS[l][hb][hs][0]->Reset();
            fProfileVsZLayerHBHS[l][hb][hs][1]->Reset();
            fProfileVsPhiLayerHBHS[l][hb][hs][0]->Reset();
            fProfileVsPhiLayerHBHS[l][hb][hs][1]->Reset();
            fSensorCenterVsZLayerHBHS[l][hb][hs][0]->Reset();
            fSensorCenterVsZLayerHBHS[l][hb][hs][1]->Reset();
            fSensorCenterVsPhiLayerHBHS[l][hb][hs][0]->Reset();
            fSensorCenterVsPhiLayerHBHS[l][hb][hs][1]->Reset();
         }       
      }
#endif        
   }           

   for(int ichipID = 0; ichipID < nSensors; ichipID++){
      fSCNetwork[NetworkChips[ichipID]]->InitResProfile();
#ifdef MONITORSENSORUNITpT               
      fSCNetwork[NetworkChips[ichipID]]->InitpTvsRes();                       
      fSCNetwork[NetworkChips[ichipID]]->InitpTvsChi();                       
      fSCNetwork[NetworkChips[ichipID]]->InitChi2();     
#endif      
      fSCNetwork[NetworkChips[ichipID]]->SetnEvents(0);      
   }                             
    
   if (fTraining) {
      Int_t nEvents = fTrainingIndex.size();//fTraining->GetN();
      std::cout<<"nEvents : "<<nEvents<<std::endl;
      for (i = 0; i < nEvents; i++) {  //Event loop
         GetEntry(fTraining->GetEntry(i));
         int Npronged = fTrainingIndex[i][1];

         int staveIndex[Npronged][nLAYER], chipIndex[Npronged][nLAYER], chipID[Npronged][nLAYER];
  	 double extended[Npronged][nLAYER][3];      
  	 double addition[Npronged][8];      
         // Define LFE error before compute dedw by event
         //std::cout<<"# Start DCDw Computation From Cost Function 1 "<<std::endl; 
         for(int t = 0; t < Npronged; t++){    
            for(int j = 0; j < 3*nLAYER; j++) {             
               neuron_O = (YNeuron *) fLastLayer.At(j); 
               neuron_O->SetNeuronIndex(t); 
               neuron_O->SetNewEvent(); 
               neuron_O->SetLFENodeIndex((3*nLAYER)*t+j);      
               neuron_O->SetLFELayerTrain(fLayerTrain); 
               int p1 = (int)(j/3);
               int p2 = (int)(j%3);        
               extended[t][p1][p2] = neuron_O->GetBranch();               						//Very Important !!!!!! 2020 08 26           
            }       
            for(int ln = 0; ln < nLAYER; ln ++){
               staveIndex[t][ln] = extended[t][ln][0];  
               chipIndex[t][ln]  = extended[t][ln][1];  
               chipID[t][ln]     = extended[t][ln][2];  
            }            
         }

         for(int t = 0; t < Npronged; t++){

            bool IsAccessorial = (fSplitReferenceSensor==-1) ? false : true;
            for(int ln = 0; ln < nLAYER; ln ++){
               if(chipID[t][ln] >= 0 && fSplitReferenceSensor == chipID[t][ln]) IsAccessorial=false;      
            } 
            if(IsAccessorial==true) continue;  
         
            for(int ln = 0; ln < nLAYER; ln ++){
               if(chipID[t][ln]<0) continue;
               fSCNetwork[chipID[t][ln]]->SetnEvents(fSCNetwork[chipID[t][ln]]->GetnEvents()+1); 
               if(fLearningMethod==YMultiLayerPerceptron::kOffsetTuneByMean) fWUpdatebyMean[chipID[t][ln]] = true;                
            }               
         } 
      }                                   
   } else if (fData) {

   }
   if(fLearningMethod==YMultiLayerPerceptron::kOffsetTuneByMean) {
      std::cout<<"PrepareDumpResiduals OffsetTuneByMean"<<std::endl;   
      for(int iID=0; iID<nSensors; iID++){    
         if(fWUpdatebyMean[iID]==false) {
            fOffsetTuning.push_back(new YOffsetTuneByMean());
         } else {   

            int l = yGEOM->GetLayer(iID);      
            int    nbinsII = l<3 ? 100 : 4;
            double rangeII = 0.5;   

            TF1* bds1_s1 = new TF1(Form("fds1_s1_Chip%d",NetworkChips[iID]),"pol1",-rangeII,+rangeII);
            TF1* bds1_s2 = new TF1(Form("fds1_s2_Chip%d",NetworkChips[iID]),"pol1",-rangeII,+rangeII);
            TF1* bds2_s1 = new TF1(Form("fds2_s1_Chip%d",NetworkChips[iID]),"pol1",-rangeII,+rangeII);
            TF1* bds2_s2 = new TF1(Form("fds2_s2_Chip%d",NetworkChips[iID]),"pol1",-rangeII,+rangeII);
            fOffsetTuning.push_back(new YOffsetTuneByMean(bds1_s1, bds1_s2, bds2_s1, bds2_s2));
         }
      }    
   }   
}
   
////////////////////////////////////////////////////////////////////////////////
/// Compute the DCDw = sum on all training events of dedw for each weight
/// normalized by the number of events.

void YMultiLayerPerceptron::ComputeDCDw() //const
{
#ifdef YMLPComputeDCDwDEBUG
   std::cout<<"YMultiLayerPerceptron::ComputeDCDw"<<std::endl;
#endif
   Int_t i,j;
   Int_t nentries;
   
   YSynapse *synapse;
   YNeuron *neuron;

   YNeuron *neuron_I;
   YNeuron *neuron_O; 
   YNeuron *neuron_A;
      
   nentries = fSynapses.GetEntriesFast();
   for (i=0;i<nentries;i++) {
      synapse = (YSynapse *) fSynapses.At(i);
      synapse->SetDCDw(0.);
   }
   nentries = fNetwork.GetEntriesFast();
   for (i=0;i<nentries;i++) {
      neuron = (YNeuron *) fNetwork.At(i);
      neuron->SetFitModel(FITMODEL);
      neuron->SetDCDw(0.);
   }

   for(int ichipID = 0; ichipID < nSensors; ichipID++){
      nentries = fSCNetwork[NetworkChips[ichipID]]->GetSynapses().GetEntriesFast();
      for (j=0;j<nentries;j++) {
         synapse = (YSynapse *) fSCNetwork[NetworkChips[ichipID]]->GetSynapses().At(j);   
         synapse->SetDCDw(0.); 
      }      
      nentries = fSCNetwork[NetworkChips[ichipID]]->GetNetwork().GetEntriesFast();
      for (j=0;j<nentries;j++) {
         neuron = (YNeuron *) fSCNetwork[NetworkChips[ichipID]]->GetNetwork().At(j);   
         neuron->SetFitModel(FITMODEL);
         neuron->SetDCDw(0.);     
      }    
   }                             
    
   
   Double_t eventWeight = 1.;
   if (fTraining) {
      Int_t nEvents = fTrainingIndex.size();//fTraining->GetN();
      std::cout<<"nEvents : "<<nEvents<<std::endl;
      for (i = 0; i < nEvents; i++) {  //Event loop
         GetEntry(fTraining->GetEntry(i));
         int Npronged = fTrainingIndex[i][1];
         
         Double_t mlpInputCenter[Npronged][nLAYER][2];  //layer input
         Double_t mlpOutputCenter[Npronged][nLAYER][3]; //layer output
         
         int staveIndex[Npronged][nLAYER], chipIndex[Npronged][nLAYER], chipID[Npronged][nLAYER];
         double input[Npronged][nLAYER][2], output[Npronged][nLAYER][3];  //prong layer axis
  	 double extended[Npronged][nLAYER][3];      
  	 double addition[Npronged][8];      
         eventWeight = fEventWeight->EvalInstance();
         eventWeight *= fCurrentTreeWeight;
         // Define LFE error before compute dedw by event
         //std::cout<<"# Start DCDw Computation From Cost Function 1 "<<std::endl; 
         for(int t = 0; t < Npronged; t++){
            //std::cout<<"(Cost 1)Event:: "<<i<<"-"<<t<<std::endl;         
            //std::cout<<"(ComputeDCDw) Event i entry track Index1 Index2:: "<<i<<" "<<t<<" "<<fTrainingIndex[i][0]<<" "<<fTrainingIndex[i][1]<<std::endl;     
            for(int j = 0; j < 8; j++) {             
               neuron_A = (YNeuron *) fAddition.At(0); 
               neuron_A->SetNeuronIndex(1000 + j);  
               neuron_A->SetNewEvent();
               addition[t][j] = neuron_A->GetBranchAddition();
#ifdef YMLPComputeDCDwDEBUG
               std::cout<<     addition[t][j] <<" ";
#endif         
            }  
#ifdef YMLPComputeDCDwDEBUG   
            std::cout<<std::endl;
#endif         
            for(int j=0; j<fNetwork.GetEntriesFast(); j++) {
               neuron = (YNeuron *)fNetwork.At(j);
               neuron->SetNeuronIndex(t);                
               neuron->SetNewEvent();
            }         
            for(int j = 0; j < 2*nLAYER; j++){               
               neuron_I = (YNeuron *) fFirstLayer.At(j);  
               neuron_I->SetNeuronIndex(t); 
               neuron_I->SetNewEvent(); 
               int p1 = (int)(j/2);
               int p2 = (int)(j%2);               
               input[t][p1][p2] = neuron_I->GetValue();
            }    
            for(int j = 0; j < 3*nLAYER; j++) {             
               neuron_O = (YNeuron *) fLastLayer.At(j); 
               neuron_O->SetNeuronIndex(t); 
               neuron_O->SetNewEvent(); 
               neuron_O->SetLFENodeIndex((3*nLAYER)*t+j);      
               neuron_O->SetLFELayerTrain(fLayerTrain); 
               int p1 = (int)(j/3);
               int p2 = (int)(j%3);        
               extended[t][p1][p2] = neuron_O->GetBranch();               						//Very Important !!!!!! 2020 08 26           
            }       
            for(int ln = 0; ln < nLAYER; ln ++){
               staveIndex[t][ln] = extended[t][ln][0];  
               chipIndex[t][ln]  = extended[t][ln][1];  
               chipID[t][ln]     = extended[t][ln][2];  
#ifdef YMLPComputeDCDwDEBUG
               std::cout<<" Layer "<<ln
                        <<" Stave "<<staveIndex[t][ln]
                        <<" Chip " << chipIndex[t][ln] 
                        <<" ID "   << chipID[t][ln]<<std::endl;
#endif
            } 
            //load Network -> YSensorCorrection
            for(int ln = 0; ln < nLAYER; ln ++){
               if(chipID[t][ln]<0) continue;               
               for(int j=0; j<fSCNetwork[chipID[t][ln]]->GetNetwork().GetEntriesFast(); j++) {
                  neuron = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetNetwork().At(j); 
                  neuron->SetNeuronIndex(t);                  
                  neuron->SetNewEvent();
               }  
                    
               for(int j = 0; j < 2; j++) {
                  neuron_I = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetFirstLayer().At(j); 
                  neuron_I->SetLFENodeIndex((3*nLAYER)*t+3*ln+j); 
                  neuron_I->ClearLFEmemory();  
               }
               
               for(int j = 0; j < 3; j++) {
                  neuron_O = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetLastLayer()[j]; 
                  int p1 = (int)(ln);    
                  int p2 = (int)(j%3);                       
                  output[t][p1][p2] = Evaluate(p2, input[t][p1], chipID[t][ln]);
                  neuron_O->SetLFENodeIndex((3*nLAYER)*t+3*ln+j);
                  neuron_O->ClearLFEmemory();   
                  neuron_O->SetLFELayerTrain(fLayerTrain);                                        
                  neuron_O = 0;   
               }
               for(int c=0; c<2; c++){
                  mlpInputCenter[t][ln][c]=0;
               }
               for(int c=0; c<3; c++){
                  mlpOutputCenter[t][ln][c] = Evaluate(c, mlpInputCenter[t][ln], chipID[t][ln]); 
               }  
            }
         }
         for(int t = 0; t < Npronged; t++){
            for(int ln = 0; ln < nLAYER; ln ++){
               if(chipID[t][ln]<0) continue;            
               for(int j = 0; j < 2; j++) {         
                  neuron_I = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetFirstLayer().At(j); 
                  for(int k = 0; k < 2*nLAYER; k++){    
                     int p1 = (int)(k/2);
                     int p2 = (int)(k%2);    
                     for(int u = 0; u < Npronged; u++){            
                        neuron_I->SetLFEinput(3*p1 + p2, u, input[u][p1][p2]);  
                     }                             
                  }        
                  for(int k =0; k < 1*nLAYER; k++){						
                     for(int u = 0; u < Npronged; u++){                                 
                        neuron_I->SetLFEinput(3*k + 2, u,0);
                     }
                  }                   
               } 
               for(int j = 0; j < 3; j++) {         
                  neuron_O = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetLastLayer().At(j);
                  for(int k = 0; k < 2*nLAYER; k++){    
                     int p1 = (int)(k/2);
                     int p2 = (int)(k%2);    
                     for(int u = 0; u < Npronged; u++){            
                        neuron_O->SetLFEinput(3*p1 + p2, u, input[u][p1][p2]);  
                     }                             
                  }
                  for(int k =0; k < 1*nLAYER; k++){						
                     for(int u = 0; u < Npronged; u++){                                 
                        neuron_O->SetLFEinput(3*k + 2, u,0);
                     }
                  }                   
                  for(int k = 0; k < 3*nLAYER; k++){
                     int p1 = (int)(k/3);                        
                     int p2 = (int)(k%3); 
                     for(int u = 0; u < Npronged; u++){                 
                        neuron_O->SetLFEoutput(k,u,output[u][p1][p2]);   
                        neuron_O->SetLFEextended(k,u,extended[u][p1][p2]);  //prong layer axis
                     }  
                  } 
                  for(int k =0; k <8; k++){						
                     for(int u = 0; u < Npronged; u++){                                        
                        neuron_O->SetLFEaddition(k, u, addition[u][k]);
                     }
                  }                  
               }   
            }
            for(int j=0; j<fNetwork.GetEntriesFast(); j++) {
               neuron = (YNeuron *)fNetwork.At(j);
               neuron->SetNewEvent();
            }                   
            for(int ln = 0; ln < nLAYER; ln ++){
               if(chipID[t][ln]<0) continue;
               for(int j=0; j<fSCNetwork[chipID[t][ln]]->GetNetwork().GetEntriesFast(); j++) {
                  neuron = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetNetwork().At(j); 
                  neuron->SetNewEvent();
                  //neuron->SetNeuronIndex(-1);                  
               }      
               
               for(int j = 0; j < 3; j++) {
                  neuron_O = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetLastLayer().At((int)(j%3));
                  neuron_O->SetLFENodeIndex((3*nLAYER)*t+3*ln+j);                              
                  neuron_O->SetNeuronIndex(-1);   
                  //neuron_O->SetNewEvent(); 
               }    
            } 
         }
         InitSCNeuronDcdw();
         CalculateEventDcdw(Npronged);         
         for(int t = 0; t < Npronged; t++){
         
            bool IsAccessorial = (fSplitReferenceSensor==-1) ? false : true;
            for(int ln = 0; ln < nLAYER; ln ++){
               if(chipID[t][ln] >= 0 && fSplitReferenceSensor == chipID[t][ln]) IsAccessorial=false;      
            } 
            if(IsAccessorial==true) continue;           
         
            for(int ln = 0; ln < nLAYER; ln ++){
               if(chipID[t][ln]<0) continue;
               for(int j = 0; j < 2; j++) {         
                  neuron_I = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetFirstLayer().At(j); 
                  neuron_I->InitSCNeuronDcdw();
                  neuron_I->DumpSCNeuronDcdw(fSCNeuronDcdw);    
               } 
               for(int j = 0; j < 3; j++) {         
                  neuron_O = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetLastLayer().At(j);
                  neuron_O->InitSCNeuronDcdw();
                  neuron_O->DumpSCNeuronDcdw(fSCNeuronDcdw);    
                  neuron_O->SetNeuronIndex(-1);                     
               }  
               nentries = fSCNetwork[chipID[t][ln]]->GetSynapses().GetEntriesFast();    
               for (j=0;j<nentries;j++) {
                  synapse = (YSynapse *) fSCNetwork[chipID[t][ln]]->GetSynapses().At(j);
                  synapse->SetDCDw( synapse->GetDCDw() + (synapse->GetDcDw()*eventWeight));
               }
               nentries = fSCNetwork[chipID[t][ln]]->GetNetwork().GetEntriesFast();
               for (j=0;j<nentries;j++) {
                  neuron = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetNetwork().At(j);
                  if(j>=2){ //No Input Update
                     neuron->SetDCDw(neuron->GetDCDw() + (neuron->GetDcDw()*eventWeight));        
                  } else {
                     neuron->SetDCDw(neuron->GetDCDw());    
                  } 
               }
            }               
         } 
      }   
      double totEvents =  fTraining->GetN() > GetEventLoss(1) ? fTraining->GetN() - GetEventLoss(1) : 1e+10; 
      for(int ichipID = 0; ichipID < nSensors; ichipID++){
         if(fSCNetwork[NetworkChips[ichipID]]->GetUpdateState()==false) continue;
         nentries = fSCNetwork[NetworkChips[ichipID]]->GetSynapses().GetEntriesFast();
         for (j=0;j<nentries;j++) {
            synapse = (YSynapse *) fSCNetwork[NetworkChips[ichipID]]->GetSynapses().At(j);   
            if(totEvents>0) {
               synapse->SetDCDw(synapse->GetDCDw() / (Double_t) totEvents ); 
            } else {
               synapse->SetDCDw(0);    
            }  
         }      
         nentries = fSCNetwork[NetworkChips[ichipID]]->GetNetwork().GetEntriesFast();
         for (j=0;j<nentries;j++) {
            neuron = (YNeuron *) fSCNetwork[NetworkChips[ichipID]]->GetNetwork().At(j);   
            neuron->SetFitModel(FITMODEL);
            if(totEvents>0) {
               neuron->SetDCDw(neuron->GetDCDw() / (Double_t) totEvents ); 
            } else {
               neuron->SetDCDw(0); 
            }
         }    
      }                             
                 
   } else if (fData) {

   }
}

void YMultiLayerPerceptron::ComputeDCDwDetectorUnit() //const
{
#ifdef YMLPComputeDCDwDEBUG
   std::cout<<"YMultiLayerPerceptron::ComputeDCDwDetectorUnit"<<std::endl;
#endif
   Int_t i,j;
   Int_t nentries;
   
   YSynapse *synapse;
   YNeuron *neuron;

   YSynapse *synapseDU[8];
   YNeuron *neuronDU[8];

   YNeuron *neuron_I;
   YNeuron *neuron_O; 
   YNeuron *neuron_A;

   YNeuron *neuron_IDU[8];
   YNeuron *neuron_ODU[8]; 
   YNeuron *neuron_ADU[8];

   nentries = fSynapses.GetEntriesFast();
   for (i=0;i<nentries;i++) {
      synapse = (YSynapse *) fSynapses.At(i);
      synapse->SetDCDw(0.);
   }
   nentries = fNetwork.GetEntriesFast();
   for (i=0;i<nentries;i++) {
      neuron = (YNeuron *) fNetwork.At(i);
      neuron->SetFitModel(FITMODEL);
      neuron->SetDCDw(0.);
   }

   for(int ichipID = 0; ichipID < nSensors; ichipID++){    
      nentries = fSCNetwork[NetworkChips[ichipID]]->GetSynapses().GetEntriesFast();
      
      int SensorID 	= ichipID;
                           
      int layer         = yGEOM->GetLayer(SensorID);
      int det           = layer < nLAYERIB ? 0 : (layer < nLAYERIB + nLAYEROB1 ? 1 : 2); 
      int detIdx        = layer < nLAYERIB ? layer : (layer < nLAYERIB + nLAYEROB1 ? layer - nLAYERIB : layer - (nLAYERIB + nLAYEROB1));
      int hb		= yGEOM->GetHalfBarrel(SensorID);  		//HalfBarrel
      int hs		= yGEOM->GetHalfStave(SensorID);  		//HalfStave               
      int stave		= yGEOM->GetStave(SensorID); 
      int staveIdx      = stave%(NStaves[layer]/2);      
      int ChipIdInStave = yGEOM->GetChipIdInStave(SensorID);
      int chipZ         = ChipIdInStave%(nHicPerStave[layer]*nChipsPerHic[layer]/NSubStave[layer]);  
      // det hs chipZ detIdx hb
      for (j=0;j<nentries;j++) {
         synapse = (YSynapse *) fSCNetwork[NetworkChips[ichipID]]->GetSynapses().At(j);   
         synapse->SetDCDw(0.); 
         
         synapseDU[0] = (YSynapse *) fDetectorUnit->fSCNetwork->GetSynapses().At(j);
         synapseDU[1] = (YSynapse *) fDetectorUnit->SubUnit[det]->fSCNetwork->GetSynapses().At(j);
         synapseDU[2] = (YSynapse *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->fSCNetwork->GetSynapses().At(j);
         synapseDU[3] = (YSynapse *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->fSCNetwork->GetSynapses().At(j);
         synapseDU[4] = (YSynapse *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->fSCNetwork->GetSynapses().At(j);
         synapseDU[5] = (YSynapse *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->fSCNetwork->GetSynapses().At(j);  
         synapseDU[6] = (YSynapse *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->fSCNetwork->GetSynapses().At(j);
         synapseDU[7] = (YSynapse *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->SubUnit[hs]->fSCNetwork->GetSynapses().At(j);
         synapseDU[0]->SetDCDw(0.);
         synapseDU[1]->SetDCDw(0.);
         synapseDU[2]->SetDCDw(0.);
         synapseDU[3]->SetDCDw(0.);
         synapseDU[4]->SetDCDw(0.);
         synapseDU[5]->SetDCDw(0.);
         synapseDU[6]->SetDCDw(0.);
         synapseDU[7]->SetDCDw(0.);         
      }      
      nentries = fSCNetwork[NetworkChips[ichipID]]->GetNetwork().GetEntriesFast();
      for (j=0;j<nentries;j++) {
         neuron = (YNeuron *) fSCNetwork[NetworkChips[ichipID]]->GetNetwork().At(j);   
         neuron->SetFitModel(FITMODEL);
         neuron->SetDCDw(0.);     
         
         neuronDU[0] = (YNeuron *) fDetectorUnit->fSCNetwork->GetNetwork().At(j);
         neuronDU[1] = (YNeuron *) fDetectorUnit->SubUnit[det]->fSCNetwork->GetNetwork().At(j);
         neuronDU[2] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->fSCNetwork->GetNetwork().At(j);
         neuronDU[3] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->fSCNetwork->GetNetwork().At(j);
         neuronDU[4] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->fSCNetwork->GetNetwork().At(j);
         neuronDU[5] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->fSCNetwork->GetNetwork().At(j);   
         neuronDU[6] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->fSCNetwork->GetNetwork().At(j);
         neuronDU[7] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->SubUnit[hs]->fSCNetwork->GetNetwork().At(j);         
         neuronDU[0]->SetFitModel(FITMODEL);
         neuronDU[0]->SetDCDw(0.);
         neuronDU[1]->SetFitModel(FITMODEL);
         neuronDU[1]->SetDCDw(0.);
         neuronDU[2]->SetFitModel(FITMODEL);
         neuronDU[2]->SetDCDw(0.);
         neuronDU[3]->SetFitModel(FITMODEL);
         neuronDU[3]->SetDCDw(0.);
         neuronDU[4]->SetFitModel(FITMODEL);
         neuronDU[4]->SetDCDw(0.);
         neuronDU[5]->SetFitModel(FITMODEL);
         neuronDU[5]->SetDCDw(0.);
         neuronDU[6]->SetFitModel(FITMODEL);
         neuronDU[6]->SetDCDw(0.);
         neuronDU[7]->SetFitModel(FITMODEL);
         neuronDU[7]->SetDCDw(0.);         
      }    
   }                             
    
   Double_t eventWeight = 1.;
   if (fTraining) {
      Int_t nEvents = fTrainingIndex.size();//fTraining->GetN();
      std::cout<<"nEvents : "<<nEvents<<std::endl;
      for (i = 0; i < nEvents; i++) {  //Event loop
         GetEntry(fTraining->GetEntry(i));
         int Npronged = fTrainingIndex[i][1];
         
         Double_t mlpInputCenter[Npronged][nLAYER][2];  //layer input
         Double_t mlpOutputCenter[Npronged][nLAYER][3]; //layer output
         
         int staveIndex[Npronged][nLAYER], chipIndex[Npronged][nLAYER], chipID[Npronged][nLAYER];
         double input[Npronged][nLAYER][2], output[Npronged][nLAYER][3];  //prong layer axis
  	 double extended[Npronged][nLAYER][3];      
  	 double addition[Npronged][8];      
         eventWeight = fEventWeight->EvalInstance();
         eventWeight *= fCurrentTreeWeight;
         // Define LFE error before compute dedw by event
         //std::cout<<"# Start DCDw Computation From Cost Function 1 "<<std::endl; 
         for(int t = 0; t < Npronged; t++){
         
            //std::cout<<"(Cost 1)Event:: "<<i<<"-"<<t<<std::endl;         
            //std::cout<<"(ComputeDCDw) Event i entry track Index1 Index2:: "<<i<<" "<<t<<" "<<fTrainingIndex[i][0]<<" "<<fTrainingIndex[i][1]<<std::endl;     
            for(int j = 0; j < 8; j++) {             
               neuron_A = (YNeuron *) fAddition.At(0); 
               neuron_A->SetNeuronIndex(1000 + j);  
               neuron_A->SetNewEvent();
               addition[t][j] = neuron_A->GetBranchAddition();
            }  
         
            for(int j=0; j<fNetwork.GetEntriesFast(); j++) {
               neuron = (YNeuron *)fNetwork.At(j);
               neuron->SetNeuronIndex(t);                
               neuron->SetNewEvent();
            }         
            for(int j = 0; j < 2*nLAYER; j++){               
               neuron_I = (YNeuron *) fFirstLayer.At(j);  
               neuron_I->SetNeuronIndex(t); 
               neuron_I->SetNewEvent(); 
               int p1 = (int)(j/2);
               int p2 = (int)(j%2);               
               input[t][p1][p2] = neuron_I->GetValue();
            }    
            for(int j = 0; j < 3*nLAYER; j++) {             
               neuron_O = (YNeuron *) fLastLayer.At(j); 
               neuron_O->SetNeuronIndex(t); 
               neuron_O->SetNewEvent(); 
               neuron_O->SetLFENodeIndex((3*nLAYER)*t+j);      
               neuron_O->SetLFELayerTrain(fLayerTrain); 
               int p1 = (int)(j/3);
               int p2 = (int)(j%3);        
               extended[t][p1][p2] = neuron_O->GetBranch();               						//Very Important !!!!!! 2020 08 26           
            }       
            for(int ln = 0; ln < nLAYER; ln ++){
               staveIndex[t][ln] = extended[t][ln][0];  
               chipIndex[t][ln]  = extended[t][ln][1];  
               chipID[t][ln]     = extended[t][ln][2];  
#ifdef YMLPComputeDCDwDEBUG
               std::cout<<" Layer "<<ln
                        <<" Stave "<<staveIndex[t][ln]
                        <<" Chip " << chipIndex[t][ln] 
                        <<" ID "   << chipID[t][ln]<<std::endl;
#endif
            } 
            //load Network -> YSensorCorrection
            for(int ln = 0; ln < nLAYER; ln ++){         
            
               if(chipID[t][ln]<0) continue;     

               int SensorID 	= chipID[t][ln]; 
               int layer        = yGEOM->GetLayer(SensorID);
               int det          = layer < nLAYERIB ? 0 : (layer < nLAYERIB + nLAYEROB1 ? 1 : 2); 
               int detIdx       = layer < nLAYERIB ? layer : (layer < nLAYERIB + nLAYEROB1 ? layer - nLAYERIB : layer - (nLAYERIB + nLAYEROB1));
               int hb		= yGEOM->GetHalfBarrel(SensorID);  		//HalfBarrel
               int hs		= yGEOM->GetHalfStave(SensorID);  		//HalfStave               
               int stave	= yGEOM->GetStave(SensorID); 
               int staveIdx     = stave%(NStaves[layer]/2);
               int ChipIdInStave = yGEOM->GetChipIdInStave(SensorID);
               int chipZ         = ChipIdInStave%(nHicPerStave[layer]*nChipsPerHic[layer]/NSubStave[layer]);  
               // det hs chipZ detIdx hb
               //int Module	= yGEOM->GetModule(SensorID);   
               double dev_sign   = (double)fSCNetwork[chipID[t][ln]]->GetMntDir();  
                         
               for(int j=0; j<fSCNetwork[chipID[t][ln]]->GetNetwork().GetEntriesFast(); j++) {
                  neuron = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetNetwork().At(j); 
                  neuron->SetNeuronIndex(t);                  
                  neuron->SetNewEvent();
                  
                  // det hs chipZ detIdx hb
                  neuronDU[0] = (YNeuron *) fDetectorUnit->fSCNetwork->GetNetwork().At(j);
                  neuronDU[1] = (YNeuron *) fDetectorUnit->SubUnit[det]->fSCNetwork->GetNetwork().At(j);
                  neuronDU[2] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->fSCNetwork->GetNetwork().At(j);
                  neuronDU[3] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->fSCNetwork->GetNetwork().At(j);
                  neuronDU[4] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->fSCNetwork->GetNetwork().At(j);
                  neuronDU[5] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->fSCNetwork->GetNetwork().At(j);                    
                  neuronDU[6] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->fSCNetwork->GetNetwork().At(j); 
                  neuronDU[7] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->SubUnit[hs]->fSCNetwork->GetNetwork().At(j);
                                    
                  neuronDU[0]->SetNeuronIndex(t);                  
                  neuronDU[0]->SetNewEvent();
                  neuronDU[1]->SetNeuronIndex(t);                  
                  neuronDU[1]->SetNewEvent();
                  neuronDU[2]->SetNeuronIndex(t);                  
                  neuronDU[2]->SetNewEvent();
                  neuronDU[3]->SetNeuronIndex(t);                  
                  neuronDU[3]->SetNewEvent();
                  neuronDU[4]->SetNeuronIndex(t);                  
                  neuronDU[4]->SetNewEvent();
                  neuronDU[5]->SetNeuronIndex(t);                  
                  neuronDU[5]->SetNewEvent();              
                  neuronDU[6]->SetNeuronIndex(t);                  
                  neuronDU[6]->SetNewEvent();    
                  neuronDU[7]->SetNeuronIndex(t);                  
                  neuronDU[7]->SetNewEvent();                                                                                                                     
               }  
                    
               for(int j = 0; j < 2; j++) {
                  neuron_I = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetFirstLayer().At(j); 
                  neuron_I->SetLFENodeIndex((3*nLAYER)*t+3*ln+j); 
                  neuron_I->ClearLFEmemory();  
                  
                  // det hs chipZ detIdx hb
                  neuron_IDU[0] = (YNeuron *) fDetectorUnit->fSCNetwork->GetFirstLayer().At(j);
                  neuron_IDU[1] = (YNeuron *) fDetectorUnit->SubUnit[det]->fSCNetwork->GetFirstLayer().At(j);
                  neuron_IDU[2] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->fSCNetwork->GetFirstLayer().At(j);
                  neuron_IDU[3] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->fSCNetwork->GetFirstLayer().At(j);
                  neuron_IDU[4] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->fSCNetwork->GetFirstLayer().At(j);
                  neuron_IDU[5] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->fSCNetwork->GetFirstLayer().At(j);   
                  neuron_IDU[6] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->fSCNetwork->GetFirstLayer().At(j);   
                  neuron_IDU[7] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->SubUnit[hs]->fSCNetwork->GetFirstLayer().At(j);   
                  
                  neuron_IDU[0]->SetLFENodeIndex((3*nLAYER)*t+3*ln+j); 
                  neuron_IDU[0]->ClearLFEmemory();       
                  neuron_IDU[1]->SetLFENodeIndex((3*nLAYER)*t+3*ln+j); 
                  neuron_IDU[1]->ClearLFEmemory(); 
                  neuron_IDU[2]->SetLFENodeIndex((3*nLAYER)*t+3*ln+j); 
                  neuron_IDU[2]->ClearLFEmemory(); 
                  neuron_IDU[3]->SetLFENodeIndex((3*nLAYER)*t+3*ln+j); 
                  neuron_IDU[3]->ClearLFEmemory(); 
                  neuron_IDU[4]->SetLFENodeIndex((3*nLAYER)*t+3*ln+j); 
                  neuron_IDU[4]->ClearLFEmemory(); 
                  neuron_IDU[5]->SetLFENodeIndex((3*nLAYER)*t+3*ln+j); 
                  neuron_IDU[5]->ClearLFEmemory(); 
                  neuron_IDU[6]->SetLFENodeIndex((3*nLAYER)*t+3*ln+j); 
                  neuron_IDU[6]->ClearLFEmemory(); 
                  neuron_IDU[7]->SetLFENodeIndex((3*nLAYER)*t+3*ln+j); 
                  neuron_IDU[7]->ClearLFEmemory();                                     
               }
               
               for(int j = 0; j < 3; j++) {
                  neuron_O = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetLastLayer()[j]; 
                  int p1 = (int)(ln);    
                  int p2 = (int)(j%3);                       
                  output[t][p1][p2] = Evaluate(p2, input[t][p1], chipID[t][ln]);
                  neuron_O->SetLFENodeIndex((3*nLAYER)*t+3*ln+j);
                  neuron_O->ClearLFEmemory();   
                  neuron_O->SetLFELayerTrain(fLayerTrain);                                        
                  neuron_O = 0;   
                  
                  // det hs chipZ detIdx hb
                  neuron_ODU[0] = (YNeuron *) fDetectorUnit->fSCNetwork->GetLastLayer().At(j);
                  neuron_ODU[1] = (YNeuron *) fDetectorUnit->SubUnit[det]->fSCNetwork->GetLastLayer().At(j);
                  neuron_ODU[2] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->fSCNetwork->GetLastLayer().At(j);
                  neuron_ODU[3] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->fSCNetwork->GetLastLayer().At(j);
                  neuron_ODU[4] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->fSCNetwork->GetLastLayer().At(j);
                  neuron_ODU[5] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->fSCNetwork->GetLastLayer().At(j);
                  neuron_ODU[6] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->fSCNetwork->GetLastLayer().At(j);
                  neuron_ODU[7] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->SubUnit[hs]->fSCNetwork->GetLastLayer().At(j);
                                                                                              

                  neuron_ODU[0]->SetLFENodeIndex((3*nLAYER)*t+3*ln+j);
                  neuron_ODU[0]->ClearLFEmemory();   
                  neuron_ODU[0]->SetLFELayerTrain(fLayerTrain);                                        
                  neuron_ODU[0] = 0; 
                  neuron_ODU[1]->SetLFENodeIndex((3*nLAYER)*t+3*ln+j);
                  neuron_ODU[1]->ClearLFEmemory();   
                  neuron_ODU[1]->SetLFELayerTrain(fLayerTrain);                                        
                  neuron_ODU[1] = 0; 
                  neuron_ODU[2]->SetLFENodeIndex((3*nLAYER)*t+3*ln+j);
                  neuron_ODU[2]->ClearLFEmemory();   
                  neuron_ODU[2]->SetLFELayerTrain(fLayerTrain);                                        
                  neuron_ODU[2] = 0; 
                  neuron_ODU[3]->SetLFENodeIndex((3*nLAYER)*t+3*ln+j);
                  neuron_ODU[3]->ClearLFEmemory();   
                  neuron_ODU[3]->SetLFELayerTrain(fLayerTrain);                                        
                  neuron_ODU[3] = 0; 
                  neuron_ODU[4]->SetLFENodeIndex((3*nLAYER)*t+3*ln+j);
                  neuron_ODU[4]->ClearLFEmemory();   
                  neuron_ODU[4]->SetLFELayerTrain(fLayerTrain);                                        
                  neuron_ODU[4] = 0; 
                  neuron_ODU[5]->SetLFENodeIndex((3*nLAYER)*t+3*ln+j);
                  neuron_ODU[5]->ClearLFEmemory();   
                  neuron_ODU[5]->SetLFELayerTrain(fLayerTrain);                                        
                  neuron_ODU[5] = 0;       
                  neuron_ODU[6]->SetLFENodeIndex((3*nLAYER)*t+3*ln+j);
                  neuron_ODU[6]->ClearLFEmemory();   
                  neuron_ODU[6]->SetLFELayerTrain(fLayerTrain);                                        
                  neuron_ODU[6] = 0; 
                  neuron_ODU[7]->SetLFENodeIndex((3*nLAYER)*t+3*ln+j);
                  neuron_ODU[7]->ClearLFEmemory();   
                  neuron_ODU[7]->SetLFELayerTrain(fLayerTrain);                                        
                  neuron_ODU[7] = 0;                                                                                                                         
               }
               for(int c=0; c<2; c++){
                  mlpInputCenter[t][ln][c]=0;
               }
               for(int c=0; c<3; c++){
                  mlpOutputCenter[t][ln][c] = Evaluate(c, mlpInputCenter[t][ln], chipID[t][ln]); 
               }  
            }
         }
         for(int t = 0; t < Npronged; t++){
            for(int ln = 0; ln < nLAYER; ln ++){
               if(chipID[t][ln]<0) continue;    
               int SensorID 	= chipID[t][ln]; 
               int layer        = yGEOM->GetLayer(SensorID);
               int det          = layer < nLAYERIB ? 0 : (layer < nLAYERIB + nLAYEROB1 ? 1 : 2); 
               int detIdx       = layer < nLAYERIB ? layer : (layer < nLAYERIB + nLAYEROB1 ? layer - nLAYERIB : layer - (nLAYERIB + nLAYEROB1));
               int hb		= yGEOM->GetHalfBarrel(SensorID);  		//HalfBarrel
               int hs		= yGEOM->GetHalfStave(SensorID);  		//HalfStave               
               int stave	= yGEOM->GetStave(SensorID); 
               int staveIdx     = stave%(NStaves[layer]/2);
               int ChipIdInStave = yGEOM->GetChipIdInStave(SensorID);
               int chipZ         = ChipIdInStave%(nHicPerStave[layer]*nChipsPerHic[layer]/NSubStave[layer]);  
               // det hs chipZ detIdx hb               
               //int Module	= yGEOM->GetModule(SensorID);   
               double dev_sign   = (double)fSCNetwork[chipID[t][ln]]->GetMntDir();                         
               for(int j = 0; j < 2; j++) {         
                  neuron_I = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetFirstLayer().At(j); 
                  
                  // det hs chipZ detIdx hb               
                  neuron_IDU[0] = (YNeuron *) fDetectorUnit->fSCNetwork->GetFirstLayer().At(j);
                  neuron_IDU[1] = (YNeuron *) fDetectorUnit->SubUnit[det]->fSCNetwork->GetFirstLayer().At(j);
                  neuron_IDU[2] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->fSCNetwork->GetFirstLayer().At(j);
                  neuron_IDU[3] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->fSCNetwork->GetFirstLayer().At(j);
                  neuron_IDU[4] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->fSCNetwork->GetFirstLayer().At(j);
                  neuron_IDU[5] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->fSCNetwork->GetFirstLayer().At(j);  
                  neuron_IDU[6] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->fSCNetwork->GetFirstLayer().At(j);
                  neuron_IDU[7] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->SubUnit[hs]->fSCNetwork->GetFirstLayer().At(j);
                  
                  for(int k = 0; k < 2*nLAYER; k++){    
                     int p1 = (int)(k/2);
                     int p2 = (int)(k%2);    
                     for(int u = 0; u < Npronged; u++){            
                        neuron_I->SetLFEinput(3*p1 + p2, u, input[u][p1][p2]);
                        neuron_IDU[0]->SetLFEinput(3*p1 + p2, u, input[u][p1][p2]);
                        neuron_IDU[1]->SetLFEinput(3*p1 + p2, u, input[u][p1][p2]);
                        neuron_IDU[2]->SetLFEinput(3*p1 + p2, u, input[u][p1][p2]);
                        neuron_IDU[3]->SetLFEinput(3*p1 + p2, u, input[u][p1][p2]);
                        neuron_IDU[4]->SetLFEinput(3*p1 + p2, u, input[u][p1][p2]);
                        neuron_IDU[5]->SetLFEinput(3*p1 + p2, u, input[u][p1][p2]);         
                        neuron_IDU[6]->SetLFEinput(3*p1 + p2, u, input[u][p1][p2]);
                        neuron_IDU[7]->SetLFEinput(3*p1 + p2, u, input[u][p1][p2]);
                     }                             
                  }        
                  for(int k =0; k < 1*nLAYER; k++){						
                     for(int u = 0; u < Npronged; u++){                                 
                        neuron_I->SetLFEinput(3*k + 2, u,0);
                        neuron_IDU[0]->SetLFEinput(3*k + 2, u,0);
                        neuron_IDU[1]->SetLFEinput(3*k + 2, u,0);
                        neuron_IDU[2]->SetLFEinput(3*k + 2, u,0);
                        neuron_IDU[3]->SetLFEinput(3*k + 2, u,0);
                        neuron_IDU[4]->SetLFEinput(3*k + 2, u,0);
                        neuron_IDU[5]->SetLFEinput(3*k + 2, u,0); 
                        neuron_IDU[6]->SetLFEinput(3*k + 2, u,0); 
                        neuron_IDU[7]->SetLFEinput(3*k + 2, u,0); 
                     }
                  }                   
               } 
               for(int j = 0; j < 3; j++) {         
                  neuron_O = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetLastLayer().At(j);
                  
                  // det hs chipZ detIdx hb               
                  neuron_ODU[0] = (YNeuron *) fDetectorUnit->fSCNetwork->GetLastLayer().At(j);
                  neuron_ODU[1] = (YNeuron *) fDetectorUnit->SubUnit[det]->fSCNetwork->GetLastLayer().At(j);
                  neuron_ODU[2] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->fSCNetwork->GetLastLayer().At(j);
                  neuron_ODU[3] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->fSCNetwork->GetLastLayer().At(j);
                  neuron_ODU[4] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->fSCNetwork->GetLastLayer().At(j);
                  neuron_ODU[5] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->fSCNetwork->GetLastLayer().At(j);                    
                  neuron_ODU[6] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->fSCNetwork->GetLastLayer().At(j);
                  neuron_ODU[7] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->SubUnit[hs]->fSCNetwork->GetLastLayer().At(j);
                  
                  for(int k = 0; k < 2*nLAYER; k++){    
                     int p1 = (int)(k/2);
                     int p2 = (int)(k%2);    
                     for(int u = 0; u < Npronged; u++){            
                        neuron_O->SetLFEinput(3*p1 + p2, u, input[u][p1][p2]);  
                        neuron_ODU[0]->SetLFEinput(3*p1 + p2, u, input[u][p1][p2]);  
                        neuron_ODU[1]->SetLFEinput(3*p1 + p2, u, input[u][p1][p2]);  
                        neuron_ODU[2]->SetLFEinput(3*p1 + p2, u, input[u][p1][p2]);  
                        neuron_ODU[3]->SetLFEinput(3*p1 + p2, u, input[u][p1][p2]);  
                        neuron_ODU[4]->SetLFEinput(3*p1 + p2, u, input[u][p1][p2]);                                                                          
                        neuron_ODU[5]->SetLFEinput(3*p1 + p2, u, input[u][p1][p2]);   
                        neuron_ODU[6]->SetLFEinput(3*p1 + p2, u, input[u][p1][p2]);
                        neuron_ODU[7]->SetLFEinput(3*p1 + p2, u, input[u][p1][p2]);
                     }                             
                  }
                  for(int k =0; k < 1*nLAYER; k++){						
                     for(int u = 0; u < Npronged; u++){                                 
                        neuron_O->SetLFEinput(3*k + 2, u,0);
                        neuron_ODU[0]->SetLFEinput(3*k + 2, u,0);
                        neuron_ODU[1]->SetLFEinput(3*k + 2, u,0);
                        neuron_ODU[2]->SetLFEinput(3*k + 2, u,0);
                        neuron_ODU[3]->SetLFEinput(3*k + 2, u,0);
                        neuron_ODU[4]->SetLFEinput(3*k + 2, u,0);
                        neuron_ODU[5]->SetLFEinput(3*k + 2, u,0);
                        neuron_ODU[6]->SetLFEinput(3*k + 2, u,0);
                        neuron_ODU[7]->SetLFEinput(3*k + 2, u,0);
                     }
                  }                   
                  for(int k = 0; k < 3*nLAYER; k++){
                     int p1 = (int)(k/3);                        
                     int p2 = (int)(k%3); 
                     for(int u = 0; u < Npronged; u++){                 
                        neuron_O->SetLFEoutput(k,u,output[u][p1][p2]);   
                        neuron_O->SetLFEextended(k,u,extended[u][p1][p2]);  //prong layer axis
                        neuron_ODU[0]->SetLFEoutput(k,u,output[u][p1][p2]);   
                        neuron_ODU[0]->SetLFEextended(k,u,extended[u][p1][p2]);  //prong layer axis  
                        neuron_ODU[1]->SetLFEoutput(k,u,output[u][p1][p2]);   
                        neuron_ODU[1]->SetLFEextended(k,u,extended[u][p1][p2]);  //prong layer axis  
                        neuron_ODU[2]->SetLFEoutput(k,u,output[u][p1][p2]);   
                        neuron_ODU[2]->SetLFEextended(k,u,extended[u][p1][p2]);  //prong layer axis  
                        neuron_ODU[3]->SetLFEoutput(k,u,output[u][p1][p2]);   
                        neuron_ODU[3]->SetLFEextended(k,u,extended[u][p1][p2]);  //prong layer axis  
                        neuron_ODU[4]->SetLFEoutput(k,u,output[u][p1][p2]);   
                        neuron_ODU[4]->SetLFEextended(k,u,extended[u][p1][p2]);  //prong layer axis  
                        neuron_ODU[5]->SetLFEoutput(k,u,output[u][p1][p2]);   
                        neuron_ODU[5]->SetLFEextended(k,u,extended[u][p1][p2]);  //prong layer axis
                        neuron_ODU[6]->SetLFEoutput(k,u,output[u][p1][p2]);   
                        neuron_ODU[6]->SetLFEextended(k,u,extended[u][p1][p2]);  //prong layer axis
                        neuron_ODU[7]->SetLFEoutput(k,u,output[u][p1][p2]);   
                        neuron_ODU[7]->SetLFEextended(k,u,extended[u][p1][p2]);  //prong layer axis
                                                                        
                     }  
                  } 
                  for(int k =0; k <8; k++){						
                     for(int u = 0; u < Npronged; u++){                                        
                        neuron_O->SetLFEaddition(k, u, addition[u][k]);
                        neuron_ODU[0]->SetLFEaddition(k, u, addition[u][k]);
                        neuron_ODU[1]->SetLFEaddition(k, u, addition[u][k]);
                        neuron_ODU[2]->SetLFEaddition(k, u, addition[u][k]);  
                        neuron_ODU[3]->SetLFEaddition(k, u, addition[u][k]);                                                
                        neuron_ODU[4]->SetLFEaddition(k, u, addition[u][k]);                                                
                        neuron_ODU[5]->SetLFEaddition(k, u, addition[u][k]);
                        neuron_ODU[6]->SetLFEaddition(k, u, addition[u][k]);
                        neuron_ODU[7]->SetLFEaddition(k, u, addition[u][k]);                                                
                     }
                  }                  
               }   
            }
            for(int j=0; j<fNetwork.GetEntriesFast(); j++) {
               neuron = (YNeuron *)fNetwork.At(j);
               neuron->SetNewEvent();
            }                   
            for(int ln = 0; ln < nLAYER; ln ++){
               if(chipID[t][ln]<0) continue;
               int SensorID 	= chipID[t][ln]; 
               int layer        = yGEOM->GetLayer(SensorID);
               int det          = layer < nLAYERIB ? 0 : (layer < nLAYERIB + nLAYEROB1 ? 1 : 2); 
               int detIdx       = layer < nLAYERIB ? layer : (layer < nLAYERIB + nLAYEROB1 ? layer - nLAYERIB : layer - (nLAYERIB + nLAYEROB1));
               int hb		= yGEOM->GetHalfBarrel(SensorID);  		//HalfBarrel
               int hs		= yGEOM->GetHalfStave(SensorID);  		//HalfStave               
               int stave	= yGEOM->GetStave(SensorID); 
               int staveIdx     = stave%(NStaves[layer]/2);
               int ChipIdInStave = yGEOM->GetChipIdInStave(SensorID);
               int chipZ         = ChipIdInStave%(nHicPerStave[layer]*nChipsPerHic[layer]/NSubStave[layer]);  
               // det hs chipZ detIdx hb                  
               //int Module	= yGEOM->GetModule(SensorID);   
               double dev_sign   = (double)fSCNetwork[chipID[t][ln]]->GetMntDir();                
               for(int j=0; j<fSCNetwork[chipID[t][ln]]->GetNetwork().GetEntriesFast(); j++) {
                  neuron = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetNetwork().At(j); 
                  neuron->SetNewEvent();
                  //neuron->SetNeuronIndex(-1); 
                   
                  // det hs chipZ detIdx hb                   
                  neuronDU[0] = (YNeuron *) fDetectorUnit->fSCNetwork->GetNetwork().At(j);
                  neuronDU[1] = (YNeuron *) fDetectorUnit->SubUnit[det]->fSCNetwork->GetNetwork().At(j);
                  neuronDU[2] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->fSCNetwork->GetNetwork().At(j);
                  neuronDU[3] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->fSCNetwork->GetNetwork().At(j);
                  neuronDU[4] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->fSCNetwork->GetNetwork().At(j);
                  neuronDU[5] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->fSCNetwork->GetNetwork().At(j);                    
                  neuronDU[6] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->fSCNetwork->GetNetwork().At(j);
                  neuronDU[7] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->SubUnit[hs]->fSCNetwork->GetNetwork().At(j);
                  
                  neuronDU[0]->SetNewEvent();
                  neuronDU[1]->SetNewEvent();
                  neuronDU[2]->SetNewEvent();
                  neuronDU[3]->SetNewEvent();
                  neuronDU[4]->SetNewEvent();
                  neuronDU[5]->SetNewEvent();   
                  neuronDU[6]->SetNewEvent(); 
                  neuronDU[7]->SetNewEvent(); 
               }      
               
               for(int j = 0; j < 3; j++) {
                  neuron_O = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetLastLayer().At(j);
                  neuron_O->SetLFENodeIndex((3*nLAYER)*t+3*ln+j);                              
                  neuron_O->SetNeuronIndex(-1);   
                  //neuron_O->SetNewEvent(); 
                  
                  // det hs chipZ detIdx hb                   
                  neuron_ODU[0] = (YNeuron *) fDetectorUnit->fSCNetwork->GetLastLayer().At(j);
                  neuron_ODU[1] = (YNeuron *) fDetectorUnit->SubUnit[det]->fSCNetwork->GetLastLayer().At(j);
                  neuron_ODU[2] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->fSCNetwork->GetLastLayer().At(j);
                  neuron_ODU[3] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->fSCNetwork->GetLastLayer().At(j);
                  neuron_ODU[4] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->fSCNetwork->GetLastLayer().At(j);
                  neuron_ODU[5] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->fSCNetwork->GetLastLayer().At(j);                    
                  neuron_ODU[6] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->fSCNetwork->GetLastLayer().At(j);
                  neuron_ODU[7] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->SubUnit[hs]->fSCNetwork->GetLastLayer().At(j);
                  
                  neuron_ODU[0]->SetLFENodeIndex((3*nLAYER)*t+3*ln+j);                              
                  neuron_ODU[0]->SetNeuronIndex(-1); 
                  neuron_ODU[1]->SetLFENodeIndex((3*nLAYER)*t+3*ln+j);                              
                  neuron_ODU[1]->SetNeuronIndex(-1); 
                  neuron_ODU[2]->SetLFENodeIndex((3*nLAYER)*t+3*ln+j);                              
                  neuron_ODU[2]->SetNeuronIndex(-1); 
                  neuron_ODU[3]->SetLFENodeIndex((3*nLAYER)*t+3*ln+j);                              
                  neuron_ODU[3]->SetNeuronIndex(-1); 
                  neuron_ODU[4]->SetLFENodeIndex((3*nLAYER)*t+3*ln+j);                              
                  neuron_ODU[4]->SetNeuronIndex(-1); 
                  neuron_ODU[5]->SetLFENodeIndex((3*nLAYER)*t+3*ln+j);                              
                  neuron_ODU[5]->SetNeuronIndex(-1);        
                  neuron_ODU[6]->SetLFENodeIndex((3*nLAYER)*t+3*ln+j);                              
                  neuron_ODU[6]->SetNeuronIndex(-1);  
                  neuron_ODU[7]->SetLFENodeIndex((3*nLAYER)*t+3*ln+j);                              
                  neuron_ODU[7]->SetNeuronIndex(-1);                                                                                                                         
               }    
            } 
         }
         InitSCNeuronDcdw();
         CalculateEventDcdw(Npronged); 
         for(int t = 0; t < Npronged; t++){
         
            bool IsAccessorial = (fSplitReferenceSensor==-1) ? false : true;
            for(int ln = 0; ln < nLAYER; ln ++){
               if(chipID[t][ln] >= 0 && fSplitReferenceSensor == chipID[t][ln]) IsAccessorial=false;      
            } 
            if(IsAccessorial==true) continue;            
         
            for(int ln = 0; ln < nLAYER; ln ++){
               int SensorID 	= chipID[t][ln];
                                          
               if(chipID[t][ln]<0) continue;

               int layer        = yGEOM->GetLayer(SensorID);
               int det          = layer < nLAYERIB ? 0 : (layer < nLAYERIB + nLAYEROB1 ? 1 : 2); 
               int detIdx       = layer < nLAYERIB ? layer : (layer < nLAYERIB + nLAYEROB1 ? layer - nLAYERIB : layer - (nLAYERIB + nLAYEROB1));
               int hb		= yGEOM->GetHalfBarrel(SensorID);  		//HalfBarrel
               int hs		= yGEOM->GetHalfStave(SensorID);  		//HalfStave               
               int stave	= yGEOM->GetStave(SensorID); 
               int staveIdx     = stave%(NStaves[layer]/2);
               int ChipIdInStave = yGEOM->GetChipIdInStave(SensorID);
               int chipZ         = ChipIdInStave%(nHicPerStave[layer]*nChipsPerHic[layer]/NSubStave[layer]);  
               // det hs chipZ detIdx hb                                 
               //int Module	= yGEOM->GetModule(SensorID);   
               double dev_sign   = (double)fSCNetwork[chipID[t][ln]]->GetMntDir();
                        
               for(int j = 0; j < 2; j++) {         
                  neuron_I = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetFirstLayer().At(j); 
                  neuron_I->InitSCNeuronDcdw();
                  neuron_I->DumpSCNeuronDcdw(fSCNeuronDcdw);    

                  // det hs chipZ detIdx hb                                       
                  neuron_IDU[0] = (YNeuron *) fDetectorUnit->fSCNetwork->GetFirstLayer().At(j);
                  neuron_IDU[1] = (YNeuron *) fDetectorUnit->SubUnit[det]->fSCNetwork->GetFirstLayer().At(j);
                  neuron_IDU[2] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->fSCNetwork->GetFirstLayer().At(j);
                  neuron_IDU[3] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->fSCNetwork->GetFirstLayer().At(j);
                  neuron_IDU[4] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->fSCNetwork->GetFirstLayer().At(j);
                  neuron_IDU[5] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->fSCNetwork->GetFirstLayer().At(j);  
                  neuron_IDU[6] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->fSCNetwork->GetFirstLayer().At(j);  
                  neuron_IDU[7] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->SubUnit[hs]->fSCNetwork->GetFirstLayer().At(j);  
                                                      
                  neuron_IDU[0]->InitSCNeuronDcdw();
                  neuron_IDU[0]->DumpSCNeuronDcdw(fSCNeuronDcdw);                                       
                  neuron_IDU[1]->InitSCNeuronDcdw();
                  neuron_IDU[1]->DumpSCNeuronDcdw(fSCNeuronDcdw);                   
                  neuron_IDU[2]->InitSCNeuronDcdw();
                  neuron_IDU[2]->DumpSCNeuronDcdw(fSCNeuronDcdw);                   
                  neuron_IDU[3]->InitSCNeuronDcdw();
                  neuron_IDU[3]->DumpSCNeuronDcdw(fSCNeuronDcdw);                                  
                  neuron_IDU[4]->InitSCNeuronDcdw();
                  neuron_IDU[4]->DumpSCNeuronDcdw(fSCNeuronDcdw);                   
                  neuron_IDU[5]->InitSCNeuronDcdw();
                  neuron_IDU[5]->DumpSCNeuronDcdw(fSCNeuronDcdw);        
                  neuron_IDU[6]->InitSCNeuronDcdw();
                  neuron_IDU[6]->DumpSCNeuronDcdw(fSCNeuronDcdw); 
                  neuron_IDU[7]->InitSCNeuronDcdw();
                  neuron_IDU[7]->DumpSCNeuronDcdw(fSCNeuronDcdw);                                                
               } 
               for(int j = 0; j < 3; j++) {         
                  neuron_O = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetLastLayer().At(j);
                  neuron_O->InitSCNeuronDcdw();
                  neuron_O->DumpSCNeuronDcdw(fSCNeuronDcdw);    
                  neuron_O->SetNeuronIndex(-1);       
                  
                  // det hs chipZ detIdx hb                                                         
                  neuron_ODU[0] = (YNeuron *) fDetectorUnit->fSCNetwork->GetLastLayer().At(j);
                  neuron_ODU[1] = (YNeuron *) fDetectorUnit->SubUnit[det]->fSCNetwork->GetLastLayer().At(j);
                  neuron_ODU[2] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->fSCNetwork->GetLastLayer().At(j);
                  neuron_ODU[3] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->fSCNetwork->GetLastLayer().At(j);
                  neuron_ODU[4] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->fSCNetwork->GetLastLayer().At(j);
                  neuron_ODU[5] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->fSCNetwork->GetLastLayer().At(j);
                  neuron_ODU[6] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->fSCNetwork->GetLastLayer().At(j);
                  neuron_ODU[7] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->SubUnit[hs]->fSCNetwork->GetLastLayer().At(j);
                  
                  neuron_ODU[0]->InitSCNeuronDcdw();
                  neuron_ODU[0]->DumpSCNeuronDcdw(fSCNeuronDcdw);    
                  neuron_ODU[0]->SetNeuronIndex(-1);                
                  neuron_ODU[1]->InitSCNeuronDcdw();
                  neuron_ODU[1]->DumpSCNeuronDcdw(fSCNeuronDcdw);       
                  neuron_ODU[1]->SetNeuronIndex(-1);                                          
                  neuron_ODU[2]->InitSCNeuronDcdw();
                  neuron_ODU[2]->DumpSCNeuronDcdw(fSCNeuronDcdw);     
                  neuron_ODU[2]->SetNeuronIndex(-1);                                         
                  neuron_ODU[3]->InitSCNeuronDcdw();
                  neuron_ODU[3]->DumpSCNeuronDcdw(fSCNeuronDcdw);    
                  neuron_ODU[3]->SetNeuronIndex(-1);                                            
                  neuron_ODU[4]->InitSCNeuronDcdw();
                  neuron_ODU[4]->DumpSCNeuronDcdw(fSCNeuronDcdw);     
                  neuron_ODU[4]->SetNeuronIndex(-1);                                      
                  neuron_ODU[5]->InitSCNeuronDcdw();
                  neuron_ODU[5]->DumpSCNeuronDcdw(fSCNeuronDcdw);       
                  neuron_ODU[5]->SetNeuronIndex(-1);            
                  neuron_ODU[6]->InitSCNeuronDcdw();
                  neuron_ODU[6]->DumpSCNeuronDcdw(fSCNeuronDcdw);     
                  neuron_ODU[6]->SetNeuronIndex(-1);                                      
                  neuron_ODU[7]->InitSCNeuronDcdw();
                  neuron_ODU[7]->DumpSCNeuronDcdw(fSCNeuronDcdw);       
                  neuron_ODU[7]->SetNeuronIndex(-1);                                                              
               }  
               nentries = fSCNetwork[chipID[t][ln]]->GetSynapses().GetEntriesFast();   
               //std::cout<<"[CHECK20220512] E"<<i<<" T"<<t<<" L"<<ln<<" C"<<chipID[t][ln]<<" DcDw(S) ";                          
               for (j=0;j<nentries;j++) {
                  synapse = (YSynapse *) fSCNetwork[chipID[t][ln]]->GetSynapses().At(j);
                  synapse->SetDCDw( synapse->GetDCDw() + (synapse->GetDcDw()*eventWeight));
                  //std::cout<<synapse->GetDcDw()<<" ";

                  // det hs chipZ detIdx hb                                                                   
                  synapseDU[0] = (YSynapse *) fDetectorUnit->fSCNetwork->GetSynapses().At(j);
                  synapseDU[1] = (YSynapse *) fDetectorUnit->SubUnit[det]->fSCNetwork->GetSynapses().At(j);
                  synapseDU[2] = (YSynapse *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->fSCNetwork->GetSynapses().At(j);
                  synapseDU[3] = (YSynapse *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->fSCNetwork->GetSynapses().At(j);
                  synapseDU[4] = (YSynapse *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->fSCNetwork->GetSynapses().At(j);
                  synapseDU[5] = (YSynapse *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->fSCNetwork->GetSynapses().At(j);             
                  synapseDU[6] = (YSynapse *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->fSCNetwork->GetSynapses().At(j);
                  synapseDU[7] = (YSynapse *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->SubUnit[hs]->fSCNetwork->GetSynapses().At(j);
                                    
                  double sDcDw = +1*(synapse->GetDcDw()*eventWeight);  
                  synapseDU[0]->SetDCDw( synapseDU[0]->GetDCDw() + sDcDw);     
                  synapseDU[1]->SetDCDw( synapseDU[1]->GetDCDw() + sDcDw);     
                  synapseDU[2]->SetDCDw( synapseDU[2]->GetDCDw() + sDcDw);     
                  synapseDU[3]->SetDCDw( synapseDU[3]->GetDCDw() + sDcDw);                        
                  synapseDU[4]->SetDCDw( synapseDU[4]->GetDCDw() + sDcDw);  
                  synapseDU[5]->SetDCDw( synapseDU[5]->GetDCDw() + sDcDw);
                  synapseDU[6]->SetDCDw( synapseDU[6]->GetDCDw() + sDcDw);                                                                                        
                  synapseDU[7]->SetDCDw( synapseDU[7]->GetDCDw() + sDcDw);
               }
               nentries = fSCNetwork[chipID[t][ln]]->GetNetwork().GetEntriesFast();
               //std::cout<<" DcDw(N) ";                 
               for (j=0;j<nentries;j++) {
                  neuron = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetNetwork().At(j);

                  // det hs chipZ detIdx hb                                                                     
                  neuronDU[0] = (YNeuron *) fDetectorUnit->fSCNetwork->GetNetwork().At(j);
                  neuronDU[1] = (YNeuron *) fDetectorUnit->SubUnit[det]->fSCNetwork->GetNetwork().At(j);
                  neuronDU[2] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->fSCNetwork->GetNetwork().At(j);
                  neuronDU[3] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->fSCNetwork->GetNetwork().At(j);
                  neuronDU[4] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->fSCNetwork->GetNetwork().At(j);
                  neuronDU[5] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->fSCNetwork->GetNetwork().At(j);  
                  neuronDU[6] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->fSCNetwork->GetNetwork().At(j);  
                  neuronDU[7] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->SubUnit[hs]->fSCNetwork->GetNetwork().At(j);
                  
                  if(j>=2){ //No Input Update
                     double nDcDw = +1*neuron->GetDcDw()*eventWeight;
                     neuron->SetDCDw(neuron->GetDCDw() + (neuron->GetDcDw()*eventWeight));
                     //std::cout<<neuron->GetDcDw()<<" ";
                     
                     neuronDU[0]->SetDCDw(neuronDU[0]->GetDCDw() + nDcDw);
                     neuronDU[1]->SetDCDw(neuronDU[1]->GetDCDw() + nDcDw);
                     neuronDU[2]->SetDCDw(neuronDU[2]->GetDCDw() + nDcDw);
                     neuronDU[3]->SetDCDw(neuronDU[3]->GetDCDw() + nDcDw);
                     neuronDU[4]->SetDCDw(neuronDU[4]->GetDCDw() + nDcDw);
                     neuronDU[5]->SetDCDw(neuronDU[5]->GetDCDw() + nDcDw);
                     neuronDU[6]->SetDCDw(neuronDU[6]->GetDCDw() + nDcDw);
                     neuronDU[7]->SetDCDw(neuronDU[7]->GetDCDw() + nDcDw);
                  } else {
                     neuron->SetDCDw(neuron->GetDCDw());   
                     
                     neuronDU[0]->SetDCDw(neuronDU[0]->GetDCDw());  
                     neuronDU[1]->SetDCDw(neuronDU[1]->GetDCDw());  
                     neuronDU[2]->SetDCDw(neuronDU[2]->GetDCDw());  
                     neuronDU[3]->SetDCDw(neuronDU[3]->GetDCDw());  
                     neuronDU[4]->SetDCDw(neuronDU[4]->GetDCDw());  
                     neuronDU[5]->SetDCDw(neuronDU[5]->GetDCDw());
                     neuronDU[6]->SetDCDw(neuronDU[6]->GetDCDw());
                     neuronDU[7]->SetDCDw(neuronDU[7]->GetDCDw());
                  } 
               }                            
            }               
         } 
      }   
      double totEvents = fTraining->GetN() > GetEventLoss(1) ? fTraining->GetN() - GetEventLoss(1) : 1e+10;                       
      std::cout<<"[ITS FULL]"<<std::endl;
      nentries = fDetectorUnit->fSCNetwork->GetSynapses().GetEntriesFast();
      for (j=0;j<nentries;j++) { 
         synapseDU[0] = (YSynapse *) fDetectorUnit->fSCNetwork->GetSynapses().At(j);  
         if(totEvents>0) {
            synapseDU[0]->SetDCDw(synapseDU[0]->GetDCDw() / (Double_t) totEvents ); 
         } else {
            synapseDU[0]->SetDCDw(0);    
         }            
      }
      nentries = fDetectorUnit->fSCNetwork->GetNetwork().GetEntriesFast();
      for (j=0;j<nentries;j++) {
         neuronDU[0] = (YNeuron *) fDetectorUnit->fSCNetwork->GetNetwork().At(j);  
         neuronDU[0]->SetFitModel(FITMODEL);
         if(totEvents>0) {
            neuronDU[0]->SetDCDw(neuronDU[0]->GetDCDw() / (Double_t) totEvents ); 
         } else {
            neuronDU[0]->SetDCDw(0);    
         }
      }      
         
      std::cout<<"[ITS IB]"<<std::endl;
      nentries = fDetectorUnit->SubUnit[0]->fSCNetwork->GetSynapses().GetEntriesFast();
      for (j=0;j<nentries;j++) { 
         synapseDU[1] = (YSynapse *) fDetectorUnit->SubUnit[0]->fSCNetwork->GetSynapses().At(j);  
         if(totEvents>0) {
            synapseDU[1]->SetDCDw(synapseDU[1]->GetDCDw() / (Double_t) totEvents ); 
         } else {
            synapseDU[1]->SetDCDw(0);    
         }              
      }
      nentries = fDetectorUnit->SubUnit[0]->fSCNetwork->GetNetwork().GetEntriesFast();
      for (j=0;j<nentries;j++) {
         neuronDU[1] = (YNeuron *) fDetectorUnit->SubUnit[0]->fSCNetwork->GetNetwork().At(j);  
         neuronDU[1]->SetFitModel(FITMODEL);                  
         if(totEvents>0) {
            neuronDU[1]->SetDCDw(neuronDU[1]->GetDCDw() / (Double_t) totEvents ); 
         } else {
            neuronDU[1]->SetDCDw(0);    
         }        
      }         
      for(int hs=0; hs<nHalfStaveIB; hs++){        
         //std::cout<<">HS "<<hs<<std::endl;
         nentries = fDetectorUnit->SubUnit[0]->SubUnit[hs]->fSCNetwork->GetSynapses().GetEntriesFast();
         for (j=0;j<nentries;j++) { 
            synapseDU[2] = (YSynapse *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->fSCNetwork->GetSynapses().At(j);  
            if(totEvents>0) {
               synapseDU[2]->SetDCDw(synapseDU[2]->GetDCDw() / (Double_t) totEvents ); 
            } else {
               synapseDU[2]->SetDCDw(0);    
            }             
         }
         nentries = fDetectorUnit->SubUnit[0]->SubUnit[hs]->fSCNetwork->GetNetwork().GetEntriesFast();
         for (j=0;j<nentries;j++) {
            neuronDU[2] = (YNeuron *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->fSCNetwork->GetNetwork().At(j);  
            neuronDU[2]->SetFitModel(FITMODEL);                  
            if(totEvents>0) {
               neuronDU[2]->SetDCDw(neuronDU[2]->GetDCDw() / (Double_t) totEvents ); 
            } else {
               neuronDU[2]->SetDCDw(0);    
            }     
         } 
         for(int cz=0; cz<nChipsPerHicIB/nHalfStaveIB; cz++){
            //std::cout<<"  >ChipZ "<<cz<<std::endl;
            nentries = fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetSynapses().GetEntriesFast();
            for (j=0;j<nentries;j++) { 
               synapseDU[3] = (YSynapse *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetSynapses().At(j);  
               if(totEvents>0) {
                  synapseDU[3]->SetDCDw(synapseDU[3]->GetDCDw() / (Double_t) totEvents ); 
               } else {
                  synapseDU[3]->SetDCDw(0);    
               }       
            }
            nentries = fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetNetwork().GetEntriesFast();
            for (j=0;j<nentries;j++) {
               neuronDU[3] = (YNeuron *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetNetwork().At(j);  
               neuronDU[3]->SetFitModel(FITMODEL);                  
               if(totEvents>0) {
                  neuronDU[3]->SetDCDw(neuronDU[3]->GetDCDw() / (Double_t) totEvents ); 
               } else {
                  neuronDU[3]->SetDCDw(0);    
               }     
            }    
            for(int l=0; l<nLAYERIB; l++){ 
               int lIB = l;
               //std::cout<<"    >Layer "<<l<<std::endl;
               nentries = fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->fSCNetwork->GetSynapses().GetEntriesFast();
               for (j=0;j<nentries;j++) { 
                  synapseDU[4] = (YSynapse *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->fSCNetwork->GetSynapses().At(j);                 
                  if(totEvents>0) {
                     synapseDU[4]->SetDCDw(synapseDU[4]->GetDCDw() / (Double_t) totEvents ); 
                  } else {
                     synapseDU[4]->SetDCDw(0);    
                  }   
               }
               nentries = fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->fSCNetwork->GetNetwork().GetEntriesFast();
               for (j=0;j<nentries;j++) {
                  neuronDU[4] = (YNeuron *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->fSCNetwork->GetNetwork().At(j);  
                  neuronDU[4]->SetFitModel(FITMODEL);                                                      
                  if(totEvents>0) {
                     neuronDU[4]->SetDCDw(neuronDU[4]->GetDCDw() / (Double_t) totEvents ); 
                  } else {
                     neuronDU[4]->SetDCDw(0);    
                  }    
               }                                                                    
               for(int hb=0; hb<2; hb++){        
                  //std::cout<<"      >HB "<<hb<<std::endl;
                  nentries = fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->fSCNetwork->GetSynapses().GetEntriesFast();
                  for (j=0;j<nentries;j++) { 
                     synapseDU[5] = (YSynapse *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->fSCNetwork->GetSynapses().At(j);  
                     if(totEvents>0) {
                        synapseDU[5]->SetDCDw(synapseDU[5]->GetDCDw() / (Double_t) totEvents ); 
                     } else {
                        synapseDU[5]->SetDCDw(0);    
                     }    
                  }
                  nentries = fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->fSCNetwork->GetNetwork().GetEntriesFast();
                  for (j=0;j<nentries;j++) {
                     neuronDU[5] = (YNeuron *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->fSCNetwork->GetNetwork().At(j);  
                     neuronDU[5]->SetFitModel(FITMODEL);                  
                     if(totEvents>0) {
                        neuronDU[5]->SetDCDw(neuronDU[5]->GetDCDw() / (Double_t) totEvents ); 
                     } else {
                        neuronDU[5]->SetDCDw(0);    
                     }         
                  }  
                  for(int st=0; st<NStaves[l]/2;st++){
                     //std::cout<<"        >Stave "<<(NStaves[l]/2)*hb + st<<std::endl; 
                     nentries = fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetSynapses().GetEntriesFast();
                     for (j=0;j<nentries;j++) { 
                        synapseDU[6] = (YSynapse *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetSynapses().At(j);  
                        if(totEvents>0) {
                           synapseDU[6]->SetDCDw(synapseDU[6]->GetDCDw() / (Double_t) totEvents ); 
                        } else {
                           synapseDU[6]->SetDCDw(0);    
                        }    
                     }
                     nentries = fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetNetwork().GetEntriesFast();
                     for (j=0;j<nentries;j++) {
                        neuronDU[6] = (YNeuron *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetNetwork().At(j);  
                        neuronDU[6]->SetFitModel(FITMODEL);                  
                        if(totEvents>0) {
                           neuronDU[6]->SetDCDw(neuronDU[6]->GetDCDw() / (Double_t) totEvents ); 
                        } else {
                           neuronDU[6]->SetDCDw(0);    
                        }         
                     }      
                     for(int bst=0; bst<nHalfStaveIB; bst++){        
                        //std::cout<<"          >SubStave "<<bst<<std::endl;
                        nentries = fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetSynapses().GetEntriesFast();
                        for (j=0;j<nentries;j++) { 
                           synapseDU[7] = (YSynapse *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetSynapses().At(j);  
                           if(totEvents>0) {
                              synapseDU[7]->SetDCDw(synapseDU[7]->GetDCDw() / (Double_t) totEvents ); 
                           } else {
                              synapseDU[7]->SetDCDw(0);    
                           }    
                        }
                        nentries = fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetNetwork().GetEntriesFast();
                        for (j=0;j<nentries;j++) {
                           neuronDU[7] = (YNeuron *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetNetwork().At(j);  
                           neuronDU[7]->SetFitModel(FITMODEL);                  
                           if(totEvents>0) {
                              neuronDU[7]->SetDCDw(neuronDU[7]->GetDCDw() / (Double_t) totEvents ); 
                           } else {
                              neuronDU[7]->SetDCDw(0);    
                           }         
                        }
                     }
                  }                       
               }       
            }          
         }
      }

      std::cout<<"[ITS OB1]"<<std::endl;
      nentries = fDetectorUnit->SubUnit[1]->fSCNetwork->GetSynapses().GetEntriesFast();
      for (j=0;j<nentries;j++) { 
         synapseDU[1] = (YSynapse *) fDetectorUnit->SubUnit[1]->fSCNetwork->GetSynapses().At(j);  
         if(totEvents>0) {
            synapseDU[1]->SetDCDw(synapseDU[1]->GetDCDw() / (Double_t) totEvents ); 
         } else {
            synapseDU[1]->SetDCDw(0);    
         }              
      }
      nentries = fDetectorUnit->SubUnit[1]->fSCNetwork->GetNetwork().GetEntriesFast();
      for (j=0;j<nentries;j++) {
         neuronDU[1] = (YNeuron *) fDetectorUnit->SubUnit[1]->fSCNetwork->GetNetwork().At(j);  
         neuronDU[1]->SetFitModel(FITMODEL);                  
         if(totEvents>0) {
            neuronDU[1]->SetDCDw(neuronDU[1]->GetDCDw() / (Double_t) totEvents ); 
         } else {
            neuronDU[1]->SetDCDw(0);    
         }        
      }         
      for(int hs=0; hs<nHalfStaveOB1; hs++){        
         //std::cout<<">HS "<<hs<<std::endl;
         nentries = fDetectorUnit->SubUnit[1]->SubUnit[hs]->fSCNetwork->GetSynapses().GetEntriesFast();
         for (j=0;j<nentries;j++) { 
            synapseDU[2] = (YSynapse *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->fSCNetwork->GetSynapses().At(j);  
            if(totEvents>0) {
               synapseDU[2]->SetDCDw(synapseDU[2]->GetDCDw() / (Double_t) totEvents ); 
            } else {
               synapseDU[2]->SetDCDw(0);    
            }             
         }
         nentries = fDetectorUnit->SubUnit[1]->SubUnit[hs]->fSCNetwork->GetNetwork().GetEntriesFast();
         for (j=0;j<nentries;j++) {
            neuronDU[2] = (YNeuron *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->fSCNetwork->GetNetwork().At(j);  
            neuronDU[2]->SetFitModel(FITMODEL);                  
            if(totEvents>0) {
               neuronDU[2]->SetDCDw(neuronDU[2]->GetDCDw() / (Double_t) totEvents ); 
            } else {
               neuronDU[2]->SetDCDw(0);    
            }     
         } 
         for(int cz=0; cz<nChipsPerHicOB1/nHalfStaveOB1; cz++){
            //std::cout<<"  >ChipZ "<<cz<<std::endl;
            nentries = fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetSynapses().GetEntriesFast();
            for (j=0;j<nentries;j++) { 
               synapseDU[3] = (YSynapse *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetSynapses().At(j);  
               if(totEvents>0) {
                  synapseDU[3]->SetDCDw(synapseDU[3]->GetDCDw() / (Double_t) totEvents ); 
               } else {
                  synapseDU[3]->SetDCDw(0);    
               }       
            }
            nentries = fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetNetwork().GetEntriesFast();
            for (j=0;j<nentries;j++) {
               neuronDU[3] = (YNeuron *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetNetwork().At(j);  
               neuronDU[3]->SetFitModel(FITMODEL);                  
               if(totEvents>0) {
                  neuronDU[3]->SetDCDw(neuronDU[3]->GetDCDw() / (Double_t) totEvents ); 
               } else {
                  neuronDU[3]->SetDCDw(0);    
               }     
            }    
            for(int l=nLAYERIB; l<nLAYERIB + nLAYEROB1; l++){ 
               int lOB = l - nLAYERIB;
               //std::cout<<"    >Layer "<<l<<std::endl;
               nentries = fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->GetSynapses().GetEntriesFast();
               for (j=0;j<nentries;j++) { 
                  synapseDU[4] = (YSynapse *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->GetSynapses().At(j);                 
                  if(totEvents>0) {
                     synapseDU[4]->SetDCDw(synapseDU[4]->GetDCDw() / (Double_t) totEvents ); 
                  } else {
                     synapseDU[4]->SetDCDw(0);    
                  }   
               }
               nentries = fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->GetNetwork().GetEntriesFast();
               for (j=0;j<nentries;j++) {
                  neuronDU[4] = (YNeuron *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->GetNetwork().At(j);  
                  neuronDU[4]->SetFitModel(FITMODEL);                                                      
                  if(totEvents>0) {
                     neuronDU[4]->SetDCDw(neuronDU[4]->GetDCDw() / (Double_t) totEvents ); 
                  } else {
                     neuronDU[4]->SetDCDw(0);    
                  }    
               }                                                                    
               for(int hb=0; hb<2; hb++){        
                  //std::cout<<"      >HB "<<hb<<std::endl;
                  nentries = fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->GetSynapses().GetEntriesFast();
                  for (j=0;j<nentries;j++) { 
                     synapseDU[5] = (YSynapse *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->GetSynapses().At(j);  
                     if(totEvents>0) {
                        synapseDU[5]->SetDCDw(synapseDU[5]->GetDCDw() / (Double_t) totEvents ); 
                     } else {
                        synapseDU[5]->SetDCDw(0);    
                     }    
                  }
                  nentries = fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->GetNetwork().GetEntriesFast();
                  for (j=0;j<nentries;j++) {
                     neuronDU[5] = (YNeuron *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->GetNetwork().At(j);  
                     neuronDU[5]->SetFitModel(FITMODEL);                  
                     if(totEvents>0) {
                        neuronDU[5]->SetDCDw(neuronDU[5]->GetDCDw() / (Double_t) totEvents ); 
                     } else {
                        neuronDU[5]->SetDCDw(0);    
                     }    
                  }
                  for(int st=0; st<NStaves[l]/2;st++){
                     //std::cout<<"        >Stave "<<(NStaves[l]/2)*hb + st<<std::endl; 
                     nentries = fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetSynapses().GetEntriesFast();
                     for (j=0;j<nentries;j++) { 
                        synapseDU[6] = (YSynapse *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetSynapses().At(j);  
                        if(totEvents>0) {
                           synapseDU[6]->SetDCDw(synapseDU[6]->GetDCDw() / (Double_t) totEvents ); 
                        } else {
                           synapseDU[6]->SetDCDw(0);    
                        }    
                     }
                     nentries = fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetNetwork().GetEntriesFast();
                     for (j=0;j<nentries;j++) {
                        neuronDU[6] = (YNeuron *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetNetwork().At(j);  
                        neuronDU[6]->SetFitModel(FITMODEL);                  
                        if(totEvents>0) {
                           neuronDU[6]->SetDCDw(neuronDU[6]->GetDCDw() / (Double_t) totEvents ); 
                        } else {
                           neuronDU[6]->SetDCDw(0);    
                        }         
                     }      
                     for(int bst=0; bst<nHalfStaveOB1; bst++){        
                        //std::cout<<"          >SubStave "<<bst<<std::endl;
                        nentries = fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetSynapses().GetEntriesFast();
                        for (j=0;j<nentries;j++) { 
                           synapseDU[7] = (YSynapse *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetSynapses().At(j);  
                           if(totEvents>0) {
                              synapseDU[7]->SetDCDw(synapseDU[7]->GetDCDw() / (Double_t) totEvents ); 
                           } else {
                              synapseDU[7]->SetDCDw(0);    
                           }    
                        }
                        nentries = fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetNetwork().GetEntriesFast();
                        for (j=0;j<nentries;j++) {
                           neuronDU[7] = (YNeuron *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetNetwork().At(j);  
                           neuronDU[7]->SetFitModel(FITMODEL);                  
                           if(totEvents>0) {
                              neuronDU[7]->SetDCDw(neuronDU[7]->GetDCDw() / (Double_t) totEvents ); 
                           } else {
                              neuronDU[7]->SetDCDw(0);    
                           }         
                        }
                     }
                  }                           
               }       
            }          
         }
      }
      
      std::cout<<"[ITS OB2]"<<std::endl;
      nentries = fDetectorUnit->SubUnit[2]->fSCNetwork->GetSynapses().GetEntriesFast();
      for (j=0;j<nentries;j++) { 
         synapseDU[1] = (YSynapse *) fDetectorUnit->SubUnit[2]->fSCNetwork->GetSynapses().At(j);  
         if(totEvents>0) {
            synapseDU[1]->SetDCDw(synapseDU[1]->GetDCDw() / (Double_t) totEvents ); 
         } else {
            synapseDU[1]->SetDCDw(0);    
         }              
      }
      nentries = fDetectorUnit->SubUnit[2]->fSCNetwork->GetNetwork().GetEntriesFast();
      for (j=0;j<nentries;j++) {
         neuronDU[1] = (YNeuron *) fDetectorUnit->SubUnit[2]->fSCNetwork->GetNetwork().At(j);  
         neuronDU[1]->SetFitModel(FITMODEL);                  
         if(totEvents>0) {
            neuronDU[1]->SetDCDw(neuronDU[1]->GetDCDw() / (Double_t) totEvents ); 
         } else {
            neuronDU[1]->SetDCDw(0);    
         }        
      }         
      for(int hs=0; hs<nHalfStaveOB2; hs++){        
         //std::cout<<">HS "<<hs<<std::endl;
         nentries = fDetectorUnit->SubUnit[2]->SubUnit[hs]->fSCNetwork->GetSynapses().GetEntriesFast();
         for (j=0;j<nentries;j++) { 
            synapseDU[2] = (YSynapse *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->fSCNetwork->GetSynapses().At(j);  
            if(totEvents>0) {
               synapseDU[2]->SetDCDw(synapseDU[2]->GetDCDw() / (Double_t) totEvents ); 
            } else {
               synapseDU[2]->SetDCDw(0);    
            }             
         }
         nentries = fDetectorUnit->SubUnit[2]->SubUnit[hs]->fSCNetwork->GetNetwork().GetEntriesFast();
         for (j=0;j<nentries;j++) {
            neuronDU[2] = (YNeuron *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->fSCNetwork->GetNetwork().At(j);  
            neuronDU[2]->SetFitModel(FITMODEL);                  
            if(totEvents>0) {
               neuronDU[2]->SetDCDw(neuronDU[2]->GetDCDw() / (Double_t) totEvents ); 
            } else {
               neuronDU[2]->SetDCDw(0);    
            }     
         } 
         for(int cz=0; cz<nChipsPerHicOB2/nHalfStaveOB2; cz++){
            //std::cout<<"  >ChipZ "<<cz<<std::endl;
            nentries = fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetSynapses().GetEntriesFast();
            for (j=0;j<nentries;j++) { 
               synapseDU[3] = (YSynapse *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetSynapses().At(j);  
               if(totEvents>0) {
                  synapseDU[3]->SetDCDw(synapseDU[3]->GetDCDw() / (Double_t) totEvents ); 
               } else {
                  synapseDU[3]->SetDCDw(0);    
               }       
            }
            nentries = fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetNetwork().GetEntriesFast();
            for (j=0;j<nentries;j++) {
               neuronDU[3] = (YNeuron *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetNetwork().At(j);  
               neuronDU[3]->SetFitModel(FITMODEL);                  
               if(totEvents>0) {
                  neuronDU[3]->SetDCDw(neuronDU[3]->GetDCDw() / (Double_t) totEvents ); 
               } else {
                  neuronDU[3]->SetDCDw(0);    
               }     
            }    
            for(int l=nLAYERIB + nLAYEROB1; l<nLAYER; l++){ 
               int lOB = l - (nLAYERIB + nLAYEROB1);
               //std::cout<<"    >Layer "<<l<<std::endl;
               nentries = fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->GetSynapses().GetEntriesFast();
               for (j=0;j<nentries;j++) { 
                  synapseDU[4] = (YSynapse *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->GetSynapses().At(j);                 
                  if(totEvents>0) {
                     synapseDU[4]->SetDCDw(synapseDU[4]->GetDCDw() / (Double_t) totEvents ); 
                  } else {
                     synapseDU[4]->SetDCDw(0);    
                  }   
               }
               nentries = fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->GetNetwork().GetEntriesFast();
               for (j=0;j<nentries;j++) {
                  neuronDU[4] = (YNeuron *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->GetNetwork().At(j);  
                  neuronDU[4]->SetFitModel(FITMODEL);                                                      
                  if(totEvents>0) {
                     neuronDU[4]->SetDCDw(neuronDU[4]->GetDCDw() / (Double_t) totEvents ); 
                  } else {
                     neuronDU[4]->SetDCDw(0);    
                  }    
               }                                                                    
               for(int hb=0; hb<2; hb++){        
                  //std::cout<<"      >HB "<<hb<<std::endl;
                  nentries = fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->GetSynapses().GetEntriesFast();
                  for (j=0;j<nentries;j++) { 
                     synapseDU[5] = (YSynapse *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->GetSynapses().At(j);  
                     if(totEvents>0) {
                        synapseDU[5]->SetDCDw(synapseDU[5]->GetDCDw() / (Double_t) totEvents ); 
                     } else {
                        synapseDU[5]->SetDCDw(0);    
                     }    
                  }
                  nentries = fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->GetNetwork().GetEntriesFast();
                  for (j=0;j<nentries;j++) {
                     neuronDU[5] = (YNeuron *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->GetNetwork().At(j);  
                     neuronDU[5]->SetFitModel(FITMODEL);                  
                     if(totEvents>0) {
                        neuronDU[5]->SetDCDw(neuronDU[5]->GetDCDw() / (Double_t) totEvents ); 
                     } else {
                        neuronDU[5]->SetDCDw(0);    
                     }    
                  }
                  for(int st=0; st<NStaves[l]/2;st++){
                     //std::cout<<"        >Stave "<<(NStaves[l]/2)*hb + st<<std::endl; 
                     nentries = fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetSynapses().GetEntriesFast();
                     for (j=0;j<nentries;j++) { 
                        synapseDU[6] = (YSynapse *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetSynapses().At(j);  
                        if(totEvents>0) {
                           synapseDU[6]->SetDCDw(synapseDU[6]->GetDCDw() / (Double_t) totEvents ); 
                        } else {
                           synapseDU[6]->SetDCDw(0);    
                        }    
                     }
                     nentries = fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetNetwork().GetEntriesFast();
                     for (j=0;j<nentries;j++) {
                        neuronDU[6] = (YNeuron *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetNetwork().At(j);  
                        neuronDU[6]->SetFitModel(FITMODEL);                  
                        if(totEvents>0) {
                           neuronDU[6]->SetDCDw(neuronDU[6]->GetDCDw() / (Double_t) totEvents ); 
                        } else {
                           neuronDU[6]->SetDCDw(0);    
                        }         
                     }      
                     for(int bst=0; bst<nHalfStaveOB2; bst++){        
                        //std::cout<<"          >SubStave "<<bst<<std::endl;
                        nentries = fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetSynapses().GetEntriesFast();
                        for (j=0;j<nentries;j++) { 
                           synapseDU[7] = (YSynapse *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetSynapses().At(j);  
                           if(totEvents>0) {
                              synapseDU[7]->SetDCDw(synapseDU[7]->GetDCDw() / (Double_t) totEvents ); 
                           } else {
                              synapseDU[7]->SetDCDw(0);    
                           }    
                        }
                        nentries = fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetNetwork().GetEntriesFast();
                        for (j=0;j<nentries;j++) {
                           neuronDU[7] = (YNeuron *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetNetwork().At(j);  
                           neuronDU[7]->SetFitModel(FITMODEL);                  
                           if(totEvents>0) {
                              neuronDU[7]->SetDCDw(neuronDU[7]->GetDCDw() / (Double_t) totEvents ); 
                           } else {
                              neuronDU[7]->SetDCDw(0);    
                           }         
                        }
                     }
                  }                         
               }       
            }          
         }
      }      

      for(int ichipID = 0; ichipID < nSensors; ichipID++){
         int SensorID 	= ichipID;
                           
         int layer      = yGEOM->GetLayer(SensorID);
         if(layer<0 || layer >nLAYER) continue;
         if(fSCNetwork[NetworkChips[ichipID]]->GetUpdateState()==false && kDetectorUnit[layer]==-1) continue;         
         int det           = layer < nLAYERIB ? 0 : (layer < nLAYERIB + nLAYEROB1 ? 1 : 2); 
         int detIdx        = layer < nLAYERIB ? layer : (layer < nLAYERIB + nLAYEROB1 ? layer - nLAYERIB : layer - (nLAYERIB + nLAYEROB1));
         int hb		= yGEOM->GetHalfBarrel(SensorID);  		//HalfBarrel
         int hs		= yGEOM->GetHalfStave(SensorID);  		//HalfStave               
         int stave	= yGEOM->GetStave(SensorID); 
         int staveIdx   = stave%(NStaves[layer]/2);
         int ChipIdInStave = yGEOM->GetChipIdInStave(SensorID);
         int chipZ         = ChipIdInStave%(nHicPerStave[layer]*nChipsPerHic[layer]/NSubStave[layer]);  
         // det hs chipZ detIdx hb         
         //int Module	= yGEOM->GetModule(SensorID);     
         double dev_sign   = (double)fSCNetwork[ichipID]->GetMntDir();  
         
         nentries = fSCNetwork[NetworkChips[ichipID]]->GetSynapses().GetEntriesFast();
         for (j=0;j<nentries;j++) {

            switch(kDetectorUnit[layer])
            {
               case -1 : 
               {
                  synapse = (YSynapse *) fSCNetwork[NetworkChips[ichipID]]->GetSynapses().At(j); 
                  if(totEvents>0) {
                     synapse->SetDCDw(synapse->GetDCDw() / (Double_t) totEvents ); 
                  } else {
                     synapse->SetDCDw(0);    
                  }          
                  break;
               }
               case 0  : 
               {
                  synapse      = (YSynapse *) fSCNetwork[NetworkChips[ichipID]]->GetSynapses().At(j);                
                  synapseDU[0] = (YSynapse *) fDetectorUnit->fSCNetwork->GetSynapses().At(j);  
                  if(totEvents>0) {
                     synapse->SetDCDw(synapseDU[0]->GetDCDw()); 
                  } else {
                     synapse->SetDCDw(0);    
                  }           
                  break;
               }                        
               case 1  : 
               {
                  synapse      = (YSynapse *) fSCNetwork[NetworkChips[ichipID]]->GetSynapses().At(j);                               
                  synapseDU[1] = (YSynapse *) fDetectorUnit->SubUnit[det]->fSCNetwork->GetSynapses().At(j);  
                  if(totEvents>0) {
                     synapse->SetDCDw(synapseDU[1]->GetDCDw()); 
                  } else {
                     synapse->SetDCDw(0);    
                  }                    
                  break;
               }  
               case 2  : 
               {
                  synapse      = (YSynapse *) fSCNetwork[NetworkChips[ichipID]]->GetSynapses().At(j);                               
                  synapseDU[2] = (YSynapse *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->fSCNetwork->GetSynapses().At(j);  
                  if(totEvents>0) {
                     synapse->SetDCDw(synapseDU[2]->GetDCDw()); 
                  } else {
                     synapse->SetDCDw(0);    
                  }                    
                  break;
               }  
               case 3  : 
               {
                  synapse      = (YSynapse *) fSCNetwork[NetworkChips[ichipID]]->GetSynapses().At(j);                               
                  synapseDU[3] = (YSynapse *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->fSCNetwork->GetSynapses().At(j);  
                  if(totEvents>0) {
                     synapse->SetDCDw(synapseDU[3]->GetDCDw()); 
                  } else {
                     synapse->SetDCDw(0);    
                  }                    
                  break;
               }  
               case 4  : 
               {
                  synapse      = (YSynapse *) fSCNetwork[NetworkChips[ichipID]]->GetSynapses().At(j);                               
                  synapseDU[4] = (YSynapse *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[hs]->fSCNetwork->GetSynapses().At(j);  
                  if(totEvents>0) {
                     synapse->SetDCDw(synapseDU[4]->GetDCDw()); 
                  } else {
                     synapse->SetDCDw(0);    
                  }                    
                  break;
               }  
               case 5  : 
               {
                  synapse      = (YSynapse *) fSCNetwork[NetworkChips[ichipID]]->GetSynapses().At(j);                               
                  synapseDU[5] = (YSynapse *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->fSCNetwork->GetSynapses().At(j);  
                  if(totEvents>0) {
                     synapse->SetDCDw(synapseDU[5]->GetDCDw()); 
                  } else {
                     synapse->SetDCDw(0);    
                  }                    
                  break;
               }      
               case 6  : 
               {
                  synapse      = (YSynapse *) fSCNetwork[NetworkChips[ichipID]]->GetSynapses().At(j);                               
                  synapseDU[6] = (YSynapse *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->fSCNetwork->GetSynapses().At(j);  
                  if(totEvents>0) {
                     synapse->SetDCDw(synapseDU[6]->GetDCDw()); 
                  } else {
                     synapse->SetDCDw(0);    
                  }                    
                  break;
               }                
               case 7  : 
               {
                  synapse      = (YSynapse *) fSCNetwork[NetworkChips[ichipID]]->GetSynapses().At(j);                               
                  synapseDU[7] = (YSynapse *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->SubUnit[hs]->fSCNetwork->GetSynapses().At(j);  
                  if(totEvents>0) {
                     synapse->SetDCDw(synapseDU[7]->GetDCDw()); 
                  } else {
                     synapse->SetDCDw(0);    
                  }                    
                  break;
               }                  
               default : 
               {
                  synapse = (YSynapse *) fSCNetwork[NetworkChips[ichipID]]->GetSynapses().At(j); 
                  if(totEvents>0) {
                     synapse->SetDCDw(synapse->GetDCDw() / (Double_t) totEvents ); 
                  } else {
                     synapse->SetDCDw(0);    
                  }                    
                  break;          
               }                                       
            }                       
         }      
         nentries = fSCNetwork[NetworkChips[ichipID]]->GetNetwork().GetEntriesFast();
         for (j=0;j<nentries;j++) {

            switch(kDetectorUnit[layer])
            {
               case -1 : 
               {
                  neuron = (YNeuron *) fSCNetwork[NetworkChips[ichipID]]->GetNetwork().At(j); 
                  neuron->SetFitModel(FITMODEL);
                  if(totEvents>0) {
                     neuron->SetDCDw(neuron->GetDCDw() / (Double_t) totEvents ); 
                  } else {
                     neuron->SetDCDw(0); 
                  }                      
                  break;
               }
               case 0  : 
               {
                  neuron      = (YNeuron *) fSCNetwork[NetworkChips[ichipID]]->GetNetwork().At(j);                
                  neuronDU[0] = (YNeuron *) fDetectorUnit->fSCNetwork->GetNetwork().At(j);  
                  neuronDU[0]->SetFitModel(FITMODEL);
                  if(totEvents>0) {
                     neuron->SetDCDw(neuronDU[0]->GetDCDw()); 
                  } else {
                     neuron->SetDCDw(0);    
                  }           
                  break;
               }                        
               case 1  : 
               {
                  neuron      = (YNeuron *) fSCNetwork[NetworkChips[ichipID]]->GetNetwork().At(j);                               
                  neuronDU[1] = (YNeuron *) fDetectorUnit->SubUnit[det]->fSCNetwork->GetNetwork().At(j);  
                  neuronDU[1]->SetFitModel(FITMODEL);                  
                  if(totEvents>0) {
                     neuron->SetDCDw(neuronDU[1]->GetDCDw()); 
                  } else {
                     neuron->SetDCDw(0);    
                  }                    
                  break;
               }  
               case 2  : 
               {
                  neuron      = (YNeuron *) fSCNetwork[NetworkChips[ichipID]]->GetNetwork().At(j);                               
                  neuronDU[2] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->fSCNetwork->GetNetwork().At(j);  
                  neuronDU[2]->SetFitModel(FITMODEL);                  
                  if(totEvents>0) {
                     neuron->SetDCDw(neuronDU[2]->GetDCDw()); 
                  } else {
                     neuron->SetDCDw(0);    
                  }                    
                  break;
               }  
               case 3  : 
               {
                  neuron      = (YNeuron *) fSCNetwork[NetworkChips[ichipID]]->GetNetwork().At(j);                               
                  neuronDU[3] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->fSCNetwork->GetNetwork().At(j);  
                  neuronDU[3]->SetFitModel(FITMODEL);                  
                  if(totEvents>0) {
                     neuron->SetDCDw(neuronDU[3]->GetDCDw()); 
                  } else {
                     neuron->SetDCDw(0);    
                  }                    
                  break;
               }  
               case 4  : 
               {
                  neuron      = (YNeuron *) fSCNetwork[NetworkChips[ichipID]]->GetNetwork().At(j);                               
                  neuronDU[4] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[hs]->fSCNetwork->GetNetwork().At(j);  
                  neuronDU[4]->SetFitModel(FITMODEL);                  
                  if(totEvents>0) {
                     neuron->SetDCDw(neuronDU[4]->GetDCDw()); 
                  } else {
                     neuron->SetDCDw(0);    
                  }                    
                  break;
               }  
               case 5  : 
               {
                  neuron      = (YNeuron *) fSCNetwork[NetworkChips[ichipID]]->GetNetwork().At(j);                               
                  neuronDU[5] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->fSCNetwork->GetNetwork().At(j);  
                  neuronDU[5]->SetFitModel(FITMODEL);                  
                  if(totEvents>0) {
                     neuron->SetDCDw(neuronDU[5]->GetDCDw()); 
                  } else {
                     neuron->SetDCDw(0);    
                  }                    
                  break;
               }         
               case 6  : 
               {
                  neuron      = (YNeuron *) fSCNetwork[NetworkChips[ichipID]]->GetNetwork().At(j);                               
                  neuronDU[6] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->fSCNetwork->GetNetwork().At(j);  
                  neuronDU[6]->SetFitModel(FITMODEL);                  
                  if(totEvents>0) {
                     neuron->SetDCDw(neuronDU[6]->GetDCDw()); 
                  } else {
                     neuron->SetDCDw(0);    
                  }                    
                  break;
               }       
               case 7  : 
               {
                  neuron      = (YNeuron *) fSCNetwork[NetworkChips[ichipID]]->GetNetwork().At(j);                               
                  neuronDU[7] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->SubUnit[hs]->fSCNetwork->GetNetwork().At(j);  
                  neuronDU[7]->SetFitModel(FITMODEL);                  
                  if(totEvents>0) {
                     neuron->SetDCDw(neuronDU[7]->GetDCDw()); 
                  } else {
                     neuron->SetDCDw(0);    
                  }                    
                  break;
               }     
               default : 
               {
                  neuron = (YNeuron *) fSCNetwork[NetworkChips[ichipID]]->GetNetwork().At(j); 
                  neuron->SetFitModel(FITMODEL);
                  if(totEvents>0) {
                     neuron->SetDCDw(neuron->GetDCDw() / (Double_t) totEvents ); 
                  } else {
                     neuron->SetDCDw(0); 
                  }                   
                  break;          
               }                                       
            }                        
         }    
      }                                           
                 
   } else if (fData) {

   }
}

void YMultiLayerPerceptron::InitSCNeuronDcdw()
{
   for(int nprong = 0; nprong < fNpronged; nprong++){
      for(int ln = 0; ln<nLAYER; ln++){         
         for(int iaxis = 0; iaxis<3; iaxis++){
            int index = (3*ln)+iaxis;
            SetSCNeuronDcdw(index,nprong,0.0);
         }
      } 
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Randomize the weights

void YMultiLayerPerceptron::Init_Randomize() const
{
   Int_t nentries = fSynapses.GetEntriesFast();
   std::cout<<"YMultiLayerPerceptron::Randomize Synapse : "<<fSynapses.GetEntriesFast()<<" Network : "<<fNetwork.GetEntriesFast()<<std::endl;
   Int_t j;
   YSynapse *synapse;
   YNeuron *neuron;
   TTimeStamp ts;
   //TRandom3 gen(ts.GetSec());
   TRandom3 gen(fRandomSeed);
   for (j=0;j<nentries;j++) {
      synapse = (YSynapse *) fSynapses.At(j);
      synapse->SetWeight(0);
   }
   nentries = fNetwork.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      neuron = (YNeuron *) fNetwork.At(j);
      neuron->SetWeight(0);
      neuron->SetFitModel(FITMODEL);      
   }
}

void YMultiLayerPerceptron::Init_RandomizeSensorCorrection() const
{
   std::cout<<"YMultiLayerPerceptron::Init_RandomizeSensorCorrection "<<std::endl;
   Int_t nentries;
   Int_t j;
   YSynapse *synapse;
   YNeuron *neuron;
   TTimeStamp ts;
   //TRandom3 gen(ts.GetSec());
   TRandom3 gen(fRandomSeed);
   for(int iID = 0; iID < nSensors; iID++){
      nentries = fSCNetwork[iID]->GetSynapses().GetEntriesFast();
      for (j=0;j<nentries;j++) {
         synapse = (YSynapse *) fSCNetwork[iID]->GetSynapses().At(j);       
         synapse->SetWeight(0);
      }  
      nentries = fSCNetwork[iID]->GetNetwork().GetEntriesFast();
      for (j=0;j<nentries;j++) {
         neuron = (YNeuron *) fSCNetwork[iID]->GetNetwork().At(j);  
         neuron->fSCNetworkNeuron = true;
         neuron->SetLFEnprong(fNpronged);                 
         neuron->SetLFEmemory(fNpronged);      
         neuron->SetWeight(0);
         neuron->SetFitModel(FITMODEL);               
      }      
   }   
}

void YMultiLayerPerceptron::Init_RandomizeSensorCorrectionDetectorUnit() const
{
   std::cout<<"YMultiLayerPerceptron::Init_RandomizeSensorCorrectionDetectorUnit "<<std::endl;
   Int_t nentries;
   Int_t j;
   YSynapse *synapse;
   YNeuron *neuron;
   YSynapse *synapseDU[8];
   YNeuron *neuronDU[8];   
   TTimeStamp ts;
   //TRandom3 gen(ts.GetSec());
   TRandom3 gen(fRandomSeed);
   for(int iID = 0; iID < nSensors; iID++){
      nentries = fSCNetwork[iID]->GetSynapses().GetEntriesFast();
      for (j=0;j<nentries;j++) {
         synapse = (YSynapse *) fSCNetwork[iID]->GetSynapses().At(j);       
         synapse->SetWeight(0);
      }  
      nentries = fSCNetwork[iID]->GetNetwork().GetEntriesFast();
      for (j=0;j<nentries;j++) {
         neuron = (YNeuron *) fSCNetwork[iID]->GetNetwork().At(j);  
         neuron->fSCNetworkNeuron = true;
         neuron->SetLFEnprong(fNpronged);                 
         neuron->SetLFEmemory(fNpronged);      
         neuron->SetWeight(0);
         neuron->SetFitModel(FITMODEL);               
      }      
   }   
   
   std::cout<<"[ITS FULL]"<<std::endl;
   nentries = fDetectorUnit->fSCNetwork->GetSynapses().GetEntriesFast();
   for (j=0;j<nentries;j++) { 
      synapseDU[0] = (YSynapse *) fDetectorUnit->fSCNetwork->GetSynapses().At(j);  
      synapseDU[0]->SetWeight(0);       
   }
   nentries = fDetectorUnit->fSCNetwork->GetNetwork().GetEntriesFast();
   for (j=0;j<nentries;j++) {
      neuronDU[0] = (YNeuron *) fDetectorUnit->fSCNetwork->GetNetwork().At(j);  
      neuronDU[0]->fSCNetworkNeuron = true;
      neuronDU[0]->SetLFEnprong(fNpronged);                 
      neuronDU[0]->SetLFEmemory(fNpronged);      
      neuronDU[0]->SetWeight(0);
      neuronDU[0]->SetFitModel(FITMODEL); 
   }      
            
   std::cout<<"[ITS IB]"<<std::endl;
   nentries = fDetectorUnit->SubUnit[0]->fSCNetwork->GetSynapses().GetEntriesFast();
   for (j=0;j<nentries;j++) { 
      synapseDU[1] = (YSynapse *) fDetectorUnit->SubUnit[0]->fSCNetwork->GetSynapses().At(j);  
      synapseDU[1]->SetWeight(0);            
   }
   nentries = fDetectorUnit->SubUnit[0]->fSCNetwork->GetNetwork().GetEntriesFast();
   for (j=0;j<nentries;j++) {
      neuronDU[1] = (YNeuron *) fDetectorUnit->SubUnit[0]->fSCNetwork->GetNetwork().At(j);  
      neuronDU[1]->fSCNetworkNeuron = true;
      neuronDU[1]->SetLFEnprong(fNpronged);                 
      neuronDU[1]->SetLFEmemory(fNpronged);      
      neuronDU[1]->SetWeight(0);
      neuronDU[1]->SetFitModel(FITMODEL); 
   }  
   for(int hs=0; hs<nHalfStaveIB; hs++){        
      //std::cout<<">HS "<<hs<<std::endl;
      nentries = fDetectorUnit->SubUnit[0]->SubUnit[hs]->fSCNetwork->GetSynapses().GetEntriesFast();
      for (j=0;j<nentries;j++) { 
         synapseDU[2] = (YSynapse *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->fSCNetwork->GetSynapses().At(j);  
         synapseDU[2]->SetWeight(0);
      }
      nentries = fDetectorUnit->SubUnit[0]->SubUnit[hs]->fSCNetwork->GetNetwork().GetEntriesFast();
      for (j=0;j<nentries;j++) {
         neuronDU[2] = (YNeuron *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->fSCNetwork->GetNetwork().At(j);  
         neuronDU[2]->fSCNetworkNeuron = true;
         neuronDU[2]->SetLFEnprong(fNpronged);                 
         neuronDU[2]->SetLFEmemory(fNpronged);      
         neuronDU[2]->SetWeight(0);
         neuronDU[2]->SetFitModel(FITMODEL); 
      } 
      for(int cz=0; cz<nChipsPerHicIB/nHalfStaveIB; cz++){
         int chipIDinStave = (nChipsPerHicIB/nHalfStaveIB)*hs + cz;      
         //std::cout<<"  >ChipZ "<<cz<<" "<<chipIDinStave<<std::endl;
         nentries = fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetSynapses().GetEntriesFast();
         for (j=0;j<nentries;j++) { 
            synapseDU[3] = (YSynapse *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetSynapses().At(j);  
            synapseDU[3]->SetWeight(0);
         }
         nentries = fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetNetwork().GetEntriesFast();
         for (j=0;j<nentries;j++) {
            neuronDU[3] = (YNeuron *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetNetwork().At(j);  
            neuronDU[3]->fSCNetworkNeuron = true;
            neuronDU[3]->SetLFEnprong(fNpronged);                 
            neuronDU[3]->SetLFEmemory(fNpronged);      
            neuronDU[3]->SetWeight(0);
            neuronDU[3]->SetFitModel(FITMODEL); 
         }    
         for(int l=0; l<nLAYERIB; l++){ 
            int lIB = l;
            //std::cout<<"    >Layer "<<l<<std::endl;
            nentries = fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->fSCNetwork->GetSynapses().GetEntriesFast();
            for (j=0;j<nentries;j++) { 
               synapseDU[4] = (YSynapse *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->fSCNetwork->GetSynapses().At(j);  
               synapseDU[4]->SetWeight(0);
            }
            nentries = fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->fSCNetwork->GetNetwork().GetEntriesFast();
            for (j=0;j<nentries;j++) {
               neuronDU[4] = (YNeuron *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->fSCNetwork->GetNetwork().At(j);  
               neuronDU[4]->fSCNetworkNeuron = true;
               neuronDU[4]->SetLFEnprong(fNpronged);                 
               neuronDU[4]->SetLFEmemory(fNpronged);      
               neuronDU[4]->SetWeight(0);
               neuronDU[4]->SetFitModel(FITMODEL); 
            }                                                                    
            for(int hb=0; hb<2; hb++){ 
               //std::cout<<"        >HB "<<hb<<std::endl; 
               nentries = fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->fSCNetwork->GetSynapses().GetEntriesFast();
               for (j=0;j<nentries;j++) { 
                  synapseDU[5] = (YSynapse *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->fSCNetwork->GetSynapses().At(j);  
                  synapseDU[5]->SetWeight(0);
               }
               nentries = fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->fSCNetwork->GetNetwork().GetEntriesFast();
               for (j=0;j<nentries;j++) {
                  neuronDU[5] = (YNeuron *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->fSCNetwork->GetNetwork().At(j);  
                  neuronDU[5]->fSCNetworkNeuron = true;
                  neuronDU[5]->SetLFEnprong(fNpronged);                 
                  neuronDU[5]->SetLFEmemory(fNpronged);      
                  neuronDU[5]->SetWeight(0);
                  neuronDU[5]->SetFitModel(FITMODEL); 
               }
               for(int st=0; st<NStaves[l]/2;st++){
                  //std::cout<<"        >Stave "<<(NStaves[l]/2)*hb + st<<std::endl; 
                  nentries = fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetSynapses().GetEntriesFast();
                  for (j=0;j<nentries;j++) { 
                     synapseDU[6] = (YSynapse *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetSynapses().At(j);  
                     synapseDU[6]->SetWeight(0);
                  }
                  nentries = fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetNetwork().GetEntriesFast();
                  for (j=0;j<nentries;j++) {
                     neuronDU[6] = (YNeuron *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetNetwork().At(j);  
                     neuronDU[6]->fSCNetworkNeuron = true;
                     neuronDU[6]->SetLFEnprong(fNpronged);                 
                     neuronDU[6]->SetLFEmemory(fNpronged);      
                     neuronDU[6]->SetWeight(0);
                     neuronDU[6]->SetFitModel(FITMODEL); 
                  }
                  for(int bst=0; bst<nHalfStaveIB; bst++){        
                     //std::cout<<"          >SubStave "<<bst<<std::endl;
                     nentries = fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetSynapses().GetEntriesFast();
                     for (j=0;j<nentries;j++) { 
                        synapseDU[7] = (YSynapse *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetSynapses().At(j);  
                        synapseDU[7]->SetWeight(0);
                     }
                     nentries = fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetNetwork().GetEntriesFast();
                     for (j=0;j<nentries;j++) {
                        neuronDU[7] = (YNeuron *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetNetwork().At(j);  
                        neuronDU[7]->fSCNetworkNeuron = true;
                        neuronDU[7]->SetLFEnprong(fNpronged);                 
                        neuronDU[7]->SetLFEmemory(fNpronged);      
                        neuronDU[7]->SetWeight(0);
                        neuronDU[7]->SetFitModel(FITMODEL); 
                     }
                  }
               }              
            }       
         }          
      }
   } 
   std::cout<<"[ITS OB1]"<<std::endl;
   nentries = fDetectorUnit->SubUnit[1]->fSCNetwork->GetSynapses().GetEntriesFast();
   for (j=0;j<nentries;j++) { 
      synapseDU[1] = (YSynapse *) fDetectorUnit->SubUnit[1]->fSCNetwork->GetSynapses().At(j);  
      synapseDU[1]->SetWeight(0);            
   }
   nentries = fDetectorUnit->SubUnit[1]->fSCNetwork->GetNetwork().GetEntriesFast();
   for (j=0;j<nentries;j++) {
      neuronDU[1] = (YNeuron *) fDetectorUnit->SubUnit[1]->fSCNetwork->GetNetwork().At(j);  
      neuronDU[1]->fSCNetworkNeuron = true;
      neuronDU[1]->SetLFEnprong(fNpronged);                 
      neuronDU[1]->SetLFEmemory(fNpronged);      
      neuronDU[1]->SetWeight(0);
      neuronDU[1]->SetFitModel(FITMODEL); 
   }  
   for(int hs=0; hs<nHalfStaveOB1; hs++){        
      //std::cout<<">HS "<<hs<<std::endl;
      nentries = fDetectorUnit->SubUnit[1]->SubUnit[hs]->fSCNetwork->GetSynapses().GetEntriesFast();
      for (j=0;j<nentries;j++) { 
         synapseDU[2] = (YSynapse *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->fSCNetwork->GetSynapses().At(j);  
         synapseDU[2]->SetWeight(0);
      }
      nentries = fDetectorUnit->SubUnit[1]->SubUnit[hs]->fSCNetwork->GetNetwork().GetEntriesFast();
      for (j=0;j<nentries;j++) {
         neuronDU[2] = (YNeuron *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->fSCNetwork->GetNetwork().At(j);  
         neuronDU[2]->fSCNetworkNeuron = true;
         neuronDU[2]->SetLFEnprong(fNpronged);                 
         neuronDU[2]->SetLFEmemory(fNpronged);      
         neuronDU[2]->SetWeight(0);
         neuronDU[2]->SetFitModel(FITMODEL); 
      } 
      for(int cz=0; cz<nChipsPerHicOB1/nHalfStaveOB1; cz++){
         int chipIDinStave = (nChipsPerHicOB1/nHalfStaveOB1)*hs + cz;      
         //std::cout<<"  >ChipZ "<<cz<<" "<<chipIDinStave<<std::endl;
         nentries = fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetSynapses().GetEntriesFast();
         for (j=0;j<nentries;j++) { 
            synapseDU[3] = (YSynapse *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetSynapses().At(j);  
            synapseDU[3]->SetWeight(0);
         }
         nentries = fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetNetwork().GetEntriesFast();
         for (j=0;j<nentries;j++) {
            neuronDU[3] = (YNeuron *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetNetwork().At(j);  
            neuronDU[3]->fSCNetworkNeuron = true;
            neuronDU[3]->SetLFEnprong(fNpronged);                 
            neuronDU[3]->SetLFEmemory(fNpronged);      
            neuronDU[3]->SetWeight(0);
            neuronDU[3]->SetFitModel(FITMODEL); 
         }    
         for(int l=nLAYERIB; l<nLAYERIB + nLAYEROB1; l++){ 
            int lOB = l-nLAYERIB;
            //std::cout<<"    >Layer "<<l<<std::endl;
            nentries = fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->GetSynapses().GetEntriesFast();
            for (j=0;j<nentries;j++) { 
               synapseDU[4] = (YSynapse *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->GetSynapses().At(j);  
               synapseDU[4]->SetWeight(0);
            }
            nentries = fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->GetNetwork().GetEntriesFast();
            for (j=0;j<nentries;j++) {
               neuronDU[4] = (YNeuron *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->GetNetwork().At(j);  
               neuronDU[4]->fSCNetworkNeuron = true;
               neuronDU[4]->SetLFEnprong(fNpronged);                 
               neuronDU[4]->SetLFEmemory(fNpronged);      
               neuronDU[4]->SetWeight(0);
               neuronDU[4]->SetFitModel(FITMODEL); 
            }                                                                    
            for(int hb=0; hb<2; hb++){ 
               //std::cout<<"        >HB "<<hb<<std::endl; 
               nentries = fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->GetSynapses().GetEntriesFast();
               for (j=0;j<nentries;j++) { 
                  synapseDU[5] = (YSynapse *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->GetSynapses().At(j);  
                  synapseDU[5]->SetWeight(0);
               }
               nentries = fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->GetNetwork().GetEntriesFast();
               for (j=0;j<nentries;j++) {
                  neuronDU[5] = (YNeuron *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->GetNetwork().At(j);  
                  neuronDU[5]->fSCNetworkNeuron = true;
                  neuronDU[5]->SetLFEnprong(fNpronged);                 
                  neuronDU[5]->SetLFEmemory(fNpronged);      
                  neuronDU[5]->SetWeight(0);
                  neuronDU[5]->SetFitModel(FITMODEL); 
               }
               for(int st=0; st<NStaves[l]/2;st++){
                  //std::cout<<"        >Stave "<<(NStaves[l]/2)*hb + st<<std::endl; 
                  nentries = fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetSynapses().GetEntriesFast();
                  for (j=0;j<nentries;j++) { 
                     synapseDU[6] = (YSynapse *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetSynapses().At(j);  
                     synapseDU[6]->SetWeight(0);
                  }
                  nentries = fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetNetwork().GetEntriesFast();
                  for (j=0;j<nentries;j++) {
                     neuronDU[6] = (YNeuron *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetNetwork().At(j);  
                     neuronDU[6]->fSCNetworkNeuron = true;
                     neuronDU[6]->SetLFEnprong(fNpronged);                 
                     neuronDU[6]->SetLFEmemory(fNpronged);      
                     neuronDU[6]->SetWeight(0);
                     neuronDU[6]->SetFitModel(FITMODEL); 
                  }
                  for(int bst=0; bst<nHalfStaveOB1; bst++){        
                     //std::cout<<"          >SubStave "<<bst<<std::endl;
                     nentries = fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetSynapses().GetEntriesFast();
                     for (j=0;j<nentries;j++) { 
                        synapseDU[7] = (YSynapse *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetSynapses().At(j);  
                        synapseDU[7]->SetWeight(0);
                     }
                     nentries = fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetNetwork().GetEntriesFast();
                     for (j=0;j<nentries;j++) {
                        neuronDU[7] = (YNeuron *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetNetwork().At(j);  
                        neuronDU[7]->fSCNetworkNeuron = true;
                        neuronDU[7]->SetLFEnprong(fNpronged);                 
                        neuronDU[7]->SetLFEmemory(fNpronged);      
                        neuronDU[7]->SetWeight(0);
                        neuronDU[7]->SetFitModel(FITMODEL); 
                     }
                  }
               }                      
            }       
         }          
      }
   } 
   std::cout<<"[ITS OB2]"<<std::endl;
   nentries = fDetectorUnit->SubUnit[2]->fSCNetwork->GetSynapses().GetEntriesFast();
   for (j=0;j<nentries;j++) { 
      synapseDU[1] = (YSynapse *) fDetectorUnit->SubUnit[2]->fSCNetwork->GetSynapses().At(j);  
      synapseDU[1]->SetWeight(0);            
   }
   nentries = fDetectorUnit->SubUnit[2]->fSCNetwork->GetNetwork().GetEntriesFast();
   for (j=0;j<nentries;j++) {
      neuronDU[1] = (YNeuron *) fDetectorUnit->SubUnit[2]->fSCNetwork->GetNetwork().At(j);  
      neuronDU[1]->fSCNetworkNeuron = true;
      neuronDU[1]->SetLFEnprong(fNpronged);                 
      neuronDU[1]->SetLFEmemory(fNpronged);      
      neuronDU[1]->SetWeight(0);
      neuronDU[1]->SetFitModel(FITMODEL); 
   }  
   for(int hs=0; hs<nHalfStaveOB2; hs++){        
      //std::cout<<">HS "<<hs<<std::endl;
      nentries = fDetectorUnit->SubUnit[2]->SubUnit[hs]->fSCNetwork->GetSynapses().GetEntriesFast();
      for (j=0;j<nentries;j++) { 
         synapseDU[2] = (YSynapse *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->fSCNetwork->GetSynapses().At(j);  
         synapseDU[2]->SetWeight(0);
      }
      nentries = fDetectorUnit->SubUnit[2]->SubUnit[hs]->fSCNetwork->GetNetwork().GetEntriesFast();
      for (j=0;j<nentries;j++) {
         neuronDU[2] = (YNeuron *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->fSCNetwork->GetNetwork().At(j);  
         neuronDU[2]->fSCNetworkNeuron = true;
         neuronDU[2]->SetLFEnprong(fNpronged);                 
         neuronDU[2]->SetLFEmemory(fNpronged);      
         neuronDU[2]->SetWeight(0);
         neuronDU[2]->SetFitModel(FITMODEL); 
      } 
      for(int cz=0; cz<nChipsPerHicOB2/nHalfStaveOB2; cz++){
         int chipIDinStave = (nChipsPerHicOB2/nHalfStaveOB2)*hs + cz;      
         //std::cout<<"  >ChipZ "<<cz<<" "<<chipIDinStave<<std::endl;
         nentries = fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetSynapses().GetEntriesFast();
         for (j=0;j<nentries;j++) { 
            synapseDU[3] = (YSynapse *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetSynapses().At(j);  
            synapseDU[3]->SetWeight(0);
         }
         nentries = fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetNetwork().GetEntriesFast();
         for (j=0;j<nentries;j++) {
            neuronDU[3] = (YNeuron *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetNetwork().At(j);  
            neuronDU[3]->fSCNetworkNeuron = true;
            neuronDU[3]->SetLFEnprong(fNpronged);                 
            neuronDU[3]->SetLFEmemory(fNpronged);      
            neuronDU[3]->SetWeight(0);
            neuronDU[3]->SetFitModel(FITMODEL); 
         }    
         for(int l=nLAYERIB + nLAYEROB1; l<nLAYER; l++){ 
            int lOB = l - (nLAYERIB + nLAYEROB1);
            //std::cout<<"    >Layer "<<l<<std::endl;
            nentries = fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->GetSynapses().GetEntriesFast();
            for (j=0;j<nentries;j++) { 
               synapseDU[4] = (YSynapse *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->GetSynapses().At(j);  
               synapseDU[4]->SetWeight(0);
            }
            nentries = fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->GetNetwork().GetEntriesFast();
            for (j=0;j<nentries;j++) {
               neuronDU[4] = (YNeuron *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->GetNetwork().At(j);  
               neuronDU[4]->fSCNetworkNeuron = true;
               neuronDU[4]->SetLFEnprong(fNpronged);                 
               neuronDU[4]->SetLFEmemory(fNpronged);      
               neuronDU[4]->SetWeight(0);
               neuronDU[4]->SetFitModel(FITMODEL); 
            }                                                                    
            for(int hb=0; hb<2; hb++){ 
               //std::cout<<"        >HB "<<hb<<std::endl; 
               nentries = fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->GetSynapses().GetEntriesFast();
               for (j=0;j<nentries;j++) { 
                  synapseDU[5] = (YSynapse *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->GetSynapses().At(j);  
                  synapseDU[5]->SetWeight(0);
               }
               nentries = fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->GetNetwork().GetEntriesFast();
               for (j=0;j<nentries;j++) {
                  neuronDU[5] = (YNeuron *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->GetNetwork().At(j);  
                  neuronDU[5]->fSCNetworkNeuron = true;
                  neuronDU[5]->SetLFEnprong(fNpronged);                 
                  neuronDU[5]->SetLFEmemory(fNpronged);      
                  neuronDU[5]->SetWeight(0);
                  neuronDU[5]->SetFitModel(FITMODEL); 
               }
               for(int st=0; st<NStaves[l]/2;st++){
                  //std::cout<<"        >Stave "<<(NStaves[l]/2)*hb + st<<std::endl; 
                  nentries = fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetSynapses().GetEntriesFast();
                  for (j=0;j<nentries;j++) { 
                     synapseDU[6] = (YSynapse *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetSynapses().At(j);  
                     synapseDU[6]->SetWeight(0);
                  }
                  nentries = fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetNetwork().GetEntriesFast();
                  for (j=0;j<nentries;j++) {
                     neuronDU[6] = (YNeuron *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetNetwork().At(j);  
                     neuronDU[6]->fSCNetworkNeuron = true;
                     neuronDU[6]->SetLFEnprong(fNpronged);                 
                     neuronDU[6]->SetLFEmemory(fNpronged);      
                     neuronDU[6]->SetWeight(0);
                     neuronDU[6]->SetFitModel(FITMODEL); 
                  }
                  for(int bst=0; bst<nHalfStaveOB2; bst++){        
                     //std::cout<<"          >SubStave "<<bst<<std::endl;
                     nentries = fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetSynapses().GetEntriesFast();
                     for (j=0;j<nentries;j++) { 
                        synapseDU[7] = (YSynapse *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetSynapses().At(j);  
                        synapseDU[7]->SetWeight(0);
                     }
                     nentries = fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetNetwork().GetEntriesFast();
                     for (j=0;j<nentries;j++) {
                        neuronDU[7] = (YNeuron *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetNetwork().At(j);  
                        neuronDU[7]->fSCNetworkNeuron = true;
                        neuronDU[7]->SetLFEnprong(fNpronged);                 
                        neuronDU[7]->SetLFEmemory(fNpronged);      
                        neuronDU[7]->SetWeight(0);
                        neuronDU[7]->SetFitModel(FITMODEL); 
                     }
                  }
               }                       
            }       
         }          
      }
   }   
   
}

void YMultiLayerPerceptron::SetNpronged(int nprong)
{
   fNpronged = nprong;
   for(int t=0; t<fNpronged; t++){
      for(int i=0; i<3*nLAYER; i++){
         fSCNeuronDcdw[i].push_back(0); //[layer][track] (cs1, cs2, cs3)                    
      }
   }
}

void YMultiLayerPerceptron::SetWeightMonitoring(const char* WeightName, int step)
{
   fWeightName = WeightName;
   fWeightName.Resize(fWeightName.Sizeof()-5);
   fWeightStep = step;
}

void YMultiLayerPerceptron::SetPrevUSL(const char* PrevUSLName)
{
   fPrevUSLName = PrevUSLName;
}

void YMultiLayerPerceptron::SetPrevWeight(const char* PrevWeightName)
{
   fPrevWeightName = PrevWeightName;
}

void YMultiLayerPerceptron::SetPrevWeightDetectorUnit(const char* PrevWeightDetectorName)
{
   fPrevWeightDetectorUnitName = PrevWeightDetectorName;
}

////////////////////////////////////////////////////////////////////////////////
/// Connects the TTree to Neurons in input and output
/// layers. The formulas associated to each neuron are created
/// and reported to the network formula manager.
/// By default, the branch is not normalised since this would degrade
/// performance for classification jobs.
/// Normalisation can be requested by putting '@' in front of the formula.

void YMultiLayerPerceptron::AttachData()
{   
   Int_t j = 0;
   YNeuron *neuron = 0;
   Bool_t normalize = false;
   fManager = new TTreeFormulaManager;

   // Set the size of the internal array of parameters of the formula
   Int_t maxop, maxpar, maxconst;
   ROOT::v5::TFormula::GetMaxima(maxop, maxpar, maxconst);
   ROOT::v5::TFormula::SetMaxima(50000, 50000, 50000);
   
   //first layer
   const TString input = TString(fStructure(0, fStructure.First(':')));
   const TObjArray *inpL = input.Tokenize(", ");
   Int_t nentries = (fFirstLayer.GetEntriesFast());
   // make sure nentries == entries in inpL
   std::cout<<"Attach Data:: input "<<input<<" nentries : "<<nentries<<" inpL->GetLast()+1 : "<<inpL->GetLast()+1<<std::endl;   
   R__ASSERT( nentries == inpL->GetLast()+1);
   for (int t=0; t<nTrackMax; t++ ) {
      for (j=0;j<nentries;j++) {
         normalize = false;
         TString branchinp= ((TObjString *)inpL->At(j))->GetString();
         int bpos = branchinp.First("[");          
         const TString brName = ((TString) ((TObjString *)inpL->At(j))->GetString()).Insert(bpos,"[" + TString::Itoa(t,10) + "]");  
         std::cout<<"AttachData:: brName input ["<<(int)(t)<<"]["<<(int)(j%(2*nLAYER))<<"] "<<brName<<std::endl;  //j -> (int)(j%(2*nLAYER))   
         neuron = (YNeuron *) fFirstLayer.At((int)(j%(2*nLAYER)));	
         neuron->SetNeuronIndex(t);					//j -> (int)(j%(2*nLAYER))
         if (brName[0]=='@')
            normalize = true;
         fManager->Add(neuron->UseBranch(fData,brName.Data() + (normalize?1:0)));
         if(!normalize) neuron->SetNormalisation(0., 1.);
      } 
   }
   delete inpL;
   std::cout<<std::endl;
   
   // last layer
   TString output = TString(
           fStructure(fStructure.Last(':') + 1,
                      fStructure.Last('|') - fStructure.Last(':') - 1));
   const TObjArray *outL = output.Tokenize(", ");
   nentries = (fLastLayer.GetEntriesFast());
   // make sure nentries == entries in outL
   std::cout<<"Attach Data:: output "<<output<<" nentries : "<<nentries<<" outL->GetLast()+1 : "<<outL->GetLast()+1<<std::endl;      
   R__ASSERT(nentries == outL->GetLast()+1);
   for (int t=0; t<nTrackMax; t++ ) {   
      for (j=0;j<nentries;j++) {
         normalize = false;
         TString branchout = ((TObjString *)outL->At(j))->GetString();
         int bpos = branchout.First("[");         
         const TString brName = ((TString) ((TObjString *)outL->At(j))->GetString()).Insert(bpos,"[" + TString::Itoa(t,10) + "]");  
         std::cout<<"AttachData:: brName output ["<<(int)(t)<<"]["<<(int)(j%(3*nLAYER))<<"] "<<brName<<std::endl;	//j -> (int)(j%(3*nLAYER))
         neuron = (YNeuron *) fLastLayer.At((int)(j%(3*nLAYER)));	
         neuron->SetNeuronIndex(t);						//j -> (int)(j%(3*nLAYER))
         if (brName[0]=='@')
            normalize = true;
         fManager->Add(neuron->UseBranch(fData,brName.Data() + (normalize?1:0)));
         if(!normalize) neuron->SetNormalisation(0., 1.);
      }
   }
   delete outL;
   std::cout<<std::endl;
   
   //additional info layer
   TString addition = TString(
           fStructure(fStructure.Last('|') + 1,
                      fStructure.Length() - fStructure.Last('|')));
   const TObjArray *addL = addition.Tokenize(", ");
   nentries = (6+2);
   // make sure nentries == entries in addL
   std::cout<<"Attach Data:: addition "<<addition<<" nentries : "<<nentries<<" addL->GetLast()+1 : "<<addL->GetLast()+1<<std::endl;      
   R__ASSERT(nentries == addL->GetLast()+1);
   for (j=0;j<nentries;j++) {
      normalize = false;
      const TString brName = ((TObjString *)addL->At(j))->GetString();  
      std::cout<<"AttachData:: brName addition ["<<(int)(j/nentries)<<"]["<<(int)(j%nentries)<<"] "<<brName<<std::endl;	
      neuron = (YNeuron *) fAddition.At(0);	
      neuron->SetNeuronIndex(1000 + j);						
      if (brName[0]=='@')
         normalize = true;
      fManager->Add(neuron->UseBranchAddition(fData,brName.Data() + (normalize?1:0)));
      if(!normalize) neuron->SetNormalisation(0., 1.);
   }
   delete addL;
   std::cout<<std::endl;


   fManager->Add((fEventWeight = new TTreeFormula("NNweight",fWeight.Data(),fData)));
   //fManager->Sync();

   // Set the old values
   //ROOT::v5::TFormula::SetMaxima(maxop, maxpar, maxconst);
}

////////////////////////////////////////////////////////////////////////////////
/// Expand the structure of the first layer

void YMultiLayerPerceptron::ExpandStructure()
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
      TTreeFormula f("sizeTestFormula",name,fData);
      // Variable size arrays are unrelialable
      if(f.GetMultiplicity()==1 && f.GetNdata()>1) {
         //Warning("YMultiLayerPerceptron::ExpandStructure()","Variable size arrays cannot be used to build implicitely an input layer. The index 0 will be assumed.");
      }
      // Check if we are coping with an array... then expand
      // The array operator used is {}. It is detected in YNeuron, and
      // passed directly as instance index of the TTreeFormula,
      // so that complex compounds made of arrays can be used without
      // parsing the details.
      else if(f.GetNdata()>1) {
         for(Int_t j=0; j<f.GetNdata(); j++) {
            if(i||j) newInput += ",";
            newInput += name;
            newInput += "{";
            newInput += j;
            newInput += "}";
         }
         continue;
      }
      if(i) newInput += ",";
      newInput += name;
   }
   delete inpL;

   // Save the result
   fStructure = newInput + ":" + hiddenAndOutput;
}

void YMultiLayerPerceptron::PrepareChipsToNetwork(int mode = 0)
{
   if(mode==0){
      NetworkChips = new int[nSensors];
      for(int ichipID =0; ichipID<nSensors; ichipID++){
         NetworkChips[ichipID] = ichipID;
      }
   } else {
      //User Define//
      NetworkChips = new int[27]; 
      //layer 0 stave 0
      NetworkChips[0] = 0;
      NetworkChips[1] = 1;
      NetworkChips[2] = 2;
      NetworkChips[3] = 3;
      NetworkChips[4] = 4;
      NetworkChips[5] = 5;
      NetworkChips[6] = 6;
      NetworkChips[7] = 7;
      NetworkChips[8] = 8;
      //layer 1 stave 0
      NetworkChips[9]  = 108;
      NetworkChips[10] = 109;
      NetworkChips[11] = 110;
      NetworkChips[12] = 111;
      NetworkChips[13] = 112;
      NetworkChips[14] = 113;
      NetworkChips[15] = 114;
      NetworkChips[16] = 115;
      NetworkChips[17] = 116;
      //layer 2 stave 0
      NetworkChips[18] = 252;
      NetworkChips[19] = 253;
      NetworkChips[20] = 254;
      NetworkChips[21] = 255;
      NetworkChips[22] = 256;
      NetworkChips[23] = 257;
      NetworkChips[24] = 258;
      NetworkChips[25] = 259;
      NetworkChips[26] = 260;
   }

}


void YMultiLayerPerceptron::SetNetworkUpdateState(std::vector<bool> chiplist)
{
   for(int iID = 0; iID <nSensors; iID++){  
      std::cout<<"chipID "<<iID<<" "<<chiplist[iID]<<std::endl;
      fSCNetwork[iID]->SetUpdateState(chiplist[iID]);
   }
}

void YMultiLayerPerceptron::InitDetectorUnitNetworkUpdateState()
{

   std::cout<<"YMultiLayerPerceptron::SetDetectorUnitNetworkUpdateState"<<std::endl;
   
   std::cout<<"[ITS IB]"<<std::endl;
   fDetectorUnit->SubUnit[0]->fSCNetwork->SetUpdateState(false);
   for(int hs=0; hs<nHalfStaveIB; hs++){        
      fDetectorUnit->SubUnit[0]->SubUnit[hs]->fSCNetwork->SetUpdateState(false);
      for(int cz=0; cz<nChipsPerHicIB/nHalfStaveIB; cz++){
         int chipIDinStave = (nChipsPerHicIB/nHalfStaveIB)*hs + cz;      
         fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->SetUpdateState(false);
         for(int l=0; l<nLAYERIB; l++){ 
            int lIB = l;
            fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->fSCNetwork->SetUpdateState(false);
            for(int hb=0; hb<2; hb++){        
               fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->fSCNetwork->SetUpdateState(false);
               for(int st=0; st<NStaves[l]/2;st++){
                  int staveID = (NStaves[l]/2)*hb + st;
                  fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->SetUpdateState(false);
                  for(int bst=0; bst<nHalfStaveIB; bst++){        
                     fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->SetUpdateState(false);
                  }
               }                
            }
         }
      }
   }
   std::cout<<"[ITS OB1]"<<std::endl;
   fDetectorUnit->SubUnit[1]->fSCNetwork->SetUpdateState(false);
   for(int hs=0; hs<nHalfStaveOB1; hs++){        
      fDetectorUnit->SubUnit[1]->SubUnit[hs]->fSCNetwork->SetUpdateState(false);
      for(int cz=0; cz<nChipsPerHicOB1/nHalfStaveOB1; cz++){
         int chipIDinStave = (nChipsPerHicOB1/nHalfStaveOB1)*hs + cz;      
         fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->SetUpdateState(false);
         for(int l=nLAYERIB; l<nLAYERIB + nLAYEROB1; l++){ 
            int lOB = l - nLAYERIB;
            fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->SetUpdateState(false);
            for(int hb=0; hb<2; hb++){        
               fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->SetUpdateState(false);
               for(int st=0; st<NStaves[l]/2;st++){
                  int staveID = (NStaves[l]/2)*hb + st;
                  fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->SetUpdateState(false);
                  for(int bst=0; bst<nHalfStaveOB1; bst++){        
                     fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->SetUpdateState(false);
                  }
               }                
            }
         }
      }
   }
   std::cout<<"[ITS OB2]"<<std::endl;
   fDetectorUnit->SubUnit[2]->fSCNetwork->SetUpdateState(false);
   for(int hs=0; hs<nHalfStaveOB2; hs++){        
      fDetectorUnit->SubUnit[2]->SubUnit[hs]->fSCNetwork->SetUpdateState(false);
      for(int cz=0; cz<nChipsPerHicOB2/nHalfStaveOB2; cz++){
         int chipIDinStave = (nChipsPerHicOB2/nHalfStaveOB2)*hs + cz;      
         fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->SetUpdateState(false);
         for(int l=nLAYERIB + nLAYEROB1; l<nLAYER; l++){ 
            int lOB = l - (nLAYERIB + nLAYEROB1);
            fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->SetUpdateState(false);
            for(int hb=0; hb<2; hb++){        
               fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->SetUpdateState(false);
               for(int st=0; st<NStaves[l]/2;st++){
                  int staveID = (NStaves[l]/2)*hb + st;
                  fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->SetUpdateState(false);
                  for(int bst=0; bst<nHalfStaveOB2; bst++){        
                     fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->SetUpdateState(false);
                  }
               }                
            }
         }
      }
   }
}

void YMultiLayerPerceptron::SetDetectorUnitNetworkUpdateStateUser()
{

   std::cout<<"YMultiLayerPerceptron::SetDetectorUnitNetworkUpdateStateUser"<<std::endl;
   
   std::cout<<"[ITS IB]"<<std::endl;
   //fDetectorUnit->SubUnit[0]->fSCNetwork->SetUpdateState(true);
   for(int hs=0; hs<nHalfStaveIB; hs++){        
      //fDetectorUnit->SubUnit[0]->SubUnit[hs]->fSCNetwork->SetUpdateState(true);
      for(int cz=0; cz<nChipsPerHicIB/nHalfStaveIB; cz++){
         int chipIDinStave = (nChipsPerHicIB/nHalfStaveIB)*hs + cz;      
         //fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->SetUpdateState(true);
         for(int l=0; l<nLAYERIB; l++){ 
            int lIB = l;
            //fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->fSCNetwork->SetUpdateState(true);
            for(int hb=0; hb<2; hb++){        
               //fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->fSCNetwork->SetUpdateState(true);
               for(int st=0; st<NStaves[l]/2;st++){
                  //fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->SetUpdateState(true);
                  for(int bst=0; bst<nHalfStaveIB; bst++){        
                     //fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->SetUpdateState(true);
                  }
               }               
            }
         }
      }
   }
   std::cout<<"[ITS OB1]"<<std::endl;
   //fDetectorUnit->SubUnit[1]->fSCNetwork->SetUpdateState(true);
   for(int hs=0; hs<nHalfStaveOB1; hs++){        
      //fDetectorUnit->SubUnit[1]->SubUnit[hs]->fSCNetwork->SetUpdateState(true);
      for(int cz=0; cz<nChipsPerHicOB1/nHalfStaveOB1; cz++){
         int chipIDinStave = (nChipsPerHicOB1/nHalfStaveOB1)*hs + cz;      
         //fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->SetUpdateState(true);
         for(int l=nLAYERIB; l<nLAYERIB + nLAYEROB1; l++){ 
            int lOB = l - nLAYERIB;
            //fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->SetUpdateState(true);
            for(int hb=0; hb<2; hb++){        
               //fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->SetUpdateState(true);
               for(int st=0; st<NStaves[l]/2;st++){
                  //fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->SetUpdateState(true);
                  for(int bst=0; bst<nHalfStaveIB; bst++){        
                     //fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->SetUpdateState(true);
                  }
               }                 
            }
         }
      }
   }
   std::cout<<"[ITS OB2]"<<std::endl;
   //fDetectorUnit->SubUnit[2]->fSCNetwork->SetUpdateState(true);
   for(int hs=0; hs<nHalfStaveOB2; hs++){        
      //fDetectorUnit->SubUnit[2]->SubUnit[hs]->fSCNetwork->SetUpdateState(true);
      for(int cz=0; cz<nChipsPerHicOB2/nHalfStaveOB2; cz++){
         int chipIDinStave = (nChipsPerHicOB2/nHalfStaveOB2)*hs + cz;      
         //fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->SetUpdateState(true);
         for(int l=nLAYERIB + nLAYEROB1; l<nLAYER; l++){ 
            int lOB = l - (nLAYERIB + nLAYEROB1);
            //fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->SetUpdateState(true);
            for(int hb=0; hb<2; hb++){        
               //fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->SetUpdateState(true);
               for(int st=0; st<NStaves[l]/2;st++){
                  //fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->SetUpdateState(true);
                  for(int bst=0; bst<nHalfStaveIB; bst++){        
                     //fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->SetUpdateState(true);
                  }
               }               
            }
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Instanciates the network from the description

void YMultiLayerPerceptron::BuildNetwork()
{
   ExpandStructure();
   TString inputA = TString(fStructure(0, fStructure.First(':')));
   TString Comma = ",";
   //TString input  = "s10,s20,s11,s21,s12,s22";//example
   TString input = "";
   for(int l =0; l< nLAYER; l++){
      input   += Comma + "s1["    + TString::Itoa(l,10) + "]"
               + Comma + "s2["    + TString::Itoa(l,10) + "]";
   }   
   input.Replace(0,1,"");
   TString hidden = TString(
           fStructure(fStructure.First(':') + 1,
                      fStructure.Last(':') - fStructure.First(':') - 1));
   //TString output = TString(
   //        fStructure(fStructure.Last(':') + 1,
   //                   fStructure.Length() - fStructure.Last(':')));
                      
   //TString output = "cs10,cs20,cs30,cs11,cs21,cs31,cs12,cs22,cs32";    //example                         
   TString output = "";
   for(int l =0; l< nLAYER; l++){
      output  += Comma + "cs1["    + TString::Itoa(l,10) + "]"
               + Comma + "cs2["    + TString::Itoa(l,10) + "]"
               + Comma + "cs3["    + TString::Itoa(l,10) + "]";
   }   
   output.Replace(0,1,"");

   std::cout<<"BuildNetwork input "<<input<<" "<<hidden<<" "<<output<<std::endl;
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
   int ninputs = inputA.CountChar('s');
   SetNpronged(ninputs/6);
   if(hidden=='0'){
      std::cout<<"BuildHiddenLayersNonCrossing No Hidden Layer ninputs : "<<6<<std::endl;   
      BuildLastLayerNonCrossing(output, 6);
   }
   BuildAddition();
   nSynapses = fSynapses.GetEntriesFast();
   nNetwork = fNetwork.GetEntriesFast();
}

void YMultiLayerPerceptron::BuildSensorCorrectionNetwork()
{
   std::cout<<"YMultiLayerPerceptron::BuildSensorCorrectionNetwork"<<std::endl;
   fSCNetwork = new YSensorCorrection *[nSensors]; 
   for(int ichipID = 0; ichipID <nSensors; ichipID++){
      int Layer           = yGEOM->GetLayer(ichipID);
      fSCNetwork[ichipID] = new YSensorCorrection(NetworkChips[ichipID]); 
      if(Layer<nLAYERIB){
         //  for IB
         //   8  7  6  5  4  3  2  1  0  (-) 
         fSCNetwork[ichipID]->SetMntDir(-1);
      } else {
         //  for OB (chipID in Module)
         //   6  5  4  3  2  1  0    	 (-)          
         //   7  8  9 10 11 12 13     	 (+)  -> Z    
         int ChipIdInModule  = yGEOM->GetChipIdInModule(ichipID); 
         short mntdir        = (ChipIdInModule<7) ? -1 : +1;
         fSCNetwork[ichipID]->SetMntDir(mntdir);
      }
   } 
}

void YMultiLayerPerceptron::BuildDetectorUnitNetwork()
{
   nChipsPerHicIB  = nHicPerStave[0]*nChipsPerHic[0];
   nChipsPerHicOB1 = nHicPerStave[3]*nChipsPerHic[3];
   nChipsPerHicOB2 = nHicPerStave[5]*nChipsPerHic[5];   

   nHalfStaveIB  = NSubStave[0];
   nHalfStaveOB1 = NSubStave[3];
   nHalfStaveOB2 = NSubStave[5];
   
   fDetectorUnit = new YDetectorUnit(0,0); 

   // Detector 0, Detector(IB/OB1/OB2) 1, HalfStave 2, ChipZ 3, Layer 4, HalfBarrel 5						   
   std::cout<<"YMultiLayerPerceptron::BuildDetectorUnitNetwork"<<std::endl;

   std::cout<<"YMultiLayerPerceptron::BuildDetectorUnitNetwork IB "<<std::endl;
   YDetectorUnit* detITSIB = new YDetectorUnit(1,0); 
   for(int hs=0; hs<nHalfStaveIB; hs++){        
      //std::cout<<">HS "<<hs<<std::endl;
      detITSIB->SubUnit.push_back(new YDetectorUnit(2,hs));    
      for(int cz=0; cz<nChipsPerHicIB/nHalfStaveIB; cz++){
         int chipIDinStave = (nChipsPerHicIB/nHalfStaveIB)*hs + cz;      
         //std::cout<<"  >ChipZ "<<cz<<" "<<chipIDinStave<<std::endl;
         detITSIB->SubUnit[hs]->SubUnit.push_back(new YDetectorUnit(3,chipIDinStave));  
         for(int l=0; l<nLAYERIB; l++){ 
            int lIB = l;
            //std::cout<<"    >Layer "<<l<<std::endl;
            detITSIB->SubUnit[hs]->SubUnit[cz]->SubUnit.push_back(new YDetectorUnit(4,l));
            for(int hb=0; hb<2; hb++){        
               //std::cout<<"      >HB "<<hb<<std::endl;
               detITSIB->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit.push_back(new YDetectorUnit(5,hb));  
               for(int st=0; st<NStaves[l]/2;st++){
                  int staveID = (NStaves[l]/2)*hb + st;
                  //std::cout<<"        >Stave "<<(NStaves[l]/2)*hb + st<<std::endl; 
                  detITSIB->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit.push_back(new YDetectorUnit(6,staveID));  
                  for(int bst=0; bst<nHalfStaveIB; bst++){        
                     //std::cout<<"          >SubStave "<<bst<<std::endl;
                     detITSIB->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->SubUnit.push_back(new YDetectorUnit(7,bst));  
                  }
               }                    
            }
         }
      }
   }
   for(int l=0; l<nLAYERIB; l++){ 
      int lIB = l;
      for(int b=0; b<nSensorsbyLayer[l];b++){
         int SensorID 	= b+ChipBoundary[l];
         int HalfBarrel	= yGEOM->GetHalfBarrel(SensorID);  
         int Stave	= yGEOM->GetStave(SensorID); 
         int StaveID    = Stave%(NStaves[l]/2);
         int HalfStave	= yGEOM->GetHalfStave(SensorID);  
         int Module	= yGEOM->GetModule(SensorID);         
         int ChipIdInStave = yGEOM->GetChipIdInStave(SensorID);
         int chipZ         = ChipIdInStave%(nHicPerStave[l]*nChipsPerHic[l]/NSubStave[l]);               
         detITSIB->chipID.push_back(SensorID);
         detITSIB->SubUnit[HalfStave]->chipID.push_back(SensorID);
         detITSIB->SubUnit[HalfStave]->SubUnit[chipZ]->chipID.push_back(SensorID);    
         detITSIB->SubUnit[HalfStave]->SubUnit[chipZ]->SubUnit[lIB]->chipID.push_back(SensorID);         
         detITSIB->SubUnit[HalfStave]->SubUnit[chipZ]->SubUnit[lIB]->SubUnit[HalfBarrel]->chipID.push_back(SensorID);      
         detITSIB->SubUnit[HalfStave]->SubUnit[chipZ]->SubUnit[lIB]->SubUnit[HalfBarrel]->SubUnit[StaveID]->chipID.push_back(SensorID);             
         detITSIB->SubUnit[HalfStave]->SubUnit[chipZ]->SubUnit[lIB]->SubUnit[HalfBarrel]->SubUnit[StaveID]->SubUnit[HalfStave]->chipID.push_back(SensorID);                                      
      }
   }
   std::cout<<"YMultiLayerPerceptron::BuildDetectorUnitNetwork OB1"<<std::endl;
   YDetectorUnit* detITSOB1 = new YDetectorUnit(1,1); 
   for(int hs=0; hs<nHalfStaveOB1; hs++){        
      //std::cout<<">HS "<<hs<<std::endl;
      detITSOB1->SubUnit.push_back(new YDetectorUnit(2,hs));    
      for(int cz=0; cz<nChipsPerHicOB1/nHalfStaveOB1; cz++){
         int chipIDinStave = (nChipsPerHicOB1/nHalfStaveOB1)*hs + cz;      
         //std::cout<<"  >ChipZ "<<cz<<" "<<chipIDinStave<<std::endl;
         detITSOB1->SubUnit[hs]->SubUnit.push_back(new YDetectorUnit(3,chipIDinStave));  
         for(int l=nLAYERIB; l<nLAYERIB + nLAYEROB1; l++){ 
            int lOB = l-nLAYERIB;
            //std::cout<<"    >Layer "<<l<<std::endl;
            detITSOB1->SubUnit[hs]->SubUnit[cz]->SubUnit.push_back(new YDetectorUnit(4,l));
            for(int hb=0; hb<2; hb++){        
               //std::cout<<"      >HB "<<hb<<std::endl;
               detITSOB1->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit.push_back(new YDetectorUnit(5,hb));    
               for(int st=0; st<NStaves[l]/2;st++){
                  int staveID = (NStaves[l]/2)*hb + st;
                  //std::cout<<"        >Stave "<<(NStaves[l]/2)*hb + st<<std::endl; 
                  detITSOB1->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit.push_back(new YDetectorUnit(6,staveID));  
                  for(int bst=0; bst<nHalfStaveOB1; bst++){        
                     //std::cout<<"          >SubStave "<<bst<<std::endl;
                     detITSOB1->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit.push_back(new YDetectorUnit(7,bst));  
                  }
               }                                
            }
         }
      }
   }
   for(int l=nLAYERIB; l<nLAYERIB + nLAYEROB1; l++){ 
      int lOB = l-nLAYERIB;
      for(int b=0; b<nSensorsbyLayer[l];b++){
         int SensorID 	= b+ChipBoundary[l];
         int HalfBarrel	= yGEOM->GetHalfBarrel(SensorID);  
         int Stave	= yGEOM->GetStave(SensorID); 
         int StaveID    = Stave%(NStaves[l]/2);
         int HalfStave	= yGEOM->GetHalfStave(SensorID);  
         int Module	= yGEOM->GetModule(SensorID);         
         int ChipIdInStave = yGEOM->GetChipIdInStave(SensorID);
         int chipZ         = ChipIdInStave%(nHicPerStave[l]*nChipsPerHic[l]/NSubStave[l]);               
         detITSOB1->chipID.push_back(SensorID);
         detITSOB1->SubUnit[HalfStave]->chipID.push_back(SensorID);
         detITSOB1->SubUnit[HalfStave]->SubUnit[chipZ]->chipID.push_back(SensorID);    
         detITSOB1->SubUnit[HalfStave]->SubUnit[chipZ]->SubUnit[lOB]->chipID.push_back(SensorID);         
         detITSOB1->SubUnit[HalfStave]->SubUnit[chipZ]->SubUnit[lOB]->SubUnit[HalfBarrel]->chipID.push_back(SensorID);          
         detITSOB1->SubUnit[HalfStave]->SubUnit[chipZ]->SubUnit[lOB]->SubUnit[HalfBarrel]->SubUnit[StaveID]->chipID.push_back(SensorID);             
         detITSOB1->SubUnit[HalfStave]->SubUnit[chipZ]->SubUnit[lOB]->SubUnit[HalfBarrel]->SubUnit[StaveID]->SubUnit[HalfStave]->chipID.push_back(SensorID);                
      }
   }

   std::cout<<"YMultiLayerPerceptron::BuildDetectorUnitNetwork OB2"<<std::endl;
   YDetectorUnit* detITSOB2 = new YDetectorUnit(1,2); 
   for(int hs=0; hs<nHalfStaveOB2; hs++){        
      //std::cout<<">HS "<<hs<<std::endl;
      detITSOB2->SubUnit.push_back(new YDetectorUnit(2,hs));    
      for(int cz=0; cz<nChipsPerHicOB2/nHalfStaveOB2; cz++){
         int chipIDinStave = (nChipsPerHicOB2/nHalfStaveOB2)*hs + cz;      
         //std::cout<<"  >ChipZ "<<cz<<" "<<chipIDinStave<<std::endl;
         detITSOB2->SubUnit[hs]->SubUnit.push_back(new YDetectorUnit(3,chipIDinStave));  
         for(int l=nLAYERIB + nLAYEROB1; l<nLAYER; l++){ 
            int lOB = l - (nLAYERIB + nLAYEROB1);
            //std::cout<<"    >Layer "<<l<<std::endl;
            detITSOB2->SubUnit[hs]->SubUnit[cz]->SubUnit.push_back(new YDetectorUnit(4,l));
            for(int hb=0; hb<2; hb++){        
               //std::cout<<"      >HB "<<hb<<std::endl;
               detITSOB2->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit.push_back(new YDetectorUnit(5,hb));   
               for(int st=0; st<NStaves[l]/2;st++){
                  int staveID = (NStaves[l]/2)*hb + st;
                  //std::cout<<"        >Stave "<<(NStaves[l]/2)*hb + st<<std::endl; 
                  detITSOB2->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit.push_back(new YDetectorUnit(6,staveID));  
                  for(int bst=0; bst<nHalfStaveOB2; bst++){        
                     //std::cout<<"          >SubStave "<<bst<<std::endl;
                     detITSOB2->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit.push_back(new YDetectorUnit(7,bst));  
                  }
               }                                 
            }
         }
      }
   }
   for(int l=nLAYERIB + nLAYEROB1; l<nLAYER; l++){ 
      int lOB = l - (nLAYERIB + nLAYEROB1);
      for(int b=0; b<nSensorsbyLayer[l];b++){
         int SensorID 	= b+ChipBoundary[l];
         int HalfBarrel	= yGEOM->GetHalfBarrel(SensorID);  
         int Stave	= yGEOM->GetStave(SensorID); 
         int StaveID    = Stave%(NStaves[l]/2);
         int HalfStave	= yGEOM->GetHalfStave(SensorID);  
         int Module	= yGEOM->GetModule(SensorID);         
         int ChipIdInStave = yGEOM->GetChipIdInStave(SensorID);
         int chipZ         = ChipIdInStave%(nHicPerStave[l]*nChipsPerHic[l]/NSubStave[l]);               
         detITSOB2->chipID.push_back(SensorID);
         detITSOB2->SubUnit[HalfStave]->chipID.push_back(SensorID);
         detITSOB2->SubUnit[HalfStave]->SubUnit[chipZ]->chipID.push_back(SensorID);    
         detITSOB2->SubUnit[HalfStave]->SubUnit[chipZ]->SubUnit[lOB]->chipID.push_back(SensorID);         
         detITSOB2->SubUnit[HalfStave]->SubUnit[chipZ]->SubUnit[lOB]->SubUnit[HalfBarrel]->chipID.push_back(SensorID);   
         detITSOB2->SubUnit[HalfStave]->SubUnit[chipZ]->SubUnit[lOB]->SubUnit[HalfBarrel]->SubUnit[StaveID]->chipID.push_back(SensorID);             
         detITSOB2->SubUnit[HalfStave]->SubUnit[chipZ]->SubUnit[lOB]->SubUnit[HalfBarrel]->SubUnit[StaveID]->SubUnit[HalfStave]->chipID.push_back(SensorID);                        
      }
   }

   fDetectorUnit->SubUnit.push_back(detITSIB);
   fDetectorUnit->SubUnit.push_back(detITSOB1);
   fDetectorUnit->SubUnit.push_back(detITSOB2);   
}

void YMultiLayerPerceptron::PrintDetectorUnitNetwork()
{

   std::cout<<"YMultiLayerPerceptron::PrintDetectorUnitNetwork"<<std::endl;
   
   std::cout<<"[ITS IB]"<<std::endl;
   fDetectorUnit->SubUnit[0]->showchipIDs();      
   for(int hs=0; hs<nHalfStaveIB; hs++){        
      std::cout<<">HS "<<hs<<std::endl;
      fDetectorUnit->SubUnit[0]->SubUnit[hs]->showchipIDs();            
      for(int cz=0; cz<nChipsPerHicIB/nHalfStaveIB; cz++){
         int chipIDinStave = (nChipsPerHicIB/nHalfStaveIB)*hs + cz;      
         std::cout<<"  >ChipZ "<<cz<<" "<<chipIDinStave<<std::endl;
         fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->showchipIDs();            
         for(int l=0; l<nLAYERIB; l++){ 
            int lIB = l;
            std::cout<<"    >Layer "<<l<<std::endl;
            fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->showchipIDs();            
            for(int hb=0; hb<2; hb++){        
               std::cout<<"      >HB "<<hb<<std::endl;
               fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->showchipIDs();      
               for(int st=0; st<NStaves[l]/2;st++){
                  int staveID = (NStaves[l]/2)*hb + st;
                  std::cout<<"        >Stave "<<(NStaves[l]/2)*hb + st<<std::endl; 
                  fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->showchipIDs();  
                  for(int bst=0; bst<nHalfStaveIB; bst++){        
                     std::cout<<"          >SubStave "<<bst<<std::endl;
                     fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->showchipIDs();  
                  }
               }                                                      
            }
         }
      }
   }
   std::cout<<"[ITS OB1]"<<std::endl;
   fDetectorUnit->SubUnit[1]->showchipIDs();      
   for(int hs=0; hs<nHalfStaveOB1; hs++){        
      std::cout<<">HS "<<hs<<std::endl;
      fDetectorUnit->SubUnit[1]->SubUnit[hs]->showchipIDs();            
      for(int cz=0; cz<nChipsPerHicOB1/nHalfStaveOB1; cz++){
         int chipIDinStave = (nChipsPerHicOB1/nHalfStaveOB1)*hs + cz;      
         std::cout<<"  >ChipZ "<<cz<<" "<<chipIDinStave<<std::endl;
         fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->showchipIDs();            
         for(int l=nLAYERIB; l<nLAYERIB + nLAYEROB1; l++){ 
            int lOB = l - nLAYERIB;
            std::cout<<"    >Layer "<<l<<std::endl;
            fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->showchipIDs();            
            for(int hb=0; hb<2; hb++){        
               std::cout<<"      >HB "<<hb<<std::endl;
               fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->showchipIDs();  
               for(int st=0; st<NStaves[l]/2;st++){
                  int staveID = (NStaves[l]/2)*hb + st;
                  std::cout<<"        >Stave "<<(NStaves[l]/2)*hb + st<<std::endl; 
                  fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->showchipIDs();  
                  for(int bst=0; bst<nHalfStaveOB1; bst++){        
                     std::cout<<"          >SubStave "<<bst<<std::endl;
                     fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->showchipIDs();  
                  }
               }                                                          
            }
         }
      }
   }
   std::cout<<"[ITS OB2]"<<std::endl;
   fDetectorUnit->SubUnit[2]->showchipIDs();      
   for(int hs=0; hs<nHalfStaveOB2; hs++){        
      std::cout<<">HS "<<hs<<std::endl;
      fDetectorUnit->SubUnit[2]->SubUnit[hs]->showchipIDs();            
      for(int cz=0; cz<nChipsPerHicOB2/nHalfStaveOB2; cz++){
         int chipIDinStave = (nChipsPerHicOB2/nHalfStaveOB2)*hs + cz;      
         std::cout<<"  >ChipZ "<<cz<<" "<<chipIDinStave<<std::endl;
         fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->showchipIDs();            
         for(int l=nLAYERIB + nLAYEROB1; l<nLAYER; l++){ 
            int lOB = l - (nLAYERIB + nLAYEROB1);
            std::cout<<"    >Layer "<<l<<std::endl;
            fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->showchipIDs();            
            for(int hb=0; hb<2; hb++){        
               std::cout<<"      >HB "<<hb<<std::endl;
               fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->showchipIDs();         
               for(int st=0; st<NStaves[l]/2;st++){
                  int staveID = (NStaves[l]/2)*hb + st;
                  std::cout<<"        >Stave "<<(NStaves[l]/2)*hb + st<<std::endl; 
                  fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->showchipIDs();  
                  for(int bst=0; bst<nHalfStaveOB2; bst++){        
                     std::cout<<"          >SubStave "<<bst<<std::endl;
                     fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->showchipIDs();  
                  }
               }                                                    
            }
         }
      }
   }  
}

////////////////////////////////////////////////////////////////////////////////
/// Instanciates the neurons in input
/// Inputs are normalised and the type is set to kOff
/// (simple forward of the formula value)

void YMultiLayerPerceptron::BuildFirstLayer(TString & input)
{
   const TObjArray *inpL = input.Tokenize(", ");
   const Int_t nneurons =inpL->GetLast()+1;
   YNeuron *neuron = 0;
   Int_t i = 0;
   std::cout<<"YMultiLayerPerceptron::BuildFirstLayer "<<nneurons<<std::endl;
   for (i = 0; i<nneurons; i++) {
      const TString name = ((TObjString *)inpL->At(i))->GetString();
      neuron = new YNeuron(YNeuron::kOff, name);
      fFirstLayer.AddLast(neuron);
      fNetwork.AddLast(neuron);
   }
   delete inpL;
}


////////////////////////////////////////////////////////////////////////////////
/// Builds hidden layers.

void YMultiLayerPerceptron::BuildHiddenLayers(TString & hidden)
{

   Int_t beg = 0;
   Int_t end = hidden.Index(":", beg + 1);
   Int_t prevStart = 0;
   Int_t prevStop = fNetwork.GetEntriesFast();
   Int_t layer = 1;
   std::cout<<"hidden beg end pStart pStop layer"<<hidden<<" "<<beg<<" "<<end<<" "<<prevStart<<" "<<prevStop<<" "<<layer<<std::endl;
   while (end != -1) {
      BuildOneHiddenLayer(hidden(beg, end - beg), layer, prevStart, prevStop, false);
      beg = end + 1;
      end = hidden.Index(":", beg + 1);
   }

   BuildOneHiddenLayer(hidden(beg, hidden.Length() - beg), layer, prevStart, prevStop, true);
}


////////////////////////////////////////////////////////////////////////////////
/// Builds a hidden layer, updates the number of layers.

void YMultiLayerPerceptron::BuildOneHiddenLayer(const TString& sNumNodes, Int_t& layer,
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
         std::cout<<"nEntries : "<<nEntries<<std::endl;
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

////////////////////////////////////////////////////////////////////////////////
/// Builds the output layer
/// Neurons are linear combinations of input, by defaul.
/// If the structure ends with "!", neurons are set up for classification,
/// ie. with a sigmoid (1 neuron) or softmax (more neurons) activation function.

void YMultiLayerPerceptron::BuildLastLayer(TString & output, Int_t prev)
{
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
   std::cout<<"BuildLastLayer A:: nneurons prevStart prevStop prev "<<nneurons<<" "<<prevStart<<" "<<prevStop<<" "<<prev<<std::endl;
   for (i = 0; i<nneurons; i++) {
      Ssiz_t nextpos=output.Index(",",pos);
      if (nextpos!=kNPOS)
         name=output(pos,nextpos-pos);
      else name=output(pos,output.Length());
      pos=nextpos+1;
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
   std::cout<<"BuildLastLayer B:: nneurons prevStop nEntries"<<nneurons<<" "<<prevStop<<" "<<nEntries<<std::endl;
   for (i = prevStop; i < nEntries; i++) {
      neuron = (YNeuron *) fNetwork[i];
      for (j = prevStop; j < nEntries; j++)
         neuron->AddInLayer((YNeuron *) fNetwork[j]);
   }
}


void YMultiLayerPerceptron::BuildLastLayerNonCrossing(TString & output, Int_t prev)
{
   std::cout<<"YMultiLayerPerceptron::BuildLastLayerNonCrossing output neurons : "<<output.CountChar(',')+1<<std::endl;
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

   int nprong = prev/6;
   std::cout<<"YMultiLayerPerceptron::BuildLastLayerNonCrossing "<<prevStop<<" "<<prevStart<<" "<<nprong<<" "<<fNpronged<<std::endl;
   Int_t prev_track = prev/1;
      
   Int_t num_NC[nLAYER];
   Int_t lindex[nLAYER+1];  
   for(int t=0; t<nLAYER; t++){
      num_NC[t] = prev_track/3;
      lindex[t]  = 0;         
   }  
   lindex[nLAYER]=0;

   Ssiz_t pos = 0;
   YNeuron *neuron;
   YSynapse *synapse;
   TString name;
   Int_t i,j;
   
   for(int t1=0; t1<nLAYER+1; t1++){
      for(int t2=0; t2<t1; t2++){
         lindex[t1] += num_NC[t2];
      }
   }

   std::cout<<"BuildLastLayerNonCrossing A:: nneurons prevStart prevStop prev "<<nneurons<<" "<<prevStart<<" "<<prevStop<<" "<<prev<<" "<<prev_track<<std::endl;
   std::cout<<"lindex ";
   for(int t=0; t<nLAYER+1; t++){
      std::cout<<lindex[t]<<" ";
   }        
   std::cout<<std::endl;

   //Xi Yi Zi 
   for(int t=0; t<nLAYER; t++){
      std::cout<<"- - - - - - - - - - - - - - - - - - - -"<<std::endl;
      for (i = 3*t; i<3*(t+1); i++) {
         Ssiz_t nextpos=output.Index(",",pos);
         if (nextpos!=kNPOS)
            name=output(pos,nextpos-pos);
         else name=output(pos,output.Length());
         pos=nextpos+1; // ORIGINAL ERROR
         neuron = new YNeuron(fOutType, name);
         //std::cout<<"name["<<i<<"] : "<<name<<" pos nextpos "<<pos<<" "<<nextpos<<std::endl;
         std::cout<<"   I["<<i<<"] pre[ ";    
         for (j = prevStart+lindex[t]; j < prevStart+lindex[t+1]; j++) {
            std::cout<<j<<" ";
            synapse = new YSynapse((YNeuron *) fNetwork[j], neuron);
            fSynapses.AddLast(synapse);
            std::cout<<"("<<fSynapses.GetEntries()-1<<") ";                        
         }      
         fLastLayer.AddLast(neuron);
         fNetwork.AddLast(neuron);
         std::cout<<" ] pst[ "<<fNetwork.GetEntries()-1<<" ]"<<std::endl;
      }
   }  
   // tell each neuron which ones are in its own layer (for Softmax)
   vStructure.push_back(fLastLayer.GetEntriesFast());
   Int_t nEntries = fNetwork.GetEntriesFast();
   std::cout<<"BuildLastLayerNonCrossing B:: nneurons prevStop nEntries"<<nneurons<<" "<<prevStop<<" "<<nEntries<<std::endl;
   //dXi dYi dZi 
   for(int t=0; t<nLAYER; t++){
      for (i = prevStop+3*t; i < prevStop+3*(t+1); i++) {
         neuron = (YNeuron *) fNetwork[i];
         for (j = prevStop+3*t; j < prevStop+3*(t+1); j++)
            neuron->AddInLayer((YNeuron *) fNetwork[j]);
      }
   }
}

void YMultiLayerPerceptron::BuildAddition(){

   YNeuron *neuron = 0;

   TString name;
   name.Form("Addition");
   neuron = new YNeuron(fType, name, "", (const char*)fextF, (const char*)fextD);
   fAddition.AddLast(neuron);

}  


Bool_t YMultiLayerPerceptron::DumpUpdateSensorList(Option_t * filename) const
{
   TString filen = filename;
   std::ostream * output;
   if (filen == "") {
      Error("YMultiLayerPerceptron::DumpWeights()","Invalid file name");
      return kFALSE;
   }
   if (filen == "-")
      output = &std::cout;
   else
      output = new std::ofstream(filen.Data());
 
   *output << "#ChipID #Ntracks #Status" << std::endl;
   for(int ic = 0; ic < ChipBoundary[nLAYER]; ic++){
      *output << NetworkChips[ic]<<" "<<fSCNetwork[NetworkChips[ic]]->GetnEvents()<<" "<<fUPDATESENSORS->GetBinContent(1+NetworkChips[ic])<<std::endl;   
   }      

   if (filen != "-") {
      ((std::ofstream *) output)->close();
      delete output;
   }
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Dumps the weights to a text file.
/// Set filename to "-" (default) to dump to the standard output

Bool_t YMultiLayerPerceptron::DumpWeights(Option_t * filename) const
{
   TString filen = filename;
   std::ostream * output;
   if (filen == "") {
      Error("YMultiLayerPerceptron::DumpWeights()","Invalid file name");
      return kFALSE;
   }
   if (filen == "-")
      output = &std::cout;
   else
      output = new std::ofstream(filen.Data());
   YNeuron *neuron = 0;
   YSynapse *synapse = 0;   
   *output << "#input normalization" << std::endl;
   Int_t nentries = fFirstLayer.GetEntriesFast();
   Int_t j=0;
   for (j=0;j<nentries;j++) {
      neuron = (YNeuron *) fFirstLayer.At(j);
      *output << neuron->GetNormalisation()[0] << " "
              << neuron->GetNormalisation()[1] << std::endl;
   }
   *output << "#output normalization" << std::endl;
   nentries = fLastLayer.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      neuron = (YNeuron *) fLastLayer.At(j);
      *output << neuron->GetNormalisation()[0] << " "
              << neuron->GetNormalisation()[1] << std::endl;
   }
   *output << "#neurons weights #synapses weights" << std::endl;
   int Nnentries = fSCNetwork[0]->GetNetwork().GetEntriesFast(); 
   int Snentries = fSCNetwork[0]->GetSynapses().GetEntriesFast();
   for(int ic = 0; ic < nSensors; ic++){
      *output << NetworkChips[ic]<<" ";
      for (j=0;j<Nnentries;j++) { 
         neuron = (YNeuron *) fSCNetwork[ic]->GetNetwork().At(j);
         *output << setprecision(10) << neuron->GetWeight() <<" ";                   
      }             
      for (j=0;j<Snentries;j++) { 
         synapse = (YSynapse *) fSCNetwork[ic]->GetSynapses().At(j);     
         *output << setprecision(10) << synapse->GetWeight() <<" ";                                                                                        
      }    
      //continue;
      for (j=0;j<3;j++) {
         *output << setprecision(10) << fSCNetwork[ic]->aR(j) <<" ";                                                                                              
      }
      for (j=0;j<3;j++) {
         *output << setprecision(10) << fSCNetwork[ic]->aT(j) <<" ";                                                                                              
      } 
      *output<<std::endl;                                        
   } 
   if (filen != "-") {
      ((std::ofstream *) output)->close();
      delete output;
   }
   return kTRUE;
}

Bool_t YMultiLayerPerceptron::DumpWeightsDetectorUnit(Option_t * filename) const
{
   TString filen = filename;
   std::ostream * output;
   if (filen == "") {
      Error("YMultiLayerPerceptron::DumpWeights()","Invalid file name");
      return kFALSE;
   }
   if (filen == "-")
      output = &std::cout;
   else
      output = new std::ofstream(filen.Data());
   YNeuron *neuron = 0;
   YSynapse *synapse = 0;   
   YNeuron *neuronDU[8];
   YSynapse *synapseDU[8];     
   
   neuronDU[0] = 0;
   neuronDU[1] = 0;
   neuronDU[2] = 0;
   neuronDU[3] = 0;
   neuronDU[4] = 0;
   neuronDU[5] = 0;
   neuronDU[6] = 0;
   neuronDU[7] = 0;
                   
   synapseDU[0] = 0;
   synapseDU[1] = 0;
   synapseDU[2] = 0;
   synapseDU[3] = 0;
   synapseDU[4] = 0;
   synapseDU[5] = 0;
   synapseDU[6] = 0;
   synapseDU[7] = 0;
    
   *output << "#input normalization" << std::endl;
   Int_t nentries = fFirstLayer.GetEntriesFast();
   Int_t j=0;
   for (j=0;j<nentries;j++) {
      neuron = (YNeuron *) fFirstLayer.At(j);
      *output << neuron->GetNormalisation()[0] << " "
              << neuron->GetNormalisation()[1] << std::endl;
   }
   *output << "#output normalization" << std::endl;
   nentries = fLastLayer.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      neuron = (YNeuron *) fLastLayer.At(j);
      *output << neuron->GetNormalisation()[0] << " "
              << neuron->GetNormalisation()[1] << std::endl;
   }
   *output << "#neurons weights #synapses weights" << std::endl;
   int Nnentries = fSCNetwork[0]->GetNetwork().GetEntriesFast(); 
   int Snentries = fSCNetwork[0]->GetSynapses().GetEntriesFast();
   
   *output << "#ITS" << std::endl;
   
   for (j=0;j<Nnentries;j++) { 
      neuronDU[0] = (YNeuron *) fDetectorUnit->fSCNetwork->GetNetwork().At(j);
      *output << neuronDU[0]->GetWeight() <<" ";
   }             
   for (j=0;j<Snentries;j++) { 
      synapseDU[0] = (YSynapse *) fDetectorUnit->fSCNetwork->GetSynapses().At(j);
      *output << synapseDU[0]->GetWeight() <<" ";
   }    
   *output<<std::endl;
 
   *output << "  #ITS IB" << std::endl;

   *output << "  ";
   for (j=0;j<Nnentries;j++) { 
      neuronDU[1] = (YNeuron *) fDetectorUnit->SubUnit[0]->fSCNetwork->GetNetwork().At(j);
      *output << neuronDU[1]->GetWeight() <<" ";
   }
   for (j=0;j<Snentries;j++) { 
      synapseDU[1] = (YSynapse *) fDetectorUnit->SubUnit[0]->fSCNetwork->GetSynapses().At(j);
      *output << synapseDU[1]->GetWeight() <<" ";
   }
   *output<<std::endl;
               
   for(int hs=0; hs<nHalfStaveIB; hs++){        
      *output << "    #Half Stave " << hs << std::endl;
      *output << "    ";
      for (j=0;j<Nnentries;j++) { 
         neuronDU[2] = (YNeuron *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->fSCNetwork->GetNetwork().At(j);
         *output << neuronDU[2]->GetWeight() <<" ";
      }             
      for (j=0;j<Snentries;j++) { 
         synapseDU[2] = (YSynapse *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->fSCNetwork->GetSynapses().At(j);
         *output << synapseDU[2]->GetWeight() <<" ";
      }    
      *output<<std::endl;   

      for(int cz=0; cz<nChipsPerHicIB/nHalfStaveIB; cz++){
         int chipIDinStave = (nChipsPerHicIB/nHalfStaveIB)*hs + cz;      
         *output << "      #ChipZ " << chipIDinStave << std::endl;
         *output << "      ";
         for (j=0;j<Nnentries;j++) { 
            neuronDU[3] = (YNeuron *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetNetwork().At(j);
            *output << neuronDU[3]->GetWeight() <<" ";
         }
         for (j=0;j<Snentries;j++) { 
            synapseDU[3] = (YSynapse *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetSynapses().At(j);
            *output << synapseDU[3]->GetWeight() <<" ";
         }    
         *output<<std::endl;

         for(int l=0; l<nLAYERIB; l++){ 
            int lIB = l;
            *output << "        #Layer " << l << std::endl;
            *output << "        ";
            for (j=0;j<Nnentries;j++) { 
               neuronDU[4] = (YNeuron *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->fSCNetwork->GetNetwork().At(j);
               *output << neuronDU[4]->GetWeight() <<" ";
            }
            for (j=0;j<Snentries;j++) { 
               synapseDU[4] = (YSynapse *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->fSCNetwork->GetSynapses().At(j);
               *output << synapseDU[4]->GetWeight() <<" ";
            }
            *output<<std::endl;
         
            for(int hb=0; hb<2; hb++){ 
               *output << "          #Half Barrel " << hb << std::endl;  
               *output << "          ";                                        
               for (j=0;j<Nnentries;j++) {
                  neuronDU[5] = (YNeuron *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->fSCNetwork->GetNetwork().At(j);
                  *output << neuronDU[5]->GetWeight() <<" ";
               }
               for (j=0;j<Snentries;j++) {
                  synapseDU[5] = (YSynapse *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->fSCNetwork->GetSynapses().At(j);
                  *output << synapseDU[5]->GetWeight() <<" ";
               }
               *output<<std::endl;
               /*
               for(int st=0; st<NStaves[l]/2;st++){
                  *output << "            #Stave " << (NStaves[l]/2)*hb + st << std::endl;  
                  *output << "            ";                                        
                  for (j=0;j<Nnentries;j++) {
                     neuronDU[6] = (YNeuron *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetNetwork().At(j);
                     *output << neuronDU[6]->GetWeight() <<" ";
                  }
                  for (j=0;j<Snentries;j++) {
                     synapseDU[6] = (YSynapse *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetSynapses().At(j);
                     *output << synapseDU[6]->GetWeight() <<" ";
                  }
                  *output<<std::endl;
                  for(int bst=0; bst<nHalfStaveIB; bst++){        
                     *output << "              #SubStave " << bst << std::endl;  
                     *output << "              ";                                        
                     for (j=0;j<Nnentries;j++) {
                        neuronDU[7] = (YNeuron *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetNetwork().At(j);
                        *output << neuronDU[7]->GetWeight() <<" ";
                     }
                     for (j=0;j<Snentries;j++) {
                        synapseDU[7] = (YSynapse *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetSynapses().At(j);
                        *output << synapseDU[7]->GetWeight() <<" ";
                     }
                     *output<<std::endl;
                  }   
               }
               */ 
            }
         }
      }
   }

   *output << "  #ITS OB1" << std::endl;

   *output << "  ";
   for (j=0;j<Nnentries;j++) { 
      neuronDU[1] = (YNeuron *) fDetectorUnit->SubUnit[1]->fSCNetwork->GetNetwork().At(j);
      *output << neuronDU[1]->GetWeight() <<" ";
   }
   for (j=0;j<Snentries;j++) { 
      synapseDU[1] = (YSynapse *) fDetectorUnit->SubUnit[1]->fSCNetwork->GetSynapses().At(j);
      *output << synapseDU[1]->GetWeight() <<" ";
   }
   *output<<std::endl;
               
   for(int hs=0; hs<nHalfStaveOB1; hs++){        
      *output << "    #Half Stave " << hs << std::endl;
      *output << "    ";
      for (j=0;j<Nnentries;j++) { 
         neuronDU[2] = (YNeuron *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->fSCNetwork->GetNetwork().At(j);
         *output << neuronDU[2]->GetWeight() <<" ";
      }             
      for (j=0;j<Snentries;j++) { 
         synapseDU[2] = (YSynapse *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->fSCNetwork->GetSynapses().At(j);
         *output << synapseDU[2]->GetWeight() <<" ";
      }    
      *output<<std::endl;   

      for(int cz=0; cz<nChipsPerHicOB1/nHalfStaveOB1; cz++){
         int chipIDinStave = (nChipsPerHicOB1/nHalfStaveOB1)*hs + cz;      
         *output << "      #ChipZ " << chipIDinStave << std::endl;
         *output << "      ";
         for (j=0;j<Nnentries;j++) { 
            neuronDU[3] = (YNeuron *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetNetwork().At(j);
            *output << neuronDU[3]->GetWeight() <<" ";
         }
         for (j=0;j<Snentries;j++) { 
            synapseDU[3] = (YSynapse *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetSynapses().At(j);
            *output << synapseDU[3]->GetWeight() <<" ";
         }    
         *output<<std::endl;

         for(int l=nLAYERIB; l<nLAYERIB + nLAYEROB1; l++){ 
            int lOB = l-nLAYERIB;
            *output << "        #Layer " << l << std::endl;
            *output << "        ";
            for (j=0;j<Nnentries;j++) { 
               neuronDU[4] = (YNeuron *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->GetNetwork().At(j);
               *output << neuronDU[4]->GetWeight() <<" ";
            }
            for (j=0;j<Snentries;j++) { 
               synapseDU[4] = (YSynapse *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->GetSynapses().At(j);
               *output << synapseDU[4]->GetWeight() <<" ";
            }
            *output<<std::endl;
         
            for(int hb=0; hb<2; hb++){ 
               *output << "          #Half Barrel " << hb << std::endl;  
               *output << "          ";                                        
               for (j=0;j<Nnentries;j++) {
                  neuronDU[5] = (YNeuron *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->GetNetwork().At(j);
                  *output << neuronDU[5]->GetWeight() <<" ";
               }
               for (j=0;j<Snentries;j++) {
                  synapseDU[5] = (YSynapse *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->GetSynapses().At(j);
                  *output << synapseDU[5]->GetWeight() <<" ";
               }
               *output<<std::endl;
               /*
               for(int st=0; st<NStaves[l]/2;st++){
                  *output << "            #Stave " << (NStaves[l]/2)*hb + st << std::endl;  
                  *output << "            ";                                        
                  for (j=0;j<Nnentries;j++) {
                     neuronDU[6] = (YNeuron *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetNetwork().At(j);
                     *output << neuronDU[6]->GetWeight() <<" ";
                  }
                  for (j=0;j<Snentries;j++) {
                     synapseDU[6] = (YSynapse *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetSynapses().At(j);
                     *output << synapseDU[6]->GetWeight() <<" ";
                  }
                  *output<<std::endl;
                  for(int bst=0; bst<nHalfStaveOB1; bst++){        
                     *output << "              #SubStave " << bst << std::endl;  
                     *output << "              ";                                        
                     for (j=0;j<Nnentries;j++) {
                        neuronDU[7] = (YNeuron *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetNetwork().At(j);
                        *output << neuronDU[7]->GetWeight() <<" ";
                     }
                     for (j=0;j<Snentries;j++) {
                        synapseDU[7] = (YSynapse *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetSynapses().At(j);
                        *output << synapseDU[7]->GetWeight() <<" ";
                     }
                     *output<<std::endl;
                  }   
               }   
               */            
            }
         }
      }
   }
   
   *output << "  #ITS OB2" << std::endl;

   *output << "  ";
   for (j=0;j<Nnentries;j++) { 
      neuronDU[1] = (YNeuron *) fDetectorUnit->SubUnit[2]->fSCNetwork->GetNetwork().At(j);
      *output << neuronDU[1]->GetWeight() <<" ";
   }
   for (j=0;j<Snentries;j++) { 
      synapseDU[1] = (YSynapse *) fDetectorUnit->SubUnit[2]->fSCNetwork->GetSynapses().At(j);
      *output << synapseDU[1]->GetWeight() <<" ";
   }
   *output<<std::endl;
               
   for(int hs=0; hs<nHalfStaveOB2; hs++){        
      *output << "    #Half Stave " << hs << std::endl;
      *output << "    ";
      for (j=0;j<Nnentries;j++) { 
         neuronDU[2] = (YNeuron *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->fSCNetwork->GetNetwork().At(j);
         *output << neuronDU[2]->GetWeight() <<" ";
      }             
      for (j=0;j<Snentries;j++) { 
         synapseDU[2] = (YSynapse *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->fSCNetwork->GetSynapses().At(j);
         *output << synapseDU[2]->GetWeight() <<" ";
      }    
      *output<<std::endl;   

      for(int cz=0; cz<nChipsPerHicOB2/nHalfStaveOB2; cz++){
         int chipIDinStave = (nChipsPerHicOB2/nHalfStaveOB2)*hs + cz;      
         *output << "      #ChipZ " << chipIDinStave << std::endl;
         *output << "      ";
         for (j=0;j<Nnentries;j++) { 
            neuronDU[3] = (YNeuron *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetNetwork().At(j);
            *output << neuronDU[3]->GetWeight() <<" ";
         }
         for (j=0;j<Snentries;j++) { 
            synapseDU[3] = (YSynapse *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetSynapses().At(j);
            *output << synapseDU[3]->GetWeight() <<" ";
         }    
         *output<<std::endl;

         for(int l=nLAYERIB + nLAYEROB1; l<nLAYER; l++){ 
            int lOB = l - (nLAYERIB + nLAYEROB1);
            *output << "        #Layer " << l << std::endl;
            *output << "        ";
            for (j=0;j<Nnentries;j++) { 
               neuronDU[4] = (YNeuron *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->GetNetwork().At(j);
               *output << neuronDU[4]->GetWeight() <<" ";
            }
            for (j=0;j<Snentries;j++) { 
               synapseDU[4] = (YSynapse *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->GetSynapses().At(j);
               *output << synapseDU[4]->GetWeight() <<" ";
            }
            *output<<std::endl;
         
            for(int hb=0; hb<2; hb++){ 
               *output << "          #Half Barrel " << hb << std::endl;  
               *output << "          ";                                        
               for (j=0;j<Nnentries;j++) {
                  neuronDU[5] = (YNeuron *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->GetNetwork().At(j);
                  *output << neuronDU[5]->GetWeight() <<" ";
               }
               for (j=0;j<Snentries;j++) {
                  synapseDU[5] = (YSynapse *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->GetSynapses().At(j);
                  *output << synapseDU[5]->GetWeight() <<" ";
               }
               *output<<std::endl;
               /*
               for(int st=0; st<NStaves[l]/2;st++){
                  *output << "            #Stave " << (NStaves[l]/2)*hb + st << std::endl;  
                  *output << "            ";                                        
                  for (j=0;j<Nnentries;j++) {
                     neuronDU[6] = (YNeuron *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetNetwork().At(j);
                     *output << neuronDU[6]->GetWeight() <<" ";
                  }
                  for (j=0;j<Snentries;j++) {
                     synapseDU[6] = (YSynapse *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetSynapses().At(j);
                     *output << synapseDU[6]->GetWeight() <<" ";
                  }
                  *output<<std::endl;
                  for(int bst=0; bst<nHalfStaveOB2; bst++){        
                     *output << "              #SubStave " << bst << std::endl;  
                     *output << "              ";                                        
                     for (j=0;j<Nnentries;j++) {
                        neuronDU[7] = (YNeuron *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetNetwork().At(j);
                        *output << neuronDU[7]->GetWeight() <<" ";
                     }
                     for (j=0;j<Snentries;j++) {
                        synapseDU[7] = (YSynapse *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetSynapses().At(j);
                        *output << synapseDU[7]->GetWeight() <<" ";
                     }
                     *output<<std::endl;
                  }   
               }      
               */           
            }
         }
      }
   }
   if (filen != "-") {
      ((std::ofstream *) output)->close();
      delete output;
   }
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// DumpResiduals
Bool_t YMultiLayerPerceptron::PrepareDumpResiduals()
{      
   std::cout<<"PrepareDumpResiduals START"<<std::endl;
   std::cout<<"PrepareDumpResiduals Monitoring Layer Level"<<std::endl;   

   for(int l = 0; l <nLAYER; l++){
      int    nbins = l<3 ? MONITOR_NbinsIB : MONITOR_NbinsOB;
      double range = l<3 ? MONITOR_RangeIB : MONITOR_RangeOB;    
    
      fChi2Layer[l]         = new TH1D(Form("fChi2Layer%d",l),Form("Chi2 Distribution Layer %d",l),100,0,20);
      fpTvsResLayer[l][0]   = new TH2D(Form("hS1pTvsResLayer%d",l),Form("pT vs #Deltas_{1} Monitoring Layer %d ; ds(cm); pT(GeV)",l),nbins,-range,+range,50,0,5);
      fpTvsResLayer[l][1]   = new TH2D(Form("hS2pTvsResLayer%d",l),Form("pT vs #Deltas_{2} Monitoring Layer %d ; ds(cm); pT(GeV)",l),nbins,-range,+range,50,0,5);            
      fpTvsChiLayer[l][0]   = new TH2D(Form("hS1pTvsChiLayer%d",l),Form("pT vs #chi s_{1} Monitoring Layer %d ; #chi; pT(GeV)",l),MONITOR_NbinsChi,-10,+10,50,0,5);
      fpTvsChiLayer[l][1]   = new TH2D(Form("hS2pTvsChiLayer%d",l),Form("pT vs #chi s_{2} Monitoring Layer %d ; #chi; pT(GeV)",l),MONITOR_NbinsChi,-10,+10,50,0,5);          
   }
#ifdef MONITORHALFSTAVEUNIT                  
   std::cout<<"PrepareDumpResiduals Monitoring Layer HalfBarrel HalfStave Level"<<std::endl;    
   double Zrange[] = {15, 15, 15, 50, 50, 80, 80};    
   fpTvsResLayerHBHS             = new TH2D ****[nLAYER];
   fpTvsChiLayerHBHS             = new TH2D ****[nLAYER];
   fResidualsVsZLayerHBHS        = new TH2D ****[nLAYER];
   fResidualsVsPhiLayerHBHS      = new TH2D ****[nLAYER];   
   fProfileVsZLayerHBHS          = new TProfile ****[nLAYER];
   fProfileVsPhiLayerHBHS        = new TProfile ****[nLAYER];  
   fSensorCenterVsZLayerHBHS     = new TProfile ****[nLAYER];
   fSensorCenterVsPhiLayerHBHS   = new TProfile ****[nLAYER];   
   for(int l = 0; l < nLAYER; l++){
      int    nbins = l<3 ? MONITOR_NbinsIB : MONITOR_NbinsOB;
      double range = l<3 ? MONITOR_RangeIB : MONITOR_RangeOB; 
         
      int nHalfBarrel = 2;
      fpTvsResLayerHBHS[l]             = new TH2D ***[nHalfBarrel];  
      fpTvsChiLayerHBHS[l]             = new TH2D ***[nHalfBarrel];  
      fResidualsVsZLayerHBHS[l]        = new TH2D ***[nHalfBarrel];  
      fResidualsVsPhiLayerHBHS[l]      = new TH2D ***[nHalfBarrel];      
      fProfileVsZLayerHBHS[l]          = new TProfile ***[nHalfBarrel];  
      fProfileVsPhiLayerHBHS[l]        = new TProfile ***[nHalfBarrel];   
      fSensorCenterVsZLayerHBHS[l]     = new TProfile ***[nHalfBarrel];  
      fSensorCenterVsPhiLayerHBHS[l]   = new TProfile ***[nHalfBarrel];  
      for(int hb = 0; hb < nHalfBarrel; hb++){
         int nHalfStave = NSubStave[l]; 
         fpTvsResLayerHBHS[l][hb]             = new TH2D **[nHalfStave];
         fpTvsChiLayerHBHS[l][hb]             = new TH2D **[nHalfStave];
         fResidualsVsZLayerHBHS[l][hb]        = new TH2D **[nHalfStave];
         fResidualsVsPhiLayerHBHS[l][hb]      = new TH2D **[nHalfStave];     
         fProfileVsZLayerHBHS[l][hb]          = new TProfile **[nHalfStave];
         fProfileVsPhiLayerHBHS[l][hb]        = new TProfile **[nHalfStave];     
         fSensorCenterVsZLayerHBHS[l][hb]     = new TProfile **[nHalfStave];
         fSensorCenterVsPhiLayerHBHS[l][hb]   = new TProfile **[nHalfStave];   
         for(int hs = 0; hs < nHalfStave; hs++){
            int nChi2 = 2;
            fpTvsResLayerHBHS[l][hb][hs]             = new TH2D *[nChi2]; 
            fpTvsChiLayerHBHS[l][hb][hs]             = new TH2D *[nChi2]; 
            fResidualsVsZLayerHBHS[l][hb][hs]        = new TH2D *[nChi2]; 
            fResidualsVsPhiLayerHBHS[l][hb][hs]      = new TH2D *[nChi2];             
            fProfileVsZLayerHBHS[l][hb][hs]          = new TProfile *[nChi2]; 
            fProfileVsPhiLayerHBHS[l][hb][hs]        = new TProfile *[nChi2];  
            fSensorCenterVsZLayerHBHS[l][hb][hs]     = new TProfile *[nChi2]; 
            fSensorCenterVsPhiLayerHBHS[l][hb][hs]   = new TProfile *[nChi2];
 
            fpTvsResLayerHBHS[l][hb][hs][0] 
            = new TH2D(Form("hS1pTvsResLayer%dHB%dHS%d",l,hb,hs),Form("pT vs #Deltas_{1} Monitoring Layer %d HB %d HS %d; ds(cm); pT(GeV)",l,hb,hs),nbins,-range,+range,50,0,5);
            fpTvsResLayerHBHS[l][hb][hs][1] 
            = new TH2D(Form("hS2pTvsResLayer%dHB%dHS%d",l,hb,hs),Form("pT vs #Deltas_{2} Monitoring Layer %d HB %d HS %d; ds(cm); pT(GeV)",l,hb,hs),nbins,-range,+range,50,0,5);
            fpTvsChiLayerHBHS[l][hb][hs][0] 
            = new TH2D(Form("hS1pTvsChiLayer%dHB%dHS%d",l,hb,hs),Form("pT vs #chi s_{1} Monitoring Layer %d HB %d HS %d; #chi; pT(GeV)",l,hb,hs),MONITOR_NbinsChi,-10,+10,50,0,5);
            fpTvsChiLayerHBHS[l][hb][hs][1] 
            = new TH2D(Form("hS2pTvsChiLayer%dHB%dHS%d",l,hb,hs),Form("pT vs #chi s_{2} Monitoring Layer %d HB %d HS %d; #chi; pT(GeV)",l,hb,hs),MONITOR_NbinsChi,-10,+10,50,0,5);    

            fResidualsVsZLayerHBHS[l][hb][hs][0]   
            = new TH2D(Form("hS1ResVsZLayer%dHB%dHS%d",l,hb,hs),Form("#Deltas_{1} vs Z Distribution Layer %d HB %d HS %d",l,hb,hs),600,-Zrange[l],+Zrange[l],nbins,-range,+range);  
            fResidualsVsZLayerHBHS[l][hb][hs][1]   
            = new TH2D(Form("hS2ResVsZLayer%dHB%dHS%d",l,hb,hs),Form("#Deltas_{2} vs Z Distribution Layer %d HB %d HS %d",l,hb,hs),600,-Zrange[l],+Zrange[l],nbins,-range,+range);    
            fResidualsVsPhiLayerHBHS[l][hb][hs][0] 
            = new TH2D(Form("hS1ResVsPhiLayer%dHB%dHS%d",l,hb,hs),Form("#Deltas_{1} vs #phi Distribution Layer %d HB %d HS %d",l,hb,hs),600,-TMath::Pi(),+TMath::Pi(),nbins,-range,+range);      
            fResidualsVsPhiLayerHBHS[l][hb][hs][1] 
            = new TH2D(Form("hS2ResVsPhiLayer%dHB%dHS%d",l,hb,hs),Form("#Deltas_{2} vs #phi Distribution Layer %d HB %d HS %d",l,hb,hs),600,-TMath::Pi(),+TMath::Pi(),nbins,-range,+range);    

            fProfileVsZLayerHBHS[l][hb][hs][0]   
            = new TProfile(Form("hS1ProfileVsZLayer%dHB%dHS%d",l,hb,hs),Form("#Deltas_{1} vs Z Profile Layer %d HB %d HS %d",l,hb,hs),200,-Zrange[l],+Zrange[l],-range,+range);
            fProfileVsZLayerHBHS[l][hb][hs][1]   
            = new TProfile(Form("hS2ProfileVsZLayer%dHB%dHS%d",l,hb,hs),Form("#Deltas_{2} vs Z Profile Layer %d HB %d HS %d",l,hb,hs),200,-Zrange[l],+Zrange[l],-range,+range);
            fProfileVsPhiLayerHBHS[l][hb][hs][0] 
            = new TProfile(Form("hS1ProfileVsPhiLayer%dHB%dHS%d",l,hb,hs),Form("#Deltas_{1} vs #phi Profile Layer %d HB %d HS %d",l,hb,hs),200,-TMath::Pi(),+TMath::Pi(),-range,+range);
            fProfileVsPhiLayerHBHS[l][hb][hs][1] 
            = new TProfile(Form("hS2ProfileVsPhiLayer%dHB%dHS%d",l,hb,hs),Form("#Deltas_{2} vs #phi Profile Layer %d HB %d HS %d",l,hb,hs),200,-TMath::Pi(),+TMath::Pi(),-range,+range);

            fSensorCenterVsZLayerHBHS[l][hb][hs][0]   
            = new TProfile(Form("hS1SensorCenterVsZLayer%dHB%dHS%d",l,hb,hs),Form("#Deltas_{1} vs Z SensorCenter Layer %d HB %d HS %d",l,hb,hs),600,-Zrange[l],+Zrange[l],-range,+range);
            fSensorCenterVsZLayerHBHS[l][hb][hs][1]   
            = new TProfile(Form("hS2SensorCenterVsZLayer%dHB%dHS%d",l,hb,hs),Form("#Deltas_{2} vs Z SensorCenter Layer %d HB %d HS %d",l,hb,hs),600,-Zrange[l],+Zrange[l],-range,+range);
            fSensorCenterVsPhiLayerHBHS[l][hb][hs][0] 
            = new TProfile(Form("hS1SensorCenterVsPhiLayer%dHB%dHS%d",l,hb,hs),Form("#Deltas_{1} vs #phi SensorCenter Layer %d HB %d HS %d",l,hb,hs),300,-TMath::Pi(),+TMath::Pi(),-range,+range);
            fSensorCenterVsPhiLayerHBHS[l][hb][hs][1] 
            = new TProfile(Form("hS2SensorCenterVsPhiLayer%dHB%dHS%d",l,hb,hs),Form("#Deltas_{2} vs #phi SensorCenter Layer %d HB %d HS %d",l,hb,hs),300,-TMath::Pi(),+TMath::Pi(),-range,+range);

            fProfileVsZLayerHBHS[l][hb][hs][0]->GetYaxis()->SetRangeUser(-range,+range);  
            fProfileVsZLayerHBHS[l][hb][hs][1]->GetYaxis()->SetRangeUser(-range,+range);  
            fProfileVsPhiLayerHBHS[l][hb][hs][0]->GetYaxis()->SetRangeUser(-range,+range);  
            fProfileVsPhiLayerHBHS[l][hb][hs][1]->GetYaxis()->SetRangeUser(-range,+range);  

            fSensorCenterVsZLayerHBHS[l][hb][hs][0]->GetYaxis()->SetRangeUser(-range,+range);  
            fSensorCenterVsZLayerHBHS[l][hb][hs][1]->GetYaxis()->SetRangeUser(-range,+range);  
            fSensorCenterVsPhiLayerHBHS[l][hb][hs][0]->GetYaxis()->SetRangeUser(-range,+range);  
            fSensorCenterVsPhiLayerHBHS[l][hb][hs][1]->GetYaxis()->SetRangeUser(-range,+range);  
         }       
      }
   }
#endif   
             
   std::cout<<"PrepareDumpResiduals END"<<std::endl;
   return kTRUE;                       
}


Bool_t YMultiLayerPerceptron::DumpResiduals(int epoch)
{
   //return kTRUE;
   std::cout<<"YMultiLayerPerceptron::DumpResiduals Start Epoch : "<<epoch<<std::endl; 
   
   // based on ~/tutorials/io/dir.C
   // create a new Root file
   TString tResName = "Residual_Epoch_At_" + TString::Itoa(epoch,10) + ".root"; 
   TFile* Residual = new TFile(tResName,"recreate");

   // create a subdirectory "tof" in this file
   TDirectory *cdResidual = Residual->mkdir("Residual");
   cdResidual->cd();    // make the "tof" directory the current directory

   TDirectory**    cdLayer = new TDirectory *[nLAYER]; 
   TDirectory***   cdStave = new TDirectory **[nLAYER]; 
   for(int l = 0; l <nLAYER; l++){
      TString tLayName = "Layer" + TString::Itoa(l,10); 
      cdLayer[l]   = cdResidual->mkdir(tLayName);
      cdStave[l]   = new TDirectory *[NStaves[l]];
      for(int s = 0; s <NStaves[l]; s++){
         std::cout<<"YMultiLayerPerceptron::DumpResiduals Prepare Layer "<< l <<" - Stave "<< s <<" Objects"<<std::endl;                  
         TString tStvName = "Stave" + TString::Itoa(s,10); 
         cdStave[l][s]   = cdLayer[l]->mkdir(tStvName);
      }
   }
   
   (TH1D*) fCostMonitor->Clone();  
   (TH2D*) fBeamXY->Clone();   
   (TH2D*) fBeamZR->Clone(); 
   (TH2D*) fVertexFitXY->Clone(); 
   (TH2D*) fVertexFitZR->Clone(); 
   (TH2D*) fVertexXY->Clone(); 
   (TH2D*) fVertexZR->Clone();   

   (TH2D*) fCostChargeSym->Clone();
   (TH1D*) fChargeSymMonitorPositive->Clone();
   (TH1D*) fChargeSymMonitorNegative->Clone();     
#ifdef MONITORONLYUPDATES
   (TH1D*) fUPDATESENSORS->Clone();   
   (TH1C*) fUPDATETRACKS->Clone();
#endif
   
   for(int l = 0; l <nLAYER; l++){
      cdLayer[l]->cd();
      (TH1D*) fChi2Layer[l]->Clone();
      (TH2D*) fpTvsResLayer[l][0]->Clone();
      (TH2D*) fpTvsResLayer[l][1]->Clone();
      (TH2D*) fpTvsChiLayer[l][0]->Clone();
      (TH2D*) fpTvsChiLayer[l][1]->Clone();  
#ifdef MONITORHALFSTAVEUNIT
      int nHalfBarrel = 2;
      for(int hb = 0; hb < nHalfBarrel; hb++){
         int nHalfStave = NSubStave[l]; 
         for(int hs = 0; hs < nHalfStave; hs++){
            (TH2D*) fpTvsResLayerHBHS[l][hb][hs][0]->Clone();
            (TH2D*) fpTvsResLayerHBHS[l][hb][hs][1]->Clone();  
            (TH2D*) fpTvsChiLayerHBHS[l][hb][hs][0]->Clone();
            (TH2D*) fpTvsChiLayerHBHS[l][hb][hs][1]->Clone();   
            (TH2D*) fResidualsVsZLayerHBHS[l][hb][hs][0]->Clone();
            (TH2D*) fResidualsVsZLayerHBHS[l][hb][hs][1]->Clone();
            (TH2D*) fResidualsVsPhiLayerHBHS[l][hb][hs][0]->Clone();
            (TH2D*) fResidualsVsPhiLayerHBHS[l][hb][hs][1]->Clone();        
            (TProfile*) fProfileVsZLayerHBHS[l][hb][hs][0]->Clone();
            (TProfile*) fProfileVsZLayerHBHS[l][hb][hs][1]->Clone();    
            (TProfile*) fProfileVsPhiLayerHBHS[l][hb][hs][0]->Clone();
            (TProfile*) fProfileVsPhiLayerHBHS[l][hb][hs][1]->Clone();  
            (TProfile*) fSensorCenterVsZLayerHBHS[l][hb][hs][0]->Clone();
            (TProfile*) fSensorCenterVsZLayerHBHS[l][hb][hs][1]->Clone();    
            (TProfile*) fSensorCenterVsPhiLayerHBHS[l][hb][hs][0]->Clone();
            (TProfile*) fSensorCenterVsPhiLayerHBHS[l][hb][hs][1]->Clone();                  
         }       
      }    
#endif         
      cdResidual->cd();      
   }   

   std::cout<<"YMultiLayerPerceptron::DumpResiduals Start of Building Histos "<<std::endl;      
   for(int ichipID = 0; ichipID < nSensors; ichipID++){      
      int layer       = yGEOM->GetLayer(NetworkChips[ichipID]);
      int staveID     = yGEOM->GetStave(NetworkChips[ichipID]);
      int chipIDstave = yGEOM->GetChipIdInStave(NetworkChips[ichipID]);  
      int chipIDlayer = yGEOM->GetChipIdInLayer(NetworkChips[ichipID]);  
      

#ifdef MONITORONLYUPDATES
      if(fUPDATESENSORS->GetBinContent(1+ichipID)!=1) continue;
#else
      if(fSCNetwork[NetworkChips[ichipID]]->GetnEvents()<Min_Cluster_by_Sensor) continue;           
#endif
      
#ifdef MONITORONLYUPDATES             
      std::cout<<" Epoch : "<<fCurrentEpoch<<
                " ChipID : "<<NetworkChips[ichipID]<<
               " nTracks : "<<fSCNetwork[NetworkChips[ichipID]]->GetnEvents()<<
                " status : "<<fUPDATESENSORS->GetBinContent(1+NetworkChips[ichipID])<<std::endl;
#endif         
         
      cdStave[layer][staveID]->cd();  
      
      (TH1D*) fSCNetwork[NetworkChips[ichipID]]->hNtracksByRejection->Clone();
      if(fLearningMethod==YMultiLayerPerceptron::kOffsetTuneByMean&&fWUpdatebyMean[NetworkChips[ichipID]]==true) {
#ifdef MONITORSENSORUNITprofile
         fSCNetwork[NetworkChips[ichipID]]->ProfileOptimizerbyMinBin(10);
         double profilehits_s1 = fSCNetwork[NetworkChips[ichipID]]->GetEntriesProfile(fSCNetwork[NetworkChips[ichipID]]->hpds1_s1);
         double profilehits_s2 = fSCNetwork[NetworkChips[ichipID]]->GetEntriesProfile(fSCNetwork[NetworkChips[ichipID]]->hpds2_s2);
         if(profilehits_s1>=30) fSCNetwork[NetworkChips[ichipID]]->hpds1_s1->Fit(fOffsetTuning[ichipID]->fds1_s1, "r");
         if(profilehits_s1>=30) fSCNetwork[NetworkChips[ichipID]]->hpds2_s1->Fit(fOffsetTuning[ichipID]->fds2_s1, "r");
         if(profilehits_s2>=30) fSCNetwork[NetworkChips[ichipID]]->hpds1_s2->Fit(fOffsetTuning[ichipID]->fds1_s2, "r");
         if(profilehits_s2>=30) fSCNetwork[NetworkChips[ichipID]]->hpds2_s2->Fit(fOffsetTuning[ichipID]->fds2_s2, "r");    
#endif
      }

#ifdef MONITORSENSORUNITprofile
      double range = 0.05;
      fSCNetwork[NetworkChips[ichipID]]->hpds1_s1->GetYaxis()->SetRangeUser(-range,+range);  
      fSCNetwork[NetworkChips[ichipID]]->hpds2_s1->GetYaxis()->SetRangeUser(-range,+range);  
      fSCNetwork[NetworkChips[ichipID]]->hpds1_s2->GetYaxis()->SetRangeUser(-range,+range);  
      fSCNetwork[NetworkChips[ichipID]]->hpds2_s2->GetYaxis()->SetRangeUser(-range,+range);     

      fSCNetwork[NetworkChips[ichipID]]->ProfileOptimizerbyMinBin(4);
      (TProfile*) fSCNetwork[NetworkChips[ichipID]]->hpds1_s1->Clone();
      (TProfile*) fSCNetwork[NetworkChips[ichipID]]->hpds2_s1->Clone();
      (TProfile*) fSCNetwork[NetworkChips[ichipID]]->hpds1_s2->Clone(); 
      (TProfile*) fSCNetwork[NetworkChips[ichipID]]->hpds2_s2->Clone(); 
#endif

#ifdef MONITORSENSORUNITpT         
      (TH2D*) fSCNetwork[NetworkChips[ichipID]]->GetpTvsResHisto(0)->Clone();
      (TH2D*) fSCNetwork[NetworkChips[ichipID]]->GetpTvsResHisto(1)->Clone();             
      (TH2D*) fSCNetwork[NetworkChips[ichipID]]->GetpTvsChiHisto(0)->Clone();
      (TH2D*) fSCNetwork[NetworkChips[ichipID]]->GetpTvsChiHisto(1)->Clone();   
      (TH1D*) fSCNetwork[NetworkChips[ichipID]]->GetChi2Histo()->Clone();               
#endif

      cdLayer[layer]->cd();  
      cdResidual->cd();
   }
   std::cout<<"YMultiLayerPerceptron::DumpResiduals End of Building Histos "<<std::endl;                  
   // save histogram hierarchy in the file
   //Residual->cd();
   Residual->Write();
   std::cout<<"YMultiLayerPerceptron::DumpResiduals End of Saving Histos "<<std::endl;                     
   delete Residual;
   std::cout<<"YMultiLayerPerceptron::DumpResiduals End of Deleting Objects "<<std::endl;   

   //Dump residual monitor root files   TString tResName = "Residual_Epoch_At_" + TString::Itoa(epoch,10) + ".root"; 
 
   return kTRUE;                    
}

Bool_t YMultiLayerPerceptron::DumpCostGradients(int epoch, Double_t** bufferArr, double threshold = 2) const
{
   //double threshold = 2; //in um 
   TString filen = "";
   if(threshold<0) filen = "CostGradient_Epoch_At_" + TString::Itoa(epoch,10) + ".txt";
   else filen = "CostGradient_Epoch_At_" + TString::Itoa(epoch,10) + "_Monitor.txt";
   std::ostream * output;
   if (filen == "") {
      Error("YMultiLayerPerceptron::DumpWeights()","Invalid file name");
      return kFALSE;
   }
   if (filen == "-")
      output = &std::cout;
   else
      output = new std::ofstream(filen.Data());   
   *output << "#Learning Rate        : " <<  fEta << std::endl; 
   *output << "#Monitoring Threshold : " <<  threshold << " um " << std::endl;    
   *output << "#Epoch #ChipID #neurons(3) #synapses(6) Cost Gradients" << std::endl;
   
   int Nnentries = fSCNetwork[0]->GetNetwork().GetEntriesFast(); 
   int Snentries = fSCNetwork[0]->GetSynapses().GetEntriesFast();
   for(int ic = 0; ic < nSensors; ic++){
      double sum_grad = 0;
      for (int j=2;j<Nnentries + Snentries + 6;j++) { 
         sum_grad += bufferArr[NetworkChips[ic]][j];
      }  
      if(std::abs(sum_grad)<1e-9) continue; 
      bool monitoring = false;

      for (int j=Nnentries + Snentries;j<Nnentries + Snentries + 6;j++) { 
         if(std::abs(bufferArr[NetworkChips[ic]][j])>threshold*1e-4) monitoring = true;             
      }       
      if(monitoring==false) continue;
      *output << "Epoch" << epoch << " ";
      *output << "Layer" << yGEOM->GetLayer(NetworkChips[ic]) << " ";
      *output << "Chip"  << NetworkChips[ic] << " " << fSCNetwork[NetworkChips[ic]]->GetnEvents() <<" hits ";
      *output << "Grad"  << " ";
      for (int j=2;j<Nnentries + Snentries;j++) { 
         *output << bufferArr[NetworkChips[ic]][j]/(-fEta) <<" ";                   
      }        
      *output << "AlignPar(R,T)"  << " ";  
      for (int j=Nnentries + Snentries;j<Nnentries + Snentries + 6;j++) { 
         *output << bufferArr[NetworkChips[ic]][j]/(-fEta) <<" ";                   
      }             
      *output << "Grad"  << " ";
      *output<<std::endl;                                        
   } 
   if (filen != "-") {
      ((std::ofstream *) output)->close();
      delete output;
   }
   return kTRUE;
}


////////////////////////////////////////////////////////////////////////////////
/// PrintWeights

Bool_t YMultiLayerPerceptron::PrintCurrentWeights()
{
   std::cout<<"YMultiLayerPerceptron::PrintCurrentWeights"<<std::endl;
   YNeuron *neuron = 0;
   YSynapse *synapse = 0; 
   std::cout << "#neurons weights #synapses weights" << std::endl;
   int Nnentries = fSCNetwork[0]->GetNetwork().GetEntriesFast(); 
   int Snentries = fSCNetwork[0]->GetSynapses().GetEntriesFast();
   for(int ic = 0; ic < nSensors; ic++){
      std::cout << NetworkChips[ic] <<" ";                                                                              
      for (int j=0;j<Nnentries;j++) { 
         neuron = (YNeuron *) fSCNetwork[ic]->GetNetwork().At(j);
         std::cout << neuron->GetWeight() <<" ";                                                                              
      }             
      for (int j=0;j<Snentries;j++) { 
         synapse = (YSynapse *) fSCNetwork[ic]->GetSynapses().At(j);     
         std::cout << synapse->GetWeight() <<" ";                                                                                        
      }    
      std::cout<<std::endl;    
   } 
   std::cout<<"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"<<std::endl;  
   return kTRUE;    
}

Bool_t YMultiLayerPerceptron::LoadOffsetSlopeCorrectionParameters(Option_t * filename)
{
   TString filen = filename;
   Double_t par;
   if (filen == "") {
      Error("LoadOffsetSlopeCorrectionParameters()","Invalid file name");
      return kFALSE;
   }
   char *buff = new char[100];
   std::ifstream input(filen.Data());
   
   
   if (!input.is_open()) {
      delete[] buff;
      return kFALSE;
   }
   if (input.peek() == std::ifstream::traits_type::eof()) {
      return kFALSE; //empty
   }
   
   //#ChipID p1A p1B p1C p1D q1A q1B q1C q1D q1E q1F q1G q1H
   input.getline(buff, 200);

   for(int ic = 0; ic < ChipBoundary[nLAYER]; ic++){
      input >> par;
      for (int ipar=0; ipar<17; ipar++) { 
         input >> par;
         fOffsetTuning[ic]->ftunePAR[ipar] = par;
      }
   }    
   delete[] buff;
   return kTRUE;
}

Bool_t YMultiLayerPerceptron::LoadUpdateSensorList(Option_t * filename)
{
   TString filen = filename;
   Double_t chipID, ntracks, status;
   if (filen == "") {
      Error("LoadUpdateSensorList()","Invalid file name");
      return kFALSE;
   }
   char *buff = new char[100];
   std::ifstream input(filen.Data());
   
   
   if (!input.is_open()) {
      delete[] buff;
      return kFALSE;
   }
   if (input.peek() == std::ifstream::traits_type::eof()) {
      return kFALSE; //empty
   }
   
   //#ChipID #Ntracks #Status
   input.getline(buff, 200);

   for(int ic = 0; ic < ChipBoundary[nLAYER]; ic++){
      input >> chipID;
      input >> ntracks;
      input >> status;
      fSCNetwork[ic]->SetnEvents(ntracks);          
      fUPDATESENSORS->SetBinContent(1+ic,status);
   }    
   delete[] buff;
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Loads the weights from a text file conforming to the format
/// defined by DumpWeights.

Bool_t YMultiLayerPerceptron::LoadWeights(Option_t * filename)
{
   TString filen = filename;
   Double_t chipID;
   Double_t w;
   if (filen == "") {
      Error("YMultiLayerPerceptron::LoadWeights()","Invalid file name");
      return kFALSE;
   }
   char *buff = new char[100];
   std::ifstream input(filen.Data());
   // input normalzation
   input.getline(buff, 100);
   TObjArrayIter *it = (TObjArrayIter *) fFirstLayer.MakeIterator();
   Float_t n1,n2;
   Int_t j, nentries;
   YNeuron *neuron = 0;
   YSynapse *synapse = 0;
   while ((neuron = (YNeuron *) it->Next())) {
      input >> n1 >> n2;
      neuron->SetNormalisation(n2,n1);
   }
   input.getline(buff, 200);
   // output normalization
   input.getline(buff, 200);
   delete it;
   it = (TObjArrayIter *) fLastLayer.MakeIterator();
   while ((neuron = (YNeuron *) it->Next())) {
      input >> n1 >> n2;
      neuron->SetNormalisation(n2,n1);
   }
   delete it;
   input.getline(buff, 200);
   // neuron weights
   input.getline(buff, 200);

   int Nnentries = fSCNetwork[0]->GetNetwork().GetEntriesFast(); 
   int Snentries = fSCNetwork[0]->GetSynapses().GetEntriesFast();
   for(int ic = 0; ic < nSensors; ic++){
      input >> NetworkChips[ic];
      for (j=0;j<Nnentries;j++) { 
         neuron = (YNeuron *) fSCNetwork[ic]->GetNetwork().At(j);
         input >> w;
         neuron->SetWeight(w);                                                                               
      }             
      for (j=0;j<Snentries;j++) { 
         synapse = (YSynapse *) fSCNetwork[ic]->GetSynapses().At(j);     
         input >> w;
         synapse->SetWeight(w); 
      }    
      //continue;
      for (j=0;j<3;j++) {
         input >> w;
         fSCNetwork[ic]->SetaR(j,w);
      }
      for (j=0;j<3;j++) {
         input >> w;
         fSCNetwork[ic]->SetaT(j,w);
      } 
      
   }    
   delete[] buff;
   return kTRUE;
}

Bool_t YMultiLayerPerceptron::LoadWeightsDetectorUnit(Option_t * filename)
{

   TString filen = filename;
   Double_t w;
   if (filen == "") {
      Error("YMultiLayerPerceptron::LoadWeights()","Invalid file name");
      return kFALSE;
   }
   char *buff = new char[400];
   std::ifstream input(filen.Data());
   // input normalzation
   input.getline(buff, 400);
   TObjArrayIter *it = (TObjArrayIter *) fFirstLayer.MakeIterator();
   Float_t n1,n2;
   Int_t j, nentries;
   YNeuron *neuron = 0;
   YSynapse *synapse = 0;
   YNeuron *neuronDU[8];
   YSynapse *synapseDU[8];     
   Int_t index;
   neuronDU[0] = 0;
   neuronDU[1] = 0;
   neuronDU[2] = 0;
   neuronDU[3] = 0;
   neuronDU[4] = 0;
   neuronDU[5] = 0;
   neuronDU[6] = 0;
   neuronDU[7] = 0;
                   
   synapseDU[0] = 0;
   synapseDU[1] = 0;
   synapseDU[2] = 0;
   synapseDU[3] = 0;
   synapseDU[4] = 0;
   synapseDU[5] = 0;
   synapseDU[6] = 0;
   synapseDU[7] = 0;   
   while ((neuron = (YNeuron *) it->Next())) {
      input >> n1 >> n2;
      neuron->SetNormalisation(n2,n1);
   }
   input.getline(buff, 400);
   // output normalization
   input.getline(buff, 400);
   delete it;
   it = (TObjArrayIter *) fLastLayer.MakeIterator();
   while ((neuron = (YNeuron *) it->Next())) {
      input >> n1 >> n2;
      neuron->SetNormalisation(n2,n1);
   }
   delete it;
   input.getline(buff, 400);
   input.getline(buff, 400);

   int Nnentries = fSCNetwork[0]->GetNetwork().GetEntriesFast(); 
   int Snentries = fSCNetwork[0]->GetSynapses().GetEntriesFast();

   input.getline(buff, 400);
   // #ITS
   
   for (j=0;j<Nnentries;j++) { 
      neuronDU[0] = (YNeuron *) fDetectorUnit->fSCNetwork->GetNetwork().At(j);
      input >> w;
      neuronDU[0]->SetWeight(w);                                                                               
   }             
   for (j=0;j<Snentries;j++) { 
      synapseDU[0] = (YSynapse *) fDetectorUnit->fSCNetwork->GetSynapses().At(j);      
      input >> w;
      synapseDU[0]->SetWeight(w); 
   }

   input.getline(buff, 400);
   // #ITS IB
   input.getline(buff, 400);
  
   for (j=0;j<Nnentries;j++) { 
      neuronDU[1] = (YNeuron *) fDetectorUnit->SubUnit[0]->fSCNetwork->GetNetwork().At(j);
      input >> w;
      neuronDU[1]->SetWeight(w);  
   }
   for (j=0;j<Snentries;j++) { 
      synapseDU[1] = (YSynapse *) fDetectorUnit->SubUnit[0]->fSCNetwork->GetSynapses().At(j);
      input >> w;
      synapseDU[1]->SetWeight(w); 
   }         
   for(int hs=0; hs<nHalfStaveIB; hs++){  
      input.getline(buff, 400);
      // #Half Stave
      input.getline(buff, 400);
  
      for (j=0;j<Nnentries;j++) { 
         neuronDU[2] = (YNeuron *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->fSCNetwork->GetNetwork().At(j);
         input >> w;
         neuronDU[2]->SetWeight(w);  
      }             
      for (j=0;j<Snentries;j++) { 
         synapseDU[2] = (YSynapse *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->fSCNetwork->GetSynapses().At(j);
         input >> w;
         synapseDU[2]->SetWeight(w); 
      }    
      for(int cz=0; cz<nChipsPerHicIB/nHalfStaveIB; cz++){
         input.getline(buff, 400);
         // #ChipZ
         input.getline(buff, 400);      

         for (j=0;j<Nnentries;j++) { 
            neuronDU[3] = (YNeuron *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetNetwork().At(j);
            input >> w;
            neuronDU[3]->SetWeight(w);  
         }
         for (j=0;j<Snentries;j++) { 
            synapseDU[3] = (YSynapse *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetSynapses().At(j);
            input >> w;
            synapseDU[3]->SetWeight(w); 
         }    
         for(int l=0; l<nLAYERIB; l++){ 
            int lIB = l;
            input.getline(buff, 400);
            // #Layer
            input.getline(buff, 400);      

            for (j=0;j<Nnentries;j++) { 
               neuronDU[4] = (YNeuron *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->fSCNetwork->GetNetwork().At(j);
               input >> w;
               neuronDU[4]->SetWeight(w); 
            }
            for (j=0;j<Snentries;j++) { 
               synapseDU[4] = (YSynapse *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->fSCNetwork->GetSynapses().At(j);
               input >> w;
               synapseDU[4]->SetWeight(w); 
            }         
            for(int hb=0; hb<2; hb++){  
               input.getline(buff, 400);
               // #Half Barrel
               input.getline(buff, 400);      

               for (j=0;j<Nnentries;j++) {
                  neuronDU[5] = (YNeuron *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->fSCNetwork->GetNetwork().At(j);
                  input >> w;
                  neuronDU[5]->SetWeight(w); 
               }
               for (j=0;j<Snentries;j++) {
                  synapseDU[5] = (YSynapse *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->fSCNetwork->GetSynapses().At(j);
                  input >> w;
                  synapseDU[5]->SetWeight(w); 
               }
               /*
               for(int st=0; st<NStaves[l]/2;st++){
                  input.getline(buff, 400);
                  // #Stave
                  input.getline(buff, 400);      
   
                  for (j=0;j<Nnentries;j++) {
                     neuronDU[6] = (YNeuron *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetNetwork().At(j);
                     input >> w;
                     neuronDU[6]->SetWeight(w); 
                  }
                  for (j=0;j<Snentries;j++) {
                     synapseDU[6] = (YSynapse *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetSynapses().At(j);
                     input >> w;
                     synapseDU[6]->SetWeight(w); 
                  }
                  for(int bst=0; bst<nHalfStaveIB; bst++){        
                     input.getline(buff, 400);
                     // #SubStave
                     input.getline(buff, 400);      
   
                     for (j=0;j<Nnentries;j++) {
                        neuronDU[7] = (YNeuron *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetNetwork().At(j);
                        input >> w;
                        neuronDU[7]->SetWeight(w); 
                     }
                     for (j=0;j<Snentries;j++) {
                        synapseDU[7] = (YSynapse *) fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetSynapses().At(j);
                        input >> w;
                        synapseDU[7]->SetWeight(w); 
                     }
                  }   
               }
               */ 
            }
         }
      }
   }

   input.getline(buff, 400);
   // #ITS OB1
   input.getline(buff, 400);
  
   for (j=0;j<Nnentries;j++) { 
      neuronDU[1] = (YNeuron *) fDetectorUnit->SubUnit[1]->fSCNetwork->GetNetwork().At(j);
      input >> w;
      neuronDU[1]->SetWeight(w);  
   }
   for (j=0;j<Snentries;j++) { 
      synapseDU[1] = (YSynapse *) fDetectorUnit->SubUnit[1]->fSCNetwork->GetSynapses().At(j);
      input >> w;
      synapseDU[1]->SetWeight(w); 
   }         
   for(int hs=0; hs<nHalfStaveOB1; hs++){  
      input.getline(buff, 400);
      // #Half Stave
      input.getline(buff, 400);
  
      for (j=0;j<Nnentries;j++) { 
         neuronDU[2] = (YNeuron *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->fSCNetwork->GetNetwork().At(j);
         input >> w;
         neuronDU[2]->SetWeight(w);  
      }             
      for (j=0;j<Snentries;j++) { 
         synapseDU[2] = (YSynapse *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->fSCNetwork->GetSynapses().At(j);
         input >> w;
         synapseDU[2]->SetWeight(w); 
      }    
      for(int cz=0; cz<nChipsPerHicOB1/nHalfStaveOB1; cz++){
         input.getline(buff, 400);
         // #ChipZ
         input.getline(buff, 400);      

         for (j=0;j<Nnentries;j++) { 
            neuronDU[3] = (YNeuron *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetNetwork().At(j);
            input >> w;
            neuronDU[3]->SetWeight(w);  
         }
         for (j=0;j<Snentries;j++) { 
            synapseDU[3] = (YSynapse *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetSynapses().At(j);
            input >> w;
            synapseDU[3]->SetWeight(w); 
         }    
         for(int l=nLAYERIB; l<nLAYERIB + nLAYEROB1; l++){ 
            int lOB = l-nLAYERIB;
            input.getline(buff, 400);
            // #Layer
            input.getline(buff, 400);      

            for (j=0;j<Nnentries;j++) { 
               neuronDU[4] = (YNeuron *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->GetNetwork().At(j);
               input >> w;
               neuronDU[4]->SetWeight(w); 
            }
            for (j=0;j<Snentries;j++) { 
               synapseDU[4] = (YSynapse *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->GetSynapses().At(j);
               input >> w;
               synapseDU[4]->SetWeight(w); 
            }         
            for(int hb=0; hb<2; hb++){  
               input.getline(buff, 400);
               // #Half Barrel
               input.getline(buff, 400);      

               for (j=0;j<Nnentries;j++) {
                  neuronDU[5] = (YNeuron *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->GetNetwork().At(j);
                  input >> w;
                  neuronDU[5]->SetWeight(w); 
               }
               for (j=0;j<Snentries;j++) {
                  synapseDU[5] = (YSynapse *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->GetSynapses().At(j);
                  input >> w;
                  synapseDU[5]->SetWeight(w); 
               }
               /*
               for(int st=0; st<NStaves[l]/2;st++){
                  input.getline(buff, 400);
                  // #Stave
                  input.getline(buff, 400);      
   
                  for (j=0;j<Nnentries;j++) {
                     neuronDU[6] = (YNeuron *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetNetwork().At(j);
                     input >> w;
                     neuronDU[6]->SetWeight(w); 
                  }
                  for (j=0;j<Snentries;j++) {
                     synapseDU[6] = (YSynapse *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetSynapses().At(j);
                     input >> w;
                     synapseDU[6]->SetWeight(w); 
                  }
                  for(int bst=0; bst<nHalfStaveOB1; bst++){        
                     input.getline(buff, 400);
                     // #SubStave
                     input.getline(buff, 400);      
   
                     for (j=0;j<Nnentries;j++) {
                        neuronDU[7] = (YNeuron *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetNetwork().At(j);
                        input >> w;
                        neuronDU[7]->SetWeight(w); 
                     }
                     for (j=0;j<Snentries;j++) {
                        synapseDU[7] = (YSynapse *) fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetSynapses().At(j);
                        input >> w;
                        synapseDU[7]->SetWeight(w); 
                     }
                  }   
               } 
               */
            }
         }
      }
   }

   input.getline(buff, 400);
   // #ITS OB2
   input.getline(buff, 400);
  
   for (j=0;j<Nnentries;j++) { 
      neuronDU[1] = (YNeuron *) fDetectorUnit->SubUnit[2]->fSCNetwork->GetNetwork().At(j);
      input >> w;
      neuronDU[1]->SetWeight(w);  
   }
   for (j=0;j<Snentries;j++) { 
      synapseDU[1] = (YSynapse *) fDetectorUnit->SubUnit[2]->fSCNetwork->GetSynapses().At(j);
      input >> w;
      synapseDU[1]->SetWeight(w); 
   }         
   for(int hs=0; hs<nHalfStaveOB2; hs++){  
      input.getline(buff, 400);
      // #Half Stave
      input.getline(buff, 400);
  
      for (j=0;j<Nnentries;j++) { 
         neuronDU[2] = (YNeuron *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->fSCNetwork->GetNetwork().At(j);
         input >> w;
         neuronDU[2]->SetWeight(w);  
      }             
      for (j=0;j<Snentries;j++) { 
         synapseDU[2] = (YSynapse *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->fSCNetwork->GetSynapses().At(j);
         input >> w;
         synapseDU[2]->SetWeight(w); 
      }    
      for(int cz=0; cz<nChipsPerHicOB2/nHalfStaveOB2; cz++){
         input.getline(buff, 400);
         // #ChipZ
         input.getline(buff, 400);      

         for (j=0;j<Nnentries;j++) { 
            neuronDU[3] = (YNeuron *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetNetwork().At(j);
            input >> w;
            neuronDU[3]->SetWeight(w);  
         }
         for (j=0;j<Snentries;j++) { 
            synapseDU[3] = (YSynapse *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->GetSynapses().At(j);
            input >> w;
            synapseDU[3]->SetWeight(w); 
         }    
         for(int l=nLAYERIB + nLAYEROB1; l<nLAYER; l++){ 
            int lOB = l - (nLAYERIB + nLAYEROB1);
            input.getline(buff, 400);
            // #Layer
            input.getline(buff, 400);      

            for (j=0;j<Nnentries;j++) { 
               neuronDU[4] = (YNeuron *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->GetNetwork().At(j);
               input >> w;
               neuronDU[4]->SetWeight(w); 
            }
            for (j=0;j<Snentries;j++) { 
               synapseDU[4] = (YSynapse *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->GetSynapses().At(j);
               input >> w;
               synapseDU[4]->SetWeight(w); 
            }         
            for(int hb=0; hb<2; hb++){  
               input.getline(buff, 400);
               // #Half Barrel
               input.getline(buff, 400);      

               for (j=0;j<Nnentries;j++) {
                  neuronDU[5] = (YNeuron *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->GetNetwork().At(j);
                  input >> w;
                  neuronDU[5]->SetWeight(w); 
               }
               for (j=0;j<Snentries;j++) {
                  synapseDU[5] = (YSynapse *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->GetSynapses().At(j);
                  input >> w;
                  synapseDU[5]->SetWeight(w); 
               }
               /*
               for(int st=0; st<NStaves[l]/2;st++){
                  input.getline(buff, 400);
                  // #Stave
                  input.getline(buff, 400);      
   
                  for (j=0;j<Nnentries;j++) {
                     neuronDU[6] = (YNeuron *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetNetwork().At(j);
                     input >> w;
                     neuronDU[6]->SetWeight(w); 
                  }
                  for (j=0;j<Snentries;j++) {
                     synapseDU[6] = (YSynapse *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->GetSynapses().At(j);
                     input >> w;
                     synapseDU[6]->SetWeight(w); 
                  }
                  for(int bst=0; bst<nHalfStaveOB2; bst++){        
                     input.getline(buff, 400);
                     // #SubStave
                     input.getline(buff, 400);      
   
                     for (j=0;j<Nnentries;j++) {
                        neuronDU[7] = (YNeuron *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetNetwork().At(j);
                        input >> w;
                        neuronDU[7]->SetWeight(w); 
                     }
                     for (j=0;j<Snentries;j++) {
                        synapseDU[7] = (YSynapse *) fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->GetSynapses().At(j);
                        input >> w;
                        synapseDU[7]->SetWeight(w); 
                     }
                  }   
               } 
               */
            }
         }
      }
   }
 
   delete[] buff;
   return kTRUE;

}

void YMultiLayerPerceptron::SetWeightsDetectorUnitUser()
{
   int Nnentries = fSCNetwork[0]->GetNetwork().GetEntriesFast(); 
   int Snentries = fSCNetwork[0]->GetSynapses().GetEntriesFast();
   YNeuron *neuronTR[2];
   YSynapse *synapseTR[2];     
   neuronTR[0] = 0;
   neuronTR[1] = 0;
   synapseTR[0] = 0;
   synapseTR[1] = 0;
   std::cout<<"YMultiLayerPerceptron::SetWeightsDetectorUnitUser"<<std::endl;
   
   std::cout<<"[ITS IB]"<<std::endl;
   //fDetectorUnit->SubUnit[0]->fSCNetwork->SetUpdateState(true);
   for(int hs=0; hs<nHalfStaveIB; hs++){        
      //fDetectorUnit->SubUnit[0]->SubUnit[hs]->fSCNetwork->SetUpdateState(true);
      for(int cz=0; cz<nChipsPerHicIB/nHalfStaveIB; cz++){
         int chipIDinStave = (nChipsPerHicIB/nHalfStaveIB)*hs + cz;      
         //fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->SetUpdateState(true);
         for(int l=0; l<nLAYERIB; l++){ 
            int lIB = l;
            //fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->fSCNetwork->SetUpdateState(true);
            for(int hb=0; hb<2; hb++){        
               //fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->fSCNetwork->SetUpdateState(true);
               for(int st=0; st<NStaves[l]/2;st++){
                  //fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->SetUpdateState(true);
                  for(int bst=0; bst<nHalfStaveIB; bst++){        
                     //fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->SetUpdateState(true);
                  }
               }               
            }
         }
      }
   }
   std::cout<<"[ITS OB1]"<<std::endl;
   //fDetectorUnit->SubUnit[1]->fSCNetwork->SetUpdateState(true);
   for(int hs=0; hs<nHalfStaveOB1; hs++){        
      //fDetectorUnit->SubUnit[1]->SubUnit[hs]->fSCNetwork->SetUpdateState(true);
      for(int cz=0; cz<nChipsPerHicOB1/nHalfStaveOB1; cz++){
         int chipIDinStave = (nChipsPerHicOB1/nHalfStaveOB1)*hs + cz;      
         //fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->SetUpdateState(true);
         for(int l=nLAYERIB; l<nLAYERIB + nLAYEROB1; l++){ 
            int lOB = l - nLAYERIB;
            //fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->SetUpdateState(true);
            for(int hb=0; hb<2; hb++){        
               //fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->SetUpdateState(true);
               for(int st=0; st<NStaves[l]/2;st++){
                  //fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->SetUpdateState(true);
                  for(int bst=0; bst<nHalfStaveIB; bst++){        
                     //fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->SetUpdateState(true);
                  }
               }                 
            }
         }
      }
   }   
   std::cout<<"[ITS OB2]"<<std::endl;
   //fDetectorUnit->SubUnit[2]->fSCNetwork->SetUpdateState(true);
   for(int hs=0; hs<nHalfStaveOB2; hs++){        
      //fDetectorUnit->SubUnit[2]->SubUnit[hs]->fSCNetwork->SetUpdateState(true);
      for(int cz=0; cz<nChipsPerHicOB2/nHalfStaveOB2; cz++){
         int chipIDinStave = (nChipsPerHicOB2/nHalfStaveOB2)*hs + cz;      
         //fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->fSCNetwork->SetUpdateState(true);
         for(int l=nLAYERIB + nLAYEROB1; l<nLAYER; l++){ 
            int lOB = l - (nLAYERIB + nLAYEROB1);
            //fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork->SetUpdateState(true);
            for(int hb=0; hb<2; hb++){        
               //fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork->SetUpdateState(true);
               for(int st=0; st<NStaves[l]/2;st++){
                  //fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork->SetUpdateState(true);
                  for(int bst=0; bst<nHalfStaveIB; bst++){        
                     //fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork->SetUpdateState(true);
                  }
               }               
            }
         }
      }
   }
}


TVector3 YMultiLayerPerceptron::GetCorrectedS2Vector(int chipID)
{
   if(chipID<0) return TVector3(-9999,-9999,-9999);   
   
   int layer      = yGEOM->GetLayer(chipID);

   double inputS[3][2];  //REF , col, row   
   double outputS[3][3];

   double input_Max[2];
   double input_Min[2]; 

   for(int s=0; s<2; s++){
      double ip, fp;   
      ip = yGEOM->GToS(chipID,yGEOM->LToG(chipID,0,0)(0),	
                              yGEOM->LToG(chipID,0,0)(1),
                              yGEOM->LToG(chipID,0,0)(2))(s);
      fp = yGEOM->GToS(chipID,yGEOM->LToG(chipID,511,1023)(0),
                              yGEOM->LToG(chipID,511,1023)(1),
                              yGEOM->LToG(chipID,511,1023)(2))(s);   
      input_Max[s] = std::max(ip,fp);
      input_Min[s] = std::min(ip,fp);                               
   }

   double row[] = {256, 256, 0};
   double col[] = {511, 1023, 511};
   
   TVector3 locSC[3];    
   TVector3 locGC[3];                                          
   for(int p=0; p<3; p++){
      TVector3 sensorS = yGEOM->GToS(chipID,yGEOM->LToG(chipID,row[p],col[p]).X(),
                                            yGEOM->LToG(chipID,row[p],col[p]).Y(),
                                            yGEOM->LToG(chipID,row[p],col[p]).Z());   

      inputS[p][0] = (double)((sensorS(0)-input_Min[0])/(input_Max[0] - input_Min[0]) - 0.5);
      inputS[p][1] = (double)((sensorS(1)-input_Min[1])/(input_Max[1] - input_Min[1]) - 0.5);

      for(int k =0; k <3; k++){         
         outputS[p][k] = Evaluate(k, inputS[p], chipID); 
      }
      
      locSC[p] = TVector3((inputS[p][0] + 0.5)*(input_Max[0]-input_Min[0])+input_Min[0] + outputS[p][0],
                          (inputS[p][1] + 0.5)*(input_Max[1]-input_Min[1])+input_Min[1] + outputS[p][1],
                          outputS[p][2]);

      locGC[p] = yGEOM->SToG(chipID, locSC[p](0), locSC[p](1), locSC[p](2)); 
   }
   
   TVector3 v2(locGC[0].X()-locGC[2].X(),
               locGC[0].Y()-locGC[2].Y(),
               locGC[0].Z()-locGC[2].Z());           

   return v2;
}

TVector3 YMultiLayerPerceptron::GetCorrectedNormalVector(int chipID)
{
   if(chipID<0) return TVector3(-9999,-9999,-9999);   
   
   int layer      = yGEOM->GetLayer(chipID);

   double inputS[3][2];  //REF , col, row   
   double outputS[3][3];

   double input_Max[2];
   double input_Min[2]; 

   for(int s=0; s<2; s++){
      double ip, fp;   
      ip = yGEOM->GToS(chipID,yGEOM->LToG(chipID,0,0)(0),	
                              yGEOM->LToG(chipID,0,0)(1),
                              yGEOM->LToG(chipID,0,0)(2))(s);
      fp = yGEOM->GToS(chipID,yGEOM->LToG(chipID,511,1023)(0),
                              yGEOM->LToG(chipID,511,1023)(1),
                              yGEOM->LToG(chipID,511,1023)(2))(s);   
      input_Max[s] = std::max(ip,fp);
      input_Min[s] = std::min(ip,fp);                               
   }

   double row[] = {256, 256, 0};
   double col[] = {511, 1023, 511};
   
   TVector3 locSC[3];    
   TVector3 locGC[3];                                          
   for(int p=0; p<3; p++){
      TVector3 sensorS = yGEOM->GToS(chipID,yGEOM->LToG(chipID,row[p],col[p]).X(),
                                            yGEOM->LToG(chipID,row[p],col[p]).Y(),
                                            yGEOM->LToG(chipID,row[p],col[p]).Z());   

      inputS[p][0] = (double)((sensorS(0)-input_Min[0])/(input_Max[0] - input_Min[0]) - 0.5);
      inputS[p][1] = (double)((sensorS(1)-input_Min[1])/(input_Max[1] - input_Min[1]) - 0.5);

      for(int k =0; k <3; k++){         
         outputS[p][k] = Evaluate(k, inputS[p], chipID); 
      }
      
      locSC[p] = TVector3((inputS[p][0] + 0.5)*(input_Max[0]-input_Min[0])+input_Min[0] + outputS[p][0],
                          (inputS[p][1] + 0.5)*(input_Max[1]-input_Min[1])+input_Min[1] + outputS[p][1],
                          outputS[p][2]);

      locGC[p] = yGEOM->SToG(chipID, locSC[p](0), locSC[p](1), locSC[p](2)); 
   }
   

   TVector3 v1(locGC[1].X()-locGC[0].X(),
               locGC[1].Y()-locGC[0].Y(),
               locGC[1].Z()-locGC[0].Z());
   TVector3 v2(locGC[2].X()-locGC[0].X(),
               locGC[2].Y()-locGC[0].Y(),
               locGC[2].Z()-locGC[0].Z());           
   TVector3 v2v1 = v2.Cross(v1);
   double m2m1 = TMath::Sqrt((v2v1.X()*v2v1.X())+(v2v1.Y()*v2v1.Y())+(v2v1.Z()*v2v1.Z()));
   TVector3 v3(v2v1.X()/m2m1,v2v1.Y()/m2m1,v2v1.Z()/m2m1);
   return v3;
}


////////////////////////////////////////////////////////////////////////////////
/// Returns the Neural Net for a given set of input parameters
/// #parameters must equal #input neurons

Double_t YMultiLayerPerceptron::Evaluate(Int_t index, Double_t *params) const
{
   TObjArrayIter *it = (TObjArrayIter *) fNetwork.MakeIterator();
   YNeuron *neuron;
   while ((neuron = (YNeuron *) it->Next()))
      neuron->SetNewEvent();
   delete it;
   it = (TObjArrayIter *) fFirstLayer.MakeIterator();
   Int_t i=0;
   while ((neuron = (YNeuron *) it->Next()))
      neuron->ForceExternalValue(params[i++]);
   delete it;
   YNeuron *out = (YNeuron *) (fLastLayer.At(index));
   if (out)
      return out->GetValue();
   else
      return 0;
}

Double_t YMultiLayerPerceptron::Evaluate(Int_t index, Double_t *params, Int_t chipID) const
{

   TObjArrayIter *it = (TObjArrayIter *) fSCNetwork[chipID]->GetNetwork().MakeIterator();
   YNeuron *neuron;
   while ((neuron = (YNeuron *) it->Next()))
      neuron->SetNewEvent();
   delete it;
   it = (TObjArrayIter *) fSCNetwork[chipID]->GetFirstLayer().MakeIterator();
   Int_t i=0;
   while ((neuron = (YNeuron *) it->Next()))
      neuron->ForceExternalValue(params[i++]);
   delete it;
   YNeuron *out = (YNeuron *) (fSCNetwork[chipID]->GetLastLayer().At(index));
   if (out)
      return out->GetValue();
   else
      return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Exports the NN as a function for any non-ROOT-dependant code
/// Supported languages are: only C++ , FORTRAN and Python (yet)
/// This feature is also usefull if you want to plot the NN as
/// a function (TF1 or TF2).

void YMultiLayerPerceptron::Export(Option_t * filename, Option_t * language) const
{
   TString lg = language;
   lg.ToUpper();
   Int_t i;
   if(GetType()==YNeuron::kExternal) {
      Warning("YMultiLayerPerceptron::Export","Request to export a network using an external function");
   }
   if (lg == "C++") {
      TString basefilename = filename;
      Int_t slash = basefilename.Last('/')+1;
      if (slash) basefilename = TString(basefilename(slash, basefilename.Length()-slash));

      TString classname = basefilename;
      TString header = filename;
      header += ".h";
      TString source = filename;
      source += ".cxx";
      std::ofstream headerfile(header);
      std::ofstream sourcefile(source);
      headerfile << "#ifndef " << basefilename << "_h" << std::endl;
      headerfile << "#define " << basefilename << "_h" << std::endl << std::endl;
      headerfile << "class " << classname << " { " << std::endl;
      headerfile << "public:" << std::endl;
      headerfile << "   " << classname << "() {}" << std::endl;
      headerfile << "   ~" << classname << "() {}" << std::endl;
      sourcefile << "#include \"" << header << "\"" << std::endl;
      sourcefile << "#include <cmath>" << std::endl << std::endl;
      headerfile << "   double Value(int index";
      sourcefile << "double " << classname << "::Value(int index";
      for (i = 0; i < fFirstLayer.GetEntriesFast(); i++) {
         headerfile << ",double in" << i;
         sourcefile << ",double in" << i;
      }
      headerfile << ");" << std::endl;
      sourcefile << ") {" << std::endl;
      for (i = 0; i < fFirstLayer.GetEntriesFast(); i++)
         sourcefile << "   input" << i << " = (in" << i << " - "
             << ((YNeuron *) fFirstLayer[i])->GetNormalisation()[1] << ")/"
             << ((YNeuron *) fFirstLayer[i])->GetNormalisation()[0] << ";"
             << std::endl;
      sourcefile << "   switch(index) {" << std::endl;
      YNeuron *neuron;
      TObjArrayIter *it = (TObjArrayIter *) fLastLayer.MakeIterator();
      Int_t idx = 0;
      while ((neuron = (YNeuron *) it->Next()))
         sourcefile << "     case " << idx++ << ":" << std::endl
                    << "         return neuron" << neuron << "();" << std::endl;
      sourcefile << "     default:" << std::endl
                 << "         return 0.;" << std::endl << "   }"
                 << std::endl;
      sourcefile << "}" << std::endl << std::endl;
      headerfile << "   double Value(int index, double* input);" << std::endl;
      sourcefile << "double " << classname << "::Value(int index, double* input) {" << std::endl;
      for (i = 0; i < fFirstLayer.GetEntriesFast(); i++)
         sourcefile << "   input" << i << " = (input[" << i << "] - "
             << ((YNeuron *) fFirstLayer[i])->GetNormalisation()[1] << ")/"
             << ((YNeuron *) fFirstLayer[i])->GetNormalisation()[0] << ";"
             << std::endl;
      sourcefile << "   switch(index) {" << std::endl;
      delete it;
      it = (TObjArrayIter *) fLastLayer.MakeIterator();
      idx = 0;
      while ((neuron = (YNeuron *) it->Next()))
         sourcefile << "     case " << idx++ << ":" << std::endl
                    << "         return neuron" << neuron << "();" << std::endl;
      sourcefile << "     default:" << std::endl
                 << "         return 0.;" << std::endl << "   }"
                 << std::endl;
      sourcefile << "}" << std::endl << std::endl;
      headerfile << "private:" << std::endl;
      for (i = 0; i < fFirstLayer.GetEntriesFast(); i++)
         headerfile << "   double input" << i << ";" << std::endl;
      delete it;
      it = (TObjArrayIter *) fNetwork.MakeIterator();
      idx = 0;
      while ((neuron = (YNeuron *) it->Next())) {
         if (!neuron->GetPre(0)) {
            headerfile << "   double neuron" << neuron << "();" << std::endl;
            sourcefile << "double " << classname << "::neuron" << neuron
                       << "() {" << std::endl;
            sourcefile << "   return input" << idx++ << ";" << std::endl;
            sourcefile << "}" << std::endl << std::endl;
         } else {
            headerfile << "   double input" << neuron << "();" << std::endl;
            sourcefile << "double " << classname << "::input" << neuron
                       << "() {" << std::endl;
            sourcefile << "   double input = " << neuron->GetWeight()
                       << ";" << std::endl;
            YSynapse *syn = 0;
            Int_t n = 0;
            while ((syn = neuron->GetPre(n++))) {
               sourcefile << "   input += synapse" << syn << "();" << std::endl;
            }
            sourcefile << "   return input;" << std::endl;
            sourcefile << "}" << std::endl << std::endl;

            headerfile << "   double neuron" << neuron << "();" << std::endl;
            sourcefile << "double " << classname << "::neuron" << neuron << "() {" << std::endl;
            sourcefile << "   double input = input" << neuron << "();" << std::endl;
            switch(neuron->GetType()) {
               case (YNeuron::kSigmoid):
                  {
                     sourcefile << "   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * ";
                     break;
                  }
               case (YNeuron::kLinear):
                  {
                     sourcefile << "   return (input * ";
                     break;
                  }
               case (YNeuron::kTanh):
                  {
                     sourcefile << "   return (tanh(input) * ";
                     break;
                  }
               case (YNeuron::kGauss):
                  {
                     sourcefile << "   return (exp(-input*input) * ";
                     break;
                  }
               case (YNeuron::kSoftmax):
                  {
                     sourcefile << "   return (exp(input) / (";
                     Int_t nn = 0;
                     YNeuron* side = neuron->GetInLayer(nn++);
                     sourcefile << "exp(input" << side << "())";
                     while ((side = neuron->GetInLayer(nn++)))
                        sourcefile << " + exp(input" << side << "())";
                     sourcefile << ") * ";
                     break;
                  }
               default:
                  {
                     sourcefile << "   return (0.0 * ";
                  }
            }
            sourcefile << neuron->GetNormalisation()[0] << ")+" ;
            sourcefile << neuron->GetNormalisation()[1] << ";" << std::endl;
            sourcefile << "}" << std::endl << std::endl;
         }
      }
      delete it;
      YSynapse *synapse = 0;
      it = (TObjArrayIter *) fSynapses.MakeIterator();
      while ((synapse = (YSynapse *) it->Next())) {
         headerfile << "   double synapse" << synapse << "();" << std::endl;
         sourcefile << "double " << classname << "::synapse"
                    << synapse << "() {" << std::endl;
         sourcefile << "   return (neuron" << synapse->GetPre()
                    << "()*" << synapse->GetWeight() << ");" << std::endl;
         sourcefile << "}" << std::endl << std::endl;
      }
      delete it;
      headerfile << "};" << std::endl << std::endl;
      headerfile << "#endif // " << basefilename << "_h" << std::endl << std::endl;
      headerfile.close();
      sourcefile.close();
      std::cout << header << " and " << source << " created." << std::endl;
   }
   else if(lg == "FORTRAN") {
      TString implicit = "      implicit double precision (a-h,n-z)\n";
      std::ofstream sigmoid("sigmoid.f");
      sigmoid         << "      double precision FUNCTION SIGMOID(X)"        << std::endl
                    << implicit
                << "      IF(X.GT.37.) THEN"                        << std::endl
                    << "         SIGMOID = 1."                        << std::endl
                << "      ELSE IF(X.LT.-709.) THEN"                << std::endl
                    << "         SIGMOID = 0."                        << std::endl
                    << "      ELSE"                                        << std::endl
                    << "         SIGMOID = 1./(1.+EXP(-X))"                << std::endl
                    << "      ENDIF"                                << std::endl
                    << "      END"                                        << std::endl;
      sigmoid.close();
      TString source = filename;
      source += ".f";
      std::ofstream sourcefile(source);

      // Header
      sourcefile << "      double precision function " << filename
                 << "(x, index)" << std::endl;
      sourcefile << implicit;
      sourcefile << "      double precision x(" <<
      fFirstLayer.GetEntriesFast() << ")" << std::endl << std::endl;

      // Last layer
      sourcefile << "C --- Last Layer" << std::endl;
      YNeuron *neuron;
      TObjArrayIter *it = (TObjArrayIter *) fLastLayer.MakeIterator();
      Int_t idx = 0;
      TString ifelseif = "      if (index.eq.";
      while ((neuron = (YNeuron *) it->Next())) {
         sourcefile << ifelseif.Data() << idx++ << ") then" << std::endl
                    << "          " << filename
                    << "=neuron" << neuron << "(x);" << std::endl;
         ifelseif = "      else if (index.eq.";
      }
      sourcefile << "      else" << std::endl
                 << "          " << filename << "=0.d0" << std::endl
                 << "      endif" << std::endl;
      sourcefile << "      end" << std::endl;

      // Network
      sourcefile << "C --- First and Hidden layers" << std::endl;
      delete it;
      it = (TObjArrayIter *) fNetwork.MakeIterator();
      idx = 0;
      while ((neuron = (YNeuron *) it->Next())) {
         sourcefile << "      double precision function neuron"
                    << neuron << "(x)" << std::endl
                    << implicit;
         sourcefile << "      double precision x("
                    << fFirstLayer.GetEntriesFast() << ")" << std::endl << std::endl;
         if (!neuron->GetPre(0)) {
            sourcefile << "      neuron" << neuron
             << " = (x(" << idx+1 << ") - "
             << ((YNeuron *) fFirstLayer[idx])->GetNormalisation()[1]
             << "d0)/"
             << ((YNeuron *) fFirstLayer[idx])->GetNormalisation()[0]
             << "d0" << std::endl;
            idx++;
         } else {
            sourcefile << "      neuron" << neuron
                       << " = " << neuron->GetWeight() << "d0" << std::endl;
            YSynapse *syn;
            Int_t n = 0;
            while ((syn = neuron->GetPre(n++)))
               sourcefile << "      neuron" << neuron
                              << " = neuron" << neuron
                          << " + synapse" << syn << "(x)" << std::endl;
            switch(neuron->GetType()) {
               case (YNeuron::kSigmoid):
                  {
                     sourcefile << "      neuron" << neuron
                                << "= (sigmoid(neuron" << neuron << ")*";
                     break;
                  }
               case (YNeuron::kLinear):
                  {
                     break;
                  }
               case (YNeuron::kTanh):
                  {
                     sourcefile << "      neuron" << neuron
                                << "= (tanh(neuron" << neuron << ")*";
                     break;
                  }
               case (YNeuron::kGauss):
                  {
                     sourcefile << "      neuron" << neuron
                                << "= (exp(-neuron" << neuron << "*neuron"
                                << neuron << "))*";
                     break;
                  }
               case (YNeuron::kSoftmax):
                  {
                     Int_t nn = 0;
                     YNeuron* side = neuron->GetInLayer(nn++);
                     sourcefile << "      div = exp(neuron" << side << "())" << std::endl;
                     while ((side = neuron->GetInLayer(nn++)))
                        sourcefile << "      div = div + exp(neuron" << side << "())" << std::endl;
                     sourcefile << "      neuron"  << neuron ;
                     sourcefile << "= (exp(neuron" << neuron << ") / div * ";
                     break;
                  }
               default:
                  {
                     sourcefile << "   neuron " << neuron << "= 0.";
                  }
            }
            sourcefile << neuron->GetNormalisation()[0] << "d0)+" ;
            sourcefile << neuron->GetNormalisation()[1] << "d0" << std::endl;
         }
         sourcefile << "      end" << std::endl;
      }
      delete it;

      // Synapses
      sourcefile << "C --- Synapses" << std::endl;
      YSynapse *synapse = 0;
      it = (TObjArrayIter *) fSynapses.MakeIterator();
      while ((synapse = (YSynapse *) it->Next())) {
         sourcefile << "      double precision function " << "synapse"
                    << synapse << "(x)\n" << implicit;
         sourcefile << "      double precision x("
                    << fFirstLayer.GetEntriesFast() << ")" << std::endl << std::endl;
         sourcefile << "      synapse" << synapse
                    << "=neuron" << synapse->GetPre()
                    << "(x)*" << synapse->GetWeight() << "d0" << std::endl;
         sourcefile << "      end" << std::endl << std::endl;
      }
      delete it;
      sourcefile.close();
      std::cout << source << " created." << std::endl;
   }
   else if(lg == "PYTHON") {
      TString classname = filename;
      TString pyfile = filename;
      pyfile += ".py";
      std::ofstream pythonfile(pyfile);
      pythonfile << "from math import exp" << std::endl << std::endl;
      pythonfile << "from math import tanh" << std::endl << std::endl;
      pythonfile << "class " << classname << ":" << std::endl;
      pythonfile << "\tdef value(self,index";
      for (i = 0; i < fFirstLayer.GetEntriesFast(); i++) {
         pythonfile << ",in" << i;
      }
      pythonfile << "):" << std::endl;
      for (i = 0; i < fFirstLayer.GetEntriesFast(); i++)
         pythonfile << "\t\tself.input" << i << " = (in" << i << " - "
             << ((YNeuron *) fFirstLayer[i])->GetNormalisation()[1] << ")/"
             << ((YNeuron *) fFirstLayer[i])->GetNormalisation()[0] << std::endl;
      YNeuron *neuron;
      TObjArrayIter *it = (TObjArrayIter *) fLastLayer.MakeIterator();
      Int_t idx = 0;
      while ((neuron = (YNeuron *) it->Next()))
         pythonfile << "\t\tif index==" << idx++
                    << ": return self.neuron" << neuron << "();" << std::endl;
      pythonfile << "\t\treturn 0." << std::endl;
      delete it;
      it = (TObjArrayIter *) fNetwork.MakeIterator();
      idx = 0;
      while ((neuron = (YNeuron *) it->Next())) {
         pythonfile << "\tdef neuron" << neuron << "(self):" << std::endl;
         if (!neuron->GetPre(0))
            pythonfile << "\t\treturn self.input" << idx++ << std::endl;
         else {
            pythonfile << "\t\tinput = " << neuron->GetWeight() << std::endl;
            YSynapse *syn;
            Int_t n = 0;
            while ((syn = neuron->GetPre(n++)))
               pythonfile << "\t\tinput = input + self.synapse"
                          << syn << "()" << std::endl;
            switch(neuron->GetType()) {
               case (YNeuron::kSigmoid):
                  {
                     pythonfile << "\t\tif input<-709. : return " << neuron->GetNormalisation()[1] << std::endl;
                     pythonfile << "\t\treturn ((1/(1+exp(-input)))*";
                     break;
                  }
               case (YNeuron::kLinear):
                  {
                     pythonfile << "\t\treturn (input*";
                     break;
                  }
               case (YNeuron::kTanh):
                  {
                     pythonfile << "\t\treturn (tanh(input)*";
                     break;
                  }
               case (YNeuron::kGauss):
                  {
                     pythonfile << "\t\treturn (exp(-input*input)*";
                     break;
                  }
               case (YNeuron::kSoftmax):
                  {
                     pythonfile << "\t\treturn (exp(input) / (";
                     Int_t nn = 0;
                     YNeuron* side = neuron->GetInLayer(nn++);
                     pythonfile << "exp(self.neuron" << side << "())";
                     while ((side = neuron->GetInLayer(nn++)))
                        pythonfile << " + exp(self.neuron" << side << "())";
                     pythonfile << ") * ";
                     break;
                  }
               default:
                  {
                     pythonfile << "\t\treturn 0.";
                  }
            }
            pythonfile << neuron->GetNormalisation()[0] << ")+" ;
            pythonfile << neuron->GetNormalisation()[1] << std::endl;
         }
      }
      delete it;
      YSynapse *synapse = 0;
      it = (TObjArrayIter *) fSynapses.MakeIterator();
      while ((synapse = (YSynapse *) it->Next())) {
         pythonfile << "\tdef synapse" << synapse << "(self):" << std::endl;
         pythonfile << "\t\treturn (self.neuron" << synapse->GetPre()
                    << "()*" << synapse->GetWeight() << ")" << std::endl;
      }
      delete it;
      pythonfile.close();
      std::cout << pyfile << " created." << std::endl;
   }
}

vector<Int_t*> YMultiLayerPerceptron::EventIndex(int type) //const
{
   vector<Int_t*> eventindex;
   YNeuron *neuron_A;    

   switch (type) {
      case 1:{
         Int_t nEvents = fTraining->GetN();      
         std::cout<<"YMultiLayerPerceptron::EventIndex Train Set "<< nEvents<< " entries"<<std::endl;         
         for (int i = 0; i < nEvents; i++) {
            GetEntry(fTraining->GetEntry(i));
            neuron_A = (YNeuron *) fAddition.At(0); 
            int index[2];
            neuron_A->SetNeuronIndex(1000 + 6);  
            int buffer = (int)neuron_A->GetBranchAddition();           
            index[0] = (int)buffer;
            neuron_A->SetNeuronIndex(1000 + 7);             
            index[1] = (int)neuron_A->GetBranchAddition();
            eventindex.push_back(new int [2]);
            eventindex[eventindex.size()-1][0] = index[0];
            eventindex[eventindex.size()-1][1] = index[1];            
            //i = i + index[1];
            fTotNTraining += index[1];
            //std::cout<<"YMultiLayerPerceptron::EventIndex "<<type<<" "<<buffer<<" "<<index[0]<<" "<<index[1]<<std::endl;
         }           
         break;
      }
      case 2:{
         Int_t nEvents = fTest->GetN();       
         std::cout<<"YMultiLayerPerceptron::EventIndex Test Set "<< nEvents<< " entries"<<std::endl;                          
         for (int i = 0; i < nEvents; i++) {
            GetEntry(fTest->GetEntry(i));
            neuron_A = (YNeuron *) fAddition.At(0);  
            int index[2];
            neuron_A->SetNeuronIndex(1000 + 6);    
            int buffer = (int)neuron_A->GetBranchAddition();                                  
            index[0] = (int)buffer;
            neuron_A->SetNeuronIndex(1000 + 7);   
            index[1] = (int)neuron_A->GetBranchAddition();
            eventindex.push_back(new int [2]);
            eventindex[eventindex.size()-1][0] = index[0];
            eventindex[eventindex.size()-1][1] = index[1];  
            //i = i + index[1];
            fTotNTest += index[1];
            //std::cout<<"YMultiLayerPerceptron::EventIndex "<<type<<" "<<buffer<<" "<<index[0]<<" "<<index[1]<<std::endl;            
         }        
         break;
      }
      default:{
         Int_t nEvents = fData->GetEntriesFast();   
         std::cout<<"YMultiLayerPerceptron::EventIndex Data Set "<< nEvents<< " entries"<<std::endl;                                                  
         for (int i = 0; i < nEvents; i++) {
            fData->GetEntry(i);
            neuron_A = (YNeuron *) fAddition.At(0); 
            int index[2];
            neuron_A->SetNeuronIndex(1000 + 6);   
            int buffer = (int)neuron_A->GetBranchAddition();                                   
            index[0] = (int)buffer;
            neuron_A->SetNeuronIndex(1000 + 7);   
            index[1] = (int)neuron_A->GetBranchAddition();
            eventindex.push_back(new int [2]);
            eventindex[eventindex.size()-1][0] = index[0];
            eventindex[eventindex.size()-1][1] = index[1];  
            //i = i + index[1];
            fTotNEvents += index[1];
            //std::cout<<"YMultiLayerPerceptron::EventIndex "<<type<<" "<<buffer<<" "<<index[0]<<" "<<index[1]<<std::endl;            
         } 
      }   
             
   }
   return eventindex;
}

////////////////////////////////////////////////////////////////////////////////
/// Shuffle the Int_t index[n] in input.
/// Input:
///   index: the array to shuffle
///   n: the size of the array
/// Output:
///   index: the shuffled indexes
/// This method is used for stochastic training

void YMultiLayerPerceptron::Shuffle(Int_t * index, Int_t n) const
{
   //TTimeStamp ts;
   //TRandom3 rnd(ts.GetSec());
   TRandom3 rnd;
   Int_t j, k;
   Int_t a = n - 1;
   for (Int_t i = 0; i < n; i++) {
      j = (Int_t) (rnd.Rndm() * a);
      k = index[j];
      index[j] = index[i];
      index[i] = k;
   }
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// One step for the stochastic method
/// buffer should contain the previous dw vector and will be updated

void YMultiLayerPerceptron::MLP_Stochastic(Double_t * buffer)
{
   Int_t nEvents = fTraining->GetN();
   Int_t *index = new Int_t[nEvents];
   Int_t i,j,nentries;
   for (i = 0; i < nEvents; i++)
      index[i] = i;
   fEta *= fEtaDecay;
   Shuffle(index, nEvents);
   YNeuron *neuron;
   YSynapse *synapse;
   for (i = 0; i < nEvents; i++) {
      GetEntry(fTraining->GetEntry(index[i]));
      // First compute DcDw for all neurons: force calculation before
      // modifying the weights.
      nentries = fFirstLayer.GetEntriesFast();
      for (j=0;j<nentries;j++) {
         neuron = (YNeuron *) fFirstLayer.At(j);
         neuron->GetDcDw();
      }
      Int_t cnt = 0;
      // Step for all neurons
      nentries = fNetwork.GetEntriesFast();
      for (j=0;j<nentries;j++) {
         neuron = (YNeuron *) fNetwork.At(j);
         buffer[cnt] = (-fEta) * (neuron->GetDcDw() + fDelta)
                       + fEpsilon * buffer[cnt];
         neuron->SetWeight(neuron->GetWeight() + buffer[cnt++]);
      }
      // Step for all synapses
      nentries = fSynapses.GetEntriesFast();
      for (j=0;j<nentries;j++) {
         synapse = (YSynapse *) fSynapses.At(j);
         buffer[cnt] = (-fEta) * (synapse->GetDcDw() + fDelta)
                       + fEpsilon * buffer[cnt];
         synapse->SetWeight(synapse->GetWeight() + buffer[cnt++]);
      }
   }
   delete[]index;
}

////////////////////////////////////////////////////////////////////////////////
/// MLP_StochasticArr

void YMultiLayerPerceptron::MLP_StochasticArr(Double_t** bufferArr)
{
   std::cout<<"MLP_StochasticArr Start "<<std::endl;
   Int_t nEvents = fTraining->GetN();
   Int_t *index = new Int_t[nEvents];
   //Int_t i,j,nentries;
   Int_t cnt = 0;        
   for (int i = 0; i < nEvents; i++)
      index[i] = i;
   fEta *= fEtaDecay;
   //Shuffle(index, nEvents);
   YNeuron *neuron;
   YSynapse *synapse;
   YNeuron *neuron_I;
   YNeuron *neuron_O;
   YNeuron *neuron_A;   
   
   for (int i = 0; i < nEvents; i++) {
      GetEntry(fTraining->GetEntry(index[i]));
      int Npronged = fTrainingIndex[index[i]][1];     
  
      Double_t mlpInputCenter[Npronged][nLAYER][2];  //layer input
      Double_t mlpOutputCenter[Npronged][nLAYER][3]; //layer output
         
      int staveIndex[Npronged][nLAYER], chipIndex[Npronged][nLAYER], chipID[Npronged][nLAYER];
      double input[Npronged][nLAYER][2], output[Npronged][nLAYER][3];  //prong layer axis
      double extended[Npronged][nLAYER][3];  
      double addition[Npronged][8];  

      for(int t = 0; t < Npronged; t++){ //             
         //std::cout<<"(StochasticArr A) Event i index[i] ntracks index :: "<<i<<" "<<index[i]<<" "<<Npronged<<" "<<t<<std::endl;         
         for(int j = 0; j < 8; j++) {             
            neuron_A = (YNeuron *) fAddition.At(0); 
            neuron_A->SetNeuronIndex(1000 + j);                 
            neuron_A->SetNewEvent();
            addition[t][j] = neuron_A->GetBranchAddition();             
         }         
   
         for(int j=0; j<fNetwork.GetEntriesFast(); j++) {
            neuron = (YNeuron *)fNetwork.At(j);
            neuron->SetNeuronIndex(t);                
            neuron->SetNewEvent();
         }
         for(int j = 0; j < 2*nLAYER; j++){               
            neuron_I = (YNeuron *) fFirstLayer.At(j);  
            neuron_I->SetNeuronIndex(t); 
            neuron_I->SetNewEvent(); 
            int p1 = (int)(j/2);
            int p2 = (int)(j%2);               
            input[t][p1][p2] = neuron_I->GetValue();
         }    
         for(int j = 0; j < 3*nLAYER; j++) {             
            neuron_O = (YNeuron *) fLastLayer.At(j); 
            neuron_O->SetNeuronIndex(t); 
            neuron_O->SetNewEvent(); 
            neuron_O->SetLFENodeIndex((3*nLAYER)*t+j);      
            neuron_O->SetLFELayerTrain(fLayerTrain);
            int p1 = (int)(j/3);
            int p2 = (int)(j%3);        
            extended[t][p1][p2] = neuron_O->GetBranch();               						//Very Important !!!!!! 2020 08 26           
            //std::cout<<j<<" "<<extended[t][p1][p2]<<std::endl;   
         }
         for(int ln = 0; ln < nLAYER; ln ++){
            staveIndex[t][ln] = extended[t][ln][0];  
            chipIndex[t][ln]  = extended[t][ln][1];  
            chipID[t][ln]     = extended[t][ln][2];  
         }     

         //load Network -> YSensorCorrection
         for(int ln = 0; ln < nLAYER; ln ++){
         
            if(chipID[t][ln]<0) continue;
            
            for(int j=0; j<fSCNetwork[chipID[t][ln]]->GetNetwork().GetEntriesFast(); j++) {
               neuron = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetNetwork().At(j); 
               neuron->SetNeuronIndex(t);                  
               neuron->SetNewEvent();
            }  
                 
            for(int j = 0; j < 2; j++) {
               neuron_I = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetFirstLayer().At(j);    
               neuron_I->SetLFENodeIndex((3*nLAYER)*t+3*ln+j);
               neuron_I->ClearLFEmemory();      
            }
               
            for(int j = 0; j < 3; j++) {
               neuron_O = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetLastLayer()[j];
               int p1 = (int)(ln);    
               int p2 = (int)(j%3);
               output[t][p1][p2] = Evaluate(p2, input[t][p1], chipID[t][ln]);
               neuron_O->SetLFENodeIndex((3*nLAYER)*t+3*ln+j);
               neuron_O->ClearLFEmemory();   
               neuron_O->SetLFELayerTrain(fLayerTrain);                                        
               neuron_O = 0;   
            }
            for(int c=0; c<2; c++){
               mlpInputCenter[t][ln][c]=0;
            }
            for(int c=0; c<3; c++){
               mlpOutputCenter[t][ln][c] = Evaluate(c, mlpInputCenter[t][ln], chipID[t][ln]); 
            }  
         }
      }     
      for(int t = 0; t < Npronged; t++){                   
         for(int ln = 0; ln < nLAYER; ln ++){ 
            if(chipID[t][ln]<0) continue; 
             
            for(int j = 0; j < 2; j++) {         
               neuron_I = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetFirstLayer().At(j);        
               for(int k = 0; k < 2*nLAYER; k++){    
                  int p1 = (int)(k/2);
                  int p2 = (int)(k%2);   
                  for(int u = 0; u < Npronged; u++){                   
                     neuron_I->SetLFEinput(3*p1 + p2, u, input[u][p1][p2]);                  
                  }               
               } 
               for(int k =0; k < 1*nLAYER; k++){						
                  for(int u = 0; u < Npronged; u++){                                 
                     neuron_I->SetLFEinput(3*k + 2, u,0);
                  }
               } 
            }
            for(int j = 0; j < 3; j++) {       
               neuron_O = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetLastLayer().At(j);
               for(int k = 0; k < 2*nLAYER; k++){    
                  int p1 = (int)(k/2);
                  int p2 = (int)(k%2);   
                  for(int u = 0; u < Npronged; u++){                      
                     neuron_O->SetLFEinput(3*p1 + p2, u, input[u][p1][p2]); 
                  }                               
               }
               for(int k =0; k < 1*nLAYER; k++){						
                  for(int u = 0; u < Npronged; u++){                                        
                     neuron_O->SetLFEinput(3*k + 2, u,0);
                  }
               }            
               for(int k = 0; k < 3*nLAYER; k++){
                  int p1 = (int)(k/3);                        
                  int p2 = (int)(k%3);    
                  for(int u = 0; u < Npronged; u++){                                          
                     neuron_O->SetLFEoutput(k,u,output[u][p1][p2]);   
                     neuron_O->SetLFEextended(k,u,extended[u][p1][p2]);  //prong layer axis

                  }                                               
               }
               for(int k =0; k < 8; k++){						
                  for(int u = 0; u < Npronged; u++){                                        
                     neuron_O->SetLFEaddition(k, u, addition[u][k]);
                  }
               }    
            } 
         } 
         for(int j=0; j<fNetwork.GetEntriesFast(); j++) {
            neuron = (YNeuron *)fNetwork.At(j);
            neuron->SetNewEvent();
         }          
         for(int ln = 0; ln < nLAYER; ln ++){
            if(chipID[t][ln]<0) continue;
            for(int j=0; j<fSCNetwork[chipID[t][ln]]->GetNetwork().GetEntriesFast(); j++) {
               neuron = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetNetwork().At(j); 
               neuron->SetNewEvent();
               //neuron->SetNeuronIndex(-1);                  
            }      
            for(int j = 0; j < 3; j++) {
               neuron_O = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetLastLayer().At((int)(j%3));
               neuron_O->SetLFENodeIndex((3*nLAYER)*t+3*ln+j);                              
               neuron_O->SetNeuronIndex(-1);   
               //neuron_O->SetNewEvent(); 
            }    
         }      
             
      } 
      InitSCNeuronDcdw();
      CalculateEventDcdw(Npronged);       
      for(int t = 0; t < Npronged; t++){                   
         for(int ln = 0; ln < nLAYER; ln ++){ 
            if(chipID[t][ln]<0) continue;
            
            for(int j = 0; j < 2; j++) {         
               neuron_I = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetFirstLayer().At(j); 
               neuron_I->InitSCNeuronDcdw();
               neuron_I->DumpSCNeuronDcdw(fSCNeuronDcdw);    
            } 
            for(int j = 0; j < 3; j++) {         
               neuron_O = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetLastLayer().At(j);
               neuron_O->InitSCNeuronDcdw();
               neuron_O->DumpSCNeuronDcdw(fSCNeuronDcdw);   
               neuron_O->SetNeuronIndex(-1);                                                     
            }  
         }
      }      
      for(int t = 0; t < Npronged; t++){  
       
         for(int ln = 0; ln < nLAYER; ln ++){
            if(chipID[t][ln]<0) continue;
            
            for (int j=0;j<fSCNetwork[chipID[t][ln]]->GetSynapses().GetEntriesFast();j++) {
               synapse = (YSynapse *) fSCNetwork[chipID[t][ln]]->GetSynapses().At(j);   
               synapse->SetDCDw(synapse->GetDcDw());          
            }      

            for (int j=0;j<fSCNetwork[chipID[t][ln]]->GetNetwork().GetEntriesFast();j++) {
               neuron = (YNeuron *) fSCNetwork[chipID[t][ln]]->GetNetwork().At(j);   
               neuron->SetDCDw(neuron->GetDcDw());  
            }
                        
            int il = ln;
            int is = staveIndex[t][ln];
            int ic = chipIndex[t][ln];
            int iID = chipID[t][ln];
            
            bool neuron_update  = false;
            bool synapse_update = false;          

            if(fUPDATESENSORS->GetBinContent(1+iID)==true) {
               neuron_update  = fSCNetwork[iID]->GetUpdateState();
               synapse_update = fSCNetwork[iID]->GetUpdateState();
            }         
             
            CalculateSCWeights(iID, fSCNetwork[iID], bufferArr[iID], neuron_update, synapse_update);
                               
         }  
      }
   }
   delete[]index;
}


////////////////////////////////////////////////////////////////////////////////
/// One step for the batch (stochastic) method.
/// DCDw should have been updated before calling this.

void YMultiLayerPerceptron::MLP_Batch(Double_t * buffer)
{
   //std::cout<<"YMultiLayerPerceptron::MLP_Batch"<<std::endl;
   fEta *= fEtaDecay;
   Int_t cnt = 0;
   TObjArrayIter *it = (TObjArrayIter *) fNetwork.MakeIterator();
   YNeuron *neuron = 0;
   // Step for all neurons
   while ((neuron = (YNeuron *) it->Next())) {
      if(fWUpdate[cnt]==true){
         buffer[cnt] = (-fEta) * (neuron->GetDCDw() + fDelta)
                    + fEpsilon * buffer[cnt];    
      } else {
         buffer[cnt] = 0;
      }
      neuron->SetWeight(neuron->GetWeight() + buffer[cnt++]);
   }
   delete it;
   it = (TObjArrayIter *) fSynapses.MakeIterator();
   YSynapse *synapse = 0;
   // Step for all synapses
   while ((synapse = (YSynapse *) it->Next())) {
      buffer[cnt] = (-fEta) * (synapse->GetDCDw() + fDelta)
                    + fEpsilon * buffer[cnt];
      synapse->SetWeight(synapse->GetWeight() + buffer[cnt++]);
   }
   delete it;
}

void YMultiLayerPerceptron::MLP_BatchArr(Double_t** bufferArr)
{
   std::cout<<"YMultiLayerPerceptron::MLP_BatchArr"<<std::endl;
   fEta *= fEtaDecay;
   Int_t j;
   Int_t cnt = 0;
   Int_t nentries=0;
   YNeuron *neuron = 0;
   YSynapse *synapse = 0;   
   // Step for all neurons  
   double paramA = 2.0;
   double paramB = 1.0;   
   double PIXAMP = 10.0; 
   for(int iID = 0; iID < nSensors; iID++){
      if(fSCNetwork[iID]->GetUpdateState()==false) continue;
      cnt = 0;
      nentries = fSCNetwork[iID]->GetNetwork().GetEntriesFast();
      for (j=0;j<nentries;j++) {
         neuron = (YNeuron *) fSCNetwork[iID]->GetNetwork().At(j);    
         if(j>=2){      
            bufferArr[iID][j] = (-fEta) * (neuron->GetDCDw() + fDelta)
                       + fEpsilon * bufferArr[iID][j];    
         } else {
            bufferArr[iID][j] = 0;
         }     
         double bufferArr_Weight = 1;//paramA/(1 + TMath::Exp(-(std::pow(bufferArr[iID][j]*1e+4/PIXAMP,2)))) - paramB;
         
         double w = neuron->GetWeight() + bufferArr_Weight*bufferArr[iID][j];     
         neuron->SetWeight(w);                                                              
      }     
      cnt = nentries;
      nentries = fSCNetwork[iID]->GetSynapses().GetEntriesFast();
      for (j=0;j<nentries;j++) {
         synapse = (YSynapse *) fSCNetwork[iID]->GetSynapses().At(j);       
         bufferArr[iID][cnt+j] = (-fEta) * (synapse->GetDCDw() + fDelta)
                       + fEpsilon * bufferArr[iID][cnt+j];
         double w = synapse->GetWeight() + bufferArr[iID][cnt+j];
         synapse->SetWeight(w);                 
      } 
      neuron = 0;
      synapse = 0;                    
   } 
}

void YMultiLayerPerceptron::MLP_BatchDetectorUnit(Double_t** bufferArr, Double_t*        bDU0, 
                                                                        Double_t**       bDU1,
                                                                        Double_t***      bDU2,
                                                                        Double_t****     bDU3,
                                                                        Double_t*****    bDU4,
                                                                        Double_t******   bDU5,
                                                                        Double_t*******  bDU6,
                                                                        Double_t******** bDU7)
{
   std::cout<<"YMultiLayerPerceptron::MLP_BatchDetectorUnit"<<std::endl;
   fEta *= fEtaDecay;
   Int_t j;
   Int_t cnt = 0;
   Int_t nentries=0;
   YNeuron *neuron = 0;
   YSynapse *synapse = 0;   
   YNeuron *neuronDU[8];
   neuronDU[0]=0;
   neuronDU[1]=0;
   neuronDU[2]=0;
   neuronDU[3]=0;
   neuronDU[4]=0;
   neuronDU[5]=0;               
   neuronDU[6]=0;               
   neuronDU[7]=0;               
             
   YSynapse *synapseDU[8];
   synapseDU[0]=0;
   synapseDU[1]=0;
   synapseDU[2]=0;
   synapseDU[3]=0;
   synapseDU[4]=0;
   synapseDU[5]=0;     
   synapseDU[6]=0;
   synapseDU[7]=0; 
       
   // Step for all neurons  
   double paramA = 2.0;
   double paramB = 1.0;   
   double PIXAMP = 10.0; 
   
   // det hs chipZ detIdx hb            
   std::cout<<"[ITS FULL]"<<std::endl;
   CalculateSCWeights(-1, fDetectorUnit->fSCNetwork, bDU0);             
   
   std::cout<<"[ITS IB]"<<std::endl;
   CalculateSCWeights(-1, fDetectorUnit->SubUnit[0]->fSCNetwork, bDU1[0]);          
   for(int hs=0; hs<nHalfStaveIB; hs++){    
      CalculateSCWeights(-1, fDetectorUnit->SubUnit[0]->SubUnit[hs]->fSCNetwork, bDU2[0][hs]);       
      for(int cz=0; cz<nChipsPerHicIB/nHalfStaveIB; cz++){
         CalculateSCWeights(-1, fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->fSCNetwork, bDU3[0][hs][cz]);
         for(int l=0; l<nLAYERIB; l++){ 
            int lIB = l;
            CalculateSCWeights(-1, fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->fSCNetwork, bDU4[0][hs][cz][lIB]);
            for(int hb=0; hb<2; hb++){  
               CalculateSCWeights(-1, fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->fSCNetwork, bDU5[0][hs][cz][lIB][hb]);            
               for(int st=0; st<NStaves[l]/2;st++){
                  CalculateSCWeights(-1, fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->fSCNetwork, bDU6[0][hs][cz][lIB][hb][st]);
                  for(int bst=0; bst<nHalfStaveIB; bst++){    
                     CalculateSCWeights(-1, fDetectorUnit->SubUnit[0]->SubUnit[hs]->SubUnit[cz]->SubUnit[lIB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork, bDU7[0][hs][cz][lIB][hb][st][bst]);
                  }
               }            
            }       
         }          
      }
   }

   std::cout<<"[ITS OB1]"<<std::endl;   
   CalculateSCWeights(-1, fDetectorUnit->SubUnit[1]->fSCNetwork, bDU1[1]);             
   for(int hs=0; hs<nHalfStaveOB1; hs++){  
      CalculateSCWeights(-1, fDetectorUnit->SubUnit[1]->SubUnit[hs]->fSCNetwork, bDU2[1][hs]);         
      for(int cz=0; cz<nChipsPerHicOB1/nHalfStaveOB1; cz++){
         CalculateSCWeights(-1, fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->fSCNetwork, bDU3[1][hs][cz]);
         for(int l=nLAYERIB; l<nLAYERIB + nLAYEROB1; l++){ 
            int lOB = l-nLAYERIB;
            CalculateSCWeights(-1, fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork, bDU4[1][hs][cz][lOB]);                        
            for(int hb=0; hb<2; hb++){  
               CalculateSCWeights(-1, fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork, bDU5[1][hs][cz][lOB][hb]);            
               for(int st=0; st<NStaves[l]/2;st++){
                  CalculateSCWeights(-1, fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork, bDU6[1][hs][cz][lOB][hb][st]);
                  for(int bst=0; bst<nHalfStaveOB1; bst++){     
                     CalculateSCWeights(-1, fDetectorUnit->SubUnit[1]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork, bDU7[1][hs][cz][lOB][hb][st][bst]);
                  }
               }               
            }       
         }          
      }
   }
   std::cout<<"[ITS OB2]"<<std::endl;   
   CalculateSCWeights(-1, fDetectorUnit->SubUnit[2]->fSCNetwork, bDU1[2]);                
   for(int hs=0; hs<nHalfStaveOB2; hs++){     
      CalculateSCWeights(-1, fDetectorUnit->SubUnit[2]->SubUnit[hs]->fSCNetwork, bDU2[2][hs]);            
      for(int cz=0; cz<nChipsPerHicOB2/nHalfStaveOB2; cz++){
         CalculateSCWeights(-1, fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->fSCNetwork, bDU3[2][hs][cz]);      
         for(int l=nLAYERIB + nLAYEROB1; l<nLAYER; l++){ 
            int lOB = l - (nLAYERIB + nLAYEROB1);
            CalculateSCWeights(-1, fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->fSCNetwork, bDU4[2][hs][cz][lOB]);
            for(int hb=0; hb<2; hb++){  
               CalculateSCWeights(-1, fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->fSCNetwork, bDU5[2][hs][cz][lOB][hb]);            
               for(int st=0; st<NStaves[l]/2;st++){
                  CalculateSCWeights(-1, fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->fSCNetwork, bDU6[2][hs][cz][lOB][hb][st]);
                  for(int bst=0; bst<nHalfStaveOB1; bst++){        
                     CalculateSCWeights(-1, fDetectorUnit->SubUnit[2]->SubUnit[hs]->SubUnit[cz]->SubUnit[lOB]->SubUnit[hb]->SubUnit[st]->SubUnit[bst]->fSCNetwork, bDU7[2][hs][cz][lOB][hb][st][bst]);
                  }
               }                
            }       
         }          
      }
   } 
   
   for(int iID = 0; iID < nSensors; iID++){
      int SensorID 	= iID;
                           
      int layer      = yGEOM->GetLayer(SensorID);
      if(layer<0 || layer >nLAYER) continue;
      //if(fSCNetwork[NetworkChips[iID]]->GetUpdateState()==false && kDetectorUnit[layer]==-1) continue;         
      int det           = layer < nLAYERIB ? 0 : (layer < nLAYERIB + nLAYEROB1 ? 1 : 2); 
      int detIdx        = layer < nLAYERIB ? layer : (layer < nLAYERIB + nLAYEROB1 ? layer - nLAYERIB : layer - (nLAYERIB + nLAYEROB1));
      int hb		= yGEOM->GetHalfBarrel(SensorID);  		//HalfBarrel
      int hs		= yGEOM->GetHalfStave(SensorID);  		//HalfStave               
      int stave		= yGEOM->GetStave(SensorID); 
      int staveIdx   	= stave%(NStaves[layer]/2);
      int ChipIdInStave = yGEOM->GetChipIdInStave(SensorID);
      int chipZ         = ChipIdInStave%(nHicPerStave[layer]*nChipsPerHic[layer]/NSubStave[layer]);  
      // det hs chipZ detIdx hb         
      //int Module	= yGEOM->GetModule(SensorID);     
      double dev_sign   = (double)fSCNetwork[iID]->GetMntDir();  
        
      if(kDetectorUnit[layer]<0){
         switch(kDetectorUnit[layer])
         {
            case -2 : 
            {                                           
               break;
            }         
            case -1 : 
            {
               short neuron_update  = 0;
               short synapse_update = 0;          

               if(fUPDATESENSORS->GetBinContent(1+iID)>0) {
                  neuron_update  = fSCNetwork[iID]->GetUpdateState();
                  synapse_update = fSCNetwork[iID]->GetUpdateState();
               }         
               
               CalculateSCWeights(iID, fSCNetwork[iID], bufferArr[iID], neuron_update, synapse_update);     

               break;
            }                                     
         }                             
      } else {
      
         cnt = 0;
         nentries = fSCNetwork[iID]->GetNetwork().GetEntriesFast();
         //std::cout<<"[MLP_BatchDetectorUnit] L"<<layer<<" "<<iID<<" "<<det<<" "<<hs<<" "<<chipZ<<" "<<detIdx<<" "<<hb<<" "<<dev_sign<<" Unit : "<<kDetectorUnit[layer]<<std::endl;         
         std::cout<<"[CHECKGRAD] L"<<layer<<" C"<<SensorID<<" DcDw(N) ";        
         for (j=0;j<nentries;j++) {
      
            bool neuron_update = false;
            switch(kDetectorUnit[layer])
            {
               case 0  : 
               {
                  neuron      = (YNeuron *) fSCNetwork[iID]->GetNetwork().At(j);    
                  neuronDU[0] = (YNeuron *) fDetectorUnit->fSCNetwork->GetNetwork().At(j);   
                  neuron_update = fDetectorUnit->fSCNetwork->GetUpdateState();               
                  if(neuron_update==true) neuron->SetWeight(neuronDU[0]->GetWeight()); 
                  break;    
               }                 
               case 1  : 
               {
                  neuron      = (YNeuron *) fSCNetwork[iID]->GetNetwork().At(j);                
                  neuronDU[1] = (YNeuron *) fDetectorUnit->SubUnit[det]->fSCNetwork->GetNetwork().At(j);  
                  neuron_update = fDetectorUnit->SubUnit[det]->fSCNetwork->GetUpdateState();                              
                  if(neuron_update==true) neuron->SetWeight(neuronDU[1]->GetWeight()); 
                  break; 
               }  
               case 2  : 
               {
                  neuron      = (YNeuron *) fSCNetwork[iID]->GetNetwork().At(j);                
                  neuronDU[2] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->fSCNetwork->GetNetwork().At(j);  
                  neuron_update = fDetectorUnit->SubUnit[det]->SubUnit[hs]->fSCNetwork->GetUpdateState();
                  if(neuron_update==true) neuron->SetWeight(neuronDU[2]->GetWeight());  
                  break; 
               }  
               case 3  : 
               {
                  neuron      = (YNeuron *) fSCNetwork[iID]->GetNetwork().At(j);                
                  neuronDU[3] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->fSCNetwork->GetNetwork().At(j);  
                  neuron_update = fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->fSCNetwork->GetUpdateState();               
                  if(neuron_update==true) neuron->SetWeight(neuronDU[3]->GetWeight()); 
                  break; 
               }  
               case 4  : 
               {
                  neuron      = (YNeuron *) fSCNetwork[iID]->GetNetwork().At(j);                
                  neuronDU[4] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->fSCNetwork->GetNetwork().At(j);  
                  neuron_update = fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->fSCNetwork->GetUpdateState();                              
                  if(neuron_update==true) neuron->SetWeight(neuronDU[4]->GetWeight()); 
                  break; 
               }  
               case 5  : 
               {
                  neuron      = (YNeuron *) fSCNetwork[iID]->GetNetwork().At(j);                
                  neuronDU[5] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->fSCNetwork->GetNetwork().At(j);  
                  neuron_update = fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->fSCNetwork->GetUpdateState();
                  if(neuron_update==true) neuron->SetWeight(neuronDU[5]->GetWeight()); 
                  break; 
               }     
               case 6  : 
               {
                  neuron      = (YNeuron *) fSCNetwork[iID]->GetNetwork().At(j);                
                  neuronDU[6] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->fSCNetwork->GetNetwork().At(j);  
                  neuron_update = fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->fSCNetwork->GetUpdateState();
                  if(neuron_update==true) neuron->SetWeight(neuronDU[6]->GetWeight()); 
                  break; 
               }                    
               case 7  : 
               {
                  neuron      = (YNeuron *) fSCNetwork[iID]->GetNetwork().At(j);                
                  neuronDU[7] = (YNeuron *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->SubUnit[hs]->fSCNetwork->GetNetwork().At(j);
                  neuron_update = fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->SubUnit[hs]->fSCNetwork->GetUpdateState();
                  if(neuron_update==true) neuron->SetWeight(neuronDU[7]->GetWeight()); 
                  break; 
               }            
               default : 
               {
                  neuron = (YNeuron *) fSCNetwork[iID]->GetNetwork().At(j);    
                  if(j>=2){      
                     bufferArr[iID][j] = (-fEta) * (neuron->GetDCDw() + fDelta)
                                + fEpsilon * bufferArr[iID][j];    
                  } else {
                     bufferArr[iID][j] = 0;
                  }             
                  double bufferArr_Weight = 1;//paramA/(1 + TMath::Exp(-(std::pow(bufferArr[iID][j]*1e+4/PIXAMP,2)))) - paramB;   
                  double w = neuron->GetWeight() + bufferArr_Weight*bufferArr[iID][j];     
                  neuron->SetWeight(w);                     
                  break;         
               }                                       
            }                                                                    
         }      
      
      
         cnt = nentries;
         nentries = fSCNetwork[iID]->GetSynapses().GetEntriesFast();
         std::cout<<"DcDw(S) ";      
         for (j=0;j<nentries;j++) {
            bool synapse_update = false;
            switch(kDetectorUnit[layer])
            {
               case 0  : 
               {
                  synapse      = (YSynapse *) fSCNetwork[iID]->GetSynapses().At(j);   
                  synapseDU[0] = (YSynapse *) fDetectorUnit->fSCNetwork->GetSynapses().At(j);  
                  synapse_update = fDetectorUnit->fSCNetwork->GetUpdateState();
                  if(synapse_update==true) synapse->SetWeight(synapseDU[0]->GetWeight());
                  break;    
               }                 
               case 1  : 
               {
                  synapse      = (YSynapse *) fSCNetwork[iID]->GetSynapses().At(j);   
                  synapseDU[1] = (YSynapse *) fDetectorUnit->SubUnit[det]->fSCNetwork->GetSynapses().At(j);  
                  synapse_update = fDetectorUnit->SubUnit[det]->fSCNetwork->GetUpdateState();
                  if(synapse_update==true) synapse->SetWeight(synapseDU[1]->GetWeight());
                  break; 
               }  
               case 2  : 
               {
                  synapse      = (YSynapse *) fSCNetwork[iID]->GetSynapses().At(j);   
                  synapseDU[2] = (YSynapse *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->fSCNetwork->GetSynapses().At(j); 
                  synapse_update = fDetectorUnit->SubUnit[det]->SubUnit[hs]->fSCNetwork->GetUpdateState();
                  if(synapse_update==true) synapse->SetWeight(synapseDU[2]->GetWeight());
                  break; 
               }  
               case 3  : 
               {
                  synapse      = (YSynapse *) fSCNetwork[iID]->GetSynapses().At(j);   
                  synapseDU[3] = (YSynapse *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->fSCNetwork->GetSynapses().At(j);  
                  synapse_update = fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->fSCNetwork->GetUpdateState();
                  if(synapse_update==true) synapse->SetWeight(synapseDU[3]->GetWeight());
                  break; 
               }  
               case 4  : 
               {
                  synapse      = (YSynapse *) fSCNetwork[iID]->GetSynapses().At(j);   
                  synapseDU[4] = (YSynapse *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->fSCNetwork->GetSynapses().At(j);  
                  synapse_update = fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->fSCNetwork->GetUpdateState();
                  if(synapse_update==true) synapse->SetWeight(synapseDU[4]->GetWeight());
                  break; 
               }  
               case 5  : 
               {
                  synapse      = (YSynapse *) fSCNetwork[iID]->GetSynapses().At(j);   
                  synapseDU[5] = (YSynapse *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->fSCNetwork->GetSynapses().At(j);  
                  synapse_update = fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->fSCNetwork->GetUpdateState();
                  if(synapse_update==true) synapse->SetWeight(synapseDU[5]->GetWeight());
                  break; 
               }  
               case 6  : 
               {
                  synapse      = (YSynapse *) fSCNetwork[iID]->GetSynapses().At(j);   
                  synapseDU[6] = (YSynapse *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->fSCNetwork->GetSynapses().At(j);  
                  synapse_update = fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->fSCNetwork->GetUpdateState();
                  if(synapse_update==true) synapse->SetWeight(synapseDU[6]->GetWeight());
                  break; 
               }                    
               case 7  : 
               {
                  synapse      = (YSynapse *) fSCNetwork[iID]->GetSynapses().At(j);   
                  synapseDU[7] = (YSynapse *) fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->SubUnit[hs]->fSCNetwork->GetSynapses().At(j);  
                  synapse_update = fDetectorUnit->SubUnit[det]->SubUnit[hs]->SubUnit[chipZ]->SubUnit[detIdx]->SubUnit[hb]->SubUnit[staveIdx]->SubUnit[hs]->fSCNetwork->GetUpdateState();               
                  if(synapse_update==true) synapse->SetWeight(synapseDU[7]->GetWeight());
                  break; 
               }              
               default : 
               {
                  synapse = (YSynapse *) fSCNetwork[iID]->GetSynapses().At(j);   
                  bufferArr[iID][cnt+j] = (-fEta) * (synapse->GetDCDw() + fDelta)
                                + fEpsilon * bufferArr[iID][cnt+j];
                  double w = synapse->GetWeight() + bufferArr[iID][cnt+j];
                  synapse->SetWeight(w);                      
                  break;         
               }                                       
            }                          
         }
         std::cout<<std::endl;              
      }
      
 

      neuron = 0;
      synapse = 0;                    
      neuronDU[0]=0;
      neuronDU[1]=0;
      neuronDU[2]=0;
      neuronDU[3]=0;
      neuronDU[4]=0;
      neuronDU[5]=0;               
      neuronDU[6]=0;
      neuronDU[7]=0;
            
      synapseDU[0]=0;
      synapseDU[1]=0;
      synapseDU[2]=0;
      synapseDU[3]=0;
      synapseDU[4]=0;
      synapseDU[5]=0;   
      synapseDU[6]=0;  
      synapseDU[7]=0;              
   } 
   std::cout<<"YMultiLayerPerceptron::MLP_BatchDetectorUnit Complete"<<std::endl;   
}

#ifdef MONITORSENSORUNITprofile
void YMultiLayerPerceptron::MLP_OffsetTuneByMean()
{
   std::cout<<"YMultiLayerPerceptron::MLP_OffsetTuneByMean"<<std::endl;
   fEta *= fEtaDecay;
   Int_t j;
   Int_t nentries=0;
   Int_t sentries=0;   
   YNeuron  *neuron  = 0;
   YSynapse *synapse = 0;      
   for(int iID = 0; iID < nSensors; iID++){
      if(fSCNetwork[iID]->GetUpdateState()==false ||fWUpdatebyMean[iID]==false) continue;

#ifdef MONITORONLYUPDATES
      if(fUPDATESENSORS->GetBinContent(1+iID)!=1) continue;
#endif
      //if(iID!=122) continue;
      if(yGEOM->GetLayer(iID)>=3) continue;
      
      int nhits_ds1_s1 = fSCNetwork[iID]->GetEntriesProfile(fSCNetwork[iID]->hpds1_s1);
      int nhits_ds1_s2 = fSCNetwork[iID]->GetEntriesProfile(fSCNetwork[iID]->hpds1_s2);
      int nhits_ds2_s1 = fSCNetwork[iID]->GetEntriesProfile(fSCNetwork[iID]->hpds2_s1);
      int nhits_ds2_s2 = fSCNetwork[iID]->GetEntriesProfile(fSCNetwork[iID]->hpds2_s2);
      
      std::cout<<"(nhits) ds1_s1 : "<<nhits_ds1_s1<<
                        " ds1_s2 : "<<nhits_ds1_s2<<
                        " ds2_s1 : "<<nhits_ds2_s1<<
                        " ds2_s2 : "<<nhits_ds2_s2<<std::endl;
                        
      std::cout<<"(Chi2/NDF) ds1_s1 : "<<fOffsetTuning[iID]->fds1_s1->GetChisquare()/fOffsetTuning[iID]->fds1_s1->GetNDF()<<
                           " ds1_s2 : "<<fOffsetTuning[iID]->fds1_s2->GetChisquare()/fOffsetTuning[iID]->fds1_s2->GetNDF()<<
                           " ds2_s1 : "<<fOffsetTuning[iID]->fds2_s1->GetChisquare()/fOffsetTuning[iID]->fds2_s1->GetNDF()<<
                           " ds2_s2 : "<<fOffsetTuning[iID]->fds2_s2->GetChisquare()/fOffsetTuning[iID]->fds2_s2->GetNDF()<<std::endl;

      std::cout<<"(Fitpar) ds1_s1 : "<<fOffsetTuning[iID]->fds1_s1->GetParameter(0)<<" "<<fOffsetTuning[iID]->fds1_s1->GetParameter(1)<<
                         " ds1_s2 : "<<fOffsetTuning[iID]->fds1_s2->GetParameter(0)<<" "<<fOffsetTuning[iID]->fds1_s2->GetParameter(1)<<
                         " ds2_s1 : "<<fOffsetTuning[iID]->fds2_s1->GetParameter(0)<<" "<<fOffsetTuning[iID]->fds2_s1->GetParameter(1)<<
                         " ds2_s2 : "<<fOffsetTuning[iID]->fds2_s2->GetParameter(0)<<" "<<fOffsetTuning[iID]->fds2_s2->GetParameter(1)<<std::endl;
      
      nentries = fSCNetwork[iID]->GetNetwork().GetEntriesFast();
      sentries = fSCNetwork[iID]->GetSynapses().GetEntriesFast();
      double wSynapse[6];
      double bNeuron[5];
   
      std::cout<<"Epoch"<<fCurrentEpoch<<" OffsetTune[0] N ";
      for (j=0;j<nentries;j++) { 
         neuron = (YNeuron *) fSCNetwork[iID]->GetNetwork().At(j);  
         std::cout << neuron->GetWeight() <<" ";  
         bNeuron[j] = neuron->GetWeight();
      }          
      std::cout<<"S ";         
      for (j=0;j<sentries;j++) { 
         synapse = (YSynapse *) fSCNetwork[iID]->GetSynapses().At(j);
         std::cout << synapse->GetWeight() <<" ";  
         wSynapse[j] = synapse->GetWeight();                                                                                      
      } 
      std::cout<<std::endl;
      
      double dcdwARR[] = {0, 0, 0, 0, 0,		// bias
                          0, 0, 0, 0, 0, 0,             // synapse
                          0, 0, 0, 0, 0, 0};		
      
      for (j=0;j<nentries;j++) {
         neuron = (YNeuron *) fSCNetwork[iID]->GetNetwork().At(j);    
         double ds_mean = 0;
         if(j==2){      
            if(fSCNetwork[iID]->GetEntriesProfile(fSCNetwork[iID]->hpds1_s1)>=50){ 
               if(fSCNetwork[iID]->GetEntriesProfile(fSCNetwork[iID]->hpds1_s2)>=50){
                  ds_mean = 0.5*(fOffsetTuning[iID]->fds1_s1->GetParameter(0) + fOffsetTuning[iID]->fds1_s2->GetParameter(0)); // ok
               } else {
                  ds_mean = fOffsetTuning[iID]->fds1_s1->GetParameter(0); // ok
               }
            } else {
               if(fSCNetwork[iID]->GetEntriesProfile(fSCNetwork[iID]->hpds1_s2)>=50){
                  ds_mean = fOffsetTuning[iID]->fds1_s2->GetParameter(0); // ok
               } else {
                  ds_mean = 0; //ok
               }    
            }
         } else if(j==3) {
            if(fSCNetwork[iID]->GetEntriesProfile(fSCNetwork[iID]->hpds2_s1)>=50){ 
               if(fSCNetwork[iID]->GetEntriesProfile(fSCNetwork[iID]->hpds2_s2)>=50){
                  ds_mean = 0.5*(fOffsetTuning[iID]->fds2_s1->GetParameter(0) + fOffsetTuning[iID]->fds2_s2->GetParameter(0)); //ok
               } else {
                  ds_mean = fOffsetTuning[iID]->fds2_s1->GetParameter(0); //ok
               }
            } else {
               if(fSCNetwork[iID]->GetEntriesProfile(fSCNetwork[iID]->hpds2_s2)>=50){
                  ds_mean = fOffsetTuning[iID]->fds2_s2->GetParameter(0); // ok
               } else {
                  ds_mean = 0; //ok
               }    
            }
         } else if(j==4) {
            if(fSCNetwork[iID]->GetEntriesProfile(fSCNetwork[iID]->hpds1_s1)>=Min_Cluster_by_Sensor&&
               fOffsetTuning[iID]->fds1_s1->GetChisquare()/fOffsetTuning[iID]->fds1_s1->GetNDF()<=5.0) ds_mean = 0.5*fOffsetTuning[iID]->fds1_s1->GetParameter(1);
         }
         dcdwARR[j] = 0.25*ds_mean;
         //double w = neuron->GetWeight() + ds_mean;     
         //neuron->SetWeight(w);                                                              
      }     

      for (j=0;j<sentries;j++) {
         synapse = (YSynapse *) fSCNetwork[iID]->GetSynapses().At(j);       
         double ds_slope = 0;         
         if(j==0){ //w11
            if(fSCNetwork[iID]->GetEntriesProfile(fSCNetwork[iID]->hpds1_s1)>=Min_Cluster_by_Sensor&&
               fOffsetTuning[iID]->fds1_s1->GetChisquare()/fOffsetTuning[iID]->fds1_s1->GetNDF()<=5.0) ds_slope = 0.5*fOffsetTuning[iID]->fds1_s1->GetParameter(1);
         } else if(j==1){ //w21
            if(fSCNetwork[iID]->GetEntriesProfile(fSCNetwork[iID]->hpds1_s2)>=Min_Cluster_by_Sensor&&
               fOffsetTuning[iID]->fds1_s2->GetChisquare()/fOffsetTuning[iID]->fds1_s2->GetNDF()<=5.0) ds_slope = 0.5*fOffsetTuning[iID]->fds1_s2->GetParameter(1);
         } else if(j==2){ //w12
            if(fSCNetwork[iID]->GetEntriesProfile(fSCNetwork[iID]->hpds2_s1)>=Min_Cluster_by_Sensor&&
               fOffsetTuning[iID]->fds2_s1->GetChisquare()/fOffsetTuning[iID]->fds2_s1->GetNDF()<=5.0) ds_slope = 0.5*fOffsetTuning[iID]->fds2_s1->GetParameter(1);
         } else if(j==3){ //w22
            if(fSCNetwork[iID]->GetEntriesProfile(fSCNetwork[iID]->hpds2_s2)>=Min_Cluster_by_Sensor&&
               fOffsetTuning[iID]->fds2_s2->GetChisquare()/fOffsetTuning[iID]->fds2_s2->GetNDF()<=5.0) ds_slope = 0.5*fOffsetTuning[iID]->fds2_s2->GetParameter(1);         
         }
         dcdwARR[nentries+j] = 0.25*ds_slope;
         //double w = synapse->GetWeight() + 0.5*ds_slope;
         //synapse->SetWeight(w);                 
      }       

      double dir = fSCNetwork[iID]->GetMntDir();
      double P1 = dir;
      double P2 = dir;
      double P3 = -1;

      double sensor_d1 = 2.991253;
      double sensor_d2 = 1.373568;
      double sensor_t2 = 0;//4.16e-4/sensor_d2;
      double sensor_y2 = 0.058128/sensor_d2;
      TMatrixD matD1(3,3);
      matD1[0] = {1/sensor_d1, 0, 0};            
      matD1[1] = {0, 1/sensor_d2, 0};            
      matD1[2] = {0, 0, 1};  
      
      TMatrixD matInvD1(3,3);
      matInvD1 = matD1;
      matInvD1.Invert();
 
      TMatrixD matD2(3,1);
      matD2[0] = {0};            
      matD2[1] = {dir*(sensor_t2 - sensor_y2)};                  
      matD2[2] = {0};  

      double Rxx, Rxy, Rxz, Tdx, Ryx, Ryy, Ryz, Tdy, Rzx, Rzy, Rzz, Tdz;
      yGEOM->GetL2GComponents(iID, Rxx, Rxy, Rxz, Tdx, Ryx, Ryy, Ryz, Tdy, Rzx, Rzy, Rzz, Tdz);

      TMatrixD matRgeom(3,3);
      matRgeom[0] = {Rxx, Rxy, Rxz};
      matRgeom[1] = {Ryx, Ryy, Ryz};
      matRgeom[2] = {Rzx, Rzy, Rzz};     

      TMatrixD matTgeom(3,1);   
      matTgeom[0] = {Tdx};
      matTgeom[1] = {Tdy};
      matTgeom[2] = {Tdz}; 

      TMatrixD matP(3,3);
      matP[0] = {P2,  0,  0};
      matP[1] = { 0, P3,  0};
      matP[2] = { 0,  0, P1};       
      
      TMatrixD matO(3,3);
      matO[0] = { 0,  1,  0};
      matO[1] = { 0,  0,  1};
      matO[2] = { 1,  0,  0}; 

      TMatrixD matRgeomPO(3,3);
      TMatrixD matInvRgeomPO(3,3);
      matRgeomPO = matRgeom * matP * matO;
      matInvRgeomPO = matRgeomPO;
      matInvRgeomPO.T();

      double tpar[17];
      for (int ipar=0; ipar<17; ipar++) { 
         tpar[ipar] = std::abs(fOffsetTuning[iID]->ftunePAR[ipar])<1.0e-4 ? 1.0e+10 : fOffsetTuning[iID]->ftunePAR[ipar];
      }
      // tpar    : p1A(0) p1B(1) p1C(2) p1D(3) q1A(4) q1B(5) q1C(6) q1D(7) q1E(8) q1F(9) q1G(10) q1H(11) r1A(12) r1B(13) r1C(14) r1D(15) r1E(16)
      
      // dcdwARR : 0(0) 0(1) db1(2) db2(3) db3(4) w11(5) w21(6) w12(7) w22(8) w13(9) w23(10)
      
      double Ralpha  = fSCNetwork[iID]->aR(0);
      double Rbeta   = fSCNetwork[iID]->aR(1);
      double Rgamma  = fSCNetwork[iID]->aR(2);
      double T1      = fSCNetwork[iID]->aT(0);
      double T2      = fSCNetwork[iID]->aT(1);
      double T3      = fSCNetwork[iID]->aT(2);  

      double dRalpha  = (dcdwARR[6]/tpar[0]/2.0) + (dcdwARR[7]/tpar[2]/2.0) + (dcdwARR[2]/tpar[4]/3.0) + (dcdwARR[3]/tpar[7]/4.0) + (dcdwARR[4]/tpar[12]/4.0);
      double dRbeta   = (dcdwARR[6]/tpar[1]/2.0) + (dcdwARR[7]/tpar[3]/2.0) + (dcdwARR[2]/tpar[5]/3.0) + (dcdwARR[3]/tpar[8]/4.0) + (dcdwARR[4]/tpar[13]/4.0);
      double dRgamma  = (dcdwARR[3]/tpar[9]/4.0) + (dcdwARR[4]/tpar[14]/4.0);
      double dT1      = (dcdwARR[3]/tpar[10]/8.0) + (dcdwARR[4]/tpar[15]/8.0);
      double dT2      = (dcdwARR[3]/tpar[11]/8.0) + (dcdwARR[4]/tpar[16]/8.0);
      double dT3      = (dcdwARR[2]/tpar[6]/3.0);

      std::cout<<"Align Parameters  : "<<Ralpha<<" "<<Rbeta<<" "<<Rgamma<<" "<<T1<<" "<<T2<<" "<<T3<<std::endl;
      std::cout<<"dAlign Parameters : "<<dRalpha<<" "<<dRbeta<<" "<<dRgamma<<" "<<dT1<<" "<<dT2<<" "<<dT3<<std::endl;
      
      Ralpha  += dRalpha;
      Rbeta   += dRbeta;
      Rgamma  += dRgamma;
      T1      += dT1;
      T2      += dT2;
      T3      += dT3;  

      TMatrixD matRalpha(3,3);         
      matRalpha[0] = {1, 0, 0};
      matRalpha[1] = {0, +std::cos(Ralpha), -std::sin(Ralpha)};
      matRalpha[2] = {0, +std::sin(Ralpha), +std::cos(Ralpha)};   
        
      TMatrixD matRbeta(3,3);
      matRbeta[0] = {+std::cos(Rbeta), 0, +std::sin(Rbeta)};
      matRbeta[1] = {0, 1, 0};
      matRbeta[2] = {-std::sin(Rbeta), 0, +std::cos(Rbeta)};       
      
      TMatrixD matRgamma(3,3);        
      matRgamma[0] = {+std::cos(Rgamma), -std::sin(Rgamma), 0};
      matRgamma[1] = {+std::sin(Rgamma), +std::cos(Rgamma), 0};
      matRgamma[2] = {0, 0, 1};
      
      TMatrixD matAR(3,3);
      matAR = matRalpha * matRbeta * matRgamma;                                      
                                   
      TMatrixD matAT(3,1);
      matAT[0] = {T1};
      matAT[1] = {T2};
      matAT[2] = {T3};
      
      std::cout<<"dcdwArr : ";
      for (j=0;j<nentries+sentries;j++) {
         std::cout<<dcdwARR[j]<<" ";
      }
      std::cout<<std::endl;  
      TMatrixD matW(3,3);
      matW = matInvRgeomPO * matAR * matRgeomPO * matInvD1;
      std::cout<<"matW "<<std::endl;
      matW.Print();
      
      TMatrixD matB(3,1);  
      matB = matInvD1 * matD2 - matInvRgeomPO * matAR * matRgeomPO * matInvD1 * matD2
           + matInvRgeomPO * matAT + matInvRgeomPO * matAR * matTgeom - matInvRgeomPO * matTgeom;
      std::cout<<"matB "<<std::endl;
      matB.Print();

      wSynapse[0] = matW[0][0] - sensor_d1;
      wSynapse[1] = matW[0][1];
      
      wSynapse[2] = matW[1][0];
      wSynapse[3] = matW[1][1] - sensor_d2;
      
      wSynapse[4] = matW[2][0];      
      wSynapse[5] = matW[2][1]; 
                             
      bNeuron[0] = 0;
      bNeuron[1] = 0;
      
      bNeuron[2] = matB[0][0];
      bNeuron[3] = matB[1][0];
      bNeuron[4] = matB[2][0];         

      for (int j=0;j<nentries;j++) {
         neuron = (YNeuron *) fSCNetwork[iID]->GetNetwork().At(j);  
         neuron->SetWeight(bNeuron[j]);
      }  

      for (int j=0;j<sentries;j++) { 
         synapse = (YSynapse *) fSCNetwork[iID]->GetSynapses().At(j); 
         synapse->SetWeight(wSynapse[j]);
      }  
      
      fSCNetwork[iID]->SetaR(0,Ralpha);
      fSCNetwork[iID]->SetaR(1,Rbeta);      
      fSCNetwork[iID]->SetaR(2,Rgamma);

      fSCNetwork[iID]->SetaT(0,T1);
      fSCNetwork[iID]->SetaT(1,T2);
      fSCNetwork[iID]->SetaT(2,T3); 

      std::cout<<"Epoch"<<fCurrentEpoch<<" OffsetTune[1] N ";
      for (j=0;j<nentries;j++) { 
         neuron = (YNeuron *) fSCNetwork[iID]->GetNetwork().At(j);  
         std::cout << neuron->GetWeight() <<" ";  
         bNeuron[j] = neuron->GetWeight();
      }          
      std::cout<<"S ";         
      for (j=0;j<sentries;j++) { 
         synapse = (YSynapse *) fSCNetwork[iID]->GetSynapses().At(j);
         std::cout << synapse->GetWeight() <<" ";  
         wSynapse[j] = synapse->GetWeight();                                                                                      
      } 
      std::cout<<std::endl;

      neuron  = 0;
      synapse = 0;      
   } 
}
#endif

void YMultiLayerPerceptron::CalculateSCWeights(int sensorID, YSensorCorrection* scn, double* dcdwARR, bool nUpdate = true, bool sUpdate = true){

   int HalfBarrel	= yGEOM->GetHalfBarrel(sensorID);  
   int Layer  		= yGEOM->GetLayer(sensorID);
   int HalfStave 	= yGEOM->GetHalfStave(sensorID);        
   int Stave	 	= yGEOM->GetStave(sensorID); 
   int Module	 	= yGEOM->GetModule(sensorID);  
   int ChipIdInModule   = yGEOM->GetChipIdInModule(sensorID); 
  
   double Update_aR[3] = {0.0, 0.0, 1.0};   
   double Update_aT[3] = {1.0, 1.0, 1.0};   

/*
   if(Layer==0){
      Update_aR[0] = 1.0;
      Update_aR[1] = 1.0;
      Update_aR[2] = 1.0;
      Update_aT[0] = 1.0;
      Update_aT[1] = 1.0;
      Update_aT[2] = 1.0;
   }
*/
   
   double paramA = 2.0;
   double paramB = 1.0;   
   double PIXAMP = 10.0; 
   double wSynapse[6];
   double bNeuron[5];
   
   double AlignParameters[2][6] = { {0, 0, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 0} };
   
   double NetworkParameters[2][11] = { {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} };
   
   int nentries = scn->GetNetwork().GetEntriesFast();
   int sentries = scn->GetSynapses().GetEntriesFast();  
   int cnt = nentries;
      
   YNeuron* neuron;
   YSynapse* synapse;

   if(sensorID>=0){  

      for (int j=0;j<nentries;j++) {
         neuron = (YNeuron *) scn->GetNetwork().At(j);  
         if(j>=2){      
            dcdwARR[j] = (-fEta) * (neuron->GetDCDw() + fDelta) + fEpsilon * dcdwARR[j];    
         } else {
            dcdwARR[j] = 0;
         }      
         bNeuron[j] = neuron->GetWeight();
      }         
      for (int j=0;j<sentries;j++) { 
         synapse = (YSynapse *) scn->GetSynapses().At(j);  
         dcdwARR[cnt+j] = (-fEta) * (synapse->GetDCDw() + fDelta) + fEpsilon * dcdwARR[cnt+j];    
         wSynapse[j] = synapse->GetWeight();
      }   
      double dir = scn->GetMntDir();
      double P1 = dir;
      double P2 = dir;
      double P3 = -1;

      double sensor_d1 = 2.991253;
      double sensor_d2 = 1.373568;
      double sensor_t2 = 0;//4.16e-4/sensor_d2;
      double sensor_y2 = 0.058128/sensor_d2;
      TMatrixD matD1(3,3);
      matD1[0] = {1/sensor_d1, 0, 0};            
      matD1[1] = {0, 1/sensor_d2, 0};            
      matD1[2] = {0, 0, 1};  
      
      TMatrixD matInvD1(3,3);
      matInvD1 = matD1;
      matInvD1.Invert();
 
      TMatrixD matD2(3,1);
      matD2[0] = {0};            
      matD2[1] = {dir*(sensor_t2 - sensor_y2)};                  
      matD2[2] = {0};  

      double Ralpha  = scn->aR(0);
      double Rbeta   = scn->aR(1);
      double Rgamma  = scn->aR(2);
      double T1      = scn->aT(0);
      double T2      = scn->aT(1);
      double T3      = scn->aT(2);  
            
      AlignParameters[0][0] = Ralpha;
      AlignParameters[0][1] = Rbeta;
      AlignParameters[0][2] = Rgamma;
      AlignParameters[0][3] = T1;
      AlignParameters[0][4] = T2;
      AlignParameters[0][5] = T3;
            
      NetworkParameters[0][0]  = bNeuron[0];
      NetworkParameters[0][1]  = bNeuron[1];
      NetworkParameters[0][2]  = bNeuron[2];
      NetworkParameters[0][3]  = bNeuron[3];    
      NetworkParameters[0][4]  = bNeuron[4];
      NetworkParameters[0][5]  = wSynapse[0];
      NetworkParameters[0][6]  = wSynapse[1];
      NetworkParameters[0][7]  = wSynapse[2];  
      NetworkParameters[0][8]  = wSynapse[3];
      NetworkParameters[0][9]  = wSynapse[4];
      NetworkParameters[0][10] = wSynapse[5];
                    
      double Rxx, Rxy, Rxz, Tdx, Ryx, Ryy, Ryz, Tdy, Rzx, Rzy, Rzz, Tdz;
      yGEOM->GetL2GComponents(sensorID, Rxx, Rxy, Rxz, Tdx, Ryx, Ryy, Ryz, Tdy, Rzx, Rzy, Rzz, Tdz);

      TMatrixD matRgeom(3,3);
      matRgeom[0] = {Rxx, Rxy, Rxz};
      matRgeom[1] = {Ryx, Ryy, Ryz};
      matRgeom[2] = {Rzx, Rzy, Rzz};     

      TMatrixD matTgeom(3,1);   
      matTgeom[0] = {Tdx};
      matTgeom[1] = {Tdy};
      matTgeom[2] = {Tdz}; 

      TMatrixD matP(3,3);
      matP[0] = {P2,  0,  0};
      matP[1] = { 0, P3,  0};
      matP[2] = { 0,  0, P1};       
      
      TMatrixD matO(3,3);
      matO[0] = { 0,  1,  0};
      matO[1] = { 0,  0,  1};
      matO[2] = { 1,  0,  0}; 

      TMatrixD matRgeomPO(3,3);
      TMatrixD matInvRgeomPO(3,3);
      matRgeomPO = matRgeom * matP * matO;
      matInvRgeomPO = matRgeomPO;
      matInvRgeomPO.T();
      
      TMatrixD matDwDRalpha(3,3);
      TMatrixD matDbDRalpha(3,1);
      matDwDRalpha[0] = {0, 0, 0};
      matDwDRalpha[1] = {std::cos(Ralpha)*std::sin(Rbeta)*std::cos(Rgamma) - std::sin(Ralpha)*std::sin(Rgamma), -std::cos(Ralpha)*std::sin(Rbeta)*std::sin(Rgamma) - std::sin(Ralpha)*std::cos(Rgamma), 0};
      matDwDRalpha[2] = {std::sin(Ralpha)*std::sin(Rbeta)*std::cos(Rgamma) + std::cos(Ralpha)*std::sin(Rgamma), -std::sin(Ralpha)*std::sin(Rbeta)*std::sin(Rgamma) + std::cos(Ralpha)*std::cos(Rgamma), 0};
      TMatrixD matRGgeomPO_DwDRalpha(3,3);
      matRGgeomPO_DwDRalpha = matInvRgeomPO * matDwDRalpha * matRgeomPO * matInvD1;
      TMatrixD matRGgeomPO_DbDRalpha(3,1);
      matRGgeomPO_DbDRalpha = matInvRgeomPO * matDwDRalpha * matTgeom - matInvRgeomPO * matDwDRalpha * matRgeomPO * matInvD1 * matD2;
      double arrDwDRalpha[6] = {
                                matRGgeomPO_DwDRalpha[0][0], matRGgeomPO_DwDRalpha[0][1],
                                matRGgeomPO_DwDRalpha[1][0], matRGgeomPO_DwDRalpha[1][1],
                                matRGgeomPO_DwDRalpha[2][0], matRGgeomPO_DwDRalpha[2][1]                                  
                               };
      double arrDbDRalpha[5] = {
                                0,
                                0,   
                                matRGgeomPO_DbDRalpha[0][0],
                                matRGgeomPO_DbDRalpha[1][0],
                                matRGgeomPO_DbDRalpha[2][0]
                               };                     
      
      TMatrixD matDwDRbeta(3,3);
      TMatrixD matDbDRbeta(3,1);
      matDwDRbeta[0] = {-std::sin(Rbeta)*std::cos(Rgamma), std::sin(Rbeta)*std::sin(Rgamma), 0};
      matDwDRbeta[1] = {std::sin(Ralpha)*std::cos(Rbeta)*std::cos(Rgamma), -std::sin(Ralpha)*std::cos(Rbeta)*std::sin(Rgamma), 0};
      matDwDRbeta[2] = {-std::cos(Ralpha)*std::cos(Rbeta)*std::cos(Rgamma), std::cos(Ralpha)*std::cos(Rbeta)*std::sin(Rgamma), 0}; 
      TMatrixD matRGgeomPO_DwDRbeta(3,3);
      matRGgeomPO_DwDRbeta = matInvRgeomPO * matDwDRbeta * matRgeomPO * matInvD1;
      TMatrixD matRGgeomPO_DbDRbeta(3,1);
      matRGgeomPO_DbDRbeta = matInvRgeomPO * matDwDRbeta * matTgeom - matInvRgeomPO * matDwDRbeta * matRgeomPO * matInvD1 * matD2;
      double arrDwDRbeta[6] = {
                                matRGgeomPO_DwDRbeta[0][0], matRGgeomPO_DwDRbeta[0][1],
                                matRGgeomPO_DwDRbeta[1][0], matRGgeomPO_DwDRbeta[1][1],
                                matRGgeomPO_DwDRbeta[2][0], matRGgeomPO_DwDRbeta[2][1]                                  
                               };
      double arrDbDRbeta[5] = {   
                                0,
                                0,         
                                matRGgeomPO_DbDRbeta[0][0],
                                matRGgeomPO_DbDRbeta[1][0],
                                matRGgeomPO_DbDRbeta[2][0]
                               };  

      // gamma rotation                                                              
      TMatrixD matDwDRgamma(3,3);
      TMatrixD matDbDRgamma(3,1);
      matDwDRgamma[0] = {-std::cos(Rbeta)*std::sin(Rgamma), -std::cos(Rbeta)*std::cos(Rgamma), 0};
      matDwDRgamma[1] = {-std::sin(Ralpha)*std::sin(Rbeta)*std::sin(Rgamma) + std::cos(Ralpha)*std::cos(Rgamma), -std::sin(Ralpha)*std::sin(Rbeta)*std::cos(Rgamma) - std::cos(Ralpha)*std::sin(Rgamma), 0};
      matDwDRgamma[2] = {std::cos(Ralpha)*std::sin(Rbeta)*std::sin(Rgamma) + std::sin(Ralpha)*std::cos(Rgamma), std::cos(Ralpha)*std::sin(Rbeta)*std::cos(Rgamma) - std::sin(Ralpha)*std::sin(Rgamma), 0};

      //      double Rxx, Rxy, Rxz, Tdx, Ryx, Ryy, Ryz, Tdy, Rzx, Rzy, Rzz, Tdz;      
      matDbDRgamma[0] = { Tdx*std::sin(Rgamma) - Tdy*std::cos(Rgamma) };
      matDbDRgamma[1] = { Tdx*std::cos(Rgamma) + Tdy*std::sin(Rgamma) }; 
      matDbDRgamma[2] = { 0 };      
            
      TMatrixD matRGgeomPO_DwDRgamma(3,3);
      matRGgeomPO_DwDRgamma = matInvRgeomPO * matDwDRgamma * matRgeomPO * matInvD1;
      TMatrixD matRGgeomPO_DbDRgamma(3,1);
      matRGgeomPO_DbDRgamma = matInvRgeomPO * matDwDRgamma * matTgeom - matInvRgeomPO * matDwDRgamma * matRgeomPO * matInvD1 * matD2 + matInvRgeomPO * matDbDRgamma;  
      double arrDwDRgamma[6] = {
                                matRGgeomPO_DwDRgamma[0][0], matRGgeomPO_DwDRgamma[0][1],
                                matRGgeomPO_DwDRgamma[1][0], matRGgeomPO_DwDRgamma[1][1],
                                matRGgeomPO_DwDRgamma[2][0], matRGgeomPO_DwDRgamma[2][1]                                  
                               };
      double arrDbDRgamma[5] = {   
                                0,
                                0,         
                                matRGgeomPO_DbDRgamma[0][0],
                                matRGgeomPO_DbDRgamma[1][0],
                                matRGgeomPO_DbDRgamma[2][0]
                               }; 

      TMatrixD matDbDT1(3,1);
      matDbDT1[0] = {1};
      matDbDT1[1] = {0};
      matDbDT1[2] = {0};   
      TMatrixD matRGgeomPO_DbDT1(3,1);
      matRGgeomPO_DbDT1 = matInvRgeomPO * matDbDT1;
      double arrDbDT1[5] = {
                            0,
                            0,         
                            matRGgeomPO_DbDT1[0][0],
                            matRGgeomPO_DbDT1[1][0],
                            matRGgeomPO_DbDT1[2][0]                         
                           };
                        
      TMatrixD matDbDT2(3,1);
      matDbDT2[0] = {0};
      matDbDT2[1] = {1};
      matDbDT2[2] = {0}; 
      TMatrixD matRGgeomPO_DbDT2(3,1);
      matRGgeomPO_DbDT2 = matInvRgeomPO * matDbDT2;
      double arrDbDT2[5] = {
                            0,
                            0,       
                            matRGgeomPO_DbDT2[0][0],
                            matRGgeomPO_DbDT2[1][0],
                            matRGgeomPO_DbDT2[2][0]                         
                           };
 
      TMatrixD matDbDT3(3,1);
      matDbDT3[0] = {0};
      matDbDT3[1] = {0};
      matDbDT3[2] = {1};  
      TMatrixD matRGgeomPO_DbDT3(3,1);     
      matRGgeomPO_DbDT3 = matInvRgeomPO * matDbDT3; 
      double arrDbDT3[5] = {
                            0,
                            0,       
                            matRGgeomPO_DbDT3[0][0],
                            matRGgeomPO_DbDT3[1][0],
                            matRGgeomPO_DbDT3[2][0]                         
                           };

      double referenceFIX = 1.0;
      //if(sensorID >= 0 && fSplitReferenceSensor == sensorID) referenceFIX = 0.0;

      double DCDRalpha = 0;
      double DCDRbeta  = 0;
      double DCDRgamma = 0;   
      for (int j=0;j<nentries;j++) {
         DCDRalpha += arrDbDRalpha[j]*dcdwARR[j];
         DCDRbeta  += arrDbDRbeta[j]*dcdwARR[j];
         DCDRgamma += arrDbDRgamma[j]*dcdwARR[j];
      }   
      for (int j=0;j<sentries;j++) { 
         DCDRalpha += arrDwDRalpha[j]*dcdwARR[cnt+j];
         DCDRbeta  += arrDwDRbeta[j]*dcdwARR[cnt+j];
         DCDRgamma += arrDwDRgamma[j]*dcdwARR[cnt+j];
      }

      double DCDT1 = 0;
      double DCDT2 = 0;
      double DCDT3 = 0; 
      for (int j=0;j<nentries;j++) {
         DCDT1 += arrDbDT1[j]*dcdwARR[j];
         DCDT2 += arrDbDT2[j]*dcdwARR[j];
         DCDT3 += arrDbDT3[j]*dcdwARR[j];
      }  

      dcdwARR[nentries + sentries + 0] = DCDRalpha;
      dcdwARR[nentries + sentries + 1] = DCDRbeta;
      dcdwARR[nentries + sentries + 2] = DCDRgamma;   
      dcdwARR[nentries + sentries + 3] = DCDT1;
      dcdwARR[nentries + sentries + 4] = DCDT2;
      dcdwARR[nentries + sentries + 5] = DCDT3; 

      Ralpha += Update_aR[0]*DCDRalpha;
      Rbeta  += Update_aR[1]*DCDRbeta;
      Rgamma += Update_aR[2]*DCDRgamma;
      T1     += Update_aT[0]*DCDT1;
      T2     += Update_aT[1]*DCDT2;
      T3     += Update_aT[2]*DCDT3;   

      AlignParameters[1][0] = Ralpha;
      AlignParameters[1][1] = Rbeta;
      AlignParameters[1][2] = Rgamma;
      AlignParameters[1][3] = T1;
      AlignParameters[1][4] = T2;
      AlignParameters[1][5] = T3;

      if(nUpdate==true) {
         scn->SetaR(0,Ralpha);
         scn->SetaR(1,Rbeta);      
         scn->SetaR(2,Rgamma);
      }
      if(sUpdate==true) {
         scn->SetaT(0,T1);
         scn->SetaT(1,T2);
         scn->SetaT(2,T3);
      }
                                   
      TMatrixD matRalpha(3,3);         
      matRalpha[0] = {1, 0, 0};
      matRalpha[1] = {0, +std::cos(Ralpha), -std::sin(Ralpha)};
      matRalpha[2] = {0, +std::sin(Ralpha), +std::cos(Ralpha)};   
        
      TMatrixD matRbeta(3,3);
      matRbeta[0] = {+std::cos(Rbeta), 0, +std::sin(Rbeta)};
      matRbeta[1] = {0, 1, 0};
      matRbeta[2] = {-std::sin(Rbeta), 0, +std::cos(Rbeta)};       
      
      TMatrixD matRgamma(3,3);        
      matRgamma[0] = {+std::cos(Rgamma), -std::sin(Rgamma), 0};
      matRgamma[1] = {+std::sin(Rgamma), +std::cos(Rgamma), 0};
      matRgamma[2] = {0, 0, 1};

      TMatrixD matAR(3,3);
      matAR = matRalpha * matRbeta * matRgamma;
      
      TMatrixD matW(3,3);
      matW = matInvRgeomPO * matAR * matRgeomPO * matInvD1;

      TMatrixD matAT(3,1);
      matAT[0] = {T1};
      matAT[1] = {T2};   
      matAT[2] = {T3};
      
      TMatrixD matB(3,1);
      matB = matInvD1 * matD2 - matInvRgeomPO * matAR * matRgeomPO * matInvD1 * matD2
           + matInvRgeomPO * matAT + matInvRgeomPO * matAR * matTgeom - matInvRgeomPO * matTgeom;

      wSynapse[0] = matW[0][0] - sensor_d1;
      wSynapse[1] = matW[0][1];
      
      wSynapse[2] = matW[1][0];
      wSynapse[3] = matW[1][1] - sensor_d2;
      
      wSynapse[4] = matW[2][0];      
      wSynapse[5] = matW[2][1]; 
                             
      bNeuron[0] = 0;
      bNeuron[1] = 0;
      
      bNeuron[2] = matB[0][0];
      bNeuron[3] = matB[1][0];
      bNeuron[4] = matB[2][0];         

      NetworkParameters[1][0]  = bNeuron[0];
      NetworkParameters[1][1]  = bNeuron[1];
      NetworkParameters[1][2]  = bNeuron[2];
      NetworkParameters[1][3]  = bNeuron[3];    
      NetworkParameters[1][4]  = bNeuron[4];
      NetworkParameters[1][5]  = wSynapse[0];
      NetworkParameters[1][6]  = wSynapse[1];
      NetworkParameters[1][7]  = wSynapse[2];  
      NetworkParameters[1][8]  = wSynapse[3];
      NetworkParameters[1][9]  = wSynapse[4];
      NetworkParameters[1][10] = wSynapse[5];

      for (int j=0;j<nentries;j++) {
         neuron = (YNeuron *) scn->GetNetwork().At(j);  
         if(nUpdate==true) neuron->SetWeight(bNeuron[j]);
      }  

      for (int j=0;j<sentries;j++) { 
         synapse = (YSynapse *) scn->GetSynapses().At(j); 
         if(sUpdate==true) synapse->SetWeight(wSynapse[j]);
      }  

// DEBUG MODE
      double dAlignParameters[6] = {
                                   AlignParameters[1][0]-AlignParameters[0][0],
                                   AlignParameters[1][1]-AlignParameters[0][1],
                                   AlignParameters[1][2]-AlignParameters[0][2],
                                   AlignParameters[1][3]-AlignParameters[0][3],
                                   AlignParameters[1][4]-AlignParameters[0][4],
                                   AlignParameters[1][5]-AlignParameters[0][5]
                                   };

      double dNetworkParameters[11] = {
                                      NetworkParameters[1][0]-NetworkParameters[0][0],
                                      NetworkParameters[1][1]-NetworkParameters[0][1],
                                      NetworkParameters[1][2]-NetworkParameters[0][2],
                                      NetworkParameters[1][3]-NetworkParameters[0][3],
                                      NetworkParameters[1][4]-NetworkParameters[0][4],
                                      NetworkParameters[1][5]-NetworkParameters[0][5],
                                      NetworkParameters[1][6]-NetworkParameters[0][6],
                                      NetworkParameters[1][7]-NetworkParameters[0][7],
                                      NetworkParameters[1][8]-NetworkParameters[0][8],
                                      NetworkParameters[1][9]-NetworkParameters[0][9],
                                      NetworkParameters[1][10]-NetworkParameters[0][10]
                                      };
                                      
      double dRalpha = dAlignParameters[0];
      double dRbeta  = dAlignParameters[1];
      double dRgamma = dAlignParameters[2];
      double dT1     = dAlignParameters[3];
      double dT2     = dAlignParameters[4];
      double dT3     = dAlignParameters[5];

      //From dAP To dNP 
       
      TMatrixD matRalpha0(3,3);         
      matRalpha0[0] = {1, 0, 0};
      matRalpha0[1] = {0, +std::cos(Ralpha-dRalpha), -std::sin(Ralpha-dRalpha)};
      matRalpha0[2] = {0, +std::sin(Ralpha-dRalpha), +std::cos(Ralpha-dRalpha)};   
        
      TMatrixD matRbeta0(3,3);
      matRbeta0[0] = {+std::cos(Rbeta-dRbeta), 0, +std::sin(Rbeta-dRbeta)};
      matRbeta0[1] = {0, 1, 0};
      matRbeta0[2] = {-std::sin(Rbeta-dRbeta), 0, +std::cos(Rbeta-dRbeta)};       
      
      TMatrixD matRgamma0(3,3);        
      matRgamma0[0] = {+std::cos(Rgamma-dRgamma), -std::sin(Rgamma-dRgamma), 0};
      matRgamma0[1] = {+std::sin(Rgamma-dRgamma), +std::cos(Rgamma-dRgamma), 0};
      matRgamma0[2] = {0, 0, 1};

      TMatrixD matAR0(3,3);
      matAR0 = matRalpha0 * matRbeta0 * matRgamma0;                                      
                                      
      TMatrixD matdW(3,3);
      matdW = matInvRgeomPO * matAR * matRgeomPO * matInvD1
            - matInvRgeomPO * matAR0* matRgeomPO * matInvD1;

      TMatrixD matAT0(3,1);
      matAT0[0] = {T1-dT1};
      matAT0[1] = {T2-dT2};   
      matAT0[2] = {T3-dT3};
      
      TMatrixD matdB(3,1);
      matdB = (matInvD1 * matD2 - matInvRgeomPO * matAR * matRgeomPO * matInvD1 * matD2
              + matInvRgeomPO * matAT + matInvRgeomPO * matAR * matTgeom - matInvRgeomPO * matTgeom)
            - (matInvD1 * matD2 - matInvRgeomPO * matAR0* matRgeomPO * matInvD1 * matD2
              + matInvRgeomPO * matAT0+ matInvRgeomPO * matAR0* matTgeom - matInvRgeomPO * matTgeom);  
              
                           
      double dw11_APtoNP = matdW[0][0];
      double dw21_APtoNP = matdW[0][1];
      double dw12_APtoNP = matdW[1][0];
      double dw22_APtoNP = matdW[1][1];
      double dw13_APtoNP = matdW[2][0];
      double dw23_APtoNP = matdW[2][1];
      
      double db1_APtoNP  = matdB[0][0];
      double db2_APtoNP  = matdB[1][0];      
      double db3_APtoNP  = matdB[2][0];
/*                                      
      std::cout<<"[ DEBUGALIGNPARAMS Epoch"<<fCurrentEpoch<<" ]   Chip"<<sensorID<<"  AP[0] "
                                                                  <<AlignParameters[0][0]<<" "<<AlignParameters[0][1]<<" "<<AlignParameters[0][2]<<" "
                                                                  <<AlignParameters[0][3]<<" "<<AlignParameters[0][4]<<" "<<AlignParameters[0][5]<<std::endl;
      std::cout<<"[ DEBUGALIGNPARAMS Epoch"<<fCurrentEpoch<<" ]   Chip"<<sensorID<<"  AP[1] "
                                                                  <<AlignParameters[1][0]<<" "<<AlignParameters[1][1]<<" "<<AlignParameters[1][2]<<" "
                                                                  <<AlignParameters[1][3]<<" "<<AlignParameters[1][4]<<" "<<AlignParameters[1][5]<<std::endl;
      std::cout<<"[ DEBUGALIGNPARAMS Epoch"<<fCurrentEpoch<<" ]   Chip"<<sensorID<<"  dAP(MDL) "
                                                                  <<dAlignParameters[0]<<" "
                                                                  <<dAlignParameters[1]<<" "
                                                                  <<dAlignParameters[2]<<" "
                                                                  <<dAlignParameters[3]<<" "
                                                                  <<dAlignParameters[4]<<" "
                                                                  <<dAlignParameters[5]<<std::endl;

      std::cout<<"[ DEBUGNETWORKPARAMS Epoch"<<fCurrentEpoch<<" ] Chip"<<sensorID<<"  NP[0] "
                                                                  <<NetworkParameters[0][2]<<" "<<NetworkParameters[0][3]<<" "<<NetworkParameters[0][4]<<" "
                                                                  <<NetworkParameters[0][5]<<" "<<NetworkParameters[0][6]<<" "
                                                                  <<NetworkParameters[0][7]<<" "<<NetworkParameters[0][8]<<" "                                                                
                                                                  <<NetworkParameters[0][9]<<" "<<NetworkParameters[0][10]<<std::endl;
      std::cout<<"[ DEBUGNETWORKPARAMS Epoch"<<fCurrentEpoch<<" ] Chip"<<sensorID<<"  NP[1] "
                                                                  <<NetworkParameters[1][2]<<" "<<NetworkParameters[1][3]<<" "<<NetworkParameters[1][4]<<" "
                                                                  <<NetworkParameters[1][5]<<" "<<NetworkParameters[1][6]<<" "
                                                                  <<NetworkParameters[1][7]<<" "<<NetworkParameters[1][8]<<" "                                                                
                                                                  <<NetworkParameters[1][9]<<" "<<NetworkParameters[1][10]<<std::endl;
      std::cout<<"[ DEBUGNETWORKPARAMS Epoch"<<fCurrentEpoch<<" ] Chip"<<sensorID<<"  dNP(MDL) "
                                                                  <<dNetworkParameters[2]<<" "
                                                                  <<dNetworkParameters[3]<<" "
                                                                  <<dNetworkParameters[4]<<" "
                                                                  <<dNetworkParameters[5]<<" "
                                                                  <<dNetworkParameters[6]<<" "
                                                                  <<dNetworkParameters[7]<<" "
                                                                  <<dNetworkParameters[8]<<" "                                                                
                                                                  <<dNetworkParameters[9]<<" "
                                                                  <<dNetworkParameters[10]<<std::endl;    

      std::cout<<"[ DEBUGNETWORKPARAMS Epoch"<<fCurrentEpoch<<" ] Chip"<<sensorID<<"  dNP(MAT) "
                                                                  <<db1_APtoNP<<" "
                                                                  <<db2_APtoNP<<" "
                                                                  <<db3_APtoNP<<" "
                                                                  <<dw11_APtoNP<<" "
                                                                  <<dw21_APtoNP<<" "
                                                                  <<dw12_APtoNP<<" "
                                                                  <<dw22_APtoNP<<" "                                                                
                                                                  <<dw13_APtoNP<<" "
                                                                  <<dw23_APtoNP<<std::endl;                                                                    
*/                                                               
                                                                                                                                
// DEBUG MODE                                           
   } else {

      for (int j=0;j<nentries;j++) {
         neuron = (YNeuron *) scn->GetNetwork().At(j);  
         if(j>=2){      
            dcdwARR[j] = (-fEta) * (neuron->GetDCDw() + fDelta) + fEpsilon * dcdwARR[j];    
         } else {
            dcdwARR[j] = 0;
         }      
         bNeuron[j] = neuron->GetWeight();
      }       
      for (int j=0;j<sentries;j++) { 
         synapse = (YSynapse *) scn->GetSynapses().At(j);  
         dcdwARR[cnt+j] = (-fEta) * (synapse->GetDCDw() + fDelta) + fEpsilon * dcdwARR[cnt+j];         
         wSynapse[j] = synapse->GetWeight();
      }
      //   0   1   2   3   4   5
      // w11 w21 w12 w22 w13 w23

      double Ralpha  = + wSynapse[5];
      double Rbeta   = - wSynapse[4]; 
      double Rgamma  = 0.5*(- wSynapse[1] + wSynapse[2]);
      
      double DwSynapseDRalpha[6] = { 
             0, 0,
             std::cos(Ralpha)*std::sin(Rbeta)*std::cos(Rgamma) - std::sin(Ralpha)*std::sin(Rgamma), -std::cos(Ralpha)*std::sin(Rbeta)*std::sin(Rgamma) - std::sin(Ralpha)*std::cos(Rgamma),
             std::sin(Ralpha)*std::sin(Rbeta)*std::cos(Rgamma) + std::cos(Ralpha)*std::sin(Rgamma), -std::sin(Ralpha)*std::sin(Rbeta)*std::sin(Rgamma) + std::cos(Ralpha)*std::cos(Rgamma)
                                   };   
      double DwSynapseDRbeta[6]  = {
             -std::sin(Rbeta)*std::cos(Rgamma), std::sin(Rbeta)*std::sin(Rgamma),
             std::sin(Ralpha)*std::cos(Rbeta)*std::cos(Rgamma), -std::sin(Ralpha)*std::cos(Rbeta)*std::sin(Rgamma),
             -std::cos(Ralpha)*std::cos(Rbeta)*std::cos(Rgamma), std::cos(Ralpha)*std::cos(Rbeta)*std::sin(Rgamma)
                                   };   
      double DwSynapseDRgamma[6] = {
             -std::cos(Rbeta)*std::sin(Rgamma), -std::cos(Rbeta)*std::cos(Rgamma),
             -std::sin(Ralpha)*std::sin(Rbeta)*std::sin(Rgamma) + std::cos(Ralpha)*std::cos(Rgamma), -std::sin(Ralpha)*std::sin(Rbeta)*std::cos(Rgamma) - std::cos(Ralpha)*std::sin(Rgamma),
             std::cos(Ralpha)*std::sin(Rbeta)*std::sin(Rgamma) + std::sin(Ralpha)*std::cos(Rgamma), std::cos(Ralpha)*std::sin(Rbeta)*std::cos(Rgamma) - std::sin(Ralpha)*std::sin(Rgamma)
                                   };   
   
      double DCDRalpha = 0;
      double DCDRbeta  = 0;
      double DCDRgamma = 0;   
      for (int j=0;j<sentries;j++) { 
         DCDRalpha += DwSynapseDRalpha[j]*dcdwARR[cnt+j];
         DCDRbeta  += DwSynapseDRbeta[j]*dcdwARR[cnt+j];
         DCDRgamma += DwSynapseDRgamma[j]*dcdwARR[cnt+j];
      }
   
      dcdwARR[nentries + sentries + 0] = DCDRalpha;
      dcdwARR[nentries + sentries + 1] = DCDRbeta;
      dcdwARR[nentries + sentries + 2] = DCDRgamma;   
      dcdwARR[nentries + sentries + 3] = 0;
      dcdwARR[nentries + sentries + 4] = 0;
      dcdwARR[nentries + sentries + 5] = 0; 
      
      Ralpha += DCDRalpha;
      Rbeta  += DCDRbeta;
      Rgamma += DCDRgamma;
   
      wSynapse[0] = std::cos(Rbeta)*std::cos(Rgamma) - 1;	
      wSynapse[1] = -std::cos(Rbeta)*std::sin(Rgamma);
   
      wSynapse[2] = std::sin(Ralpha)*std::sin(Rbeta)*std::cos(Rgamma) + std::cos(Ralpha)*std::sin(Rgamma);
      wSynapse[3] = -std::sin(Ralpha)*std::sin(Rbeta)*std::sin(Rgamma) + std::cos(Ralpha)*std::cos(Rgamma) - 1;
   
      wSynapse[4] = -std::cos(Ralpha)*std::sin(Rbeta)*std::cos(Rgamma) + std::sin(Ralpha)*std::sin(Rgamma);
      wSynapse[5] = std::cos(Ralpha)*std::sin(Rbeta)*std::sin(Rgamma) + std::sin(Ralpha)*std::cos(Rgamma);
       
      for (int j=0;j<nentries;j++) {
         neuron = (YNeuron *) scn->GetNetwork().At(j);  
         double bufferArr_Weight = 1;//paramA/(1 + TMath::Exp(-(std::pow(dcdwARR[j]*1e+4/PIXAMP,2)))) - paramB;
         bNeuron[j] = neuron->GetWeight() + bufferArr_Weight*dcdwARR[j];            
         if(nUpdate==true) neuron->SetWeight(bNeuron[j]);
      }  
    
      for (int j=0;j<sentries;j++) { 
         synapse = (YSynapse *) scn->GetSynapses().At(j); 
         if(sUpdate==true) synapse->SetWeight(wSynapse[j]);
      } 
   }

}

////////////////////////////////////////////////////////////////////////////////
/// Sets the weights to a point along a line
/// Weights are set to [origin + (dist * dir)].

void YMultiLayerPerceptron::MLP_Line(Double_t * origin, Double_t * dir, Double_t dist)
{
   Int_t idx = 0;
   YNeuron *neuron = 0;
   YSynapse *synapse = 0;
   TObjArrayIter *it = (TObjArrayIter *) fNetwork.MakeIterator();
   while ((neuron = (YNeuron *) it->Next())) {
      neuron->SetWeight(origin[idx] + (dir[idx] * dist));
      idx++;
   }
   delete it;
   it = (TObjArrayIter *) fSynapses.MakeIterator();
   while ((synapse = (YSynapse *) it->Next())) {
      synapse->SetWeight(origin[idx] + (dir[idx] * dist));
      idx++;
   }
   delete it;
}

void YMultiLayerPerceptron::MLP_LineArr(Double_t** originArr, Double_t** dirArr, Double_t dist)
{
   //std::cout<<"YMultiLayerPerceptron::MLP_LineArr"<<std::endl;
   Int_t idx = 0;
   Int_t j, nentries = 0;
   YNeuron *neuron = 0;
   YSynapse *synapse = 0;   
   for(int iID = 0; iID < nSensors; iID++){
      idx = 0;
      nentries = fSCNetwork[iID]->GetNetwork().GetEntriesFast();
      for (j=0;j<nentries;j++) {
         neuron = (YNeuron *) fSCNetwork[iID]->GetNetwork().At(j);    
         double w = originArr[iID][j] + (dirArr[iID][j] * dist);
         neuron->SetWeight(w);  
      }     
      idx = nentries;            
      nentries = fSCNetwork[iID]->GetSynapses().GetEntriesFast();
      for (j=0;j<nentries;j++) {
         synapse = (YSynapse *) fSCNetwork[iID]->GetSynapses().At(j);       
         double w = originArr[iID][idx+j] + (dirArr[iID][idx+j] * dist);
         synapse->SetWeight(w);     
      }   
      neuron = 0;
      synapse = 0;      
   }        
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the search direction to steepest descent.

void YMultiLayerPerceptron::SteepestDir(Double_t * dir)
{
   Int_t idx = 0;
   YNeuron *neuron = 0;
   YSynapse *synapse = 0;
   TObjArrayIter *it = (TObjArrayIter *) fNetwork.MakeIterator();
   while ((neuron = (YNeuron *) it->Next()))
      dir[idx++] = -neuron->GetDCDw();
   delete it;
   it = (TObjArrayIter *) fSynapses.MakeIterator();
   while ((synapse = (YSynapse *) it->Next()))
      dir[idx++] = -synapse->GetDCDw();
   delete it;
}

void YMultiLayerPerceptron::SteepestDirArr(Double_t** dirArr)
{   
   //std::cout<<"YMultiLayerPerceptron::SteepestDirArr"<<std::endl;
   Int_t j;
   Int_t idx =0;  
   Int_t nentries =0;  
   YSynapse *synapse;
   YNeuron *neuron;
   for(int iID = 0; iID < nSensors; iID++){
      idx = 0;
      nentries = fSCNetwork[iID]->GetNetwork().GetEntriesFast();
      for (j=0;j<nentries;j++) {
         neuron = (YNeuron *) fSCNetwork[iID]->GetNetwork().At(j);   
         dirArr[iID][j] = -neuron->GetDCDw();                              
      }     
      idx = nentries;            
      nentries = fSCNetwork[iID]->GetSynapses().GetEntriesFast();
      for (j=0;j<nentries;j++) {
         synapse = (YSynapse *) fSCNetwork[iID]->GetSynapses().At(j);       
         dirArr[iID][idx+j] = -synapse->GetDCDw();   
      } 
      neuron = 0;
      synapse = 0;                        
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Search along the line defined by direction.
/// buffer is not used but is updated with the new dw
/// so that it can be used by a later stochastic step.
/// It returns true if the line search fails.

bool YMultiLayerPerceptron::LineSearch(Double_t * direction, Double_t * buffer)
{
   Int_t idx = 0;
   Int_t j,nentries;
   YNeuron *neuron = 0;
   YSynapse *synapse = 0;
   // store weights before line search
   Double_t *origin = new Double_t[fNetwork.GetEntriesFast() +
                                   fSynapses.GetEntriesFast()];
   nentries = fNetwork.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      neuron = (YNeuron *) fNetwork.At(j);
      origin[idx++] = neuron->GetWeight();
   }
   nentries = fSynapses.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      synapse = (YSynapse *) fSynapses.At(j);
      origin[idx++] = synapse->GetWeight();
   }
   // try to find a triplet (alpha1, alpha2, alpha3) such that
   // Error(alpha1)>Error(alpha2)<Error(alpha3)
   Double_t err1 = GetCost(kTraining);
   Double_t alpha1 = 0.;
   Double_t alpha2 = fLastAlpha;
   if (alpha2 < 0.01)
      alpha2 = 0.01;
   if (alpha2 > 2.0)
      alpha2 = 2.0;
   Double_t alpha3 = alpha2;
   MLP_Line(origin, direction, alpha2);
   Double_t err2 = GetCost(kTraining);
   Double_t err3 = err2;
   Bool_t bingo = false;
   Int_t icount;
   if (err1 > err2) {
      for (icount = 0; icount < 100; icount++) {
         alpha3 *= fTau;
         MLP_Line(origin, direction, alpha3);
         err3 = GetCost(kTraining);
         if (err3 > err2) {
            bingo = true;
            break;
         }
         alpha1 = alpha2;
         err1 = err2;
         alpha2 = alpha3;
         err2 = err3;
      }
      if (!bingo) {
         std::cout<<"LineSearch Type A fLastAlpha : "<<fLastAlpha<<std::endl;
         MLP_Line(origin, direction, 0.);
         delete[]origin;
         return true;
      }
   } else {
      for (icount = 0; icount < 100; icount++) {
         alpha2 /= fTau;
         MLP_Line(origin, direction, alpha2);
         err2 = GetCost(kTraining);
         if (err1 > err2) {
            bingo = true;
            break;
         }
         alpha3 = alpha2;
         err3 = err2;
      }
      if (!bingo) {
         std::cout<<"LineSearch Type B fLastAlpha : "<<fLastAlpha<<std::endl;
         MLP_Line(origin, direction, 0.);
         delete[]origin;
         fLastAlpha = 0.05;
         return true;
      }
   }
   // Sets the weights to the bottom of parabola
   fLastAlpha = 0.5 * (alpha1 + alpha3 -
                (err3 - err1) / ((err3 - err2) / (alpha3 - alpha2)
                - (err2 - err1) / (alpha2 - alpha1)));
   fLastAlpha = fLastAlpha < 100000 ? fLastAlpha : 100000;
   MLP_Line(origin, direction, fLastAlpha);
   GetCost(kTraining);
   // Stores weight changes (can be used by a later stochastic step)
   idx = 0;
   nentries = fNetwork.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      neuron = (YNeuron *) fNetwork.At(j);
      buffer[idx] = neuron->GetWeight() - origin[idx];
      idx++;
   }
   nentries = fSynapses.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      synapse = (YSynapse *) fSynapses.At(j);
      buffer[idx] = synapse->GetWeight() - origin[idx];
      idx++;
   }
   std::cout<<"LineSearch Type C fLastAlpha : "<<fLastAlpha<<" Tau : "<<fTau<<std::endl;   
   delete[]origin;
   return false;
}

bool YMultiLayerPerceptron::LineSearchArr(Double_t** directionArr, Double_t** bufferArr)
{
   Int_t idx = 0;
   Int_t j,nentries;
   YNeuron *neuron = 0;
   YSynapse *synapse = 0;
   // store weights before line search
   Int_t els = fSCNetwork[0]->GetNetwork().GetEntriesFast() + fSCNetwork[0]->GetSynapses().GetEntriesFast();
   Double_t **originArr;
   originArr = new Double_t *[nSensors];
   for(int iID = 0; iID <nSensors; iID++){
      originArr[iID] = new Double_t [els];
   }
   for(int iID = 0; iID <nSensors; iID++){
      idx = 0;
      nentries = fSCNetwork[iID]->GetNetwork().GetEntriesFast();
      for (j=0;j<nentries;j++) {
         neuron = (YNeuron *) fSCNetwork[iID]->GetNetwork().At(j);    
         originArr[iID][j] = neuron->GetWeight();                                                 
      }     
      idx = nentries;            
      nentries = fSCNetwork[iID]->GetSynapses().GetEntriesFast();
      for (j=0;j<nentries;j++) {
         synapse = (YSynapse *) fSCNetwork[iID]->GetSynapses().At(j);       
         originArr[iID][idx+j] = synapse->GetWeight();     
      }      
   }   
   for(int iID = 0; iID <nSensors; iID++){
      nentries = fSCNetwork[iID]->GetSynapses().GetEntriesFast();
      for (int j=0;j<nentries;j++) {
         synapse = (YSynapse *) fSCNetwork[iID]->GetSynapses().At(j);       
      }  
      nentries = fSCNetwork[iID]->GetNetwork().GetEntriesFast();
      for (int j=0;j<nentries;j++) {
         neuron = (YNeuron *) fSCNetwork[iID]->GetNetwork().At(j);       
      }       
   }
   for(int iID = 0; iID <nSensors; iID++){
      nentries = fSCNetwork[iID]->GetSynapses().GetEntriesFast();
      for (int j=0;j<nentries;j++) {
         synapse = (YSynapse *) fSCNetwork[iID]->GetSynapses().At(j);       
      }  
      nentries = fSCNetwork[iID]->GetNetwork().GetEntriesFast();
      for (int j=0;j<nentries;j++) {
         neuron = (YNeuron *) fSCNetwork[iID]->GetNetwork().At(j);       
      }      
   }   
   // try to find a triplet (alpha1, alpha2, alpha3) such that
   // Error(alpha1)>Error(alpha2)<Error(alpha3)
   Double_t err1 = GetCost(kTraining);
   Double_t alpha1 = 0.;
   Double_t alpha2 = fLastAlpha;
   if (alpha2 < 0.01)
      alpha2 = 0.01;
   if (alpha2 > 2.0)
      alpha2 = 2.0;
   Double_t alpha3 = alpha2;
   MLP_LineArr(originArr, directionArr, alpha2);
   for(int iID = 0; iID <nSensors; iID++){
      nentries = fSCNetwork[iID]->GetSynapses().GetEntriesFast();
      for (int j=0;j<nentries;j++) {
         synapse = (YSynapse *) fSCNetwork[iID]->GetSynapses().At(j);       
      }  
      nentries = fSCNetwork[iID]->GetNetwork().GetEntriesFast();
      for (int j=0;j<nentries;j++) {
         neuron = (YNeuron *) fSCNetwork[iID]->GetNetwork().At(j);       

      }      
   }
   for(int iID = 0; iID <nSensors; iID++){
      nentries = fSCNetwork[iID]->GetSynapses().GetEntriesFast();
      for (int j=0;j<nentries;j++) {
         synapse = (YSynapse *) fSCNetwork[iID]->GetSynapses().At(j);       
      }  
      nentries = fSCNetwork[iID]->GetNetwork().GetEntriesFast();
      for (int j=0;j<nentries;j++) {
         neuron = (YNeuron *) fSCNetwork[iID]->GetNetwork().At(j);       
      }       
   }   
   Double_t err2 = GetCost(kTraining);
   Double_t err3 = err2;
   Bool_t bingo = false;
   Int_t icount;
   if (err1 > err2) {
      for (icount = 0; icount < 100; icount++) {
         alpha3 *= fTau;
         MLP_LineArr(originArr, directionArr, alpha3);
         err3 = GetCost(kTraining);
         if (err3 > err2) {
            bingo = true;
            break;
         }
         alpha1 = alpha2;
         err1 = err2;
         alpha2 = alpha3;
         err2 = err3;
      }
      if (!bingo) {
         //std::cout<<"LineSearch Type A fLastAlpha : "<<fLastAlpha<<std::endl;
         MLP_LineArr(originArr, directionArr, 0.);
         //delete[]originArr;
         for(int iID = 0; iID <nSensors; iID++){
            delete [] originArr[iID];   
         } 
         delete [] originArr;
         return true;
      }
   } else {
      for (icount = 0; icount < 100; icount++) {
         alpha2 /= fTau;
         MLP_LineArr(originArr, directionArr, alpha2);
         err2 = GetCost(kTraining);
         if (err1 > err2) {
            bingo = true;
            break;
         }
         alpha3 = alpha2;
         err3 = err2;
      }
      if (!bingo) {
         //std::cout<<"LineSearch Type B fLastAlpha : "<<fLastAlpha<<std::endl;
         MLP_LineArr(originArr, directionArr, 0.);
         //delete[]originArr;
         for(int iID = 0; iID <nSensors; iID++){
            delete [] originArr[iID];   
         } 
         delete [] originArr;
         fLastAlpha = 0.05;
         return true;
      }
   }
   // Sets the weights to the bottom of parabola
   fLastAlpha = 0.5 * (alpha1 + alpha3 -
                (err3 - err1) / ((err3 - err2) / (alpha3 - alpha2)
                - (err2 - err1) / (alpha2 - alpha1)));
   fLastAlpha = fLastAlpha < 100000 ? fLastAlpha : 100000;
   MLP_LineArr(originArr, directionArr, fLastAlpha);
   for(int iID = 0; iID <nSensors; iID++){
      int nentries = fSCNetwork[iID]->GetSynapses().GetEntriesFast();
      for (int j=0;j<nentries;j++) {
         synapse = (YSynapse *) fSCNetwork[iID]->GetSynapses().At(j);       
      }  
      nentries = fSCNetwork[iID]->GetNetwork().GetEntriesFast();
      for (int j=0;j<nentries;j++) {
         neuron = (YNeuron *) fSCNetwork[iID]->GetNetwork().At(j);       
      }      
   }
   for(int iID = 0; iID <nSensors; iID++){
      int nentries = fSCNetwork[iID]->GetSynapses().GetEntriesFast();
      for (int j=0;j<nentries;j++) {
         synapse = (YSynapse *) fSCNetwork[iID]->GetSynapses().At(j);       
      }  
      nentries = fSCNetwork[iID]->GetNetwork().GetEntriesFast();
      for (int j=0;j<nentries;j++) {
         neuron = (YNeuron *) fSCNetwork[iID]->GetNetwork().At(j);       
      }      
   }
   GetCost(kTraining);
   // Stores weight changes (can be used by a later stochastic step)
   for(int iID = 0; iID <nSensors; iID++){
      idx = 0;
      nentries = fSCNetwork[iID]->GetNetwork().GetEntriesFast();
      for (j=0;j<nentries;j++) {
         neuron = (YNeuron *) fSCNetwork[iID]->GetNetwork().At(j);      
         bufferArr[iID][j] = neuron->GetWeight() - originArr[iID][j];     
      }     
      idx = nentries;            
      nentries = fSCNetwork[iID]->GetSynapses().GetEntriesFast();
      for (j=0;j<nentries;j++) {
         synapse = (YSynapse *) fSCNetwork[iID]->GetSynapses().At(j);       
         bufferArr[iID][idx+j] = synapse->GetWeight() - originArr[iID][idx+j];     
      }                     
   
   } 
 
   //std::cout<<"LineSearch Type C fLastAlpha : "<<fLastAlpha<<" Tau : "<<fTau<<std::endl;   

   for(int iID = 0; iID <nSensors; iID++){
      delete [] originArr[iID];   
   } 
   delete [] originArr;      
   
   return false;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the search direction to conjugate gradient direction
/// beta should be:
///  ||g_{(t+1)}||^2 / ||g_{(t)}||^2                   (Fletcher-Reeves)
///  g_{(t+1)} (g_{(t+1)}-g_{(t)}) / ||g_{(t)}||^2     (Ribiere-Polak)

void YMultiLayerPerceptron::ConjugateGradientsDir(Double_t * dir, Double_t beta)
{
   Int_t idx = 0;
   Int_t j,nentries;
   YNeuron *neuron = 0;
   YSynapse *synapse = 0;
   nentries = fNetwork.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      neuron = (YNeuron *) fNetwork.At(j);
      dir[idx] = -neuron->GetDCDw() + beta * dir[idx];
      idx++;
   }
   nentries = fSynapses.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      synapse = (YSynapse *) fSynapses.At(j);
      dir[idx] = -synapse->GetDCDw() + beta * dir[idx];
      idx++;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Computes the hessian matrix using the BFGS update algorithm.
/// from gamma (g_{(t+1)}-g_{(t)}) and delta (w_{(t+1)}-w_{(t)}).
/// It returns true if such a direction could not be found
/// (if gamma and delta are orthogonal).

bool YMultiLayerPerceptron::GetBFGSH(TMatrixD & bfgsh, TMatrixD & gamma, TMatrixD & delta)
{
   TMatrixD gd(gamma, TMatrixD::kTransposeMult, delta);
   if ((Double_t) gd[0][0] == 0.)
      return true;
   TMatrixD aHg(bfgsh, TMatrixD::kMult, gamma);
   TMatrixD tmp(gamma, TMatrixD::kTransposeMult, bfgsh);
   TMatrixD gHg(gamma, TMatrixD::kTransposeMult, aHg);
   Double_t a = 1 / (Double_t) gd[0][0];
   Double_t f = 1 + ((Double_t) gHg[0][0] * a);
   TMatrixD res( TMatrixD(delta, TMatrixD::kMult,
                TMatrixD(TMatrixD::kTransposed, delta)));
   res *= f;
   res -= (TMatrixD(delta, TMatrixD::kMult, tmp) +
           TMatrixD(aHg, TMatrixD::kMult,
                   TMatrixD(TMatrixD::kTransposed, delta)));
   res *= a;
   bfgsh += res;
   return false;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the gamma (g_{(t+1)}-g_{(t)}) and delta (w_{(t+1)}-w_{(t)}) vectors
/// Gamma is computed here, so ComputeDCDw cannot have been called before,
/// and delta is a direct translation of buffer into a TMatrixD.

void YMultiLayerPerceptron::SetGammaDelta(TMatrixD & gamma, TMatrixD & delta,
                                          Double_t * buffer)
{
   Int_t els = fNetwork.GetEntriesFast() + fSynapses.GetEntriesFast();
   Int_t idx = 0;
   Int_t j,nentries;
   YNeuron *neuron = 0;
   YSynapse *synapse = 0;
   nentries = fNetwork.GetEntriesFast();
#ifdef YMLPDEBUG
   std::cout<<"YMultiLayerPerceptron::SetGammaDelta neurons"<<std::endl;
#endif
   for (j=0;j<nentries;j++) {
      neuron = (YNeuron *) fNetwork.At(j);
      gamma[idx++][0] = -neuron->GetDCDw();
   }
   nentries = fSynapses.GetEntriesFast();
#ifdef YMLPDEBUG
   std::cout<<"YMultiLayerPerceptron::SetGammaDelta synapses"<<std::endl;
#endif
   for (j=0;j<nentries;j++) {
      synapse = (YSynapse *) fSynapses.At(j);
      gamma[idx++][0] = -synapse->GetDCDw();
   }
#ifdef YMLPDEBUG
   std::cout<<"YMultiLayerPerceptron::SetGammaDelta A"<<std::endl;
#endif
   for (Int_t i = 0; i < els; i++)
      delta[i].Assign(buffer[i]);
   //delta.SetElements(buffer,"F");
#ifdef YMLPDEBUG 
   std::cout<<"YMultiLayerPerceptron::SetGammaDelta B"<<std::endl;
#endif
   ComputeDCDw();
  

   idx = 0;
   nentries = fNetwork.GetEntriesFast();
#ifdef YMLPDEBUG
   std::cout<<"YMultiLayerPerceptron::SetGammaDelta C"<<std::endl;
#endif
   for (j=0;j<nentries;j++) {
      neuron = (YNeuron *) fNetwork.At(j);
      gamma[idx++][0] += neuron->GetDCDw();
   }
#ifdef YMLPDEBUG
   std::cout<<"YMultiLayerPerceptron::SetGammaDelta D"<<std::endl;
#endif
   nentries = fSynapses.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      synapse = (YSynapse *) fSynapses.At(j);
      gamma[idx++][0] += synapse->GetDCDw();
   }
#ifdef YMLPDEBUG
   std::cout<<"YMultiLayerPerceptron::SetGammaDelta E"<<std::endl;
#endif
}

////////////////////////////////////////////////////////////////////////////////
/// scalar product between gradient and direction
/// = derivative along direction

Double_t YMultiLayerPerceptron::DerivDir(Double_t * dir)
{
   Int_t idx = 0;
   Int_t j,nentries;
   Double_t output = 0;
   YNeuron *neuron = 0;
   YSynapse *synapse = 0;
   nentries = fNetwork.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      neuron = (YNeuron *) fNetwork.At(j);
      output += neuron->GetDCDw() * dir[idx++];
   }
   nentries = fSynapses.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      synapse = (YSynapse *) fSynapses.At(j);
      output += synapse->GetDCDw() * dir[idx++];
   }
   return output;
}

////////////////////////////////////////////////////////////////////////////////
/// Computes the direction for the BFGS algorithm as the product
/// between the Hessian estimate (bfgsh) and the dir.

void YMultiLayerPerceptron::BFGSDir(TMatrixD & bfgsh, Double_t * dir)
{
   Int_t els = fNetwork.GetEntriesFast() + fSynapses.GetEntriesFast();
   TMatrixD dedw(els, 1);
   Int_t idx = 0;
   Int_t j,nentries;
   YNeuron *neuron = 0;
   YSynapse *synapse = 0;
   nentries = fNetwork.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      neuron = (YNeuron *) fNetwork.At(j);
      dedw[idx++][0] = neuron->GetDCDw();
   }
   nentries = fSynapses.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      synapse = (YSynapse *) fSynapses.At(j);
      dedw[idx++][0] = synapse->GetDCDw();
   }
   TMatrixD direction(bfgsh, TMatrixD::kMult, dedw);
   for (Int_t i = 0; i < els; i++)
      dir[i] = -direction[i][0];
   //direction.GetElements(dir,"F");
}

////////////////////////////////////////////////////////////////////////////////
/// Draws the network structure.
/// Neurons are depicted by a blue disk, and synapses by
/// lines connecting neurons.
/// The line width is proportionnal to the weight.

void YMultiLayerPerceptron::Draw(Option_t * /*option*/)
{
#define NeuronSize 2.5

   Int_t nLayers = fStructure.CountChar(':')+1;
   Float_t xStep = 1./(nLayers+1.);
   Int_t layer;
   for(layer=0; layer< nLayers-1; layer++) {
      Float_t nNeurons_this = 0;
      if(layer==0) {
         TString input      = TString(fStructure(0, fStructure.First(':')));
         nNeurons_this = input.CountChar(',')+1;
      }
      else {
         Int_t cnt=0;
         TString hidden = TString(fStructure(fStructure.First(':') + 1,fStructure.Last(':') - fStructure.First(':') - 1));
         Int_t beg = 0;
         Int_t end = hidden.Index(":", beg + 1);
         while (end != -1) {
            Int_t num = atoi(TString(hidden(beg, end - beg)).Data());
            cnt++;
            beg = end + 1;
            end = hidden.Index(":", beg + 1);
            if(layer==cnt) nNeurons_this = num;
         }
         Int_t num = atoi(TString(hidden(beg, hidden.Length() - beg)).Data());
         cnt++;
         if(layer==cnt) nNeurons_this = num;
      }
      Float_t nNeurons_next = 0;
      if(layer==nLayers-2) {
         TString output = TString(fStructure(fStructure.Last(':') + 1,fStructure.Length() - fStructure.Last(':')));
         nNeurons_next = output.CountChar(',')+1;
      }
      else {
         Int_t cnt=0;
         TString hidden = TString(fStructure(fStructure.First(':') + 1,fStructure.Last(':') - fStructure.First(':') - 1));
         Int_t beg = 0;
         Int_t end = hidden.Index(":", beg + 1);
         while (end != -1) {
            Int_t num = atoi(TString(hidden(beg, end - beg)).Data());
            cnt++;
            beg = end + 1;
            end = hidden.Index(":", beg + 1);
            if(layer+1==cnt) nNeurons_next = num;
         }
         Int_t num = atoi(TString(hidden(beg, hidden.Length() - beg)).Data());
         cnt++;
         if(layer+1==cnt) nNeurons_next = num;
      }
      Float_t yStep_this = 1./(nNeurons_this+1.);
      Float_t yStep_next = 1./(nNeurons_next+1.);
      TObjArrayIter* it = (TObjArrayIter *) fSynapses.MakeIterator();
      YSynapse *theSynapse = 0;
      Float_t maxWeight = 0;
      while ((theSynapse = (YSynapse *) it->Next()))
         maxWeight = maxWeight < theSynapse->GetWeight() ? theSynapse->GetWeight() : maxWeight;
      delete it;
      it = (TObjArrayIter *) fSynapses.MakeIterator();
      for(Int_t neuron1=0; neuron1<nNeurons_this; neuron1++) {
         for(Int_t neuron2=0; neuron2<nNeurons_next; neuron2++) {
            TLine* synapse = new TLine(xStep*(layer+1),yStep_this*(neuron1+1),xStep*(layer+2),yStep_next*(neuron2+1));
            synapse->Draw();
            theSynapse = (YSynapse *) it->Next();
            if (!theSynapse) continue;
            synapse->SetLineWidth(Int_t((theSynapse->GetWeight()/maxWeight)*10.));
            synapse->SetLineStyle(1);
            if(((TMath::Abs(theSynapse->GetWeight())/maxWeight)*10.)<0.5) synapse->SetLineStyle(2);
            if(((TMath::Abs(theSynapse->GetWeight())/maxWeight)*10.)<0.25) synapse->SetLineStyle(3);
         }
      }
      delete it;
   }
   for(layer=0; layer< nLayers; layer++) {
      Float_t nNeurons = 0;
      if(layer==0) {
         TString input      = TString(fStructure(0, fStructure.First(':')));
         nNeurons = input.CountChar(',')+1;
      }
      else if(layer==nLayers-1) {
         TString output = TString(fStructure(fStructure.Last(':') + 1,fStructure.Length() - fStructure.Last(':')));
         nNeurons = output.CountChar(',')+1;
      }
      else {
         Int_t cnt=0;
         TString hidden = TString(fStructure(fStructure.First(':') + 1,fStructure.Last(':') - fStructure.First(':') - 1));
         Int_t beg = 0;
         Int_t end = hidden.Index(":", beg + 1);
         while (end != -1) {
            Int_t num = atoi(TString(hidden(beg, end - beg)).Data());
            cnt++;
            beg = end + 1;
            end = hidden.Index(":", beg + 1);
            if(layer==cnt) nNeurons = num;
         }
         Int_t num = atoi(TString(hidden(beg, hidden.Length() - beg)).Data());
         cnt++;
         if(layer==cnt) nNeurons = num;
      }
      Float_t yStep = 1./(nNeurons+1.);
      for(Int_t neuron=0; neuron<nNeurons; neuron++) {
         TMarker* m = new TMarker(xStep*(layer+1),yStep*(neuron+1),20);
         m->SetMarkerColor(4);
         m->SetMarkerSize(NeuronSize);
         m->Draw();
      }
   }
   const TString input = TString(fStructure(0, fStructure.First(':')));
   const TObjArray *inpL = input.Tokenize(" ,");
   const Int_t nrItems = inpL->GetLast()+1;
   Float_t yStep = 1./(nrItems+1);
   for (Int_t item = 0; item < nrItems; item++) {
      const TString brName = ((TObjString *)inpL->At(item))->GetString();
      TText* label = new TText(0.5*xStep,yStep*(item+1),brName.Data());
      label->Draw();
   }
   delete inpL;

   Int_t numOutNodes=fLastLayer.GetEntriesFast();
   yStep=1./(numOutNodes+1);
   for (Int_t outnode=0; outnode<numOutNodes; outnode++) {
      YNeuron* neuron=(YNeuron*)fLastLayer[outnode];
      if (neuron && neuron->GetName()) {
         TText* label = new TText(xStep*nLayers,
                                  yStep*(outnode+1),
                                  neuron->GetName());
         label->Draw();
      }
   }
}

void YMultiLayerPerceptron::DrawNonCrossing(Option_t * /*option*/)
{
#define NeuronSize 2.5

   Int_t nLayers = fStructure.CountChar(':')+1;
   Float_t xStep = 1./(nLayers+1.);
   Int_t layer;
   for(layer=0; layer< nLayers-1; layer++) {
      Float_t nNeurons_this = 0;
      if(layer==0) {
         TString input      = TString(fStructure(0, fStructure.First(':')));
         nNeurons_this = input.CountChar(',')+1;
      }
      else {
         Int_t cnt=0;
         TString hidden = TString(fStructure(fStructure.First(':') + 1,fStructure.Last(':') - fStructure.First(':') - 1));
         Int_t beg = 0;
         Int_t end = hidden.Index(":", beg + 1);
         while (end != -1) {
            Int_t num = atoi(TString(hidden(beg, end - beg)).Data());
            cnt++;
            beg = end + 1;
            end = hidden.Index(":", beg + 1);
            if(layer==cnt) nNeurons_this = num;
         }
         Int_t num = atoi(TString(hidden(beg, hidden.Length() - beg)).Data());
         cnt++;
         if(layer==cnt) nNeurons_this = num;
      }
      Float_t nNeurons_next = 0;
      if(layer==nLayers-2) {
         TString output = TString(fStructure(fStructure.Last(':') + 1,fStructure.Length() - fStructure.Last(':')));
         nNeurons_next = output.CountChar(',')+1;
      }
      else {
         Int_t cnt=0;
         TString hidden = TString(fStructure(fStructure.First(':') + 1,fStructure.Last(':') - fStructure.First(':') - 1));
         Int_t beg = 0;
         Int_t end = hidden.Index(":", beg + 1);
         while (end != -1) {
            Int_t num = atoi(TString(hidden(beg, end - beg)).Data());
            cnt++;
            beg = end + 1;
            end = hidden.Index(":", beg + 1);
            if(layer+1==cnt) nNeurons_next = num;
         }
         Int_t num = atoi(TString(hidden(beg, hidden.Length() - beg)).Data());
         cnt++;
         if(layer+1==cnt) nNeurons_next = num;
      }
      Float_t yStep_this = 1./(nNeurons_this+1.);
      Float_t yStep_next = 1./(nNeurons_next+1.);
      TObjArrayIter* it = (TObjArrayIter *) fSynapses.MakeIterator();
      YSynapse *theSynapse = 0;
      Float_t maxWeight = 0;
      while ((theSynapse = (YSynapse *) it->Next()))
         maxWeight = maxWeight < theSynapse->GetWeight() ? theSynapse->GetWeight() : maxWeight;
      delete it;
      it = (TObjArrayIter *) fSynapses.MakeIterator();

      Int_t nNeurons_this0 = nNeurons_this/3;
      Int_t nNeurons_this2 = nNeurons_this/3;
      Int_t nNeurons_this1 = nNeurons_this - (nNeurons_this0 + nNeurons_this2);
      Int_t nNeurons_next0 = nNeurons_next/3;
      Int_t nNeurons_next2 = nNeurons_next/3;
      Int_t nNeurons_next1 = nNeurons_next - (nNeurons_next0 + nNeurons_next2);    
      std::cout<<"DrawNonCrossing nNeurons::_this : "<<nNeurons_this<<" "<<nNeurons_this0<<" "<<nNeurons_this1<<" "<<nNeurons_this2<<std::endl;
      std::cout<<"DrawNonCrossing nNeurons::_next : "<<nNeurons_next<<" "<<nNeurons_next0<<" "<<nNeurons_next1<<" "<<nNeurons_next2<<std::endl;
      
      for(Int_t neuron1=0; neuron1<nNeurons_this; neuron1++) {
         if(neuron1>=0&&neuron1<nNeurons_this0){
            for(Int_t neuron2=0; neuron2<nNeurons_next0; neuron2++) {
               TLine* synapse = new TLine(xStep*(layer+1),yStep_this*(neuron1+1),xStep*(layer+2),yStep_next*(neuron2+1));
               synapse->Draw();
               theSynapse = (YSynapse *) it->Next();
               if (!theSynapse) continue;
               synapse->SetLineWidth(Int_t((theSynapse->GetWeight()/maxWeight)*10.));
               synapse->SetLineStyle(1);
               if(((TMath::Abs(theSynapse->GetWeight())/maxWeight)*10.)<0.5) synapse->SetLineStyle(2);
               if(((TMath::Abs(theSynapse->GetWeight())/maxWeight)*10.)<0.25) synapse->SetLineStyle(3);
            }     
         } else if(neuron1>=nNeurons_this0&&neuron1<nNeurons_this0+nNeurons_this1){
            for(Int_t neuron2=nNeurons_next0; neuron2<nNeurons_next0+nNeurons_next1; neuron2++) {
               TLine* synapse = new TLine(xStep*(layer+1),yStep_this*(neuron1+1),xStep*(layer+2),yStep_next*(neuron2+1));
               synapse->Draw();
               theSynapse = (YSynapse *) it->Next();
               if (!theSynapse) continue;
               synapse->SetLineWidth(Int_t((theSynapse->GetWeight()/maxWeight)*10.));
               synapse->SetLineStyle(1);
               if(((TMath::Abs(theSynapse->GetWeight())/maxWeight)*10.)<0.5) synapse->SetLineStyle(2);
               if(((TMath::Abs(theSynapse->GetWeight())/maxWeight)*10.)<0.25) synapse->SetLineStyle(3);
            }  
         } else if(neuron1>=nNeurons_this0+nNeurons_this1&&neuron1<nNeurons_this0+nNeurons_this1+nNeurons_this2){
            for(Int_t neuron2=nNeurons_next0+nNeurons_next1; neuron2<nNeurons_next0+nNeurons_next1+nNeurons_next2; neuron2++) {
               TLine* synapse = new TLine(xStep*(layer+1),yStep_this*(neuron1+1),xStep*(layer+2),yStep_next*(neuron2+1));
               synapse->Draw();
               theSynapse = (YSynapse *) it->Next();
               if (!theSynapse) continue;
               synapse->SetLineWidth(Int_t((theSynapse->GetWeight()/maxWeight)*10.));
               synapse->SetLineStyle(1);
               if(((TMath::Abs(theSynapse->GetWeight())/maxWeight)*10.)<0.5) synapse->SetLineStyle(2);
               if(((TMath::Abs(theSynapse->GetWeight())/maxWeight)*10.)<0.25) synapse->SetLineStyle(3);
            }       
         }
      }
      delete it;
   }
   for(layer=0; layer< nLayers; layer++) {
      Float_t nNeurons = 0;
      if(layer==0) {
         TString input      = TString(fStructure(0, fStructure.First(':')));
         nNeurons = input.CountChar(',')+1;
      }
      else if(layer==nLayers-1) {
         TString output = TString(fStructure(fStructure.Last(':') + 1,fStructure.Length() - fStructure.Last(':')));
         nNeurons = output.CountChar(',')+1;
      }
      else {
         Int_t cnt=0;
         TString hidden = TString(fStructure(fStructure.First(':') + 1,fStructure.Last(':') - fStructure.First(':') - 1));
         Int_t beg = 0;
         Int_t end = hidden.Index(":", beg + 1);
         while (end != -1) {
            Int_t num = atoi(TString(hidden(beg, end - beg)).Data());
            cnt++;
            beg = end + 1;
            end = hidden.Index(":", beg + 1);
            if(layer==cnt) nNeurons = num;
         }
         Int_t num = atoi(TString(hidden(beg, hidden.Length() - beg)).Data());
         cnt++;
         if(layer==cnt) nNeurons = num;
      }
      Float_t yStep = 1./(nNeurons+1.);
      for(Int_t neuron=0; neuron<nNeurons; neuron++) {
         TMarker* m = new TMarker(xStep*(layer+1),yStep*(neuron+1),20);
         m->SetMarkerColor(4);
         m->SetMarkerSize(NeuronSize);
         m->Draw();
      }
   }
   const TString input = TString(fStructure(0, fStructure.First(':')));
   const TObjArray *inpL = input.Tokenize(" ,");
   const Int_t nrItems = inpL->GetLast()+1;
   Float_t yStep = 1./(nrItems+1);
   std::cout<<"DrawNonCrossing nrItems : "<<nrItems<<std::endl;
   for (Int_t item = 0; item < nrItems; item++) {
      const TString brName = ((TObjString *)inpL->At(item))->GetString();
      TText* label = new TText(0.5*xStep,yStep*(item+1),brName.Data());
      std::cout<<item<<" "<<brName<<std::endl;      
      label->Draw();
   }
   delete inpL;

   const TString output = TString(fStructure(fStructure.Last(':') + 1,fStructure.Length() - fStructure.Last(':')));
   const TObjArray *outpL = output.Tokenize(" ,");

   Int_t numOutNodes=fLastLayer.GetEntriesFast();
   yStep=1./(numOutNodes+1);
   std::cout<<"DrawNonCrossing numOutNodes : "<<numOutNodes<<std::endl;
   for (Int_t outnode=0; outnode<numOutNodes; outnode++) {
      const TString brName = ((TObjString *)outpL->At(outnode))->GetString();
      std::cout<<outnode<<" "<<brName<<std::endl;

      TText* label = new TText(xStep*nLayers,
                                  yStep*(outnode+1),
                                  brName.Data());
         
      label->Draw();
      
   }
   delete outpL;
}

void YMultiLayerPerceptron::InitEventLoss(){

   fEventLoss = 0;

}

void YMultiLayerPerceptron::AddEventLoss(){
   fEventLoss++;
} 


void YMultiLayerPerceptron::SetEventLoss(int type){ //0 : total, 1 : training, 2 : test
   if (type==0) { 
      fTotNEventsLoss   = fEventLoss;
   } else if (type==1) {
      fTotNTrainingLoss = fEventLoss;
   } else if (type==2) {
      fTotNTestLoss     = fEventLoss;
   } else {

   }
   fEventLoss = 0;
}

int YMultiLayerPerceptron::GetEventLoss(int type){ //0 : total, 1 : training, 2 : test

   if (type==0) { 
      return fTotNEventsLoss;
   } else if (type==1) {
      return fTotNTrainingLoss;
   } else if (type==2) {
      return fTotNTestLoss;
   } else {
      return 0;
   }

}

void YMultiLayerPerceptron::InitTrackLoss(){

   fTrackLoss = 0;

}

void YMultiLayerPerceptron::AddTrackLoss(){
   fTrackLoss++;
} 


void YMultiLayerPerceptron::SetTrackLoss(int type){ //0 : total, 1 : training, 2 : test
   if (type==0) { 
      fTotNTracksLoss   = fTrackLoss;
   } else if (type==1) {
      fTotNTrainingTrackLoss = fTrackLoss;
   } else if (type==2) {
      fTotNTestTrackLoss     = fTrackLoss;
   } else {

   }
   fTrackLoss = 0;
}

int YMultiLayerPerceptron::GetTrackLoss(int type){ //0 : total, 1 : training, 2 : test

   if (type==0) { 
      return fTotNTracksLoss;
   } else if (type==1) {
      return fTotNTrainingTrackLoss;
   } else if (type==2) {
      return fTotNTestTrackLoss;
   } else {
      return 0;
   }

}

////////////////////////////////////////////////////////////////////////////////
/// SetSplitRefernceSensor

void YMultiLayerPerceptron::SetSplitReferenceSensor(int layer, int chipIDinlayer)
{
   if(layer>=0 && layer < nLAYER) {
      fSplitReferenceSensor = ChipBoundary[layer] + chipIDinlayer;
   } else {
      fSplitReferenceSensor = -1;
   }
}

