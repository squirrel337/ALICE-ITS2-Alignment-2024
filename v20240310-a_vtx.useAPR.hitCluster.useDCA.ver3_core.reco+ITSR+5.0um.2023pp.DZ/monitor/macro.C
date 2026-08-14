#include "data_simul.h"

#define nLAYERIB 3
#define nLAYEROB 4
#define nLAYER 7

#define nPSTAVE 20
#define nQUAD 4

#define TEXTSIZE 0.05
#define LATEXSIZE 0.06
const int NSubStave[nLAYER] = { 1, 1, 1, 2, 2, 2, 2 };
const int NStaves[nLAYER] = { 12, 16, 20, 24, 30, 42, 48 };
const int nHicPerStave[nLAYER] = { 1, 1, 1, 8, 8, 14, 14 };
const int nChipsPerHic[nLAYER] = { 9, 9, 9, 14, 14, 14, 14 };
const int ChipBoundary[nLAYER + 1] = { 0, 108, 252, 432, 3120, 6480, 14712, 24120 };
const int StaveBoundary[nLAYER + 1] = { 0, 12, 28, 48, 72, 102, 144, 192 };

int nSensorsbyLayer[nLAYER]= {	ChipBoundary[1] - ChipBoundary[0],
				ChipBoundary[2] - ChipBoundary[1],
				ChipBoundary[3] - ChipBoundary[2],
				ChipBoundary[4] - ChipBoundary[3],
				ChipBoundary[5] - ChipBoundary[4],
				ChipBoundary[6] - ChipBoundary[5],
				ChipBoundary[7] - ChipBoundary[6]};

void macro_plot_impact_parameter_z_2D(double rangeX = 0.06, double momCut = 0.0, int MagFieldSign = +1, bool chargePlot=false){

   int nMonitors = 2;
   TString tFormula[nMonitors];
   tFormula[0] = "fip[0]:vtxZ";
   tFormula[1] = "fip[1]:vtxZ"; 

   TH2D* hpTvsIP[2][3];
   TProfile* hpfIP[2][3];
   
   hpfIP[0][0] = new TProfile("hpfIPD",    "Impact Parameter D vs z; z(cm); D(cm)",30,-15,15,-rangeX,rangeX);
   hpfIP[0][1] = new TProfile("hpfIPD_pos","Impact Parameter D vs z; z(cm); D(cm)",30,-15,15,-rangeX,rangeX);
   hpfIP[0][2] = new TProfile("hpfIPD_neg","Impact Parameter D vs z; z(cm); D(cm)",30,-15,15,-rangeX,rangeX);
   hpfIP[1][0] = new TProfile("hpfIPZ",    "Impact Parameter Z vs z; z(cm); Z(cm)",30,-15,15,-rangeX,rangeX);
   hpfIP[1][1] = new TProfile("hpfIPZ_pos","Impact Parameter Z vs z; z(cm); Z(cm)",30,-15,15,-rangeX,rangeX);
   hpfIP[1][2] = new TProfile("hpfIPZ_neg","Impact Parameter Z vs z; z(cm); Z(cm)",30,-15,15,-rangeX,rangeX);
       
   ResMonitor->Draw(Form("%s>>hpfIPD",    (const char*)tFormula[0]), Form("TMath::Abs(fip[0])<%g&&pT>=%g",rangeX,momCut),"goff");   
   ResMonitor->Draw(Form("%s>>hpfIPD_pos",(const char*)tFormula[0]), Form("TMath::Abs(fip[0])<%g&&pT>=%g&&(%d*cuvR<0)",rangeX,momCut,MagFieldSign),"goff"); 
   ResMonitor->Draw(Form("%s>>hpfIPD_neg",(const char*)tFormula[0]), Form("TMath::Abs(fip[0])<%g&&pT>=%g&&(%d*cuvR>0)",rangeX,momCut,MagFieldSign),"goff"); 
   ResMonitor->Draw(Form("%s>>hpfIPZ",    (const char*)tFormula[1]), Form("TMath::Abs(fip[1])<%g&&pT>=%g",rangeX,momCut),"goff"); 
   ResMonitor->Draw(Form("%s>>hpfIPZ_pos",(const char*)tFormula[1]), Form("TMath::Abs(fip[1])<%g&&pT>=%g&&(%d*cuvR<0)",rangeX,momCut,MagFieldSign),"goff"); 
   ResMonitor->Draw(Form("%s>>hpfIPZ_neg",(const char*)tFormula[1]), Form("TMath::Abs(fip[1])<%g&&pT>=%g&&(%d*cuvR>0)",rangeX,momCut,MagFieldSign),"goff"); 
   
   double pT_max = 2.0;
   int nBins_pT = (int)(pT_max/0.1);
   hpTvsIP[0][0] = new TH2D("hpTvsIPD",    "p_{T} vs Impact Parameter D; D(cm); p_{T}(GeV)",2*100,-2*rangeX,2*rangeX,nBins_pT,0,pT_max);
   hpTvsIP[0][1] = new TH2D("hpTvsIPD_pos","p_{T} vs Impact Parameter D; D(cm); p_{T}(GeV)",2*100,-2*rangeX,2*rangeX,nBins_pT,0,pT_max);
   hpTvsIP[0][2] = new TH2D("hpTvsIPD_neg","p_{T} vs Impact Parameter D; D(cm); p_{T}(GeV)",2*100,-2*rangeX,2*rangeX,nBins_pT,0,pT_max);
   hpTvsIP[1][0] = new TH2D("hpTvsIPZ",    "p_{T} vs Impact Parameter D; Z(cm); p_{T}(GeV)",2*100,-2*rangeX,2*rangeX,nBins_pT,0,pT_max);
   hpTvsIP[1][1] = new TH2D("hpTvsIPZ_pos","p_{T} vs Impact Parameter D; Z(cm); p_{T}(GeV)",2*100,-2*rangeX,2*rangeX,nBins_pT,0,pT_max);
   hpTvsIP[1][2] = new TH2D("hpTvsIPZ_neg","p_{T} vs Impact Parameter D; Z(cm); p_{T}(GeV)",2*100,-2*rangeX,2*rangeX,nBins_pT,0,pT_max);

   ResMonitor->Draw(Form("pT:fip[0]>>hpTvsIPD"),     Form("TMath::Abs(fip[0])<%g&&pT>=%g&&pT<%g",rangeX,momCut,pT_max),"goff");   
   ResMonitor->Draw(Form("pT:fip[0]>>hpTvsIPD_pos"), Form("TMath::Abs(fip[0])<%g&&pT>=%g&&pT<%g&&(%d*cuvR<0)",rangeX,momCut,pT_max,MagFieldSign),"goff"); 
   ResMonitor->Draw(Form("pT:fip[0]>>hpTvsIPD_neg"), Form("TMath::Abs(fip[0])<%g&&pT>=%g&&pT<%g&&(%d*cuvR>0)",rangeX,momCut,pT_max,MagFieldSign),"goff"); 
   ResMonitor->Draw(Form("pT:fip[1]>>hpTvsIPZ"),     Form("TMath::Abs(fip[1])<%g&&pT>=%g&&pT<%g",rangeX,momCut,pT_max),"goff"); 
   ResMonitor->Draw(Form("pT:fip[1]>>hpTvsIPZ_pos"), Form("TMath::Abs(fip[1])<%g&&pT>=%g&&pT<%g&&(%d*cuvR<0)",rangeX,momCut,pT_max,MagFieldSign),"goff"); 
   ResMonitor->Draw(Form("pT:fip[1]>>hpTvsIPZ_neg"), Form("TMath::Abs(fip[1])<%g&&pT>=%g&&pT<%g&&(%d*cuvR>0)",rangeX,momCut,pT_max,MagFieldSign),"goff"); 
   
   TF1* fit_gaus = new TF1("fit_gaus","gaus");   
   TH1D* hIP_sigma_vs_pT[2][3];
   hIP_sigma_vs_pT[0][0] = new TH1D(Form("hIPD_sigma_vs_pT"),    Form("#sigma(IP D) vs p_{T} ; p_{T}(GeV); #sigma(IP D)(cm)"),nBins_pT,0,pT_max);
   hIP_sigma_vs_pT[0][1] = new TH1D(Form("hIPD_pos_sigma_vs_pT"),Form("#sigma(IP D) vs p_{T} ; p_{T}(GeV); #sigma(IP D)(cm)"),nBins_pT,0,pT_max);
   hIP_sigma_vs_pT[0][2] = new TH1D(Form("hIPD_neg_sigma_vs_pT"),Form("#sigma(IP D) vs p_{T} ; p_{T}(GeV); #sigma(IP D)(cm)"),nBins_pT,0,pT_max);
   
   hIP_sigma_vs_pT[1][0] = new TH1D(Form("hIPZ_sigma_vs_pT"),    Form("#sigma(IP Z) vs p_{T} ; p_{T}(GeV); #sigma(IP Z)(cm)"),nBins_pT,0,pT_max);  
   hIP_sigma_vs_pT[1][1] = new TH1D(Form("hIPZ_pos_sigma_vs_pT"),Form("#sigma(IP Z) vs p_{T} ; p_{T}(GeV); #sigma(IP Z)(cm)"),nBins_pT,0,pT_max);
   hIP_sigma_vs_pT[1][2] = new TH1D(Form("hIPZ_neg_sigma_vs_pT"),Form("#sigma(IP Z) vs p_{T} ; p_{T}(GeV); #sigma(IP Z)(cm)"),nBins_pT,0,pT_max);
   
   for(int axis = 0; axis < 2 ; axis++){   
      for(int pbin = 0; pbin < (int)(nBins_pT); pbin++){
         if(pbin<1) continue;
         bool switchComponent[] = {false, false, false};
         double sigma[] = {0, 0, 0};
         double sigmaError[] {0, 0, 0};
         if(hpTvsIP[axis][0]->ProjectionX("",pbin+1,pbin+1)->GetEntries()>50) {
            switchComponent[0] = true;
            hpTvsIP[axis][0]->ProjectionX("",pbin+1,pbin+1)->Fit(fit_gaus);
            sigma[0]      = fit_gaus->GetParameter(2); 
            sigmaError[0] = fit_gaus->GetParError(2); 
         }
         if(hpTvsIP[axis][1]->ProjectionX("",pbin+1,pbin+1)->GetEntries()>50) {
            switchComponent[0] = true;
            hpTvsIP[axis][1]->ProjectionX("",pbin+1,pbin+1)->Fit(fit_gaus);
            sigma[1]      = fit_gaus->GetParameter(2); 
            sigmaError[1] = fit_gaus->GetParError(2);             
         }
         if(hpTvsIP[axis][2]->ProjectionX("",pbin+1,pbin+1)->GetEntries()>50) {
            switchComponent[2] = true;
            hpTvsIP[axis][2]->ProjectionX("",pbin+1,pbin+1)->Fit(fit_gaus);
            sigma[2] = fit_gaus->GetParameter(2); 
            sigmaError[2] = fit_gaus->GetParError(2);                         
         }        
         
         if(chargePlot==false) {
            if(switchComponent[0]==true){
               hIP_sigma_vs_pT[axis][0]->SetBinContent(pbin+1, sigma[0]); 
               hIP_sigma_vs_pT[axis][0]->SetBinError(pbin+1, sigmaError[0]); 
            }
         } else {
            if(switchComponent[0]==true){
               hIP_sigma_vs_pT[axis][0]->SetBinContent(pbin+1, sigma[0]); 
               //hIP_sigma_vs_pT[axis][1]->SetBinContent(pbin+1, sigma[1]); 
               //hIP_sigma_vs_pT[axis][2]->SetBinContent(pbin+1, sigma[2]);   
               hIP_sigma_vs_pT[axis][0]->SetBinError(pbin+1, sigmaError[0]); 
               //hIP_sigma_vs_pT[axis][1]->SetBinError(pbin+1, sigmaError[1]); 
               //hIP_sigma_vs_pT[axis][2]->SetBinError(pbin+1, sigmaError[2]);            
               
                                
            }  
         }
      }
   }
   
   hpfIP[0][0]->SetMarkerColor(kBlack);   hpfIP[0][0]->SetLineColor(kBlack);
   hpfIP[0][1]->SetMarkerColor(kRed);     hpfIP[0][1]->SetLineColor(kRed);
   hpfIP[0][2]->SetMarkerColor(kBlue);    hpfIP[0][2]->SetLineColor(kBlue);
   hpfIP[1][0]->SetMarkerColor(kBlack);   hpfIP[1][0]->SetLineColor(kBlack);
   hpfIP[1][1]->SetMarkerColor(kRed);     hpfIP[1][1]->SetLineColor(kRed);
   hpfIP[1][2]->SetMarkerColor(kBlue);    hpfIP[1][2]->SetLineColor(kBlue);     

   hIP_sigma_vs_pT[0][0]->SetMarkerColor(kBlack);   hIP_sigma_vs_pT[0][0]->SetLineColor(kBlack);
   hIP_sigma_vs_pT[0][1]->SetMarkerColor(kRed);     hIP_sigma_vs_pT[0][1]->SetLineColor(kRed);
   hIP_sigma_vs_pT[0][2]->SetMarkerColor(kBlue);    hIP_sigma_vs_pT[0][2]->SetLineColor(kBlue);
   hIP_sigma_vs_pT[1][0]->SetMarkerColor(kBlack);   hIP_sigma_vs_pT[1][0]->SetLineColor(kBlack);
   hIP_sigma_vs_pT[1][1]->SetMarkerColor(kRed);     hIP_sigma_vs_pT[1][1]->SetLineColor(kRed);
   hIP_sigma_vs_pT[1][2]->SetMarkerColor(kBlue);    hIP_sigma_vs_pT[1][2]->SetLineColor(kBlue);    

   
   TCanvas* canvas_vtall = new TCanvas("canvas_vtall","canvas_vtall",2*800,nMonitors*500);
   canvas_vtall->Divide((int)2,(int)nMonitors,0.001,0.001);  
   
   canvas_vtall->cd(1);
   canvas_vtall->cd(1)->SetGridx();
   canvas_vtall->cd(1)->SetGridy();
   hpfIP[0][0]->GetYaxis()->SetRangeUser(-rangeX,rangeX);   
   hpfIP[0][0]->Draw("");
   if(chargePlot==true) {
      hpfIP[0][1]->Draw("same");
      hpfIP[0][2]->Draw("same");   
   }

   canvas_vtall->cd(2);
   canvas_vtall->cd(2)->SetGridx();
   canvas_vtall->cd(2)->SetGridy();   
   hpfIP[1][0]->GetYaxis()->SetRangeUser(-rangeX,rangeX);      
   hpfIP[1][0]->Draw("");
   if(chargePlot==true) {
      hpfIP[1][1]->Draw("same");
      hpfIP[1][2]->Draw("same");   
   }   
   
   TF1* f1_IPD = new TF1("f1_IPD","TMath::Sqrt([0]*[0] + (([1]*[1])/(x*x)))",0.1,pT_max);
   canvas_vtall->cd(3);
   hIP_sigma_vs_pT[0][0]->GetYaxis()->SetRangeUser(0,rangeX);  
   
   hIP_sigma_vs_pT[0][0]->Fit("f1_IPD","rQ");  
   gStyle->SetOptFit(111111);
   hIP_sigma_vs_pT[0][0]->Draw("");
   if(chargePlot==true) {
      //hIP_sigma_vs_pT[0][1]->Draw("same");
      //hIP_sigma_vs_pT[0][2]->Draw("same");   
   }

   TF1* f1_IPZ = new TF1("f1_IPZ","TMath::Sqrt([0]*[0] + (([1]*[1])/(x*x)))",0.1,pT_max);
   canvas_vtall->cd(4);
   hIP_sigma_vs_pT[1][0]->GetYaxis()->SetRangeUser(0,rangeX);   
   hIP_sigma_vs_pT[1][0]->Fit("f1_IPZ","rQ");  
   gStyle->SetOptFit(111111);         
   hIP_sigma_vs_pT[1][0]->Draw("");
   if(chargePlot==true) {
      //hIP_sigma_vs_pT[1][1]->Draw("same");
      //hIP_sigma_vs_pT[1][2]->Draw("same");   
   }      
   canvas_vtall->SaveAs(Form("Plot_ImpactParameterVsZ_allpTCut%g.gif",momCut),"gif");
}

void macro_plot_impact_parameter_phi_2D(double rangeX = 0.06, double momCut = 0.0, int MagFieldSign = +1, bool chargePlot=false){

   int nMonitors = 2;
   TString tFormula[nMonitors];
   tFormula[0] = "fip[0]:phi";
   tFormula[1] = "fip[1]:phi"; 

   TH2D* hpTvsIP[2][3];
   TProfile* hpfIP[2][3];
   
   hpfIP[0][0] = new TProfile("hpfIPD",    "Impact Parameter D vs #phi; #phi(rad); D(cm)",20,-TMath::Pi(),TMath::Pi(),-rangeX,rangeX);
   hpfIP[0][1] = new TProfile("hpfIPD_pos","Impact Parameter D vs #phi; #phi(rad); D(cm)",20,-TMath::Pi(),TMath::Pi(),-rangeX,rangeX);
   hpfIP[0][2] = new TProfile("hpfIPD_neg","Impact Parameter D vs #phi; #phi(rad); D(cm)",20,-TMath::Pi(),TMath::Pi(),-rangeX,rangeX);
   hpfIP[1][0] = new TProfile("hpfIPZ",    "Impact Parameter Z vs #phi; #phi(rad); Z(cm)",20,-TMath::Pi(),TMath::Pi(),-rangeX,rangeX);
   hpfIP[1][1] = new TProfile("hpfIPZ_pos","Impact Parameter Z vs #phi; #phi(rad); Z(cm)",20,-TMath::Pi(),TMath::Pi(),-rangeX,rangeX);
   hpfIP[1][2] = new TProfile("hpfIPZ_neg","Impact Parameter Z vs #phi; #phi(rad); Z(cm)",20,-TMath::Pi(),TMath::Pi(),-rangeX,rangeX);
       
   ResMonitor->Draw(Form("%s>>hpfIPD",    (const char*)tFormula[0]), Form("TMath::Abs(fip[0])<%g&&pT>=%g",rangeX,momCut),"goff");   
   ResMonitor->Draw(Form("%s>>hpfIPD_pos",(const char*)tFormula[0]), Form("TMath::Abs(fip[0])<%g&&pT>=%g&&(%d*cuvR<0)",rangeX,momCut,MagFieldSign),"goff"); 
   ResMonitor->Draw(Form("%s>>hpfIPD_neg",(const char*)tFormula[0]), Form("TMath::Abs(fip[0])<%g&&pT>=%g&&(%d*cuvR>0)",rangeX,momCut,MagFieldSign),"goff"); 
   ResMonitor->Draw(Form("%s>>hpfIPZ",    (const char*)tFormula[1]), Form("TMath::Abs(fip[1])<%g&&pT>=%g",rangeX,momCut),"goff"); 
   ResMonitor->Draw(Form("%s>>hpfIPZ_pos",(const char*)tFormula[1]), Form("TMath::Abs(fip[1])<%g&&pT>=%g&&(%d*cuvR<0)",rangeX,momCut,MagFieldSign),"goff"); 
   ResMonitor->Draw(Form("%s>>hpfIPZ_neg",(const char*)tFormula[1]), Form("TMath::Abs(fip[1])<%g&&pT>=%g&&(%d*cuvR>0)",rangeX,momCut,MagFieldSign),"goff"); 
   
   double pT_max = 2.0;
   int nBins_pT = (int)(pT_max/0.1);
   hpTvsIP[0][0] = new TH2D("hpTvsIPD",    "p_{T} vs Impact Parameter D; D(cm); p_{T}(GeV)",2*100,-2*rangeX,2*rangeX,nBins_pT,0,pT_max);
   hpTvsIP[0][1] = new TH2D("hpTvsIPD_pos","p_{T} vs Impact Parameter D; D(cm); p_{T}(GeV)",2*100,-2*rangeX,2*rangeX,nBins_pT,0,pT_max);
   hpTvsIP[0][2] = new TH2D("hpTvsIPD_neg","p_{T} vs Impact Parameter D; D(cm); p_{T}(GeV)",2*100,-2*rangeX,2*rangeX,nBins_pT,0,pT_max);
   hpTvsIP[1][0] = new TH2D("hpTvsIPZ",    "p_{T} vs Impact Parameter D; Z(cm); p_{T}(GeV)",2*100,-2*rangeX,2*rangeX,nBins_pT,0,pT_max);
   hpTvsIP[1][1] = new TH2D("hpTvsIPZ_pos","p_{T} vs Impact Parameter D; Z(cm); p_{T}(GeV)",2*100,-2*rangeX,2*rangeX,nBins_pT,0,pT_max);
   hpTvsIP[1][2] = new TH2D("hpTvsIPZ_neg","p_{T} vs Impact Parameter D; Z(cm); p_{T}(GeV)",2*100,-2*rangeX,2*rangeX,nBins_pT,0,pT_max);

   ResMonitor->Draw(Form("pT:fip[0]>>hpTvsIPD"),     Form("TMath::Abs(fip[0])<%g&&pT>=%g&&pT<%g",rangeX,momCut,pT_max),"goff");   
   ResMonitor->Draw(Form("pT:fip[0]>>hpTvsIPD_pos"), Form("TMath::Abs(fip[0])<%g&&pT>=%g&&pT<%g&&(%d*cuvR<0)",rangeX,momCut,pT_max,MagFieldSign),"goff"); 
   ResMonitor->Draw(Form("pT:fip[0]>>hpTvsIPD_neg"), Form("TMath::Abs(fip[0])<%g&&pT>=%g&&pT<%g&&(%d*cuvR>0)",rangeX,momCut,pT_max,MagFieldSign),"goff"); 
   ResMonitor->Draw(Form("pT:fip[1]>>hpTvsIPZ"),     Form("TMath::Abs(fip[1])<%g&&pT>=%g&&pT<%g",rangeX,momCut,pT_max),"goff"); 
   ResMonitor->Draw(Form("pT:fip[1]>>hpTvsIPZ_pos"), Form("TMath::Abs(fip[1])<%g&&pT>=%g&&pT<%g&&(%d*cuvR<0)",rangeX,momCut,pT_max,MagFieldSign),"goff"); 
   ResMonitor->Draw(Form("pT:fip[1]>>hpTvsIPZ_neg"), Form("TMath::Abs(fip[1])<%g&&pT>=%g&&pT<%g&&(%d*cuvR>0)",rangeX,momCut,pT_max,MagFieldSign),"goff"); 
   
   TF1* fit_gaus = new TF1("fit_gaus","gaus");   
   TH1D* hIP_sigma_vs_pT[2][3];
   hIP_sigma_vs_pT[0][0] = new TH1D(Form("hIPD_sigma_vs_pT"),    Form("#sigma(IP D) vs p_{T} ; p_{T}(GeV); #sigma(IP D)(cm)"),nBins_pT,0,pT_max);
   hIP_sigma_vs_pT[0][1] = new TH1D(Form("hIPD_pos_sigma_vs_pT"),Form("#sigma(IP D) vs p_{T} ; p_{T}(GeV); #sigma(IP D)(cm)"),nBins_pT,0,pT_max);
   hIP_sigma_vs_pT[0][2] = new TH1D(Form("hIPD_neg_sigma_vs_pT"),Form("#sigma(IP D) vs p_{T} ; p_{T}(GeV); #sigma(IP D)(cm)"),nBins_pT,0,pT_max);
   
   hIP_sigma_vs_pT[1][0] = new TH1D(Form("hIPZ_sigma_vs_pT"),    Form("#sigma(IP Z) vs p_{T} ; p_{T}(GeV); #sigma(IP Z)(cm)"),nBins_pT,0,pT_max);  
   hIP_sigma_vs_pT[1][1] = new TH1D(Form("hIPZ_pos_sigma_vs_pT"),Form("#sigma(IP Z) vs p_{T} ; p_{T}(GeV); #sigma(IP Z)(cm)"),nBins_pT,0,pT_max);
   hIP_sigma_vs_pT[1][2] = new TH1D(Form("hIPZ_neg_sigma_vs_pT"),Form("#sigma(IP Z) vs p_{T} ; p_{T}(GeV); #sigma(IP Z)(cm)"),nBins_pT,0,pT_max);
   
   for(int axis = 0; axis < 2 ; axis++){   
      for(int pbin = 0; pbin < (int)(nBins_pT); pbin++){
         if(pbin<1) continue;
         bool switchComponent[] = {false, false, false};
         double sigma[] = {0, 0, 0};
         double sigmaError[] {0, 0, 0};
         if(hpTvsIP[axis][0]->ProjectionX("",pbin+1,pbin+1)->GetEntries()>50) {
            switchComponent[0] = true;
            hpTvsIP[axis][0]->ProjectionX("",pbin+1,pbin+1)->Fit(fit_gaus);
            sigma[0]      = fit_gaus->GetParameter(2); 
            sigmaError[0] = fit_gaus->GetParError(2); 
         }
         if(hpTvsIP[axis][1]->ProjectionX("",pbin+1,pbin+1)->GetEntries()>50) {
            switchComponent[0] = true;
            hpTvsIP[axis][1]->ProjectionX("",pbin+1,pbin+1)->Fit(fit_gaus);
            sigma[1]      = fit_gaus->GetParameter(2); 
            sigmaError[1] = fit_gaus->GetParError(2);             
         }
         if(hpTvsIP[axis][2]->ProjectionX("",pbin+1,pbin+1)->GetEntries()>50) {
            switchComponent[2] = true;
            hpTvsIP[axis][2]->ProjectionX("",pbin+1,pbin+1)->Fit(fit_gaus);
            sigma[2] = fit_gaus->GetParameter(2); 
            sigmaError[2] = fit_gaus->GetParError(2);                         
         }        
         
         if(chargePlot==false) {
            if(switchComponent[0]==true){
               hIP_sigma_vs_pT[axis][0]->SetBinContent(pbin+1, sigma[0]); 
               hIP_sigma_vs_pT[axis][0]->SetBinError(pbin+1, sigmaError[0]); 
            }
         } else {
            if(switchComponent[0]==true){
               hIP_sigma_vs_pT[axis][0]->SetBinContent(pbin+1, sigma[0]); 
               //hIP_sigma_vs_pT[axis][1]->SetBinContent(pbin+1, sigma[1]); 
               //hIP_sigma_vs_pT[axis][2]->SetBinContent(pbin+1, sigma[2]);   
               hIP_sigma_vs_pT[axis][0]->SetBinError(pbin+1, sigmaError[0]); 
               //hIP_sigma_vs_pT[axis][1]->SetBinError(pbin+1, sigmaError[1]); 
               //hIP_sigma_vs_pT[axis][2]->SetBinError(pbin+1, sigmaError[2]);            
               
                                
            }  
         }
      }
   }
   
   hpfIP[0][0]->SetMarkerColor(kBlack);   hpfIP[0][0]->SetLineColor(kBlack);
   hpfIP[0][1]->SetMarkerColor(kRed);     hpfIP[0][1]->SetLineColor(kRed);
   hpfIP[0][2]->SetMarkerColor(kBlue);    hpfIP[0][2]->SetLineColor(kBlue);
   hpfIP[1][0]->SetMarkerColor(kBlack);   hpfIP[1][0]->SetLineColor(kBlack);
   hpfIP[1][1]->SetMarkerColor(kRed);     hpfIP[1][1]->SetLineColor(kRed);
   hpfIP[1][2]->SetMarkerColor(kBlue);    hpfIP[1][2]->SetLineColor(kBlue);     

   hIP_sigma_vs_pT[0][0]->SetMarkerColor(kBlack);   hIP_sigma_vs_pT[0][0]->SetLineColor(kBlack);
   hIP_sigma_vs_pT[0][1]->SetMarkerColor(kRed);     hIP_sigma_vs_pT[0][1]->SetLineColor(kRed);
   hIP_sigma_vs_pT[0][2]->SetMarkerColor(kBlue);    hIP_sigma_vs_pT[0][2]->SetLineColor(kBlue);
   hIP_sigma_vs_pT[1][0]->SetMarkerColor(kBlack);   hIP_sigma_vs_pT[1][0]->SetLineColor(kBlack);
   hIP_sigma_vs_pT[1][1]->SetMarkerColor(kRed);     hIP_sigma_vs_pT[1][1]->SetLineColor(kRed);
   hIP_sigma_vs_pT[1][2]->SetMarkerColor(kBlue);    hIP_sigma_vs_pT[1][2]->SetLineColor(kBlue);    

   
   TCanvas* canvas_vtall = new TCanvas("canvas_vtall","canvas_vtall",2*800,nMonitors*500);
   canvas_vtall->Divide((int)2,(int)nMonitors,0.001,0.001);  
   
   canvas_vtall->cd(1);
   canvas_vtall->cd(1)->SetGridx();
   canvas_vtall->cd(1)->SetGridy();
   hpfIP[0][0]->GetYaxis()->SetRangeUser(-rangeX,rangeX);   
   hpfIP[0][0]->Draw("");
   if(chargePlot==true) {
      hpfIP[0][1]->Draw("same");
      hpfIP[0][2]->Draw("same");   
   }

   canvas_vtall->cd(2);
   canvas_vtall->cd(2)->SetGridx();
   canvas_vtall->cd(2)->SetGridy();   
   hpfIP[1][0]->GetYaxis()->SetRangeUser(-rangeX,rangeX);      
   hpfIP[1][0]->Draw("");
   if(chargePlot==true) {
      hpfIP[1][1]->Draw("same");
      hpfIP[1][2]->Draw("same");   
   }   
   
   TF1* f1_IPD = new TF1("f1_IPD","TMath::Sqrt([0]*[0] + (([1]*[1])/(x*x)))",0.1,pT_max);
   canvas_vtall->cd(3);
   hIP_sigma_vs_pT[0][0]->GetYaxis()->SetRangeUser(0,rangeX);  
   
   hIP_sigma_vs_pT[0][0]->Fit("f1_IPD","rQ");  
   gStyle->SetOptFit(111111);
   hIP_sigma_vs_pT[0][0]->Draw("");
   if(chargePlot==true) {
      //hIP_sigma_vs_pT[0][1]->Draw("same");
      //hIP_sigma_vs_pT[0][2]->Draw("same");   
   }

   TF1* f1_IPZ = new TF1("f1_IPZ","TMath::Sqrt([0]*[0] + (([1]*[1])/(x*x)))",0.1,pT_max);
   canvas_vtall->cd(4);
   hIP_sigma_vs_pT[1][0]->GetYaxis()->SetRangeUser(0,rangeX);   
   hIP_sigma_vs_pT[1][0]->Fit("f1_IPZ","rQ");  
   gStyle->SetOptFit(111111);         
   hIP_sigma_vs_pT[1][0]->Draw("");
   if(chargePlot==true) {
      //hIP_sigma_vs_pT[1][1]->Draw("same");
      //hIP_sigma_vs_pT[1][2]->Draw("same");   
   }      
   canvas_vtall->SaveAs(Form("Plot_ImpactParameterVsPhi_allpTCut%g.gif",momCut),"gif");
}

void macro_plot_vtx_track_phi_summary(double rangeX = 0.1, TString ChargeOpt="", double momCut = 0.0){

   // DET // Quad1, 2, 3 vs Phi
   int nMonitors = 3;
   TString tFormula[nMonitors];
   //1. vtx(xy) x track(xy) at L2 in r   vs phi ; scalar product
   tFormula[0] = "(vtxfitX*fgX[2]+vtxfitY*fgY[2])/TMath::Sqrt(fgX[2]*fgX[2]+fgY[2]*fgY[2])";  
   //2. vtx(xy) x track(xy) at L2 in phi vs phi ; cross product
   tFormula[1] = "(vtxfitX*fgY[2]-vtxfitY*fgX[2])/TMath::Sqrt(fgX[2]*fgX[2]+fgY[2]*fgY[2])";
   //3. dvtxZ vs phi at L2
   tFormula[2] = "vtxfitZ-vtxZ";
   
   int max[nMonitors][nQUAD];

   double valueQUAD[2][nQUAD] = {{0, 0.5*TMath::Pi(), -1.0*TMath::Pi(), -0.5*TMath::Pi()},{0.5*TMath::Pi(), 1.0*TMath::Pi(), -0.5*TMath::Pi(), 0}};
   
   TH2D* hdxDET[nQUAD];
   TH1D* hdx[nMonitors][nQUAD];
   TF1*  fit[nMonitors][nQUAD];
   double fitpar[nMonitors][nQUAD][3]; // (entries, mean, sigma)
   TLatex *tlpar[nMonitors][nQUAD][4];  
  
   TCanvas* canvas_vtall = new TCanvas("canvas_vtall","canvas_vtall",5*600,nMonitors*500);
   canvas_vtall->Divide((int)5,(int)nMonitors,0.001,0.001);
  
   int f;
   f=0;
   for(int q = 0; q < 5; q++){

      int index = (nQUAD+1)*f + q + 1;

      canvas_vtall->cd(index);
      canvas_vtall->cd(index)->SetGridx();
      canvas_vtall->cd(index)->SetGridy();
      gStyle->SetOptStat(0);

      double range = 0.25*rangeX;
      double res = 0.25*0.002; 
      
      if(q==0){
         gStyle->SetOptStat(111111);      
         hdxDET[f] = new TH2D(Form("hdxVTRvsPhiQ%d",q),Form("Normalized Vertex_{xy} #bullet Track_{xy} vs Phi (All Quadrants); phi; dx"),600,-TMath::Pi(),TMath::Pi(),(int)2*(range/res),-range,+range);
         ResMonitor->Draw(Form("(%s):fgPhi[2]>>hdxVTRvsPhiQ%d",(const char*)tFormula[f], q),
                          Form("TMath::Abs(fgPhi[2])<10&&TMath::Abs(%s)<%g&&pT>=%g%s",
                               (const char*)tFormula[f],range,momCut,(const char*)ChargeOpt),"colz");    
                               
         hdxDET[f]->GetXaxis()->SetLabelSize(TEXTSIZE);
         hdxDET[f]->GetYaxis()->SetLabelSize(TEXTSIZE);
         hdxDET[f]->GetXaxis()->SetTitleSize(TEXTSIZE);
         hdxDET[f]->GetYaxis()->SetTitleSize(TEXTSIZE);
         hdxDET[f]->GetXaxis()->SetTitleOffset(1.0);         
         hdxDET[f]->GetYaxis()->SetTitleOffset(1.0);                        
      } else {

         hdx[f][q-1] = new TH1D(Form("hdxVTRQ%d",q),Form("Normalized Vertex_{xy} #bullet Track_{xy} (Quadrant%d); dx; N",q),(int)2*(range/res),-range,+range);
         ResMonitor->Draw(Form("(%s)>>hdxVTRQ%d",(const char*)tFormula[f], q),
                          Form("TMath::Abs(fgPhi[2])<10&&TMath::Abs(%s)<%g&&fgPhi[2]>%g&&fgPhi[2]<=%g&&pT>=%g%s",
                               (const char*)tFormula[f],range,valueQUAD[0][q-1],valueQUAD[1][q-1],momCut,(const char*)ChargeOpt));
         max[f][q-1] = hdx[f][q-1]->GetMaximum();                       
         fit[f][q-1] = new TF1(Form("fdxVTRQ%d",q),"gaus",-0.5*range,+0.5*range);

         hdx[f][q-1]->GetXaxis()->SetLabelSize(TEXTSIZE);
         hdx[f][q-1]->GetYaxis()->SetLabelSize(TEXTSIZE);
         hdx[f][q-1]->GetXaxis()->SetTitleSize(TEXTSIZE);
         hdx[f][q-1]->GetYaxis()->SetTitleSize(TEXTSIZE);
         hdx[f][q-1]->GetXaxis()->SetTitleOffset(1.0);         
         hdx[f][q-1]->GetYaxis()->SetTitleOffset(1.0);  

         hdx[f][q-1]->Fit(Form("fdxVTRQ%d",q));
         fitpar[f][q-1][0] = hdx[f][q-1]->GetEntries();
         fitpar[f][q-1][1] = fit[f][q-1]->GetParameter(1);
         fitpar[f][q-1][2] = fit[f][q-1]->GetParameter(2); 

         hdx[f][q-1]->SetMarkerColor(kBlack);         
         hdx[f][q-1]->SetLineColor(kBlack);           
         hdx[f][q-1]->Draw("");
         fit[f][q-1]->Draw("same");
         tlpar[f][q-1][0] = new TLatex(range*0.25, max[f][q-1]*0.9, Form("Entries : %d", (int)(fitpar[f][q-1][0])));
         tlpar[f][q-1][0]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][0]->SetTextColor(kBlack);
         tlpar[f][q-1][0]->Draw("same");
         tlpar[f][q-1][1] = new TLatex(range*0.25, max[f][q-1]*0.75, Form("mean : %4.4g", (fitpar[f][q-1][1])));
         tlpar[f][q-1][1]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][1]->SetTextColor(kBlack);
         tlpar[f][q-1][1]->Draw("same");
         tlpar[f][q-1][2] = new TLatex(range*0.25, max[f][q-1]*0.6, Form("stddev : %4.4g", (fitpar[f][q-1][2])));
         tlpar[f][q-1][2]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][2]->SetTextColor(kBlack);
         tlpar[f][q-1][2]->Draw("same");   
         tlpar[f][q-1][3] = new TLatex(range*0.25, max[f][q-1]*0.45, Form("stddev : %4.4g", (data_simul_vtx_track_vs_phi[f][q-1])));
         tlpar[f][q-1][3]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][3]->SetTextColor(kBlue);
         tlpar[f][q-1][3]->Draw("same");                  
      }
   }
   
   f=1;
   for(int q = 0; q < 5; q++){

      int index = (nQUAD+1)*f + q + 1;

      canvas_vtall->cd(index);
      canvas_vtall->cd(index)->SetGridx();
      canvas_vtall->cd(index)->SetGridy();
      gStyle->SetOptStat(0);

      double range = rangeX;
      double res = 0.002;  
      
      if(q==0){
         gStyle->SetOptStat(111111);      
         hdxDET[f] = new TH2D(Form("hdxVTPhivsPhiQ%d",q),Form("Normalized Vertex_{xy} #times Track_{xy} vs Phi (All Quadrants); phi; dx"),600,-TMath::Pi(),TMath::Pi(),(int)2*(range/res),-range,+range);
         ResMonitor->Draw(Form("(%s):fgPhi[2]>>hdxVTPhivsPhiQ%d",(const char*)tFormula[f], q),
                          Form("TMath::Abs(fgPhi[2])<10&&TMath::Abs(%s)<%g&&pT>=%g%s",
                               (const char*)tFormula[f],range,momCut,(const char*)ChargeOpt),"colz");    
                               
         hdxDET[f]->GetXaxis()->SetLabelSize(TEXTSIZE);
         hdxDET[f]->GetYaxis()->SetLabelSize(TEXTSIZE);
         hdxDET[f]->GetXaxis()->SetTitleSize(TEXTSIZE);
         hdxDET[f]->GetYaxis()->SetTitleSize(TEXTSIZE);
         hdxDET[f]->GetXaxis()->SetTitleOffset(1.0);         
         hdxDET[f]->GetYaxis()->SetTitleOffset(1.0);                        
      } else {

         hdx[f][q-1] = new TH1D(Form("hdxVTPhiQ%d",q),Form("Normalized Vertex_{xy} #times Track_{xy} (Quadrant%d); dx; N",q),(int)2*(range/res),-range,+range);
         ResMonitor->Draw(Form("(%s)>>hdxVTPhiQ%d",(const char*)tFormula[f], q),
                          Form("TMath::Abs(fgPhi[2])<10&&TMath::Abs(%s)<%g&&fgPhi[2]>%g&&fgPhi[2]<=%g&&pT>=%g%s",
                               (const char*)tFormula[f],range,valueQUAD[0][q-1],valueQUAD[1][q-1],momCut,(const char*)ChargeOpt));
         max[f][q-1] = hdx[f][q-1]->GetMaximum();                       
         fit[f][q-1] = new TF1(Form("fdxVTPhiQ%d",q),"gaus",-0.5*range,+0.5*range);

         hdx[f][q-1]->GetXaxis()->SetLabelSize(TEXTSIZE);
         hdx[f][q-1]->GetYaxis()->SetLabelSize(TEXTSIZE);
         hdx[f][q-1]->GetXaxis()->SetTitleSize(TEXTSIZE);
         hdx[f][q-1]->GetYaxis()->SetTitleSize(TEXTSIZE);
         hdx[f][q-1]->GetXaxis()->SetTitleOffset(1.0);         
         hdx[f][q-1]->GetYaxis()->SetTitleOffset(1.0);  

         hdx[f][q-1]->Fit(Form("fdxVTPhiQ%d",q));
         fitpar[f][q-1][0] = hdx[f][q-1]->GetEntries();
         fitpar[f][q-1][1] = fit[f][q-1]->GetParameter(1);
         fitpar[f][q-1][2] = fit[f][q-1]->GetParameter(2); 

         hdx[f][q-1]->SetMarkerColor(kBlack);         
         hdx[f][q-1]->SetLineColor(kBlack);           
         hdx[f][q-1]->Draw("");
         fit[f][q-1]->Draw("same");
         tlpar[f][q-1][0] = new TLatex(range*0.25, max[f][q-1]*0.9, Form("Entries : %d", (int)(fitpar[f][q-1][0])));
         tlpar[f][q-1][0]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][0]->SetTextColor(kBlack);
         tlpar[f][q-1][0]->Draw("same");
         tlpar[f][q-1][1] = new TLatex(range*0.25, max[f][q-1]*0.75, Form("mean : %4.4g", (fitpar[f][q-1][1])));
         tlpar[f][q-1][1]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][1]->SetTextColor(kBlack);
         tlpar[f][q-1][1]->Draw("same");
         tlpar[f][q-1][2] = new TLatex(range*0.25, max[f][q-1]*0.6, Form("stddev : %4.4g", (fitpar[f][q-1][2])));
         tlpar[f][q-1][2]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][2]->SetTextColor(kBlack);
         tlpar[f][q-1][2]->Draw("same");   
         tlpar[f][q-1][3] = new TLatex(range*0.25, max[f][q-1]*0.45, Form("stddev : %4.4g", (data_simul_vtx_track_vs_phi[f][q-1])));
         tlpar[f][q-1][3]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][3]->SetTextColor(kBlue);
         tlpar[f][q-1][3]->Draw("same");                  
      }
   }
   
   f=2;
   for(int q = 0; q < 5; q++){

      int index = (nQUAD+1)*f + q + 1;

      canvas_vtall->cd(index);
      canvas_vtall->cd(index)->SetGridx();
      canvas_vtall->cd(index)->SetGridy();
      gStyle->SetOptStat(0);

      double range = rangeX*5.0;
      double res = 0.005;  
      
      if(q==0){
         gStyle->SetOptStat(111111);      
         hdxDET[f] = new TH2D(Form("hdxVzvsPhiQ%d",q),Form("#DeltaVertex_{z} vs Phi (All Quadrants); phi; #DeltaVertex_{z}"),600,-TMath::Pi(),TMath::Pi(),(int)2*(range/res),-range,+range);
         ResMonitor->Draw(Form("(%s):fgPhi[2]>>hdxVzvsPhiQ%d",(const char*)tFormula[f], q),
                          Form("TMath::Abs(fgPhi[2])<10&&TMath::Abs(%s)<%g&&pT>=%g%s",
                               (const char*)tFormula[f],range,momCut,(const char*)ChargeOpt),"colz");    
                               
         hdxDET[f]->GetXaxis()->SetLabelSize(TEXTSIZE);
         hdxDET[f]->GetYaxis()->SetLabelSize(TEXTSIZE);
         hdxDET[f]->GetXaxis()->SetTitleSize(TEXTSIZE);
         hdxDET[f]->GetYaxis()->SetTitleSize(TEXTSIZE);
         hdxDET[f]->GetXaxis()->SetTitleOffset(1.0);         
         hdxDET[f]->GetYaxis()->SetTitleOffset(1.0);                        
      } else {

         hdx[f][q-1] = new TH1D(Form("hdxVzQ%d",q),Form("#DeltaVertex_{z} (Quadrant%d); #DeltaVertex_{z}; N",q),(int)2*(range/res),-range,+range);
         ResMonitor->Draw(Form("(%s)>>hdxVzQ%d",(const char*)tFormula[f], q),
                          Form("TMath::Abs(fgPhi[2])<10&&TMath::Abs(%s)<%g&&fgPhi[2]>%g&&fgPhi[2]<=%g&&pT>=%g%s",
                               (const char*)tFormula[f],range,valueQUAD[0][q-1],valueQUAD[1][q-1],momCut,(const char*)ChargeOpt));
         max[f][q-1] = hdx[f][q-1]->GetMaximum();                       
         fit[f][q-1] = new TF1(Form("fdxVzQ%d",q),"gaus",-0.5*range,+0.5*range);

         hdx[f][q-1]->GetXaxis()->SetLabelSize(TEXTSIZE);
         hdx[f][q-1]->GetYaxis()->SetLabelSize(TEXTSIZE);
         hdx[f][q-1]->GetXaxis()->SetTitleSize(TEXTSIZE);
         hdx[f][q-1]->GetYaxis()->SetTitleSize(TEXTSIZE);
         hdx[f][q-1]->GetXaxis()->SetTitleOffset(1.0);         
         hdx[f][q-1]->GetYaxis()->SetTitleOffset(1.0);  

         hdx[f][q-1]->Fit(Form("fdxVzQ%d",q));
         fitpar[f][q-1][0] = hdx[f][q-1]->GetEntries();
         fitpar[f][q-1][1] = fit[f][q-1]->GetParameter(1);
         fitpar[f][q-1][2] = fit[f][q-1]->GetParameter(2); 

         hdx[f][q-1]->SetMarkerColor(kBlack);         
         hdx[f][q-1]->SetLineColor(kBlack);           
         hdx[f][q-1]->Draw("");
         fit[f][q-1]->Draw("same");
         tlpar[f][q-1][0] = new TLatex(range*0.25, max[f][q-1]*0.9, Form("Entries : %d", (int)(fitpar[f][q-1][0])));
         tlpar[f][q-1][0]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][0]->SetTextColor(kBlack);
         tlpar[f][q-1][0]->Draw("same");
         tlpar[f][q-1][1] = new TLatex(range*0.25, max[f][q-1]*0.75, Form("mean : %4.4g", (fitpar[f][q-1][1])));
         tlpar[f][q-1][1]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][1]->SetTextColor(kBlack);
         tlpar[f][q-1][1]->Draw("same");
         tlpar[f][q-1][2] = new TLatex(range*0.25, max[f][q-1]*0.6, Form("stddev : %4.4g", (fitpar[f][q-1][2])));
         tlpar[f][q-1][2]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][2]->SetTextColor(kBlack);
         tlpar[f][q-1][2]->Draw("same");       
         tlpar[f][q-1][3] = new TLatex(range*0.25, max[f][q-1]*0.45, Form("stddev : %4.4g", (data_simul_vtx_track_vs_phi[f][q-1])));
         tlpar[f][q-1][3]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][3]->SetTextColor(kBlue);
         tlpar[f][q-1][3]->Draw("same");                
      }
   }
   canvas_vtall->SaveAs(Form("Plot_vtx-track_vs_phi_allpTCut%g.gif",momCut),"gif");
}

void macro_plot_vtx_track_z_summary(double rangeX = 0.1, TString ChargeOpt="", double momCut = 0.0){

   // DET // Quad1, 2, 3 vs Phi
   int nMonitors = 3;
   TString tFormula[nMonitors];
   //1. vtx(xy) x track(xy) at L2 in r   vs phi ; scalar product
   tFormula[0] = "(vtxfitX*fgX[2]+vtxfitY*fgY[2])/TMath::Sqrt(fgX[2]*fgX[2]+fgY[2]*fgY[2])";  
   //2. vtx(xy) x track(xy) at L2 in phi vs phi ; cross product
   tFormula[1] = "(vtxfitX*fgY[2]-vtxfitY*fgX[2])/TMath::Sqrt(fgX[2]*fgX[2]+fgY[2]*fgY[2])";
   //3. dvtxZ vs phi at L2
   tFormula[2] = "vtxfitZ-vtxZ";
   
   int max[nMonitors][nQUAD];

   double valueQUAD[2][nQUAD] = {{-15, -7.5, 0, 7.5},{-7.5, 0, 7.5, 15}};
   
   TH2D* hdxDET[nQUAD];
   TH1D* hdx[nMonitors][nQUAD];
   TF1*  fit[nMonitors][nQUAD];
   double fitpar[nMonitors][nQUAD][3]; // (entries, mean, sigma)
   TLatex *tlpar[nMonitors][nQUAD][4];  
  
   TCanvas* canvas_vtall = new TCanvas("canvas_vtall","canvas_vtall",5*600,nMonitors*500);
   canvas_vtall->Divide((int)5,(int)nMonitors,0.001,0.001);
  
   int f;
   f=0;
   for(int q = 0; q < 5; q++){

      int index = (nQUAD+1)*f + q + 1;

      canvas_vtall->cd(index);
      canvas_vtall->cd(index)->SetGridx();
      canvas_vtall->cd(index)->SetGridy();
      gStyle->SetOptStat(0);

      double range = 0.25*rangeX;
      double res = 0.25*0.002;  
      
      if(q==0){
         gStyle->SetOptStat(111111);      
         hdxDET[f] = new TH2D(Form("hdxVTRvsZQ%d",q),Form("Normalized Vertex_{xy} #bullet Track_{xy} vs Z (All Z); z; dx"),1000,-15,15,(int)2*(range/res),-range,+range);
         ResMonitor->Draw(Form("(%s):fgZ[2]>>hdxVTRvsZQ%d",(const char*)tFormula[f], q),
                          Form("TMath::Abs(fgZ[2])<30&&TMath::Abs(%s)<%g&&pT>=%g%s",
                               (const char*)tFormula[f],range,momCut,(const char*)ChargeOpt),"colz");    
                               
         hdxDET[f]->GetXaxis()->SetLabelSize(TEXTSIZE);
         hdxDET[f]->GetYaxis()->SetLabelSize(TEXTSIZE);
         hdxDET[f]->GetXaxis()->SetTitleSize(TEXTSIZE);
         hdxDET[f]->GetYaxis()->SetTitleSize(TEXTSIZE);
         hdxDET[f]->GetXaxis()->SetTitleOffset(1.0);         
         hdxDET[f]->GetYaxis()->SetTitleOffset(1.0);                        
      } else {

         hdx[f][q-1] = new TH1D(Form("hdxVTRQ%d",q),Form("Normalized Vertex_{xy} #bullet Track_{xy} (Z range %d); dx; N",q),(int)2*(range/res),-range,+range);
         ResMonitor->Draw(Form("(%s)>>hdxVTRQ%d",(const char*)tFormula[f], q),
                          Form("TMath::Abs(fgZ[2])<30&&TMath::Abs(%s)<%g&&fgZ[2]>%g&&fgZ[2]<=%g&&pT>=%g%s",
                               (const char*)tFormula[f],range,valueQUAD[0][q-1],valueQUAD[1][q-1],momCut,(const char*)ChargeOpt));
         max[f][q-1] = hdx[f][q-1]->GetMaximum();                       
         fit[f][q-1] = new TF1(Form("fdxVTRQ%d",q),"gaus",-0.5*range,+0.5*range);

         hdx[f][q-1]->GetXaxis()->SetLabelSize(TEXTSIZE);
         hdx[f][q-1]->GetYaxis()->SetLabelSize(TEXTSIZE);
         hdx[f][q-1]->GetXaxis()->SetTitleSize(TEXTSIZE);
         hdx[f][q-1]->GetYaxis()->SetTitleSize(TEXTSIZE);
         hdx[f][q-1]->GetXaxis()->SetTitleOffset(1.0);         
         hdx[f][q-1]->GetYaxis()->SetTitleOffset(1.0);  

         hdx[f][q-1]->Fit(Form("fdxVTRQ%d",q));
         fitpar[f][q-1][0] = hdx[f][q-1]->GetEntries();
         fitpar[f][q-1][1] = fit[f][q-1]->GetParameter(1);
         fitpar[f][q-1][2] = fit[f][q-1]->GetParameter(2); 

         hdx[f][q-1]->SetMarkerColor(kBlack);         
         hdx[f][q-1]->SetLineColor(kBlack);           
         hdx[f][q-1]->Draw("");
         fit[f][q-1]->Draw("same");
         tlpar[f][q-1][0] = new TLatex(range*0.25, max[f][q-1]*0.9, Form("Entries : %d", (int)(fitpar[f][q-1][0])));
         tlpar[f][q-1][0]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][0]->SetTextColor(kBlack);
         tlpar[f][q-1][0]->Draw("same");
         tlpar[f][q-1][1] = new TLatex(range*0.25, max[f][q-1]*0.75, Form("mean : %4.4g", (fitpar[f][q-1][1])));
         tlpar[f][q-1][1]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][1]->SetTextColor(kBlack);
         tlpar[f][q-1][1]->Draw("same");
         tlpar[f][q-1][2] = new TLatex(range*0.25, max[f][q-1]*0.6, Form("stddev : %4.4g", (fitpar[f][q-1][2])));
         tlpar[f][q-1][2]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][2]->SetTextColor(kBlack);
         tlpar[f][q-1][2]->Draw("same");     
         tlpar[f][q-1][3] = new TLatex(range*0.25, max[f][q-1]*0.45, Form("stddev : %4.4g", (data_simul_vtx_track_vs_z[f][q-1])));
         tlpar[f][q-1][3]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][3]->SetTextColor(kBlue);
         tlpar[f][q-1][3]->Draw("same");              
      }
   }
   
   f=1;
   for(int q = 0; q < 5; q++){

      int index = (nQUAD+1)*f + q + 1;

      canvas_vtall->cd(index);
      canvas_vtall->cd(index)->SetGridx();
      canvas_vtall->cd(index)->SetGridy();
      gStyle->SetOptStat(0);

      double range = rangeX;
      double res = 0.002;  
      
      if(q==0){
         gStyle->SetOptStat(111111);      
         hdxDET[f] = new TH2D(Form("hdxVTPhivsZQ%d",q),Form("Normalized Vertex_{xy} #times Track_{xy} vs Z (All Z); z; dx"),1000,-15,15,(int)2*(range/res),-range,+range);
         ResMonitor->Draw(Form("(%s):fgZ[2]>>hdxVTPhivsZQ%d",(const char*)tFormula[f], q),
                          Form("TMath::Abs(fgZ[2])<30&&TMath::Abs(%s)<%g&&pT>=%g%s",
                               (const char*)tFormula[f],range,momCut,(const char*)ChargeOpt),"colz");    
                               
         hdxDET[f]->GetXaxis()->SetLabelSize(TEXTSIZE);
         hdxDET[f]->GetYaxis()->SetLabelSize(TEXTSIZE);
         hdxDET[f]->GetXaxis()->SetTitleSize(TEXTSIZE);
         hdxDET[f]->GetYaxis()->SetTitleSize(TEXTSIZE);
         hdxDET[f]->GetXaxis()->SetTitleOffset(1.0);         
         hdxDET[f]->GetYaxis()->SetTitleOffset(1.0);                        
      } else {

         hdx[f][q-1] = new TH1D(Form("hdxVTPhiQ%d",q),Form("Normalized Vertex_{xy} #times Track_{xy} (Z range %d); dx; N",q),(int)2*(range/res),-range,+range);
         ResMonitor->Draw(Form("(%s)>>hdxVTPhiQ%d",(const char*)tFormula[f], q),
                          Form("TMath::Abs(fgZ[2])<30&&TMath::Abs(%s)<%g&&fgZ[2]>%g&&fgZ[2]<=%g&&pT>=%g%s",
                               (const char*)tFormula[f],range,valueQUAD[0][q-1],valueQUAD[1][q-1],momCut,(const char*)ChargeOpt));
         max[f][q-1] = hdx[f][q-1]->GetMaximum();                       
         fit[f][q-1] = new TF1(Form("fdxVTPhiQ%d",q),"gaus",-0.5*range,+0.5*range);

         hdx[f][q-1]->GetXaxis()->SetLabelSize(TEXTSIZE);
         hdx[f][q-1]->GetYaxis()->SetLabelSize(TEXTSIZE);
         hdx[f][q-1]->GetXaxis()->SetTitleSize(TEXTSIZE);
         hdx[f][q-1]->GetYaxis()->SetTitleSize(TEXTSIZE);
         hdx[f][q-1]->GetXaxis()->SetTitleOffset(1.0);         
         hdx[f][q-1]->GetYaxis()->SetTitleOffset(1.0);  

         hdx[f][q-1]->Fit(Form("fdxVTPhiQ%d",q));
         fitpar[f][q-1][0] = hdx[f][q-1]->GetEntries();
         fitpar[f][q-1][1] = fit[f][q-1]->GetParameter(1);
         fitpar[f][q-1][2] = fit[f][q-1]->GetParameter(2); 

         hdx[f][q-1]->SetMarkerColor(kBlack);         
         hdx[f][q-1]->SetLineColor(kBlack);           
         hdx[f][q-1]->Draw("");
         fit[f][q-1]->Draw("same");
         tlpar[f][q-1][0] = new TLatex(range*0.25, max[f][q-1]*0.9, Form("Entries : %d", (int)(fitpar[f][q-1][0])));
         tlpar[f][q-1][0]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][0]->SetTextColor(kBlack);
         tlpar[f][q-1][0]->Draw("same");
         tlpar[f][q-1][1] = new TLatex(range*0.25, max[f][q-1]*0.75, Form("mean : %4.4g", (fitpar[f][q-1][1])));
         tlpar[f][q-1][1]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][1]->SetTextColor(kBlack);
         tlpar[f][q-1][1]->Draw("same");
         tlpar[f][q-1][2] = new TLatex(range*0.25, max[f][q-1]*0.6, Form("stddev : %4.4g", (fitpar[f][q-1][2])));
         tlpar[f][q-1][2]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][2]->SetTextColor(kBlack);
         tlpar[f][q-1][2]->Draw("same");    
         tlpar[f][q-1][3] = new TLatex(range*0.25, max[f][q-1]*0.45, Form("stddev : %4.4g", (data_simul_vtx_track_vs_z[f][q-1])));
         tlpar[f][q-1][3]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][3]->SetTextColor(kBlue);
         tlpar[f][q-1][3]->Draw("same");                  
      }
   }
   
   f=2;
   for(int q = 0; q < 5; q++){

      int index = (nQUAD+1)*f + q + 1;

      canvas_vtall->cd(index);
      canvas_vtall->cd(index)->SetGridx();
      canvas_vtall->cd(index)->SetGridy();
      gStyle->SetOptStat(0);

      double range = rangeX*5.0;
      double res = 0.005;  
      
      if(q==0){
         gStyle->SetOptStat(111111);      
         hdxDET[f] = new TH2D(Form("hdxVzvsZQ%d",q),Form("#DeltaVertex_{z} vs Z (All Z); z; #DeltaVertex_{z}"),1000,-15,15,(int)2*(range/res),-range,+range);
         ResMonitor->Draw(Form("(%s):fgZ[2]>>hdxVzvsZQ%d",(const char*)tFormula[f], q),
                          Form("TMath::Abs(fgZ[2])<30&&TMath::Abs(%s)<%g&&pT>=%g%s",
                               (const char*)tFormula[f],range,momCut,(const char*)ChargeOpt),"colz");    
                               
         hdxDET[f]->GetXaxis()->SetLabelSize(TEXTSIZE);
         hdxDET[f]->GetYaxis()->SetLabelSize(TEXTSIZE);
         hdxDET[f]->GetXaxis()->SetTitleSize(TEXTSIZE);
         hdxDET[f]->GetYaxis()->SetTitleSize(TEXTSIZE);
         hdxDET[f]->GetXaxis()->SetTitleOffset(1.0);         
         hdxDET[f]->GetYaxis()->SetTitleOffset(1.0);                        
      } else {

         hdx[f][q-1] = new TH1D(Form("hdxVzQ%d",q),Form("#DeltaVertex_{z} (Z range %d); #DeltaVertex_{z}; N",q),(int)2*(range/res),-range,+range);
         ResMonitor->Draw(Form("(%s)>>hdxVzQ%d",(const char*)tFormula[f], q),
                          Form("TMath::Abs(fgZ[2])<30&&TMath::Abs(%s)<%g&&fgZ[2]>%g&&fgZ[2]<=%g&&pT>=%g%s",
                               (const char*)tFormula[f],range,valueQUAD[0][q-1],valueQUAD[1][q-1],momCut,(const char*)ChargeOpt));
         max[f][q-1] = hdx[f][q-1]->GetMaximum();                       
         fit[f][q-1] = new TF1(Form("fdxVzQ%d",q),"gaus",-0.5*range,+0.5*range);

         hdx[f][q-1]->GetXaxis()->SetLabelSize(TEXTSIZE);
         hdx[f][q-1]->GetYaxis()->SetLabelSize(TEXTSIZE);
         hdx[f][q-1]->GetXaxis()->SetTitleSize(TEXTSIZE);
         hdx[f][q-1]->GetYaxis()->SetTitleSize(TEXTSIZE);
         hdx[f][q-1]->GetXaxis()->SetTitleOffset(1.0);         
         hdx[f][q-1]->GetYaxis()->SetTitleOffset(1.0);  

         hdx[f][q-1]->Fit(Form("fdxVzQ%d",q));
         fitpar[f][q-1][0] = hdx[f][q-1]->GetEntries();
         fitpar[f][q-1][1] = fit[f][q-1]->GetParameter(1);
         fitpar[f][q-1][2] = fit[f][q-1]->GetParameter(2); 

         hdx[f][q-1]->SetMarkerColor(kBlack);         
         hdx[f][q-1]->SetLineColor(kBlack);           
         hdx[f][q-1]->Draw("");
         fit[f][q-1]->Draw("same");
         tlpar[f][q-1][0] = new TLatex(range*0.25, max[f][q-1]*0.9, Form("Entries : %d", (int)(fitpar[f][q-1][0])));
         tlpar[f][q-1][0]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][0]->SetTextColor(kBlack);
         tlpar[f][q-1][0]->Draw("same");
         tlpar[f][q-1][1] = new TLatex(range*0.25, max[f][q-1]*0.75, Form("mean : %4.4g", (fitpar[f][q-1][1])));
         tlpar[f][q-1][1]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][1]->SetTextColor(kBlack);
         tlpar[f][q-1][1]->Draw("same");
         tlpar[f][q-1][2] = new TLatex(range*0.25, max[f][q-1]*0.6, Form("stddev : %4.4g", (fitpar[f][q-1][2])));
         tlpar[f][q-1][2]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][2]->SetTextColor(kBlack);
         tlpar[f][q-1][2]->Draw("same");       
         tlpar[f][q-1][3] = new TLatex(range*0.25, max[f][q-1]*0.45, Form("stddev : %4.4g", (data_simul_vtx_track_vs_z[f][q-1])));
         tlpar[f][q-1][3]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][3]->SetTextColor(kBlue);
         tlpar[f][q-1][3]->Draw("same");               
      }
   }
   canvas_vtall->SaveAs(Form("Plot_vtx-track_vs_z_allpTCut%g.gif",momCut),"gif");
}

void macro_plot_vtx_track_theta_summary(double rangeX = 0.1, TString ChargeOpt="", double momCut = 0.0){

   // DET // Quad1, 2, 3 vs Phi
   int nMonitors = 3;
   TString tFormula[nMonitors];
   //1. vtx(xy) x track(xy) at L2 in r   vs phi ; scalar product
   tFormula[0] = "(vtxfitX*fgX[2]+vtxfitY*fgY[2])/TMath::Sqrt(fgX[2]*fgX[2]+fgY[2]*fgY[2])";  
   //2. vtx(xy) x track(xy) at L2 in phi vs phi ; cross product
   tFormula[1] = "(vtxfitX*fgY[2]-vtxfitY*fgX[2])/TMath::Sqrt(fgX[2]*fgX[2]+fgY[2]*fgY[2])";
   //3. dvtxZ vs phi at L2
   tFormula[2] = "vtxfitZ-vtxZ";
   
   TString fL2Theta = "TMath::ATan2(TMath::Sqrt(fgX[2]*fgX[2]+fgY[2]*fgY[2]),fgZ[2]-vtxZ)";
   
   int max[nMonitors][nQUAD];

   double valueQUAD[2][nQUAD] = {{TMath::Pi()/8.0, 0.5*(TMath::Pi()/8.0 + TMath::Pi()/2.0), TMath::Pi()/2.0, 0.5*(7.0*TMath::Pi()/8.0 + TMath::Pi()/2.0)},
                                 {0.5*(TMath::Pi()/8.0 + TMath::Pi()/2.0), TMath::Pi()/2.0, 0.5*(7.0*TMath::Pi()/8.0 + TMath::Pi()/2.0), 7.0*TMath::Pi()/8.0}};
   
   TH2D* hdxDET[nQUAD];
   TH1D* hdx[nMonitors][nQUAD];
   TF1*  fit[nMonitors][nQUAD];
   double fitpar[nMonitors][nQUAD][3]; // (entries, mean, sigma)
   TLatex *tlpar[nMonitors][nQUAD][4];  
  
   TCanvas* canvas_vtall = new TCanvas("canvas_vtall","canvas_vtall",5*600,nMonitors*500);
   canvas_vtall->Divide((int)5,(int)nMonitors,0.001,0.001);
  
   int f;
   f=0;
   for(int q = 0; q < 5; q++){

      int index = (nQUAD+1)*f + q + 1;

      canvas_vtall->cd(index);
      canvas_vtall->cd(index)->SetGridx();
      canvas_vtall->cd(index)->SetGridy();
      gStyle->SetOptStat(0);

      double range = 0.25*rangeX;
      double res = 0.25*0.002;  
      
      if(q==0){
         gStyle->SetOptStat(111111);      
         hdxDET[f] = new TH2D(Form("hdxVTRvsThetaQ%d",q),
                              Form("Normalized Vertex_{xy} #bullet Track_{xy} vs Theta (All Theta); theta; dx"),600,0,TMath::Pi(),(int)2*(range/res),-range,+range);
         ResMonitor->Draw(Form("(%s):%s>>hdxVTRvsThetaQ%d",(const char*)tFormula[f], (const char*)fL2Theta, q),
                          Form("TMath::Abs(fgPhi[2])<10&&TMath::Abs(%s)<30&&TMath::Abs(%s)<%g&&pT>=%g%s",
                               (const char*)fL2Theta,(const char*)tFormula[f],range,momCut,(const char*)ChargeOpt),"colz");    
                               
         hdxDET[f]->GetXaxis()->SetLabelSize(TEXTSIZE);
         hdxDET[f]->GetYaxis()->SetLabelSize(TEXTSIZE);
         hdxDET[f]->GetXaxis()->SetTitleSize(TEXTSIZE);
         hdxDET[f]->GetYaxis()->SetTitleSize(TEXTSIZE);
         hdxDET[f]->GetXaxis()->SetTitleOffset(1.0);         
         hdxDET[f]->GetYaxis()->SetTitleOffset(1.0);                        
      } else {

         hdx[f][q-1] = new TH1D(Form("hdxVTRQ%d",q),Form("Normalized Vertex_{xy} #bullet Track_{xy} (Theta range %d); dx; N",q),(int)2*(range/res),-range,+range);
         ResMonitor->Draw(Form("(%s)>>hdxVTRQ%d",(const char*)tFormula[f], q),
                          Form("TMath::Abs(fgPhi[2])<10&&TMath::Abs(%s)<30&&TMath::Abs(%s)<%g&&%s>%g&&%s<=%g&&pT>=%g%s",
                               (const char*)fL2Theta,(const char*)tFormula[f],range,(const char*)fL2Theta,valueQUAD[0][q-1],(const char*)fL2Theta,valueQUAD[1][q-1],momCut,(const char*)ChargeOpt));
         max[f][q-1] = hdx[f][q-1]->GetMaximum();                       
         fit[f][q-1] = new TF1(Form("fdxVTRQ%d",q),"gaus",-0.5*range,+0.5*range);

         hdx[f][q-1]->GetXaxis()->SetLabelSize(TEXTSIZE);
         hdx[f][q-1]->GetYaxis()->SetLabelSize(TEXTSIZE);
         hdx[f][q-1]->GetXaxis()->SetTitleSize(TEXTSIZE);
         hdx[f][q-1]->GetYaxis()->SetTitleSize(TEXTSIZE);
         hdx[f][q-1]->GetXaxis()->SetTitleOffset(1.0);         
         hdx[f][q-1]->GetYaxis()->SetTitleOffset(1.0);  

         hdx[f][q-1]->Fit(Form("fdxVTRQ%d",q));
         fitpar[f][q-1][0] = hdx[f][q-1]->GetEntries();
         fitpar[f][q-1][1] = fit[f][q-1]->GetParameter(1);
         fitpar[f][q-1][2] = fit[f][q-1]->GetParameter(2); 

         hdx[f][q-1]->SetMarkerColor(kBlack);         
         hdx[f][q-1]->SetLineColor(kBlack);           
         hdx[f][q-1]->Draw("");
         fit[f][q-1]->Draw("same");
         tlpar[f][q-1][0] = new TLatex(range*0.25, max[f][q-1]*0.9, Form("Entries : %d", (int)(fitpar[f][q-1][0])));
         tlpar[f][q-1][0]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][0]->SetTextColor(kBlack);
         tlpar[f][q-1][0]->Draw("same");
         tlpar[f][q-1][1] = new TLatex(range*0.25, max[f][q-1]*0.75, Form("mean : %4.4g", (fitpar[f][q-1][1])));
         tlpar[f][q-1][1]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][1]->SetTextColor(kBlack);
         tlpar[f][q-1][1]->Draw("same");
         tlpar[f][q-1][2] = new TLatex(range*0.25, max[f][q-1]*0.6, Form("stddev : %4.4g", (fitpar[f][q-1][2])));
         tlpar[f][q-1][2]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][2]->SetTextColor(kBlack);
         tlpar[f][q-1][2]->Draw("same");    
         tlpar[f][q-1][3] = new TLatex(range*0.25, max[f][q-1]*0.45, Form("stddev : %4.4g", (data_simul_vtx_track_vs_theta[f][q-1])));
         tlpar[f][q-1][3]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][3]->SetTextColor(kBlue);
         tlpar[f][q-1][3]->Draw("same");                     
      }
   }
   
   f=1;
   for(int q = 0; q < 5; q++){

      int index = (nQUAD+1)*f + q + 1;

      canvas_vtall->cd(index);
      canvas_vtall->cd(index)->SetGridx();
      canvas_vtall->cd(index)->SetGridy();
      gStyle->SetOptStat(0);

      double range = rangeX;
      double res = 0.002;  
      
      if(q==0){
         gStyle->SetOptStat(111111);      
         hdxDET[f] = new TH2D(Form("hdxVTPhivsThetaQ%d",q),
                              Form("Normalized Vertex_{xy} #times Track_{xy} vs Theta (All Theta); theta; dx"),600,0,TMath::Pi(),(int)2*(range/res),-range,+range);
         ResMonitor->Draw(Form("(%s):%s>>hdxVTPhivsThetaQ%d",(const char*)tFormula[f], (const char*)fL2Theta, q),
                          Form("TMath::Abs(fgPhi[2])<10&&TMath::Abs(%s)<30&&TMath::Abs(%s)<%g&&pT>=%g%s",
                               (const char*)fL2Theta,(const char*)tFormula[f],range,momCut,(const char*)ChargeOpt),"colz");    
                               
         hdxDET[f]->GetXaxis()->SetLabelSize(TEXTSIZE);
         hdxDET[f]->GetYaxis()->SetLabelSize(TEXTSIZE);
         hdxDET[f]->GetXaxis()->SetTitleSize(TEXTSIZE);
         hdxDET[f]->GetYaxis()->SetTitleSize(TEXTSIZE);
         hdxDET[f]->GetXaxis()->SetTitleOffset(1.0);         
         hdxDET[f]->GetYaxis()->SetTitleOffset(1.0);                        
      } else {

         hdx[f][q-1] = new TH1D(Form("hdxVTPhiQ%d",q),Form("Normalized Vertex_{xy} #times Track_{xy} (Theta range %d); dx; N",q),(int)2*(range/res),-range,+range);
         ResMonitor->Draw(Form("(%s)>>hdxVTPhiQ%d",(const char*)tFormula[f], q),
                          Form("TMath::Abs(fgPhi[2])<10&&TMath::Abs(%s)<30&&TMath::Abs(%s)<%g&&%s>%g&&%s<=%g&&pT>=%g%s",
                               (const char*)fL2Theta,(const char*)tFormula[f],range,(const char*)fL2Theta,valueQUAD[0][q-1],(const char*)fL2Theta,valueQUAD[1][q-1],momCut,(const char*)ChargeOpt));
         max[f][q-1] = hdx[f][q-1]->GetMaximum();                       
         fit[f][q-1] = new TF1(Form("fdxVTPhiQ%d",q),"gaus",-0.5*range,+0.5*range);

         hdx[f][q-1]->GetXaxis()->SetLabelSize(TEXTSIZE);
         hdx[f][q-1]->GetYaxis()->SetLabelSize(TEXTSIZE);
         hdx[f][q-1]->GetXaxis()->SetTitleSize(TEXTSIZE);
         hdx[f][q-1]->GetYaxis()->SetTitleSize(TEXTSIZE);
         hdx[f][q-1]->GetXaxis()->SetTitleOffset(1.0);         
         hdx[f][q-1]->GetYaxis()->SetTitleOffset(1.0);  

         hdx[f][q-1]->Fit(Form("fdxVTPhiQ%d",q));
         fitpar[f][q-1][0] = hdx[f][q-1]->GetEntries();
         fitpar[f][q-1][1] = fit[f][q-1]->GetParameter(1);
         fitpar[f][q-1][2] = fit[f][q-1]->GetParameter(2); 

         hdx[f][q-1]->SetMarkerColor(kBlack);         
         hdx[f][q-1]->SetLineColor(kBlack);           
         hdx[f][q-1]->Draw("");
         fit[f][q-1]->Draw("same");
         tlpar[f][q-1][0] = new TLatex(range*0.25, max[f][q-1]*0.9, Form("Entries : %d", (int)(fitpar[f][q-1][0])));
         tlpar[f][q-1][0]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][0]->SetTextColor(kBlack);
         tlpar[f][q-1][0]->Draw("same");
         tlpar[f][q-1][1] = new TLatex(range*0.25, max[f][q-1]*0.75, Form("mean : %4.4g", (fitpar[f][q-1][1])));
         tlpar[f][q-1][1]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][1]->SetTextColor(kBlack);
         tlpar[f][q-1][1]->Draw("same");
         tlpar[f][q-1][2] = new TLatex(range*0.25, max[f][q-1]*0.6, Form("stddev : %4.4g", (fitpar[f][q-1][2])));
         tlpar[f][q-1][2]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][2]->SetTextColor(kBlack);
         tlpar[f][q-1][2]->Draw("same");  
         tlpar[f][q-1][3] = new TLatex(range*0.25, max[f][q-1]*0.45, Form("stddev : %4.4g", (data_simul_vtx_track_vs_theta[f][q-1])));
         tlpar[f][q-1][3]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][3]->SetTextColor(kBlue);
         tlpar[f][q-1][3]->Draw("same");                       
      }
   }
   
   f=2;
   for(int q = 0; q < 5; q++){

      int index = (nQUAD+1)*f + q + 1;

      canvas_vtall->cd(index);
      canvas_vtall->cd(index)->SetGridx();
      canvas_vtall->cd(index)->SetGridy();
      gStyle->SetOptStat(0);

      double range = rangeX*5.0;
      double res = 0.005;  
      
      if(q==0){
         gStyle->SetOptStat(111111);      
         hdxDET[f] = new TH2D(Form("hdxVzvsThetaQ%d",q),Form("#DeltaVertex_{z} vs Theta (All Theta); theta; #DeltaVertex_{z}"),600,0,TMath::Pi(),(int)2*(range/res),-range,+range);
         ResMonitor->Draw(Form("(%s):%s>>hdxVzvsThetaQ%d",(const char*)tFormula[f], (const char*)fL2Theta, q),
                          Form("TMath::Abs(fgPhi[2])<10&&TMath::Abs(%s)<30&&TMath::Abs(%s)<%g&&pT>=%g%s",
                               (const char*)fL2Theta,(const char*)tFormula[f],range,momCut,(const char*)ChargeOpt),"colz");    
                               
         hdxDET[f]->GetXaxis()->SetLabelSize(TEXTSIZE);
         hdxDET[f]->GetYaxis()->SetLabelSize(TEXTSIZE);
         hdxDET[f]->GetXaxis()->SetTitleSize(TEXTSIZE);
         hdxDET[f]->GetYaxis()->SetTitleSize(TEXTSIZE);
         hdxDET[f]->GetXaxis()->SetTitleOffset(1.0);         
         hdxDET[f]->GetYaxis()->SetTitleOffset(1.0);                        
      } else {

         hdx[f][q-1] = new TH1D(Form("hdxVzQ%d",q),Form("#DeltaVertex_{z} (Theta range %d); #DeltaVertex_{z}; N",q),(int)2*(range/res),-range,+range);
         ResMonitor->Draw(Form("(%s)>>hdxVzQ%d",(const char*)tFormula[f], q),
                          Form("TMath::Abs(fgPhi[2])<10&&TMath::Abs(%s)<30&&TMath::Abs(%s)<%g&&%s>%g&&%s<=%g&&pT>=%g%s",
                               (const char*)fL2Theta,(const char*)tFormula[f],range,(const char*)fL2Theta,valueQUAD[0][q-1],(const char*)fL2Theta,valueQUAD[1][q-1],momCut,(const char*)ChargeOpt));
         max[f][q-1] = hdx[f][q-1]->GetMaximum();                       
         fit[f][q-1] = new TF1(Form("fdxVzQ%d",q),"gaus",-0.5*range,+0.5*range);

         hdx[f][q-1]->GetXaxis()->SetLabelSize(TEXTSIZE);
         hdx[f][q-1]->GetYaxis()->SetLabelSize(TEXTSIZE);
         hdx[f][q-1]->GetXaxis()->SetTitleSize(TEXTSIZE);
         hdx[f][q-1]->GetYaxis()->SetTitleSize(TEXTSIZE);
         hdx[f][q-1]->GetXaxis()->SetTitleOffset(1.0);         
         hdx[f][q-1]->GetYaxis()->SetTitleOffset(1.0);  

         hdx[f][q-1]->Fit(Form("fdxVzQ%d",q));
         fitpar[f][q-1][0] = hdx[f][q-1]->GetEntries();
         fitpar[f][q-1][1] = fit[f][q-1]->GetParameter(1);
         fitpar[f][q-1][2] = fit[f][q-1]->GetParameter(2); 

         hdx[f][q-1]->SetMarkerColor(kBlack);         
         hdx[f][q-1]->SetLineColor(kBlack);           
         hdx[f][q-1]->Draw("");
         fit[f][q-1]->Draw("same");
         tlpar[f][q-1][0] = new TLatex(range*0.25, max[f][q-1]*0.9, Form("Entries : %d", (int)(fitpar[f][q-1][0])));
         tlpar[f][q-1][0]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][0]->SetTextColor(kBlack);
         tlpar[f][q-1][0]->Draw("same");
         tlpar[f][q-1][1] = new TLatex(range*0.25, max[f][q-1]*0.75, Form("mean : %4.4g", (fitpar[f][q-1][1])));
         tlpar[f][q-1][1]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][1]->SetTextColor(kBlack);
         tlpar[f][q-1][1]->Draw("same");
         tlpar[f][q-1][2] = new TLatex(range*0.25, max[f][q-1]*0.6, Form("stddev : %4.4g", (fitpar[f][q-1][2])));
         tlpar[f][q-1][2]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][2]->SetTextColor(kBlack);
         tlpar[f][q-1][2]->Draw("same");     
         tlpar[f][q-1][3] = new TLatex(range*0.25, max[f][q-1]*0.45, Form("stddev : %4.4g", (data_simul_vtx_track_vs_theta[f][q-1])));
         tlpar[f][q-1][3]->SetTextSize(LATEXSIZE);
         tlpar[f][q-1][3]->SetTextColor(kBlue);
         tlpar[f][q-1][3]->Draw("same");                
      }
   }
   canvas_vtall->SaveAs(Form("Plot_vtx-track_vs_theta_allpTCut%g.gif",momCut),"gif");
}

void macro_plot_vtx_track_r_by_pstave(double rangeX = 0.1, TString ChargeOpt="", double momCut = 0.0){

   int max[nPSTAVE];

   TH1D* hdx[nPSTAVE];
   TF1*  fit[nPSTAVE];
   double fitpar[nPSTAVE][3]; // (entries, mean, sigma)
   TLatex *tlpar[nPSTAVE][3];

   TCanvas* canvas_vtr = new TCanvas("canvas_vtr","canvas_vtr",5*600,4*400);
   canvas_vtr->Divide((int)5,(int)4,0.001,0.001);
   
   for(int t = 0; t < 20; t++){
      canvas_vtr->cd(1 + t);   

      canvas_vtr->cd(1 + t)->SetGridx();
      canvas_vtr->cd(1 + t)->SetGridy();
      gStyle->SetOptStat(0);

      double range = 0.25*rangeX;
      double res = 0.25*0.002;  
           
      TString tFormula = "(vtxfitX*fgX[2]+vtxfitY*fgY[2])/TMath::Sqrt(fgX[2]*fgX[2]+fgY[2]*fgY[2])";
      hdx[t] = new TH1D(Form("hdxpstv%d",t),Form("Normalized Vertex_{xy} #bullet Track_{xy} (pStave%d); dx; N",t),(int)2*(range/res),-range,+range);
      ResMonitor->Draw(Form("(%s)>>hdxpstv%d",(const char*)tFormula, t),
                       Form("TMath::Abs(fgPhi[2])<10&&TMath::Abs(%s)<%g&&fstv[2]==%d&&pT>=%g%s",(const char*)tFormula,range,t,momCut,(const char*)ChargeOpt));
      max[t] = hdx[t]->GetMaximum();                       
      fit[t] = new TF1(Form("fdxpstv%d",t),"gaus",-0.5*range,+0.5*range);

      hdx[t]->GetXaxis()->SetLabelSize(TEXTSIZE);
      hdx[t]->GetYaxis()->SetLabelSize(TEXTSIZE);
      hdx[t]->GetXaxis()->SetTitleSize(TEXTSIZE);
      hdx[t]->GetYaxis()->SetTitleSize(TEXTSIZE);
      hdx[t]->GetXaxis()->SetTitleOffset(1.0);         
      hdx[t]->GetYaxis()->SetTitleOffset(1.0);  

      hdx[t]->Fit(Form("fdxpstv%d",t),"r");
      fitpar[t][0] = hdx[t]->GetEntries();
      fitpar[t][1] = fit[t]->GetParameter(1);
      fitpar[t][2] = fit[t]->GetParameter(2); 

      hdx[t]->SetMarkerColor(kBlack);         
      hdx[t]->SetLineColor(kBlack);           
      hdx[t]->Draw("");
      fit[t]->Draw("same");
      tlpar[t][0] = new TLatex(range*0.25, max[t]*0.9, Form("Entries : %d", (int)(fitpar[t][0])));
      tlpar[t][0]->SetTextSize(LATEXSIZE);
      tlpar[t][0]->SetTextColor(kBlack);
      tlpar[t][0]->Draw("same");
      tlpar[t][1] = new TLatex(range*0.25, max[t]*0.75, Form("mean : %4.4g", (fitpar[t][1])));
      tlpar[t][1]->SetTextSize(LATEXSIZE);
      tlpar[t][1]->SetTextColor(kBlack);
      tlpar[t][1]->Draw("same");
      tlpar[t][2] = new TLatex(range*0.25, max[t]*0.6, Form("stddev : %4.4g", (fitpar[t][2])));
      tlpar[t][2]->SetTextSize(LATEXSIZE);
      tlpar[t][2]->SetTextColor(kBlack);
      tlpar[t][2]->Draw("same");  
   }
   canvas_vtr->SaveAs(Form("Plot_vtx-track_r_by_pstavepTCut%g.gif",momCut),"gif");
}

void macro_plot_vtx_track_phi_by_pstave(double rangeX = 0.1, TString ChargeOpt="", double momCut = 0.0){

   int max[nPSTAVE];

   TH1D* hdx[nPSTAVE];
   TF1*  fit[nPSTAVE];
   double fitpar[nPSTAVE][3]; // (entries, mean, sigma)
   TLatex *tlpar[nPSTAVE][3];

   TCanvas* canvas_vtphi = new TCanvas("canvas_vtphi","canvas_vtphi",5*600,4*400);
   canvas_vtphi->Divide((int)5,(int)4,0.001,0.001);
   
   for(int t = 0; t < 20; t++){
      canvas_vtphi->cd(1 + t);   

      canvas_vtphi->cd(1 + t)->SetGridx();
      canvas_vtphi->cd(1 + t)->SetGridy();
      gStyle->SetOptStat(0);

      double range = rangeX;
      double res = 0.002;  

      TString tFormula = "(vtxfitX*fgY[2]-vtxfitY*fgX[2])/TMath::Sqrt(fgX[2]*fgX[2]+fgY[2]*fgY[2])";
      hdx[t] = new TH1D(Form("hdxpstv%d",t),Form("Normalized Vertex_{xy} #times Track_{xy} (pStave%d); dx; N",t),(int)2*(range/res),-range,+range);
      ResMonitor->Draw(Form("(%s)>>hdxpstv%d",(const char*)tFormula, t),
                       Form("TMath::Abs(fgPhi[2])<10&&TMath::Abs(%s)<%g&&fstv[2]==%d&&pT>=%g%s",(const char*)tFormula,range,t,momCut,(const char*)ChargeOpt));
      max[t] = hdx[t]->GetMaximum();                       
      fit[t] = new TF1(Form("fdxpstv%d",t),"gaus",-0.5*range,+0.5*range);

      hdx[t]->GetXaxis()->SetLabelSize(TEXTSIZE);
      hdx[t]->GetYaxis()->SetLabelSize(TEXTSIZE);
      hdx[t]->GetXaxis()->SetTitleSize(TEXTSIZE);
      hdx[t]->GetYaxis()->SetTitleSize(TEXTSIZE);
      hdx[t]->GetXaxis()->SetTitleOffset(1.0);         
      hdx[t]->GetYaxis()->SetTitleOffset(1.0);  

      hdx[t]->Fit(Form("fdxpstv%d",t),"r");
      fitpar[t][0] = hdx[t]->GetEntries();
      fitpar[t][1] = fit[t]->GetParameter(1);
      fitpar[t][2] = fit[t]->GetParameter(2); 

      hdx[t]->SetMarkerColor(kBlack);         
      hdx[t]->SetLineColor(kBlack);           
      hdx[t]->Draw("");
      fit[t]->Draw("same");
      tlpar[t][0] = new TLatex(range*0.25, max[t]*0.9, Form("Entries : %d", (int)(fitpar[t][0])));
      tlpar[t][0]->SetTextSize(LATEXSIZE);
      tlpar[t][0]->SetTextColor(kBlack);
      tlpar[t][0]->Draw("same");
      tlpar[t][1] = new TLatex(range*0.25, max[t]*0.75, Form("mean : %4.4g", (fitpar[t][1])));
      tlpar[t][1]->SetTextSize(LATEXSIZE);
      if(TMath::Abs(fitpar[t][1])<0.005) tlpar[t][1]->SetTextColor(kBlack);
      else tlpar[t][1]->SetTextColor(kRed);
      tlpar[t][1]->Draw("same");
      tlpar[t][2] = new TLatex(range*0.25, max[t]*0.6, Form("stddev : %4.4g", (fitpar[t][2])));
      tlpar[t][2]->SetTextSize(LATEXSIZE);
      tlpar[t][2]->SetTextColor(kBlack);
      tlpar[t][2]->Draw("same");  
   }
   canvas_vtphi->SaveAs(Form("Plot_vtx-track_phi_by_pstavepTCut%g.gif",momCut),"gif");
}

void macro_plot_vtxz_diff_by_pstave(double rangeX = 0.1, TString ChargeOpt="", double momCut = 0.0){

   int max[nPSTAVE];

   TH1D* hdx[nPSTAVE];
   TF1*  fit[nPSTAVE];
   double fitpar[nPSTAVE][3]; // (entries, mean, sigma)
   TLatex *tlpar[nPSTAVE][3];

   TCanvas* canvas_vtphi = new TCanvas("canvas_vtphi","canvas_vtphi",5*600,4*400);
   canvas_vtphi->Divide((int)5,(int)4,0.001,0.001);
   
   for(int t = 0; t < 20; t++){
      canvas_vtphi->cd(1 + t);   

      canvas_vtphi->cd(1 + t)->SetGridx();
      canvas_vtphi->cd(1 + t)->SetGridy();
      gStyle->SetOptStat(0);

      double range = rangeX*5;
      double res = 0.005;  

      TString tFormula = "vtxfitZ-vtxZ";
      hdx[t] = new TH1D(Form("hdxpstv%d",t),Form("#DeltaVertex_{z} (pStave%d); #DeltaVertex_{z}; N",t),(int)2*(range/res),-range,+range);
      ResMonitor->Draw(Form("(%s)>>hdxpstv%d",(const char*)tFormula, t),
                       Form("TMath::Abs(fgPhi[2])<10&&TMath::Abs(%s)<%g&&fstv[2]==%d&&pT>=%g%s",(const char*)tFormula,range,t,momCut,(const char*)ChargeOpt));
      max[t] = hdx[t]->GetMaximum();                       
      fit[t] = new TF1(Form("fdxpstv%d",t),"gaus",-0.5*range,+0.5*range);

      hdx[t]->GetXaxis()->SetLabelSize(TEXTSIZE);
      hdx[t]->GetYaxis()->SetLabelSize(TEXTSIZE);
      hdx[t]->GetXaxis()->SetTitleSize(TEXTSIZE);
      hdx[t]->GetYaxis()->SetTitleSize(TEXTSIZE);
      hdx[t]->GetXaxis()->SetTitleOffset(1.0);         
      hdx[t]->GetYaxis()->SetTitleOffset(1.0);  

      hdx[t]->Fit(Form("fdxpstv%d",t));
      fitpar[t][0] = hdx[t]->GetEntries();
      fitpar[t][1] = fit[t]->GetParameter(1);
      fitpar[t][2] = fit[t]->GetParameter(2); 

      hdx[t]->SetMarkerColor(kBlack);         
      hdx[t]->SetLineColor(kBlack);           
      hdx[t]->Draw("");
      fit[t]->Draw("same");
      tlpar[t][0] = new TLatex(range*0.25, max[t]*0.9, Form("Entries : %d", (int)(fitpar[t][0])));
      tlpar[t][0]->SetTextSize(LATEXSIZE);
      tlpar[t][0]->SetTextColor(kBlack);
      tlpar[t][0]->Draw("same");
      tlpar[t][1] = new TLatex(range*0.25, max[t]*0.75, Form("mean : %4.4g", (fitpar[t][1])));
      tlpar[t][1]->SetTextSize(LATEXSIZE);
      tlpar[t][1]->SetTextColor(kBlack);
      tlpar[t][1]->Draw("same");
      tlpar[t][2] = new TLatex(range*0.25, max[t]*0.6, Form("stddev : %4.4g", (fitpar[t][2])));
      tlpar[t][2]->SetTextSize(LATEXSIZE);
      tlpar[t][2]->SetTextColor(kBlack);
      tlpar[t][2]->Draw("same");  
   }
   canvas_vtphi->SaveAs(Form("Plot_vtxz_diff_by_pstavepTCut%g.gif",momCut),"gif");
}

void macro_plot_pT_vs_vtx_track_r_by_pstave(double rangeX = 0.1, TString ChargeOpt="", double momCut = 0.0){

   int max[nPSTAVE];

   TH2D* hdx[nPSTAVE];
   TF1*  fit[nPSTAVE];
   double fitpar[nPSTAVE][3]; // (entries, mean, sigma)
   TLatex *tlpar[nPSTAVE][3];

   TCanvas* canvas_vtr = new TCanvas("canvas_vtr","canvas_vtr",5*600,4*400);
   canvas_vtr->Divide((int)5,(int)4,0.001,0.001);
   
   for(int t = 0; t < 20; t++){
      canvas_vtr->cd(1 + t);   

      canvas_vtr->cd(1 + t)->SetGridx();
      canvas_vtr->cd(1 + t)->SetGridy();
      gStyle->SetOptStat(111111);

      double range = 0.25*rangeX;
      double res = 0.25*0.002;  
           
      TString tFormula = "(vtxfitX*fgX[2]+vtxfitY*fgY[2])/TMath::Sqrt(fgX[2]*fgX[2]+fgY[2]*fgY[2])";
      hdx[t] = new TH2D(Form("hdxpstv%d",t),Form("p_{T} vs Normalized Vertex_{xy} #bullet Track_{xy} (pStave%d); dx; N",t),(int)2*(range/res),-range,+range,80,0,4);
      ResMonitor->Draw(Form("pT:(%s)>>hdxpstv%d",(const char*)tFormula, t),
                       Form("TMath::Abs(fgPhi[2])<10&&TMath::Abs(%s)<%g&&fstv[2]==%d&&pT>=%g%s",(const char*)tFormula,range,t,momCut,(const char*)ChargeOpt));

      hdx[t]->GetXaxis()->SetLabelSize(TEXTSIZE);
      hdx[t]->GetYaxis()->SetLabelSize(TEXTSIZE);
      hdx[t]->GetXaxis()->SetTitleSize(TEXTSIZE);
      hdx[t]->GetYaxis()->SetTitleSize(TEXTSIZE);
      hdx[t]->GetXaxis()->SetTitleOffset(1.0);         
      hdx[t]->GetYaxis()->SetTitleOffset(1.0);  

      hdx[t]->SetMarkerColor(kBlack);         
      hdx[t]->SetLineColor(kBlack);           
      hdx[t]->Draw("");

   }
   canvas_vtr->SaveAs(Form("Plot_pT_vs_vtx-track_r_by_pstavepTCut%g.gif",momCut),"gif");
}

void macro_plot_pT_vs_vtx_track_phi_by_pstave(double rangeX = 0.1, TString ChargeOpt="", double momCut = 0.0){

   int max[nPSTAVE];

   TH2D* hdx[nPSTAVE];

   TCanvas* canvas_vtphi = new TCanvas("canvas_vtphi","canvas_vtphi",5*600,4*400);
   canvas_vtphi->Divide((int)5,(int)4,0.001,0.001);
   
   for(int t = 0; t < 20; t++){
      canvas_vtphi->cd(1 + t);   

      canvas_vtphi->cd(1 + t)->SetGridx();
      canvas_vtphi->cd(1 + t)->SetGridy();
      gStyle->SetOptStat(111111);

      double range = rangeX;
      double res = 0.002;  

      TString tFormula = "(vtxfitX*fgY[2]-vtxfitY*fgX[2])/TMath::Sqrt(fgX[2]*fgX[2]+fgY[2]*fgY[2])";
      hdx[t] = new TH2D(Form("hdxpstv%d",t),Form("p_{T} vs Normalized Vertex_{xy} #times Track_{xy} (pStave%d); dx; N",t),(int)2*(range/res),-range,+range,80,0,4);
      ResMonitor->Draw(Form("pT:(%s)>>hdxpstv%d",(const char*)tFormula, t),
                       Form("TMath::Abs(fgPhi[2])<10&&TMath::Abs(%s)<%g&&fstv[2]==%d&&pT>=%g%s",(const char*)tFormula,range,t,momCut,(const char*)ChargeOpt));

      hdx[t]->GetXaxis()->SetLabelSize(TEXTSIZE);
      hdx[t]->GetYaxis()->SetLabelSize(TEXTSIZE);
      hdx[t]->GetXaxis()->SetTitleSize(TEXTSIZE);
      hdx[t]->GetYaxis()->SetTitleSize(TEXTSIZE);
      hdx[t]->GetXaxis()->SetTitleOffset(1.0);         
      hdx[t]->GetYaxis()->SetTitleOffset(1.0);  

      hdx[t]->SetMarkerColor(kBlack);         
      hdx[t]->SetLineColor(kBlack);           
      hdx[t]->Draw(""); 
   }
   canvas_vtphi->SaveAs(Form("Plot_pT_vs_vtx-track_phi_by_pstavepTCut%g.gif",momCut),"gif");
}

void macro_plot_vtx_all(double rangeX = 0.1, TString ChargeOpt="", double momCut = 0.0){

   macro_plot_vtx_track_phi_summary(rangeX,ChargeOpt,momCut);
   macro_plot_vtx_track_z_summary(rangeX,ChargeOpt,momCut);
   macro_plot_vtx_track_theta_summary(rangeX,ChargeOpt,momCut);   
   macro_plot_vtx_track_r_by_pstave(rangeX,ChargeOpt,momCut);
   macro_plot_vtx_track_phi_by_pstave(rangeX,ChargeOpt,momCut);
   macro_plot_pT_vs_vtx_track_r_by_pstave(rangeX,ChargeOpt,momCut);
   macro_plot_pT_vs_vtx_track_phi_by_pstave(rangeX,ChargeOpt,momCut);
   macro_plot_vtxz_diff_by_pstave(rangeX,ChargeOpt,momCut);
   macro_plot_impact_parameter_z_2D(0.06, 0.0, +1, true);
   macro_plot_impact_parameter_phi_2D(0.06, 0.0, +1, true);
}
				
void macro_plot_residuals_by_layer(double rangeIB1 = 0.015, double rangeIB2 = 0.015, double rangeOB1 = 0.050, double rangeOB2 = 0.015, TString ChargeOpt="", double momCut = 0.0){

   int max[nLAYER][2];

   TH1D* hds[nLAYER][2];
   TF1*  fit[nLAYER][2];
   double fitpar[nLAYER][2][3]; // Layer axis (entries, mean, sigma)
   TLatex *tlpar[nLAYER][2][4];

   TCanvas* canvasIB = new TCanvas("canvasIB","canvasIB",2*600,3*400);
   canvasIB->Divide((int)2,(int)3,0.001,0.001);
   
   for(int il = 0; il < nLAYERIB; il++){
      for(int ia = 0; ia < 2; ia++){

         int index = 2*il + ia + 1;

         canvasIB->cd(index);
         canvasIB->cd(index)->SetGridx();
         canvasIB->cd(index)->SetGridy();
         gStyle->SetOptStat(0);
         
         double range = ia==0 ? rangeIB1 : rangeIB2;
         double res = std::abs(range)<0.03 ? 0.0005 : 0.002;         
         hds[il][ia] = new TH1D(Form("hds%dL%d",ia+1,il),Form("ds%d (Layer%d); ds%d(cm); N",ia+1,il,ia+1),(int)2*(range/res),-range,+range);
         ResMonitor->Draw(Form("fds%d[%d]>>hds%dL%d",ia+1,il,ia+1,il),Form("TMath::Abs(fds%d[%d][])<%g&&pT>=%g%s",ia+1,il,range,momCut,(const char*)ChargeOpt));
         max[il][ia] = hds[il][ia]->GetMaximum();
         fit[il][ia] = new TF1(Form("fds%dL%d",ia+1,il),"gaus",-range,+range);
    
         hds[il][ia]->GetXaxis()->SetLabelSize(TEXTSIZE);
         hds[il][ia]->GetYaxis()->SetLabelSize(TEXTSIZE);
         hds[il][ia]->GetXaxis()->SetTitleSize(TEXTSIZE);
         hds[il][ia]->GetYaxis()->SetTitleSize(TEXTSIZE);
         hds[il][ia]->GetXaxis()->SetTitleOffset(1.0);         
         hds[il][ia]->GetYaxis()->SetTitleOffset(1.0);         

         hds[il][ia]->Fit(Form("fds%dL%d",ia+1,il));
         fitpar[il][ia][0] = hds[il][ia]->GetEntries();
         fitpar[il][ia][1] = fit[il][ia]->GetParameter(1);
         fitpar[il][ia][2] = fit[il][ia]->GetParameter(2); 

         hds[il][ia]->SetMarkerColor(kBlack);         
         hds[il][ia]->SetLineColor(kBlack);           
         hds[il][ia]->Draw("");
         fit[il][ia]->Draw("same");
         tlpar[il][ia][0] = new TLatex(range*0.25, max[il][ia]*0.9, Form("Entries : %d", (int)(fitpar[il][ia][0])));
         tlpar[il][ia][0]->SetTextSize(LATEXSIZE);
         tlpar[il][ia][0]->SetTextColor(kBlack);
         tlpar[il][ia][0]->Draw("same");
         tlpar[il][ia][1] = new TLatex(range*0.25, max[il][ia]*0.75, Form("mean : %4.4g", (fitpar[il][ia][1])));
         tlpar[il][ia][1]->SetTextSize(LATEXSIZE);
         tlpar[il][ia][1]->SetTextColor(kBlack);
         tlpar[il][ia][1]->Draw("same");
         tlpar[il][ia][2] = new TLatex(range*0.25, max[il][ia]*0.6, Form("stddev : %4.4g", (fitpar[il][ia][2])));
         tlpar[il][ia][2]->SetTextSize(LATEXSIZE);
         tlpar[il][ia][2]->SetTextColor(kBlack);
         tlpar[il][ia][2]->Draw("same");
         if(TMath::Abs(momCut)<1e-5) { 
            tlpar[il][ia][3] = new TLatex(range*0.25, max[il][ia]*0.45, Form("stddev : %4.4g", ((ia == 0) ? data_simul_ds1_IB[il] : data_simul_ds2_IB[il])));
         } else if(TMath::Abs(momCut-0.3)<1e-5) {
            tlpar[il][ia][3] = new TLatex(range*0.25, max[il][ia]*0.45, Form("stddev : %4.4g", ((ia == 0) ? data_simul_ds1_IB_pTCut300MeV[il] : data_simul_ds2_IB_pTCut300MeV[il])));
         } else if(TMath::Abs(momCut-0.5)<1e-5) {
            tlpar[il][ia][3] = new TLatex(range*0.25, max[il][ia]*0.45, Form("stddev : %4.4g", ((ia == 0) ? data_simul_ds1_IB_pTCut500MeV[il] : data_simul_ds2_IB_pTCut500MeV[il])));
         } else if(TMath::Abs(momCut-1.0)<1e-5) {
            tlpar[il][ia][3] = new TLatex(range*0.25, max[il][ia]*0.45, Form("stddev : %4.4g", ((ia == 0) ? data_simul_ds1_IB_pTCut1GeV[il] : data_simul_ds2_IB_pTCut1GeV[il])));
         } else if(TMath::Abs(momCut-2.0)<1e-5) {
            tlpar[il][ia][3] = new TLatex(range*0.25, max[il][ia]*0.45, Form("stddev : %4.4g", ((ia == 0) ? data_simul_ds1_IB_pTCut2GeV[il] : data_simul_ds2_IB_pTCut2GeV[il])));
         }
         tlpar[il][ia][3]->SetTextSize(LATEXSIZE);
         tlpar[il][ia][3]->SetTextColor(kBlue);
         tlpar[il][ia][3]->Draw("same");              
      } 
   }
   canvasIB->SaveAs(Form("Plot_ds_IB_pTCut%g.gif",momCut),"gif");

   TCanvas* canvasOB = new TCanvas("canvasOB","canvasOB",2*600,4*400);
   canvasOB->Divide((int)2,(int)4,0.001,0.001);
   
   for(int il = nLAYERIB; il < nLAYER; il++){
      for(int ia = 0; ia < 2; ia++){

         int index = 2*(il - nLAYERIB) + ia + 1;

         canvasOB->cd(index);
         canvasOB->cd(index)->SetGridx();
         canvasOB->cd(index)->SetGridy();
         gStyle->SetOptStat(0);
         
         double range = ia==0 ? rangeOB1 : rangeOB2;
         double res = std::abs(range)<0.03 ? 0.0005 : 0.002;
         hds[il][ia] = new TH1D(Form("hds%dL%d",ia+1,il),Form("ds%d (Layer%d); ds%d(cm); N",ia+1,il,ia+1),(int)2*(range/res),-range,+range);
         ResMonitor->Draw(Form("fds%d[%d]>>hds%dL%d",ia+1,il,ia+1,il),Form("TMath::Abs(fds%d[%d][])<%g&&pT>=%g%s",ia+1,il,range,momCut,(const char*)ChargeOpt));
         max[il][ia] = hds[il][ia]->GetMaximum();
         fit[il][ia] = new TF1(Form("fds%dL%d",ia+1,il),"gaus",-range,+range);
     
         hds[il][ia]->GetXaxis()->SetLabelSize(TEXTSIZE);
         hds[il][ia]->GetYaxis()->SetLabelSize(TEXTSIZE);
         hds[il][ia]->GetXaxis()->SetTitleSize(TEXTSIZE);
         hds[il][ia]->GetYaxis()->SetTitleSize(TEXTSIZE);
         hds[il][ia]->GetXaxis()->SetTitleOffset(1.0);         
         hds[il][ia]->GetYaxis()->SetTitleOffset(1.0);         

         hds[il][ia]->Fit(Form("fds%dL%d",ia+1,il));
         fitpar[il][ia][0] = hds[il][ia]->GetEntries();
         fitpar[il][ia][1] = fit[il][ia]->GetParameter(1);
         fitpar[il][ia][2] = fit[il][ia]->GetParameter(2); 
     
         hds[il][ia]->SetMarkerColor(kBlack);         
         hds[il][ia]->SetLineColor(kBlack);           
         hds[il][ia]->Draw("");
         tlpar[il][ia][0] = new TLatex(range*0.25, max[il][ia]*0.9, Form("Entries : %d", (int)(fitpar[il][ia][0])));
         tlpar[il][ia][0]->SetTextSize(LATEXSIZE);
         tlpar[il][ia][0]->SetTextColor(kBlack);
         tlpar[il][ia][0]->Draw("same");
         tlpar[il][ia][1] = new TLatex(range*0.25, max[il][ia]*0.75, Form("mean : %4.4g", (fitpar[il][ia][1])));
         tlpar[il][ia][1]->SetTextSize(LATEXSIZE);
         tlpar[il][ia][1]->SetTextColor(kBlack);
         tlpar[il][ia][1]->Draw("same");
         tlpar[il][ia][2] = new TLatex(range*0.25, max[il][ia]*0.6, Form("stddev : %4.4g", (fitpar[il][ia][2])));
         tlpar[il][ia][2]->SetTextSize(LATEXSIZE);
         tlpar[il][ia][2]->SetTextColor(kBlack);
         tlpar[il][ia][2]->Draw("same");  
         if(TMath::Abs(momCut)<1e-5) { 
            tlpar[il][ia][3] = new TLatex(range*0.25, max[il][ia]*0.45, Form("stddev : %4.4g", ((ia == 0) ? data_simul_ds1_OB[il-nLAYERIB] : data_simul_ds2_OB[il-nLAYERIB])));
         } else if(TMath::Abs(momCut-0.3)<1e-5) {
            tlpar[il][ia][3] = new TLatex(range*0.25, max[il][ia]*0.45, Form("stddev : %4.4g", ((ia == 0) ? data_simul_ds1_OB_pTCut300MeV[il-nLAYERIB] : data_simul_ds2_OB_pTCut300MeV[il-nLAYERIB])));
         } else if(TMath::Abs(momCut-0.5)<1e-5) {
            tlpar[il][ia][3] = new TLatex(range*0.25, max[il][ia]*0.45, Form("stddev : %4.4g", ((ia == 0) ? data_simul_ds1_OB_pTCut500MeV[il-nLAYERIB] : data_simul_ds2_OB_pTCut500MeV[il-nLAYERIB])));
         } else if(TMath::Abs(momCut-1.0)<1e-5) {
            tlpar[il][ia][3] = new TLatex(range*0.25, max[il][ia]*0.45, Form("stddev : %4.4g", ((ia == 0) ? data_simul_ds1_OB_pTCut1GeV[il-nLAYERIB] : data_simul_ds2_OB_pTCut1GeV[il-nLAYERIB])));
         } else if(TMath::Abs(momCut-2.0)<1e-5) {
            tlpar[il][ia][3] = new TLatex(range*0.25, max[il][ia]*0.45, Form("stddev : %4.4g", ((ia == 0) ? data_simul_ds1_OB_pTCut2GeV[il-nLAYERIB] : data_simul_ds2_OB_pTCut2GeV[il-nLAYERIB])));
         }
         tlpar[il][ia][3]->SetTextSize(LATEXSIZE);
         tlpar[il][ia][3]->SetTextColor(kBlue);
         tlpar[il][ia][3]->Draw("same");  
      } 
   }
   canvasOB->SaveAs(Form("Plot_ds_OB_pTCut%g.gif",momCut),"gif");

}

void macro_plot_residuals_by_layer_with_momentum(double rangeIB1 = 0.05, double rangeIB2 = 0.05, double rangeOB1 = 0.2, double rangeOB2 = 0.05, TString ChargeOpt="", double momCut = 20.0, double momBin = 0.1){

   TH2D* hpds[nLAYER][2];
   TH1D* hsigma[nLAYER][2];
   
   TF1* fit_gaus = new TF1("fit_gaus","gaus");
   
   TF1*  fit[nLAYER][2];
   double fitpar[nLAYER][2][3]; // Layer axis (mean, sigma)
   TLatex *tlpar[nLAYER][2][2];

   TCanvas* canvas = new TCanvas("canvas","canvas",2*800,2*400);

   
   for(int il = 0; il < nLAYER; il++){
      canvas->Clear();   
      canvas->Divide((int)2,(int)2,0.001,0.001);
      for(int ia = 0; ia < 2; ia++){

         int index = ia + 1;

         canvas->cd(index);
         canvas->cd(index)->SetGridx();
         canvas->cd(index)->SetGridy();
         gStyle->SetOptStat(1111);
         
         double range = ia==0 ? ( il<3 ? rangeIB1 : rangeOB1 ) : ( il<3 ? rangeIB2 : rangeOB2 );
         double res = std::abs(range)<0.1 ? 0.0005 : 0.002;         
         hpds[il][ia] = new TH2D(Form("hp_vs_ds%dL%d",ia+1,il),Form("p vs ds%d (Layer%d); ds%d(cm); p(GeV)",ia+1,il,ia+1),(int)2*(range/res),-range,+range,(int)(momCut/momBin),0,momCut);
         ResMonitor->Draw(Form("p:fds%d[%d][]>>hp_vs_ds%dL%d",ia+1,il,ia+1,il),Form("TMath::Abs(fds%d[%d][])<%g%s",ia+1,il,range,(const char*)ChargeOpt));
    
         hpds[il][ia]->GetXaxis()->SetLabelSize(TEXTSIZE);
         hpds[il][ia]->GetYaxis()->SetLabelSize(TEXTSIZE);
         hpds[il][ia]->GetXaxis()->SetTitleSize(TEXTSIZE);
         hpds[il][ia]->GetYaxis()->SetTitleSize(TEXTSIZE);
         hpds[il][ia]->GetXaxis()->SetTitleOffset(1.0);         
         hpds[il][ia]->GetYaxis()->SetTitleOffset(1.1);         

         hpds[il][ia]->SetMarkerColor(kBlack);         
         hpds[il][ia]->SetLineColor(kBlack);           
         hpds[il][ia]->Draw("");
               
      } 
      for(int ia = 0; ia < 2; ia++){
         int index = ia + 3;

         canvas->cd(index);
         canvas->cd(index)->SetGridx();
         canvas->cd(index)->SetGridy();   

         double range = ia==0 ? ( il<3 ? rangeIB1 : rangeOB1 ) : ( il<3 ? rangeIB2 : rangeOB2 );
         double res = std::abs(range)<0.1 ? 0.0005 : 0.002;             
         hsigma[il][ia] = new TH1D(Form("hfit%dL%d",ia+1,il),Form("#sigma(ds%d) vs p (Layer%d); p(GeV); #sigma(ds%d)(cm)",ia+1,il,ia+1),(int)(momCut/momBin),0,momCut);
         for(int pbin = 0; pbin < (int)(momCut/momBin); pbin++){
            if(hpds[il][ia]->ProjectionX("",pbin+1,pbin+1)->GetEntries()<100) continue;
            hpds[il][ia]->ProjectionX("",pbin+1,pbin+1)->Fit(fit_gaus);
            double mean = fit_gaus->GetParameter(1);
            double sigma = fit_gaus->GetParameter(2);
            hsigma[il][ia]->SetBinContent(pbin+1, sigma);
         }
         
         //fit[il][ia] = new TF1(Form("fpInv_s%dL%d",ia+1,il),"[0]/x",0.25,5.0);          
         fit[il][ia] = new TF1(Form("fpInv_s%dL%d",ia+1,il),"TMath::Sqrt([0]*[0] + (([1]*[1])/(x*x)))",0.25,5.0);         
         double sigma_range = ia==0 ? ( il<3 ? 0.005 : 0.05 ) : ( il<3 ? 0.005 : 0.05 );
         hsigma[il][ia]->Fit(fit[il][ia]);
         fit[il][ia]->SetRange(0.2,20);
         
         hsigma[il][ia]->SetMaximum(sigma_range);
         hsigma[il][ia]->GetXaxis()->SetLabelSize(TEXTSIZE);
         hsigma[il][ia]->GetYaxis()->SetLabelSize(TEXTSIZE);
         hsigma[il][ia]->GetXaxis()->SetTitleSize(TEXTSIZE);
         hsigma[il][ia]->GetYaxis()->SetTitleSize(TEXTSIZE);
         hsigma[il][ia]->GetXaxis()->SetTitleOffset(1.0);         
         hsigma[il][ia]->GetYaxis()->SetTitleOffset(1.1);         

         hsigma[il][ia]->GetYaxis()->SetNdivisions(505);         

         hsigma[il][ia]->SetMarkerColor(kBlack);         
         hsigma[il][ia]->SetLineColor(kBlack);           
         hsigma[il][ia]->Draw("");     

         tlpar[il][ia][0] = new TLatex(momCut*0.6, sigma_range*0.80, Form("p0 : %4.4g", fit[il][ia]->GetParameter(0)));
         tlpar[il][ia][0]->SetTextSize(LATEXSIZE);
         tlpar[il][ia][0]->SetTextColor(kRed);
         tlpar[il][ia][0]->Draw("same");
         tlpar[il][ia][1] = new TLatex(momCut*0.6, sigma_range*0.65, Form("p1 : %4.4g", fit[il][ia]->GetParameter(1)));
         tlpar[il][ia][1]->SetTextSize(LATEXSIZE);
         tlpar[il][ia][1]->SetTextColor(kRed);
         tlpar[il][ia][1]->Draw("same");
         
      }
      
      canvas->SaveAs(Form("Plot_p_vs_ds_Layer%d.gif",il),"gif");
   }


}


void macro1(int patchID = 28, int mode = 0, double range = 0.015){

   int Nds = -1;
   int NchipID = -1;
   if(mode == 0){ 		//IB vs IB
      Nds 	= nLAYERIB;
      NchipID	= nLAYERIB;
   } else if (mode == 1){	//IB vs OB
      Nds 	= nLAYERIB;
      NchipID	= nLAYEROB;  
   } else if (mode == 2){	//OB vs IB
      Nds 	= nLAYEROB;
      NchipID	= nLAYERIB;
   } else if (mode == 3){	//OB vs OB
      Nds 	= nLAYEROB;
      NchipID	= nLAYEROB; 
   } else {
      return;
   }

   TCanvas* canvas = new TCanvas("canvas","canvas",NchipID*600,Nds*400);
   canvas->Divide((int)NchipID,(int)Nds,0.001,0.001);

   TH2D* hds1_vs_chipID[Nds][NchipID];
   
   for(int id = 0; id < nLAYERIB; id++){
      for(int ic = 0; ic < nLAYERIB; ic++){

         int index = nLAYERIB*id + ic + 1;

         canvas->cd(index);
         canvas->cd(index)->SetGridx();
         canvas->cd(index)->SetGridy();

         hds1_vs_chipID[id][ic] 
         = new TH2D(Form("hds1L%d_vs_chipIDL%d",id,ic),Form("ds1(L%d) vs_chipID(L%d); chipID; ds1(cm)",id,ic),
         nSensorsbyLayer[ic],ChipBoundary[ic],ChipBoundary[ic+1],(int)2*(range/0.001),-range,+range);
         ResMonitor->Draw(Form("fds1[%d][]:fchipID[%d][]>>hds1L%d_vs_chipIDL%d",id,ic,id,ic),Form("TMath::Abs(fds1[%d][])<%g&&(fchipID[%d][]>=0)",id,range,ic));
         
         double xmin = ChipBoundary[ic];
         double xmax = ChipBoundary[ic+1];
         bool find_xmin = false;
         bool find_xmax = false;
         
         TH1D* hprojX = (TH1D*) hds1_vs_chipID[id][ic]->ProjectionX()->Clone();

         for(int ix = 0; ix < nSensorsbyLayer[ic]; ix++){
            if(find_xmin==true && find_xmax==true) continue;
            if(find_xmin==false) {
               if(hprojX->GetBinContent(ix+1)>0) {
                  find_xmin = true;
                  xmin = ix;
               }
            }
            if(find_xmax==false) {
               if(hprojX->GetBinContent(nSensorsbyLayer[ic] - ix)>0) {
                  find_xmax = true;
                  xmax = nSensorsbyLayer[ic] - ix - 1;
               }
            }           
         }
         
         int min_stave = (int) xmin/(nHicPerStave[ic]*nChipsPerHic[ic]);
         int max_stave = (int) xmax/(nHicPerStave[ic]*nChipsPerHic[ic]) + 1;
         hds1_vs_chipID[id][ic]->GetXaxis()->SetLabelSize(TEXTSIZE);
         hds1_vs_chipID[id][ic]->GetYaxis()->SetLabelSize(TEXTSIZE);
         hds1_vs_chipID[id][ic]->GetXaxis()->SetTitleSize(TEXTSIZE);
         hds1_vs_chipID[id][ic]->GetYaxis()->SetTitleSize(TEXTSIZE);
         hds1_vs_chipID[id][ic]->GetXaxis()->SetTitleOffset(1.0);         
         hds1_vs_chipID[id][ic]->GetYaxis()->SetTitleOffset(1.0);         
         
         hds1_vs_chipID[id][ic]->GetXaxis()->SetRangeUser(ChipBoundary[ic] + nHicPerStave[ic]*nChipsPerHic[ic]*min_stave, ChipBoundary[ic] + nHicPerStave[ic]*nChipsPerHic[ic]*max_stave);
         hds1_vs_chipID[id][ic]->SetMarkerColor(kBlack);         
         hds1_vs_chipID[id][ic]->SetLineColor(kBlack);           
         hds1_vs_chipID[id][ic]->Draw("box");
      } 
   }
   canvas->SaveAs(Form("Plot_ds1_vs_chipID_P%03d_mode%d.gif",patchID,mode),"gif");

}

void macro2(int patchID = 28, int mode = 0, double range = 0.015){

   int Nds = -1;
   int NchipID = -1;
   if(mode == 0){ 		//IB vs IB
      Nds 	= nLAYERIB;
      NchipID	= nLAYERIB;
   } else if (mode == 1){	//IB vs OB
      Nds 	= nLAYERIB;
      NchipID	= nLAYEROB;  
   } else if (mode == 2){	//OB vs IB
      Nds 	= nLAYEROB;
      NchipID	= nLAYERIB;
   } else if (mode == 3){	//OB vs OB
      Nds 	= nLAYEROB;
      NchipID	= nLAYEROB; 
   } else {
      return;
   }

   TCanvas* canvas = new TCanvas("canvas","canvas",NchipID*600,Nds*400);
   canvas->Divide((int)NchipID,(int)Nds,0.001,0.001);

   TH2D* hds2_vs_chipID[Nds][NchipID];
   
   for(int id = 0; id < nLAYERIB; id++){
      for(int ic = 0; ic < nLAYERIB; ic++){

         int index = nLAYERIB*id + ic + 1;

         canvas->cd(index);
         canvas->cd(index)->SetGridx();
         canvas->cd(index)->SetGridy();

         hds2_vs_chipID[id][ic] 
         = new TH2D(Form("hds2L%d_vs_chipIDL%d",id,ic),Form("ds2(L%d) vs_chipID(L%d); chipID; ds2(cm)",id,ic),
         nSensorsbyLayer[ic],ChipBoundary[ic],ChipBoundary[ic+1],(int)2*(range/0.001),-range,+range);
         ResMonitor->Draw(Form("fds2[%d][]:fchipID[%d][]>>hds2L%d_vs_chipIDL%d",id,ic,id,ic),Form("TMath::Abs(fds2[%d][])<%g&&(fchipID[%d][]>=0)",id,range,ic));
         
         double xmin = ChipBoundary[ic];
         double xmax = ChipBoundary[ic+1];
         bool find_xmin = false;
         bool find_xmax = false;
         
         TH1D* hprojX = (TH1D*) hds2_vs_chipID[id][ic]->ProjectionX()->Clone();

         for(int ix = 0; ix < nSensorsbyLayer[ic]; ix++){
            if(find_xmin==true && find_xmax==true) continue;
            if(find_xmin==false) {
               if(hprojX->GetBinContent(ix+1)>0) {
                  find_xmin = true;
                  xmin = ix;
               }
            }
            if(find_xmax==false) {
               if(hprojX->GetBinContent(nSensorsbyLayer[ic] - ix)>0) {
                  find_xmax = true;
                  xmax = nSensorsbyLayer[ic] - ix - 1;
               }
            }           
         }
         
         int min_stave = (int) xmin/(nHicPerStave[ic]*nChipsPerHic[ic]);
         int max_stave = (int) xmax/(nHicPerStave[ic]*nChipsPerHic[ic]) + 1;
         hds2_vs_chipID[id][ic]->GetXaxis()->SetLabelSize(TEXTSIZE);
         hds2_vs_chipID[id][ic]->GetYaxis()->SetLabelSize(TEXTSIZE);
         hds2_vs_chipID[id][ic]->GetXaxis()->SetTitleSize(TEXTSIZE);
         hds2_vs_chipID[id][ic]->GetYaxis()->SetTitleSize(TEXTSIZE);
         hds2_vs_chipID[id][ic]->GetXaxis()->SetTitleOffset(1.0);         
         hds2_vs_chipID[id][ic]->GetYaxis()->SetTitleOffset(1.0);         
         
         hds2_vs_chipID[id][ic]->GetXaxis()->SetRangeUser(ChipBoundary[ic] + nHicPerStave[ic]*nChipsPerHic[ic]*min_stave, ChipBoundary[ic] + nHicPerStave[ic]*nChipsPerHic[ic]*max_stave);
         hds2_vs_chipID[id][ic]->SetMarkerColor(kBlack);         
         hds2_vs_chipID[id][ic]->SetLineColor(kBlack);  
         hds2_vs_chipID[id][ic]->Draw("box");
      } 
   }
   canvas->SaveAs(Form("Plot_ds2_vs_chipID_P%03d_mode%d.gif",patchID,mode),"gif");

}


void macro_plot_Residual2D(){

   TH2D* hIB_ds1_vs_z[3];
   TH2D* hIB_ds1_vs_phi[3];
   TH2D* hOB_ds1_vs_z[4];
   TH2D* hOB_ds1_vs_phi[4];   
   TH2D* hIB_ds2_vs_z[3];
   TH2D* hIB_ds2_vs_phi[3];
   TH2D* hOB_ds2_vs_z[4];
   TH2D* hOB_ds2_vs_phi[4];    

   TProfile* pIB_ds1_vs_z[3];
   TProfile* pIB_ds1_vs_phi[3];
   TProfile* pOB_ds1_vs_z[4];
   TProfile* pOB_ds1_vs_phi[4];   
   TProfile* pIB_ds2_vs_z[3];
   TProfile* pIB_ds2_vs_phi[3];
   TProfile* pOB_ds2_vs_z[4];
   TProfile* pOB_ds2_vs_phi[4];  

   double mrangeZ[] = {15, 15, 15, 50, 50, 85, 85};

   double mrangeD[] = {0.06, 0.06, 0.06, 0.1, 0.1, 0.1, 0.1};
   for(int iIB = 0; iIB < nLAYERIB; iIB++){ 
      int idx = iIB;
      
      hIB_ds1_vs_z[idx]   
      = new TH2D(Form("hIB_ds1_vs_z_L%d",iIB),Form("ds_{1} vs z Layer %d",iIB),600,-mrangeZ[iIB],+mrangeZ[iIB],400,-mrangeD[iIB],+mrangeD[iIB]);
      hIB_ds1_vs_phi[idx] 
      = new TH2D(Form("hIB_ds1_vs_phi_L%d",iIB),Form("ds_{1} vs #phi Layer %d",iIB),600,-TMath::Pi(),+TMath::Pi(),400,-mrangeD[iIB],+mrangeD[iIB]);     
      hIB_ds2_vs_z[idx]   
      = new TH2D(Form("hIB_ds2_vs_z_L%d",iIB),Form("ds_{1} vs z Layer %d",iIB),600,-mrangeZ[iIB],+mrangeZ[iIB],400,-mrangeD[iIB],+mrangeD[iIB]);
      hIB_ds2_vs_phi[idx] 
      = new TH2D(Form("hIB_ds2_vs_phi_L%d",iIB),Form("ds_{1} vs #phi Layer %d",iIB),600,-TMath::Pi(),+TMath::Pi(),400,-mrangeD[iIB],+mrangeD[iIB]);     

      pIB_ds1_vs_z[idx]   
      = new TProfile(Form("pIB_ds1_vs_z_L%d",iIB),Form("ds_{1} vs z Profile Layer %d",iIB),4*nHicPerStave[iIB]*nChipsPerHic[iIB],-mrangeZ[iIB],+mrangeZ[iIB],-mrangeD[iIB],+mrangeD[iIB]);
      pIB_ds1_vs_phi[idx] 
      = new TProfile(Form("pIB_ds1_vs_phi_L%d",iIB),Form("ds_{1} vs #phi Profile Layer %d",iIB),4*nHicPerStave[iIB]*nChipsPerHic[iIB],-TMath::Pi(),+TMath::Pi(),-mrangeD[iIB],+mrangeD[iIB]);
      pIB_ds2_vs_z[idx]   
      = new TProfile(Form("pIB_ds2_vs_z_L%d",iIB),Form("ds_{1} vs z Profile Layer %d",iIB),4*nHicPerStave[iIB]*nChipsPerHic[iIB],-mrangeZ[iIB],+mrangeZ[iIB],-mrangeD[iIB],+mrangeD[iIB]);
      pIB_ds2_vs_phi[idx] 
      = new TProfile(Form("pIB_ds2_vs_phi_L%d",iIB),Form("ds_{1} vs #phi Profile Layer %d",iIB),4*nHicPerStave[iIB]*nChipsPerHic[iIB],-TMath::Pi(),+TMath::Pi(),-mrangeD[iIB],+mrangeD[iIB]);
   }  
   
   for(int iOB = nLAYERIB; iOB < nLAYER; iOB++){
      int idx = iOB - nLAYERIB;

      hOB_ds1_vs_z[idx]   
      = new TH2D(Form("hOB_ds1_vs_z_L%d",iOB),Form("ds_{1} vs z Layer %d",iOB),600,-mrangeZ[iOB],+mrangeZ[iOB],400,-mrangeD[iOB],+mrangeD[iOB]);
      hOB_ds1_vs_phi[idx] 
      = new TH2D(Form("hOB_ds1_vs_phi_L%d",iOB),Form("ds_{1} vs #phi Layer %d",iOB),600,-TMath::Pi(),+TMath::Pi(),400,-mrangeD[iOB],+mrangeD[iOB]);     
      hOB_ds2_vs_z[idx]   
      = new TH2D(Form("hOB_ds2_vs_z_L%d",iOB),Form("ds_{1} vs z Layer %d",iOB),600,-mrangeZ[iOB],+mrangeZ[iOB],400,-mrangeD[iOB],+mrangeD[iOB]);
      hOB_ds2_vs_phi[idx] 
      = new TH2D(Form("hOB_ds2_vs_phi_L%d",iOB),Form("ds_{1} vs #phi Layer %d",iOB),600,-TMath::Pi(),+TMath::Pi(),400,-mrangeD[iOB],+mrangeD[iOB]);     

      pOB_ds1_vs_z[idx]   
      = new TProfile(Form("pOB_ds1_vs_z_L%d",iOB),Form("ds_{1} vs z Profile Layer %d",iOB),nHicPerStave[iOB]*nChipsPerHic[iOB]/7,-mrangeZ[iOB],+mrangeZ[iOB],-mrangeD[iOB],+mrangeD[iOB]);
      pOB_ds1_vs_phi[idx] 
      = new TProfile(Form("pOB_ds1_vs_phi_L%d",iOB),Form("ds_{1} vs #phi Profile Layer %d",iOB),nHicPerStave[iOB]*nChipsPerHic[iOB]/7,-TMath::Pi(),+TMath::Pi(),-mrangeD[iOB],+mrangeD[iOB]);
      pOB_ds2_vs_z[idx]   
      = new TProfile(Form("pOB_ds2_vs_z_L%d",iOB),Form("ds_{1} vs z Profile Layer %d",iOB),nHicPerStave[iOB]*nChipsPerHic[iOB]/7,-mrangeZ[iOB],+mrangeZ[iOB],-mrangeD[iOB],+mrangeD[iOB]);
      pOB_ds2_vs_phi[idx] 
      = new TProfile(Form("pOB_ds2_vs_phi_L%d",iOB),Form("ds_{1} vs #phi Profile Layer %d",iOB),nHicPerStave[iOB]*nChipsPerHic[iOB]/7,-TMath::Pi(),+TMath::Pi(),-mrangeD[iOB],+mrangeD[iOB]);
   }   
            

   TCanvas *canvas_IB1 = new TCanvas("canvas_IB1","canvas_IB1",1600,900);
   canvas_IB1->Divide(2,3);

   canvas_IB1->cd(1);
   gStyle->SetOptStat(0);
   ResMonitor->Draw("fds1[0]:fgZ[0]>>hIB_ds1_vs_z_L0","fchipID[0]>=0&&TMath::Abs(fds1[0])<0.06","goff");
   ResMonitor->Draw("fds1[0]:fgZ[0]>>pIB_ds1_vs_z_L0","fchipID[0]>=0&&TMath::Abs(fds1[0])<0.06","goff");   
   hIB_ds1_vs_z[0]->Draw();
   pIB_ds1_vs_z[0]->Draw("same");
   canvas_IB1->cd(2);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds1[0]:fgPhi[0]>>hIB_ds1_vs_phi_L0","fchipID[0]>=0&&TMath::Abs(fds1[0])<0.06","goff");
   ResMonitor->Draw("fds1[0]:fgPhi[0]>>pIB_ds1_vs_phi_L0","fchipID[0]>=0&&TMath::Abs(fds1[0])<0.06","goff");
   hIB_ds1_vs_phi[0]->Draw();
   pIB_ds1_vs_phi[0]->Draw("same");
   
   canvas_IB1->cd(3);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds1[1]:fgZ[1]>>hIB_ds1_vs_z_L1","fchipID[1]>=0&&TMath::Abs(fds1[1])<0.06","goff");
   ResMonitor->Draw("fds1[1]:fgZ[1]>>pIB_ds1_vs_z_L1","fchipID[1]>=0&&TMath::Abs(fds1[1])<0.06","goff");   
   hIB_ds1_vs_z[1]->Draw();   
   pIB_ds1_vs_z[1]->Draw("same");      
   canvas_IB1->cd(4);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds1[1]:fgPhi[1]>>hIB_ds1_vs_phi_L1","fchipID[1]>=0&&TMath::Abs(fds1[1])<0.06","goff");
   ResMonitor->Draw("fds1[1]:fgPhi[1]>>pIB_ds1_vs_phi_L1","fchipID[1]>=0&&TMath::Abs(fds1[1])<0.06","goff");   
   hIB_ds1_vs_phi[1]->Draw();
   pIB_ds1_vs_phi[1]->Draw("same");
    
   canvas_IB1->cd(5);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds1[2]:fgZ[2]>>hIB_ds1_vs_z_L2","fchipID[2]>=0&&TMath::Abs(fds1[2])<0.06","goff");
   ResMonitor->Draw("fds1[2]:fgZ[2]>>pIB_ds1_vs_z_L2","fchipID[2]>=0&&TMath::Abs(fds1[2])<0.06","goff");   
   hIB_ds1_vs_z[2]->Draw(); 
   pIB_ds1_vs_z[2]->Draw("same"); 
   canvas_IB1->cd(6);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds1[2]:fgPhi[2]>>hIB_ds1_vs_phi_L2","fchipID[2]>=0&&TMath::Abs(fds1[2])<0.06","goff");
   ResMonitor->Draw("fds1[2]:fgPhi[2]>>pIB_ds1_vs_phi_L2","fchipID[2]>=0&&TMath::Abs(fds1[2])<0.06","goff");
   hIB_ds1_vs_phi[2]->Draw(); 
   pIB_ds1_vs_phi[2]->Draw("same"); 
   canvas_IB1->SaveAs("Res_ds1_IB.gif","gif");
   
   TCanvas *canvas_OB1 = new TCanvas("canvas_OB1","canvas_OB1",1600,1200);
   canvas_OB1->Divide(2,4);
   canvas_OB1->cd(1);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds1[3]:fgZ[3]>>hOB_ds1_vs_z_L3","fchipID[3]>=0&&TMath::Abs(fds1[3])<0.1","goff");
   ResMonitor->Draw("fds1[3]:fgZ[3]>>pOB_ds1_vs_z_L3","fchipID[3]>=0&&TMath::Abs(fds1[3])<0.1","goff");   
   hOB_ds1_vs_z[0]->Draw();  
   pOB_ds1_vs_z[0]->Draw("same");       
   canvas_OB1->cd(2);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds1[3]:fgPhi[3]>>hOB_ds1_vs_phi_L3","fchipID[3]>=0&&TMath::Abs(fds1[3])<0.1","goff");
   ResMonitor->Draw("fds1[3]:fgPhi[3]>>pOB_ds1_vs_phi_L3","fchipID[3]>=0&&TMath::Abs(fds1[3])<0.1","goff");   
   hOB_ds1_vs_phi[0]->Draw();
   pOB_ds1_vs_phi[0]->Draw("same");
   
   canvas_OB1->cd(3);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds1[4]:fgZ[4]>>hOB_ds1_vs_z_L4","fchipID[4]>=0&&TMath::Abs(fds1[4])<0.1","goff");
   ResMonitor->Draw("fds1[4]:fgZ[4]>>pOB_ds1_vs_z_L4","fchipID[4]>=0&&TMath::Abs(fds1[4])<0.1","goff");   
   hOB_ds1_vs_z[1]->Draw();      
   pOB_ds1_vs_z[1]->Draw("same");         
   canvas_OB1->cd(4);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds1[4]:fgPhi[4]>>hOB_ds1_vs_phi_L4","fchipID[4]>=0&&TMath::Abs(fds1[4])<0.1","goff");
   ResMonitor->Draw("fds1[4]:fgPhi[4]>>pOB_ds1_vs_phi_L4","fchipID[4]>=0&&TMath::Abs(fds1[4])<0.1","goff");   
   hOB_ds1_vs_phi[1]->Draw();   
   pOB_ds1_vs_phi[1]->Draw("same");  
    
   canvas_OB1->cd(5);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds1[5]:fgZ[5]>>hOB_ds1_vs_z_L5","fchipID[5]>=0&&TMath::Abs(fds1[5])<0.1","goff");
   ResMonitor->Draw("fds1[5]:fgZ[5]>>pOB_ds1_vs_z_L5","fchipID[5]>=0&&TMath::Abs(fds1[5])<0.1","goff");   
   hOB_ds1_vs_z[2]->Draw();   
   pOB_ds1_vs_z[2]->Draw("same");      
   canvas_OB1->cd(6);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds1[5]:fgPhi[5]>>hOB_ds1_vs_phi_L5","fchipID[5]>=0&&TMath::Abs(fds1[5])<0.1","goff");
   ResMonitor->Draw("fds1[5]:fgPhi[5]>>pOB_ds1_vs_phi_L5","fchipID[5]>=0&&TMath::Abs(fds1[5])<0.1","goff");   
   hOB_ds1_vs_phi[2]->Draw();   
   pOB_ds1_vs_phi[2]->Draw("same");  
    
   canvas_OB1->cd(7);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds1[6]:fgZ[6]>>hOB_ds1_vs_z_L6","fchipID[6]>=0&&TMath::Abs(fds1[6])<0.1","goff");
   ResMonitor->Draw("fds1[6]:fgZ[6]>>pOB_ds1_vs_z_L6","fchipID[6]>=0&&TMath::Abs(fds1[6])<0.1","goff");
   hOB_ds1_vs_z[3]->Draw();   
   pOB_ds1_vs_z[3]->Draw("same");      
   canvas_OB1->cd(8);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds1[6]:fgPhi[6]>>hOB_ds1_vs_phi_L6","fchipID[6]>=0&&TMath::Abs(fds1[6])<0.1","goff");
   ResMonitor->Draw("fds1[6]:fgPhi[6]>>pOB_ds1_vs_phi_L6","fchipID[6]>=0&&TMath::Abs(fds1[6])<0.1","goff");   
   hOB_ds1_vs_phi[3]->Draw();  
   pOB_ds1_vs_phi[3]->Draw("same");       
   canvas_OB1->SaveAs("Res_ds1_OB.gif","gif");

   TCanvas *canvas_IB2 = new TCanvas("canvas_IB2","canvas_IB2",1600,900);
   canvas_IB2->Divide(2,3);

   canvas_IB2->cd(1);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds2[0]:fgZ[0]>>hIB_ds2_vs_z_L0","fchipID[0]>=0&&TMath::Abs(fds2[0])<0.06","goff");
   ResMonitor->Draw("fds2[0]:fgZ[0]>>pIB_ds2_vs_z_L0","fchipID[0]>=0&&TMath::Abs(fds2[0])<0.06","goff");   
   hIB_ds2_vs_z[0]->Draw();      
   pIB_ds2_vs_z[0]->Draw("same");         
   canvas_IB2->cd(2);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds2[0]:fgPhi[0]>>hIB_ds2_vs_phi_L0","fchipID[0]>=0&&TMath::Abs(fds2[0])<0.06","goff");
   ResMonitor->Draw("fds2[0]:fgPhi[0]>>pIB_ds2_vs_phi_L0","fchipID[0]>=0&&TMath::Abs(fds2[0])<0.06","goff");   
   hIB_ds2_vs_phi[0]->Draw();   
   pIB_ds2_vs_phi[0]->Draw("same");   
   
   canvas_IB2->cd(3);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds2[1]:fgZ[1]>>hIB_ds2_vs_z_L1","fchipID[1]>=0&&TMath::Abs(fds2[1])<0.06","goff");
   ResMonitor->Draw("fds2[1]:fgZ[1]>>pIB_ds2_vs_z_L1","fchipID[1]>=0&&TMath::Abs(fds2[1])<0.06","goff");   
   hIB_ds2_vs_z[1]->Draw();    
   pIB_ds2_vs_z[1]->Draw("same");                 
   canvas_IB2->cd(4);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds2[1]:fgPhi[1]>>hIB_ds2_vs_phi_L1","fchipID[1]>=0&&TMath::Abs(fds2[1])<0.06","goff");
   ResMonitor->Draw("fds2[1]:fgPhi[1]>>pIB_ds2_vs_phi_L1","fchipID[1]>=0&&TMath::Abs(fds2[1])<0.06","goff");   
   hIB_ds2_vs_phi[1]->Draw();   
   pIB_ds2_vs_phi[1]->Draw("same");   
   
   canvas_IB2->cd(5);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds2[2]:fgZ[2]>>hIB_ds2_vs_z_L2","fchipID[2]>=0&&TMath::Abs(fds2[2])<0.06","goff");
   ResMonitor->Draw("fds2[2]:fgZ[2]>>pIB_ds2_vs_z_L2","fchipID[2]>=0&&TMath::Abs(fds2[2])<0.06","goff");   
   hIB_ds2_vs_z[2]->Draw();         
   pIB_ds2_vs_z[2]->Draw("same");         
   canvas_IB2->cd(6);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds2[2]:fgPhi[2]>>hIB_ds2_vs_phi_L2","fchipID[2]>=0&&TMath::Abs(fds2[2])<0.06","goff");
   ResMonitor->Draw("fds2[2]:fgPhi[2]>>pIB_ds2_vs_phi_L2","fchipID[2]>=0&&TMath::Abs(fds2[2])<0.06","goff");   
   hIB_ds2_vs_phi[2]->Draw();  
   pIB_ds2_vs_phi[2]->Draw("same");       
   canvas_IB2->SaveAs("Res_ds2_IB.gif","gif");

   TCanvas *canvas_OB2 = new TCanvas("canvas_OB2","canvas_OB2",1600,1200);
   canvas_OB2->Divide(2,4);
   canvas_OB2->cd(1);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds2[3]:fgZ[3]>>hOB_ds2_vs_z_L3","fchipID[3]>=0&&TMath::Abs(fds2[3])<0.1","goff");
   ResMonitor->Draw("fds2[3]:fgZ[3]>>pOB_ds2_vs_z_L3","fchipID[3]>=0&&TMath::Abs(fds2[3])<0.1","goff");   
   hOB_ds2_vs_z[0]->Draw(); 
   pOB_ds2_vs_z[0]->Draw("same");    
   canvas_OB2->cd(2);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds2[3]:fgPhi[3]>>hOB_ds2_vs_phi_L3","fchipID[3]>=0&&TMath::Abs(fds2[3])<0.1","goff");
   ResMonitor->Draw("fds2[3]:fgPhi[3]>>pOB_ds2_vs_phi_L3","fchipID[3]>=0&&TMath::Abs(fds2[3])<0.1","goff");   
   hOB_ds2_vs_phi[0]->Draw(); 
   pOB_ds2_vs_phi[0]->Draw("same"); 
   
   canvas_OB2->cd(3);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds2[4]:fgZ[4]>>hOB_ds2_vs_z_L4","fchipID[4]>=0&&TMath::Abs(fds2[4])<0.1","goff");
   ResMonitor->Draw("fds2[4]:fgZ[4]>>pOB_ds2_vs_z_L4","fchipID[4]>=0&&TMath::Abs(fds2[4])<0.1","goff");   
   hOB_ds2_vs_z[1]->Draw(); 
   pOB_ds2_vs_z[1]->Draw("same");    
   canvas_OB2->cd(4);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds2[4]:fgPhi[4]>>hOB_ds2_vs_phi_L4","fchipID[4]>=0&&TMath::Abs(fds2[4])<0.1","goff");
   ResMonitor->Draw("fds2[4]:fgPhi[4]>>pOB_ds2_vs_phi_L4","fchipID[4]>=0&&TMath::Abs(fds2[4])<0.1","goff");   
   hOB_ds2_vs_phi[1]->Draw(); 
   pOB_ds2_vs_phi[1]->Draw("same"); 
   
   canvas_OB2->cd(5);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds2[5]:fgZ[5]>>hOB_ds2_vs_z_L5","fchipID[5]>=0&&TMath::Abs(fds2[5])<0.1","goff");
   ResMonitor->Draw("fds2[5]:fgZ[5]>>pOB_ds2_vs_z_L5","fchipID[5]>=0&&TMath::Abs(fds2[5])<0.1","goff");   
   hOB_ds2_vs_z[2]->Draw(); 
   pOB_ds2_vs_z[2]->Draw("same");    
   canvas_OB2->cd(6);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds2[5]:fgPhi[5]>>hOB_ds2_vs_phi_L5","fchipID[5]>=0&&TMath::Abs(fds2[5])<0.1","goff");
   ResMonitor->Draw("fds2[5]:fgPhi[5]>>pOB_ds2_vs_phi_L5","fchipID[5]>=0&&TMath::Abs(fds2[5])<0.1","goff");   
   hOB_ds2_vs_phi[2]->Draw(); 
   pOB_ds2_vs_phi[2]->Draw("same"); 
   
   canvas_OB2->cd(7);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds2[6]:fgZ[6]>>hOB_ds2_vs_z_L6","fchipID[6]>=0&&TMath::Abs(fds2[6])<0.1","goff");
   ResMonitor->Draw("fds2[6]:fgZ[6]>>pOB_ds2_vs_z_L6","fchipID[6]>=0&&TMath::Abs(fds2[6])<0.1","goff");   
   hOB_ds2_vs_z[3]->Draw(); 
   pOB_ds2_vs_z[3]->Draw("same");    
   canvas_OB2->cd(8);
   gStyle->SetOptStat(0);   
   ResMonitor->Draw("fds2[6]:fgPhi[6]>>hOB_ds2_vs_phi_L6","fchipID[6]>=0&&TMath::Abs(fds2[6])<0.1","goff");
   ResMonitor->Draw("fds2[6]:fgPhi[6]>>pOB_ds2_vs_phi_L6","fchipID[6]>=0&&TMath::Abs(fds2[6])<0.1","goff");   
   hOB_ds2_vs_phi[3]->Draw();    
   pOB_ds2_vs_phi[3]->Draw("same");       
   canvas_OB2->SaveAs("Res_ds2_OB.gif","gif");

};

void macro_plot_ResidualProfile(){

   TH2D* hIB_ds1_vs_z[3];
   TH2D* hIB_ds1_vs_phi[3];
   TH2D* hOB_ds1_vs_z[4];
   TH2D* hOB_ds1_vs_phi[4];   
   TH2D* hIB_ds2_vs_z[3];
   TH2D* hIB_ds2_vs_phi[3];
   TH2D* hOB_ds2_vs_z[4];
   TH2D* hOB_ds2_vs_phi[4];    

   TProfile* pIB_ds1_vs_z[3];
   TProfile* pIB_ds1_vs_phi[3];
   TProfile* pOB_ds1_vs_z[4];
   TProfile* pOB_ds1_vs_phi[4];   
   TProfile* pIB_ds2_vs_z[3];
   TProfile* pIB_ds2_vs_phi[3];
   TProfile* pOB_ds2_vs_z[4];
   TProfile* pOB_ds2_vs_phi[4];  

   double mrangeZ[] = {15, 15, 15, 50, 50, 85, 85};

   double mrangeD[] = {0.015, 0.015, 0.015, 0.025, 0.025, 0.025, 0.025};
   for(int iIB = 0; iIB < nLAYERIB; iIB++){ 
      int idx = iIB;
      
      hIB_ds1_vs_z[idx]   
      = new TH2D(Form("hIB_ds1_vs_z_L%d",iIB),Form("ds_{1} vs z Layer %d",iIB),600,-mrangeZ[iIB],+mrangeZ[iIB],400,-mrangeD[iIB],+mrangeD[iIB]);
      hIB_ds1_vs_phi[idx] 
      = new TH2D(Form("hIB_ds1_vs_phi_L%d",iIB),Form("ds_{1} vs #phi Layer %d",iIB),600,-TMath::Pi(),+TMath::Pi(),400,-mrangeD[iIB],+mrangeD[iIB]);     
      hIB_ds2_vs_z[idx]   
      = new TH2D(Form("hIB_ds2_vs_z_L%d",iIB),Form("ds_{1} vs z Layer %d",iIB),600,-mrangeZ[iIB],+mrangeZ[iIB],400,-mrangeD[iIB],+mrangeD[iIB]);
      hIB_ds2_vs_phi[idx] 
      = new TH2D(Form("hIB_ds2_vs_phi_L%d",iIB),Form("ds_{1} vs #phi Layer %d",iIB),600,-TMath::Pi(),+TMath::Pi(),400,-mrangeD[iIB],+mrangeD[iIB]);     

      pIB_ds1_vs_z[idx]   
      = new TProfile(Form("pIB_ds1_vs_z_L%d",iIB),Form("ds_{1} vs z Profile Layer %d",iIB),4*nHicPerStave[iIB]*nChipsPerHic[iIB],-mrangeZ[iIB],+mrangeZ[iIB],-mrangeD[iIB],+mrangeD[iIB]);
      pIB_ds1_vs_phi[idx] 
      = new TProfile(Form("pIB_ds1_vs_phi_L%d",iIB),Form("ds_{1} vs #phi Profile Layer %d",iIB),4*nHicPerStave[iIB]*nChipsPerHic[iIB],-TMath::Pi(),+TMath::Pi(),-mrangeD[iIB],+mrangeD[iIB]);
      pIB_ds2_vs_z[idx]   
      = new TProfile(Form("pIB_ds2_vs_z_L%d",iIB),Form("ds_{1} vs z Profile Layer %d",iIB),4*nHicPerStave[iIB]*nChipsPerHic[iIB],-mrangeZ[iIB],+mrangeZ[iIB],-mrangeD[iIB],+mrangeD[iIB]);
      pIB_ds2_vs_phi[idx] 
      = new TProfile(Form("pIB_ds2_vs_phi_L%d",iIB),Form("ds_{1} vs #phi Profile Layer %d",iIB),4*nHicPerStave[iIB]*nChipsPerHic[iIB],-TMath::Pi(),+TMath::Pi(),-mrangeD[iIB],+mrangeD[iIB]);
   }  
   
   for(int iOB = nLAYERIB; iOB < nLAYER; iOB++){
      int idx = iOB - nLAYERIB;

      hOB_ds1_vs_z[idx]   
      = new TH2D(Form("hOB_ds1_vs_z_L%d",iOB),Form("ds_{1} vs z Layer %d",iOB),600,-mrangeZ[iOB],+mrangeZ[iOB],400,-mrangeD[iOB],+mrangeD[iOB]);
      hOB_ds1_vs_phi[idx] 
      = new TH2D(Form("hOB_ds1_vs_phi_L%d",iOB),Form("ds_{1} vs #phi Layer %d",iOB),600,-TMath::Pi(),+TMath::Pi(),400,-mrangeD[iOB],+mrangeD[iOB]);     
      hOB_ds2_vs_z[idx]   
      = new TH2D(Form("hOB_ds2_vs_z_L%d",iOB),Form("ds_{1} vs z Layer %d",iOB),600,-mrangeZ[iOB],+mrangeZ[iOB],400,-mrangeD[iOB],+mrangeD[iOB]);
      hOB_ds2_vs_phi[idx] 
      = new TH2D(Form("hOB_ds2_vs_phi_L%d",iOB),Form("ds_{1} vs #phi Layer %d",iOB),600,-TMath::Pi(),+TMath::Pi(),400,-mrangeD[iOB],+mrangeD[iOB]);     

      pOB_ds1_vs_z[idx]   
      = new TProfile(Form("pOB_ds1_vs_z_L%d",iOB),Form("ds_{1} vs z Profile Layer %d",iOB),nHicPerStave[iOB]*nChipsPerHic[iOB]/7,-mrangeZ[iOB],+mrangeZ[iOB],-mrangeD[iOB],+mrangeD[iOB]);
      pOB_ds1_vs_phi[idx] 
      = new TProfile(Form("pOB_ds1_vs_phi_L%d",iOB),Form("ds_{1} vs #phi Profile Layer %d",iOB),nHicPerStave[iOB]*nChipsPerHic[iOB]/7,-TMath::Pi(),+TMath::Pi(),-mrangeD[iOB],+mrangeD[iOB]);
      pOB_ds2_vs_z[idx]   
      = new TProfile(Form("pOB_ds2_vs_z_L%d",iOB),Form("ds_{1} vs z Profile Layer %d",iOB),nHicPerStave[iOB]*nChipsPerHic[iOB]/7,-mrangeZ[iOB],+mrangeZ[iOB],-mrangeD[iOB],+mrangeD[iOB]);
      pOB_ds2_vs_phi[idx] 
      = new TProfile(Form("pOB_ds2_vs_phi_L%d",iOB),Form("ds_{1} vs #phi Profile Layer %d",iOB),nHicPerStave[iOB]*nChipsPerHic[iOB]/7,-TMath::Pi(),+TMath::Pi(),-mrangeD[iOB],+mrangeD[iOB]);
   }   
            

   TCanvas *canvas_IB1 = new TCanvas("canvas_IB1","canvas_IB1",1600,900);
   canvas_IB1->Divide(2,3);

   canvas_IB1->cd(1); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);
   //ResMonitor->Draw("fds1[0]:fgZ[0]>>hIB_ds1_vs_z_L0","fchipID[0]>=0&&TMath::Abs(fds1[0])<0.06","goff");
   ResMonitor->Draw("fds1[0]:fgZ[0]>>pIB_ds1_vs_z_L0","fchipID[0]>=0&&TMath::Abs(fds1[0])<0.06","goff");   
   hIB_ds1_vs_z[0]->Draw();
   pIB_ds1_vs_z[0]->Draw("same");
   canvas_IB1->cd(2); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds1[0]:fgPhi[0]>>hIB_ds1_vs_phi_L0","fchipID[0]>=0&&TMath::Abs(fds1[0])<0.06","goff");
   ResMonitor->Draw("fds1[0]:fgPhi[0]>>pIB_ds1_vs_phi_L0","fchipID[0]>=0&&TMath::Abs(fds1[0])<0.06","goff");
   hIB_ds1_vs_phi[0]->Draw();
   pIB_ds1_vs_phi[0]->Draw("same");
   
   canvas_IB1->cd(3); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds1[1]:fgZ[1]>>hIB_ds1_vs_z_L1","fchipID[1]>=0&&TMath::Abs(fds1[1])<0.06","goff");
   ResMonitor->Draw("fds1[1]:fgZ[1]>>pIB_ds1_vs_z_L1","fchipID[1]>=0&&TMath::Abs(fds1[1])<0.06","goff");   
   hIB_ds1_vs_z[1]->Draw();   
   pIB_ds1_vs_z[1]->Draw("same");      
   canvas_IB1->cd(4); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds1[1]:fgPhi[1]>>hIB_ds1_vs_phi_L1","fchipID[1]>=0&&TMath::Abs(fds1[1])<0.06","goff");
   ResMonitor->Draw("fds1[1]:fgPhi[1]>>pIB_ds1_vs_phi_L1","fchipID[1]>=0&&TMath::Abs(fds1[1])<0.06","goff");   
   hIB_ds1_vs_phi[1]->Draw();
   pIB_ds1_vs_phi[1]->Draw("same");
    
   canvas_IB1->cd(5); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds1[2]:fgZ[2]>>hIB_ds1_vs_z_L2","fchipID[2]>=0&&TMath::Abs(fds1[2])<0.06","goff");
   ResMonitor->Draw("fds1[2]:fgZ[2]>>pIB_ds1_vs_z_L2","fchipID[2]>=0&&TMath::Abs(fds1[2])<0.06","goff");   
   hIB_ds1_vs_z[2]->Draw(); 
   pIB_ds1_vs_z[2]->Draw("same"); 
   canvas_IB1->cd(6); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds1[2]:fgPhi[2]>>hIB_ds1_vs_phi_L2","fchipID[2]>=0&&TMath::Abs(fds1[2])<0.06","goff");
   ResMonitor->Draw("fds1[2]:fgPhi[2]>>pIB_ds1_vs_phi_L2","fchipID[2]>=0&&TMath::Abs(fds1[2])<0.06","goff");
   hIB_ds1_vs_phi[2]->Draw(); 
   pIB_ds1_vs_phi[2]->Draw("same"); 
   canvas_IB1->SaveAs("Res_ds1_IB_Profile.gif","gif");
   
   TCanvas *canvas_OB1 = new TCanvas("canvas_OB1","canvas_OB1",1600,1200);
   canvas_OB1->Divide(2,4);
   canvas_OB1->cd(1); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds1[3]:fgZ[3]>>hOB_ds1_vs_z_L3","fchipID[3]>=0&&TMath::Abs(fds1[3])<0.1","goff");
   ResMonitor->Draw("fds1[3]:fgZ[3]>>pOB_ds1_vs_z_L3","fchipID[3]>=0&&TMath::Abs(fds1[3])<0.1","goff");   
   hOB_ds1_vs_z[0]->Draw();  
   pOB_ds1_vs_z[0]->Draw("same");       
   canvas_OB1->cd(2); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds1[3]:fgPhi[3]>>hOB_ds1_vs_phi_L3","fchipID[3]>=0&&TMath::Abs(fds1[3])<0.1","goff");
   ResMonitor->Draw("fds1[3]:fgPhi[3]>>pOB_ds1_vs_phi_L3","fchipID[3]>=0&&TMath::Abs(fds1[3])<0.1","goff");   
   hOB_ds1_vs_phi[0]->Draw();
   pOB_ds1_vs_phi[0]->Draw("same");
   
   canvas_OB1->cd(3); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds1[4]:fgZ[4]>>hOB_ds1_vs_z_L4","fchipID[4]>=0&&TMath::Abs(fds1[4])<0.1","goff");
   ResMonitor->Draw("fds1[4]:fgZ[4]>>pOB_ds1_vs_z_L4","fchipID[4]>=0&&TMath::Abs(fds1[4])<0.1","goff");   
   hOB_ds1_vs_z[1]->Draw();      
   pOB_ds1_vs_z[1]->Draw("same");         
   canvas_OB1->cd(4); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds1[4]:fgPhi[4]>>hOB_ds1_vs_phi_L4","fchipID[4]>=0&&TMath::Abs(fds1[4])<0.1","goff");
   ResMonitor->Draw("fds1[4]:fgPhi[4]>>pOB_ds1_vs_phi_L4","fchipID[4]>=0&&TMath::Abs(fds1[4])<0.1","goff");   
   hOB_ds1_vs_phi[1]->Draw();   
   pOB_ds1_vs_phi[1]->Draw("same");  
    
   canvas_OB1->cd(5); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds1[5]:fgZ[5]>>hOB_ds1_vs_z_L5","fchipID[5]>=0&&TMath::Abs(fds1[5])<0.1","goff");
   ResMonitor->Draw("fds1[5]:fgZ[5]>>pOB_ds1_vs_z_L5","fchipID[5]>=0&&TMath::Abs(fds1[5])<0.1","goff");   
   hOB_ds1_vs_z[2]->Draw();   
   pOB_ds1_vs_z[2]->Draw("same");      
   canvas_OB1->cd(6); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds1[5]:fgPhi[5]>>hOB_ds1_vs_phi_L5","fchipID[5]>=0&&TMath::Abs(fds1[5])<0.1","goff");
   ResMonitor->Draw("fds1[5]:fgPhi[5]>>pOB_ds1_vs_phi_L5","fchipID[5]>=0&&TMath::Abs(fds1[5])<0.1","goff");   
   hOB_ds1_vs_phi[2]->Draw();   
   pOB_ds1_vs_phi[2]->Draw("same");  
    
   canvas_OB1->cd(7); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds1[6]:fgZ[6]>>hOB_ds1_vs_z_L6","fchipID[6]>=0&&TMath::Abs(fds1[6])<0.1","goff");
   ResMonitor->Draw("fds1[6]:fgZ[6]>>pOB_ds1_vs_z_L6","fchipID[6]>=0&&TMath::Abs(fds1[6])<0.1","goff");
   hOB_ds1_vs_z[3]->Draw();   
   pOB_ds1_vs_z[3]->Draw("same");      
   canvas_OB1->cd(8); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds1[6]:fgPhi[6]>>hOB_ds1_vs_phi_L6","fchipID[6]>=0&&TMath::Abs(fds1[6])<0.1","goff");
   ResMonitor->Draw("fds1[6]:fgPhi[6]>>pOB_ds1_vs_phi_L6","fchipID[6]>=0&&TMath::Abs(fds1[6])<0.1","goff");   
   hOB_ds1_vs_phi[3]->Draw();  
   pOB_ds1_vs_phi[3]->Draw("same");       
   canvas_OB1->SaveAs("Res_ds1_OB_Profile.gif","gif");

   TCanvas *canvas_IB2 = new TCanvas("canvas_IB2","canvas_IB2",1600,900);
   canvas_IB2->Divide(2,3);

   canvas_IB2->cd(1); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds2[0]:fgZ[0]>>hIB_ds2_vs_z_L0","fchipID[0]>=0&&TMath::Abs(fds2[0])<0.06","goff");
   ResMonitor->Draw("fds2[0]:fgZ[0]>>pIB_ds2_vs_z_L0","fchipID[0]>=0&&TMath::Abs(fds2[0])<0.06","goff");   
   hIB_ds2_vs_z[0]->Draw();      
   pIB_ds2_vs_z[0]->Draw("same");         
   canvas_IB2->cd(2); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds2[0]:fgPhi[0]>>hIB_ds2_vs_phi_L0","fchipID[0]>=0&&TMath::Abs(fds2[0])<0.06","goff");
   ResMonitor->Draw("fds2[0]:fgPhi[0]>>pIB_ds2_vs_phi_L0","fchipID[0]>=0&&TMath::Abs(fds2[0])<0.06","goff");   
   hIB_ds2_vs_phi[0]->Draw();   
   pIB_ds2_vs_phi[0]->Draw("same");   
   
   canvas_IB2->cd(3); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds2[1]:fgZ[1]>>hIB_ds2_vs_z_L1","fchipID[1]>=0&&TMath::Abs(fds2[1])<0.06","goff");
   ResMonitor->Draw("fds2[1]:fgZ[1]>>pIB_ds2_vs_z_L1","fchipID[1]>=0&&TMath::Abs(fds2[1])<0.06","goff");   
   hIB_ds2_vs_z[1]->Draw();    
   pIB_ds2_vs_z[1]->Draw("same");                 
   canvas_IB2->cd(4); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds2[1]:fgPhi[1]>>hIB_ds2_vs_phi_L1","fchipID[1]>=0&&TMath::Abs(fds2[1])<0.06","goff");
   ResMonitor->Draw("fds2[1]:fgPhi[1]>>pIB_ds2_vs_phi_L1","fchipID[1]>=0&&TMath::Abs(fds2[1])<0.06","goff");   
   hIB_ds2_vs_phi[1]->Draw();   
   pIB_ds2_vs_phi[1]->Draw("same");   
   
   canvas_IB2->cd(5); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds2[2]:fgZ[2]>>hIB_ds2_vs_z_L2","fchipID[2]>=0&&TMath::Abs(fds2[2])<0.06","goff");
   ResMonitor->Draw("fds2[2]:fgZ[2]>>pIB_ds2_vs_z_L2","fchipID[2]>=0&&TMath::Abs(fds2[2])<0.06","goff");   
   hIB_ds2_vs_z[2]->Draw();         
   pIB_ds2_vs_z[2]->Draw("same");         
   canvas_IB2->cd(6); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds2[2]:fgPhi[2]>>hIB_ds2_vs_phi_L2","fchipID[2]>=0&&TMath::Abs(fds2[2])<0.06","goff");
   ResMonitor->Draw("fds2[2]:fgPhi[2]>>pIB_ds2_vs_phi_L2","fchipID[2]>=0&&TMath::Abs(fds2[2])<0.06","goff");   
   hIB_ds2_vs_phi[2]->Draw();  
   pIB_ds2_vs_phi[2]->Draw("same");       
   canvas_IB2->SaveAs("Res_ds2_IB_Profile.gif","gif");

   TCanvas *canvas_OB2 = new TCanvas("canvas_OB2","canvas_OB2",1600,1200);
   canvas_OB2->Divide(2,4);
   canvas_OB2->cd(1); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds2[3]:fgZ[3]>>hOB_ds2_vs_z_L3","fchipID[3]>=0&&TMath::Abs(fds2[3])<0.1","goff");
   ResMonitor->Draw("fds2[3]:fgZ[3]>>pOB_ds2_vs_z_L3","fchipID[3]>=0&&TMath::Abs(fds2[3])<0.1","goff");   
   hOB_ds2_vs_z[0]->Draw(); 
   pOB_ds2_vs_z[0]->Draw("same");    
   canvas_OB2->cd(2); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds2[3]:fgPhi[3]>>hOB_ds2_vs_phi_L3","fchipID[3]>=0&&TMath::Abs(fds2[3])<0.1","goff");
   ResMonitor->Draw("fds2[3]:fgPhi[3]>>pOB_ds2_vs_phi_L3","fchipID[3]>=0&&TMath::Abs(fds2[3])<0.1","goff");   
   hOB_ds2_vs_phi[0]->Draw(); 
   pOB_ds2_vs_phi[0]->Draw("same"); 
   
   canvas_OB2->cd(3); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds2[4]:fgZ[4]>>hOB_ds2_vs_z_L4","fchipID[4]>=0&&TMath::Abs(fds2[4])<0.1","goff");
   ResMonitor->Draw("fds2[4]:fgZ[4]>>pOB_ds2_vs_z_L4","fchipID[4]>=0&&TMath::Abs(fds2[4])<0.1","goff");   
   hOB_ds2_vs_z[1]->Draw(); 
   pOB_ds2_vs_z[1]->Draw("same");    
   canvas_OB2->cd(4); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds2[4]:fgPhi[4]>>hOB_ds2_vs_phi_L4","fchipID[4]>=0&&TMath::Abs(fds2[4])<0.1","goff");
   ResMonitor->Draw("fds2[4]:fgPhi[4]>>pOB_ds2_vs_phi_L4","fchipID[4]>=0&&TMath::Abs(fds2[4])<0.1","goff");   
   hOB_ds2_vs_phi[1]->Draw(); 
   pOB_ds2_vs_phi[1]->Draw("same"); 
   
   canvas_OB2->cd(5); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds2[5]:fgZ[5]>>hOB_ds2_vs_z_L5","fchipID[5]>=0&&TMath::Abs(fds2[5])<0.1","goff");
   ResMonitor->Draw("fds2[5]:fgZ[5]>>pOB_ds2_vs_z_L5","fchipID[5]>=0&&TMath::Abs(fds2[5])<0.1","goff");   
   hOB_ds2_vs_z[2]->Draw(); 
   pOB_ds2_vs_z[2]->Draw("same");    
   canvas_OB2->cd(6); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds2[5]:fgPhi[5]>>hOB_ds2_vs_phi_L5","fchipID[5]>=0&&TMath::Abs(fds2[5])<0.1","goff");
   ResMonitor->Draw("fds2[5]:fgPhi[5]>>pOB_ds2_vs_phi_L5","fchipID[5]>=0&&TMath::Abs(fds2[5])<0.1","goff");   
   hOB_ds2_vs_phi[2]->Draw(); 
   pOB_ds2_vs_phi[2]->Draw("same"); 
   
   canvas_OB2->cd(7); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds2[6]:fgZ[6]>>hOB_ds2_vs_z_L6","fchipID[6]>=0&&TMath::Abs(fds2[6])<0.1","goff");
   ResMonitor->Draw("fds2[6]:fgZ[6]>>pOB_ds2_vs_z_L6","fchipID[6]>=0&&TMath::Abs(fds2[6])<0.1","goff");   
   hOB_ds2_vs_z[3]->Draw(); 
   pOB_ds2_vs_z[3]->Draw("same");    
   canvas_OB2->cd(8); gPad->SetGridx(); gPad->SetGridy();
   gStyle->SetOptStat(0);   
   //ResMonitor->Draw("fds2[6]:fgPhi[6]>>hOB_ds2_vs_phi_L6","fchipID[6]>=0&&TMath::Abs(fds2[6])<0.1","goff");
   ResMonitor->Draw("fds2[6]:fgPhi[6]>>pOB_ds2_vs_phi_L6","fchipID[6]>=0&&TMath::Abs(fds2[6])<0.1","goff");   
   hOB_ds2_vs_phi[3]->Draw();    
   pOB_ds2_vs_phi[3]->Draw("same");       
   canvas_OB2->SaveAs("Res_ds2_OB_Profile.gif","gif");


};
