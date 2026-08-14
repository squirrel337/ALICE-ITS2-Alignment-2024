
#define nTrials 1

int trialCase[] = {0}; 
TString runNumber[] = {"#"};
TCanvas* canvas;

TFile* file[2][nTrials];
TTree* tree_ResMonitor[nTrials];

TH1D* hChi2TOT[nTrials];
TH2D* hS1ResVsZ[3][2][nTrials]; // layer , hb, trials = opt 1
TH2D* hS1ResVsPhi[3][2][nTrials]; // layer , hb, trials = opt 1
TH2D* hS2ResVsZ[3][2][nTrials]; // layer , hb, trials = opt 2
TH2D* hS2ResVsPhi[3][2][nTrials]; // layer , hb, trials = opt 2


TH2D* hS2ResVsPhiL0[nTrials]; // layer , hb, trials = opt 2
TH2D* hS2ResVsPhiL2[nTrials]; // layer , hb, trials = opt 2
TH2D* hS2ResVsPhiL0L2Sum[nTrials]; // layer , hb, trials = opt 2

TH2D* hS2ResVsPhiLAYER[nTrials][7][20]; // trial ,layer, pstv
TH2D* hS2ResVsPhiLAYERALL[nTrials][7];
TF1* fS2ResLAYER[nTrials][7][20]; // trial ,layer, pstv

TH2D* hvtxXvsPhi[nTrials];
TH2D* hvtxYvsPhi[nTrials];
TH1D* hvtxX_by_phi[nTrials][8];
TH1D* hvtxY_by_phi[nTrials][8];
TF1*  fvtxX_by_phi[nTrials][8];
TF1*  fvtxY_by_phi[nTrials][8];

TH1D* hCurvChargepT[nTrials]; // layer , hb, trials = opt 1
TH2D* hCurvChargepT2D[nTrials]; // layer , hb, trials = opt 1

TH1D* hCurvChargepT_by_pstv[nTrials][20]; // layer , hb, trials = opt 1
TH2D* hCurvChargepT2D_by_pstv[nTrials][20]; // layer , hb, trials = opt 1
TF1*  fCurvChargepT_by_pstv[nTrials][20]; // layer , hb, trials = opt 1
TH1D* hpT[3][nTrials];
TH1D* hpT_by_pstv[3][nTrials][20];
TLatex* tl_hpT_by_pstv[3][20];

TH1D htemp1D[nTrials][7];
TH2D htemp2D[nTrials][7];

TString analyDIR = ".";

void load(){
   for(int t = 0; t < nTrials; t++){
      file[0][t] = new TFile(Form("%s/Residual_Epoch_At_-1.root",(const char*) analyDIR),"read");
      hChi2TOT[t] = (TH1D*) file[0][t]->Get(Form("Residual/hChi2TOT"));             
      for(int lIB = 0; lIB < 3; lIB++){
         for(int hb = 0; hb < 2; hb++){
            hS1ResVsZ[lIB][hb][t]   = (TH2D*) file[0][t]->Get(Form("Residual/Layer%d/hS1ResVsZLayer%dHB%dHS0",lIB,lIB,hb)); 
            hS1ResVsPhi[lIB][hb][t] = (TH2D*) file[0][t]->Get(Form("Residual/Layer%d/hS1ResVsPhiLayer%dHB%dHS0",lIB,lIB,hb)); 
            hS2ResVsZ[lIB][hb][t]   = (TH2D*) file[0][t]->Get(Form("Residual/Layer%d/hS2ResVsZLayer%dHB%dHS0",lIB,lIB,hb)); 
            hS2ResVsPhi[lIB][hb][t] = (TH2D*) file[0][t]->Get(Form("Residual/Layer%d/hS2ResVsPhiLayer%dHB%dHS0",lIB,lIB,hb)); 
         }
      }      
      
      file[1][t] = new TFile(Form("%s/Residual_Monitor_Epoch_At_-1.root",(const char*) analyDIR),"read");
      tree_ResMonitor[t] = (TTree*) file[1][t]->GetObjectUnchecked("ResMonitor");

      int nbins = 400;
      double range = 0.2;
      
      hS2ResVsPhiL0[t]      = new TH2D(Form("hS2ResVsPhiL%d_B%d",0,trialCase[t]),Form("#Deltas_{2} vs #phi L%d",0),600,-TMath::Pi(),+TMath::Pi(),nbins,-range,+range);  
      hS2ResVsPhiL2[t]      = new TH2D(Form("hS2ResVsPhiL%d_B%d",2,trialCase[t]),Form("#Deltas_{2} vs #phi L%d",2),600,-TMath::Pi(),+TMath::Pi(),nbins,-range,+range);  
      hS2ResVsPhiL0L2Sum[t] = new TH2D(Form("hS2ResVsPhiL%dL%dsum_B%d",0,2,trialCase[t]),Form("#Deltas_{2} vs #phi L%d (+) L%d",0,2),600,-TMath::Pi(),+TMath::Pi(),nbins,-range,+range);  

      tree_ResMonitor[t]->Draw(Form("fds2[%d]:fgPhi[2]>>hS2ResVsPhiL%d_B%d",0,0,trialCase[t]),Form("fgPhi[2]>-10&&std::abs(fds2[%d])<%g",0,range),"goff");
      tree_ResMonitor[t]->Draw(Form("fds2[%d]:fgPhi[2]>>hS2ResVsPhiL%d_B%d",2,2,trialCase[t]),Form("fgPhi[2]>-10&&std::abs(fds2[%d])<%g",2,range),"goff");
      tree_ResMonitor[t]->Draw(Form("fds2[%d]+fds2[%d]:fgPhi[2]>>hS2ResVsPhiL%dL%dsum_B%d",0,2,0,2,trialCase[t]),Form("fgPhi[2]>-10&&std::abs(fds2[%d])<%g&&std::abs(fds2[%d])<%g",0,range,2,range),"goff");
            
      hCurvChargepT[t] = new TH1D(Form("hCurvChargepT_B%d",trialCase[t]),"1/p_{T}*sign; 1/p_{T}*sign; N",50,-2,2);
      tree_ResMonitor[t]->Draw(Form("1/pT*cuvR/std::abs(cuvR)>>hCurvChargepT_B%d",trialCase[t]),"fgPhi[2]>-10&&std::abs(1/pT)<2","goff");
     
      hCurvChargepT2D[t] = new TH2D(Form("hCurvChargepT2D_B%d",trialCase[t]),"1/p_{T}*sign vs phi; phi; 1/p_{T}*sign",100,-TMath::Pi(),TMath::Pi(),50,-2,2);
      tree_ResMonitor[t]->Draw(Form("1/pT*cuvR/std::abs(cuvR):fgPhi[2]>>hCurvChargepT2D_B%d",trialCase[t]),"fgPhi[2]>-10&&std::abs(1/pT)<2","goff");
 
      for(int layer = 0; layer < 7; layer++){
         hS2ResVsPhiLAYERALL[t][layer] = new TH2D(Form("hS2ResVsPhiL%dALL_B%d",layer,trialCase[t]),
                                                  Form("#Deltas_{2} vs #phi L%d Trial%d",layer,trialCase[t]),600,-TMath::Pi(),+TMath::Pi(),nbins,-range,+range);  
         tree_ResMonitor[t]->Draw(Form("fds2[%d]:fgPhi[%d]>>hS2ResVsPhiL%dALL_B%d",layer,layer,layer,trialCase[t]),
                                  Form("fgPhi[%d]>-10&&std::abs(fds2[%d])<%g",layer,layer,range),"goff");                                     
      } 
     
      for(int pstv = 0; pstv < 20; pstv++){

         for(int layer = 0; layer < 7; layer++){
            hS2ResVsPhiLAYER[t][layer][pstv] = new TH2D(Form("hS2ResVsPhiL%d_pstv%d_B%d",layer,pstv,trialCase[t]),
                                                        Form("#Deltas_{2} vs #phi L%d pstv%d Trial%d",layer,pstv,trialCase[t]),600,-TMath::Pi(),+TMath::Pi(),nbins,-range,+range);  
            tree_ResMonitor[t]->Draw(Form("fds2[%d]:fgPhi[2]>>hS2ResVsPhiL%d_pstv%d_B%d",layer,layer,pstv,trialCase[t]),
                                     Form("fgPhi[2]>-10&&std::abs(fds2[%d])<%g&&fstv[%d]==%d",layer,range,2,pstv),"goff");
                                     
            fS2ResLAYER[t][layer][pstv] = new TF1(Form("fS2ResL%d_pstv%d_B%d",layer,pstv,trialCase[t]),"gaus",-range,+range);                
                                     
         }
         
         hCurvChargepT_by_pstv[t][pstv] = new TH1D(Form("hCurvChargepT_by_pstv%d_B%d",pstv,trialCase[t]),"1/p_{T}*sign; 1/p_{T}*sign; N",50,-2,2);
         tree_ResMonitor[t]->Draw(Form("1/pT*cuvR/std::abs(cuvR)>>hCurvChargepT_by_pstv%d_B%d",pstv,trialCase[t]),Form("fgPhi[2]>-10&&std::abs(1/pT)<2&&fstv[2]==%d",pstv),"goff"); 
         fCurvChargepT_by_pstv[t][pstv] = new TF1(Form("fCurvChargepT_by_pstv%d_B%d",pstv,trialCase[t]),"pol2",-1.5,1.5);
         hCurvChargepT_by_pstv[t][pstv]->Fit(Form("fCurvChargepT_by_pstv%d_B%d",pstv,trialCase[t]),"rQ");
        
         hCurvChargepT2D_by_pstv[t][pstv] = new TH2D(Form("hCurvChargepT2D_by_pstv%d_B%d",pstv,trialCase[t]),"1/p_{T}*sign vs phi; phi; 1/p_{T}*sign",100,-TMath::Pi(),TMath::Pi(),50,-2,2);
         tree_ResMonitor[t]->Draw(Form("1/pT*cuvR/std::abs(cuvR):fgPhi[2]>>hCurvChargepT2D_by_pstv%d_B%d",pstv,trialCase[t]),Form("fgPhi[2]>-10&&std::abs(1/pT)<2&&fstv[2]==%d",pstv),"goff");    
      }     
     
      
      hpT[0][t] = new TH1D(Form("hpT_B%d",trialCase[t]),Form("p_{T} Distribution RUN%s; p_{T} (GeV); N",(const char*)runNumber[t]),200, 0, 10);
      hpT[1][t] = new TH1D(Form("hpT_positive_B%d",trialCase[t]),Form("p_{T} Distribution RUN%s; p_{T} (GeV); N",(const char*)runNumber[t]),200, 0, 10);
      hpT[2][t] = new TH1D(Form("hpT_negative_B%d",trialCase[t]),Form("p_{T} Distribution RUN%s; p_{T} (GeV); N",(const char*)runNumber[t]),200, 0, 10);
      tree_ResMonitor[t]->Draw(Form("pT>>hpT_B%d",trialCase[t]),
                               Form(""),"goff");   
      tree_ResMonitor[t]->Draw(Form("pT>>hpT_positive_B%d",trialCase[t]),
                               Form("cuvR<0"),"goff"); 
      tree_ResMonitor[t]->Draw(Form("pT>>hpT_negative_B%d",trialCase[t]),
                               Form("cuvR>0"),"goff");  
                               
      for(int pstv = 0; pstv < 20; pstv++){
         hpT_by_pstv[0][t][pstv] = new TH1D(Form("hpT_by_pstv%d_B%d",pstv,trialCase[t]),Form("p_{T} Distribution RUN%s [pstv%d]; p_{T} (GeV); N",(const char*)runNumber[t],pstv),200, 0, 10);
         hpT_by_pstv[1][t][pstv] = new TH1D(Form("hpT_positive_by_pstv%d_B%d",pstv,trialCase[t]),Form("p_{T} Distribution RUN%s [pstv%d]; p_{T} (GeV); N",(const char*)runNumber[t],pstv),200, 0, 10);
         hpT_by_pstv[2][t][pstv] = new TH1D(Form("hpT_negative_by_pstv%d_B%d",pstv,trialCase[t]),Form("p_{T} Distribution RUN%s [pstv%d]; p_{T} (GeV); N",(const char*)runNumber[t],pstv),200, 0, 10);
         tree_ResMonitor[t]->Draw(Form("pT>>hpT_by_pstv%d_B%d",pstv,trialCase[t]),
                                  Form("fstv[2]==%d",pstv),"goff");   
         tree_ResMonitor[t]->Draw(Form("pT>>hpT_positive_by_pstv%d_B%d",pstv,trialCase[t]),
                                  Form("cuvR<0&&fstv[2]==%d",pstv),"goff"); 
         tree_ResMonitor[t]->Draw(Form("pT>>hpT_negative_by_pstv%d_B%d",pstv,trialCase[t]),
                                  Form("cuvR>0&&fstv[2]==%d",pstv),"goff");  
      }                              
                               
      hvtxXvsPhi[t] = new TH2D(Form("hvtxXvsPhi_B%d",trialCase[t]),"vtxX vs Phi; Phi(rad); vtxX(cm)",200,-TMath::Pi(),TMath::Pi(),400,-0.1,0.1);
      hvtxYvsPhi[t] = new TH2D(Form("hvtxYvsPhi_B%d",trialCase[t]),"vtxY vs Phi; Phi(rad); vtxX(cm)",200,-TMath::Pi(),TMath::Pi(),400,-0.1,0.1);                    
      tree_ResMonitor[t]->Draw(Form("vtxfitX:fgPhi[2]>>hvtxXvsPhi_B%d",trialCase[t]),"TMath::Abs(fgPhi[2])<10","goff");      
      tree_ResMonitor[t]->Draw(Form("vtxfitY:fgPhi[2]>>hvtxYvsPhi_B%d",trialCase[t]),"TMath::Abs(fgPhi[2])<10","goff");

      for(int sphi = 0; sphi < 8; sphi++){
         double phiRange_min = -TMath::Pi() + 2.0*TMath::Pi()/8.0 * sphi;
         double phiRange_max = -TMath::Pi() + 2.0*TMath::Pi()/8.0 * (sphi + 1);
         hvtxX_by_phi[t][sphi] = new TH1D(Form("hvtxX_by_phi%d_B%d",sphi,trialCase[t]),"vtxX distribution; vtxX(cm); N",400,-0.1,0.1);
         hvtxY_by_phi[t][sphi] = new TH1D(Form("hvtxY_by_phi%d_B%d",sphi,trialCase[t]),"vtxY distribution; vtxX(cm); N",400,-0.1,0.1);  
         tree_ResMonitor[t]->Draw(Form("vtxfitX>>hvtxX_by_phi%d_B%d",sphi,trialCase[t]),Form("TMath::Abs(fgPhi[2])<10&&(fgPhi[2]>%g&&fgPhi[2]<=%g)",phiRange_min,phiRange_max),"goff");      
         tree_ResMonitor[t]->Draw(Form("vtxfitY>>hvtxY_by_phi%d_B%d",sphi,trialCase[t]),Form("TMath::Abs(fgPhi[2])<10&&(fgPhi[2]>%g&&fgPhi[2]<=%g)",phiRange_min,phiRange_max),"goff");       
         fvtxX_by_phi[t][sphi] = new TF1(Form("fvtxX_by_phi%d_B%d",sphi,trialCase[t]),"gaus",-0.1,0.1);
         fvtxY_by_phi[t][sphi] = new TF1(Form("fvtxY_by_phi%d_B%d",sphi,trialCase[t]),"gaus",-0.1,0.1);        
      }
   } 
   canvas = new TCanvas("canvas","canvas",2000,900);
}

void plot_TotalChi2(){
   canvas->Clear();
   //canvas->Divide(3,3);
   for(int t = 0; t < nTrials; t++){
      canvas->cd(t+1);
      hChi2TOT[t]->Draw("");
   }
}

void plot_CurvChargepT(){
   canvas->Clear();
   //canvas->Divide(3,3);
   for(int t = 0; t < nTrials; t++){
      canvas->cd(t+1);
      hCurvChargepT[t]->Draw("box");
   }
}

void plot_CurvChargepT2D(TString opt = ""){
   canvas->Clear();
   //canvas->Divide(3,3);
   for(int t = 0; t < nTrials; t++){
      canvas->cd(t+1);
      hCurvChargepT2D[t]->Draw(Form("%s",(const char*)opt));
   }
}

void plot_CurvChargepT_by_pstv(int tr = -1){
   if(tr==-1) return;
   canvas->Clear();
   canvas->Divide(5,4);
   for(int pstv = 0; pstv < 20; pstv++){
      canvas->cd(pstv+1);
      gStyle->SetOptFit(111111);
      hCurvChargepT_by_pstv[tr][pstv]->Draw("");
      std::cout<<"pstv "<<pstv<<" p0 "<<fCurvChargepT_by_pstv[tr][pstv]->GetParameter(0)<<
                                " p1 "<<fCurvChargepT_by_pstv[tr][pstv]->GetParameter(1)<<
                                " p2 "<<fCurvChargepT_by_pstv[tr][pstv]->GetParameter(2)<<
                                " midX "<<-0.5*(fCurvChargepT_by_pstv[tr][pstv]->GetParameter(1)/fCurvChargepT_by_pstv[tr][pstv]->GetParameter(2))<<std::endl; 
      
   }
}

void plot_CurvChargepT2D_by_pstv(int tr = -1){
   if(tr==-1) return;
   canvas->Clear();
   canvas->Divide(5,4);
   for(int pstv = 0; pstv < 20; pstv++){
      canvas->cd(pstv+1);
      hCurvChargepT2D_by_pstv[tr][pstv]->Draw("box");
   }
}

void plot_vtxXVsPhi(TString opt = "", double scale =0.1){
   canvas->Clear();
   //canvas->Divide(3,3);
   for(int t = 0; t < nTrials; t++){
      canvas->cd(t+1);
      hvtxXvsPhi[t]->GetYaxis()->SetRangeUser(-scale,+scale);
      hvtxXvsPhi[t]->Draw(Form("%s",(const char*)opt));
   }
}

void plot_vtxYVsPhi(TString opt = "", double scale =0.1){
   canvas->Clear();
   //canvas->Divide(3,3);
   for(int t = 0; t < nTrials; t++){
      canvas->cd(t+1);
      hvtxYvsPhi[t]->GetYaxis()->SetRangeUser(-scale,+scale);      
      hvtxYvsPhi[t]->Draw(Form("%s",(const char*)opt));
   }
}

void plot_vtxX(TString opt = ""){
   canvas->Clear();
   //canvas->Divide(3,3);
   for(int t = 0; t < nTrials; t++){
      canvas->cd(t+1);
      hvtxXvsPhi[t]->ProjectionY()->Draw(Form("%s",(const char*)opt));
   }
}

void plot_vtxY(TString opt = ""){
   canvas->Clear();
   //canvas->Divide(3,3);
   for(int t = 0; t < nTrials; t++){
      canvas->cd(t+1);
      hvtxYvsPhi[t]->ProjectionY()->Draw(Form("%s",(const char*)opt));
   }
}

void plot_vtxX_by_phi(int tr=-1, double scale =0.1){
   if(tr==-1) return;
   canvas->Clear();
   canvas->Divide(4,2);
   for(int sphi = 0; sphi < 8; sphi++){
      canvas->cd(sphi+1);
      gPad->SetGridx();      
      gPad->SetGridy();      
      hvtxX_by_phi[tr][sphi]->GetXaxis()->SetRangeUser(-scale,+scale);
      fvtxX_by_phi[tr][sphi]->SetRange(-scale,+scale);
      hvtxX_by_phi[tr][sphi]->Fit(Form("fvtxX_by_phi%d_B%d",sphi,trialCase[tr]),"QN");
      hvtxX_by_phi[tr][sphi]->Draw();
      std::cout<<"sphi"<<sphi<<" Mean "<<hvtxX_by_phi[tr][sphi]->GetMean()<<" StdDev "<<hvtxX_by_phi[tr][sphi]->GetStdDev()<<" [Fit] Mean "<<fvtxX_by_phi[tr][sphi]->GetParameter(1)<<std::endl;
   }
}

void plot_vtxY_by_phi(int tr=-1, double scale =0.1){
   if(tr==-1) return;
   canvas->Clear();
   canvas->Divide(4,2);
   for(int sphi = 0; sphi < 8; sphi++){
      canvas->cd(sphi+1);
      gPad->SetGridx();      
      gPad->SetGridy();
      hvtxY_by_phi[tr][sphi]->GetXaxis()->SetRangeUser(-scale,+scale);     
      fvtxY_by_phi[tr][sphi]->SetRange(-scale,+scale);
      hvtxY_by_phi[tr][sphi]->Fit(Form("fvtxY_by_phi%d_B%d",sphi,trialCase[tr]),"QN");      
      hvtxY_by_phi[tr][sphi]->Draw();
      std::cout<<"sphi"<<sphi<<" Mean "<<hvtxX_by_phi[tr][sphi]->GetMean()<<" StdDev "<<hvtxX_by_phi[tr][sphi]->GetStdDev()<<" [Fit] Mean "<<fvtxY_by_phi[tr][sphi]->GetParameter(1)<<std::endl;
   }
}



void plotS1ResVsPhi(int layer = 0, int qr = 0, double rangeY1 = -0.2, double rangeY2 = 0.2){

   double rangeX1 = -TMath::Pi(); double rangeX2 = TMath::Pi();

   if(qr==1){
      rangeX1 = 0 - 0.05;
      rangeX2 = 0.5*TMath::Pi() + 0.05;
   } else if(qr==2) {
      rangeX1 = 0.5*TMath::Pi() - 0.05;
      rangeX2 = TMath::Pi(); 
   } else if(qr==3) {
      rangeX1 = -TMath::Pi();
      rangeX2 = -0.5*TMath::Pi() + 0.05;
   } else if(qr==4) {
      rangeX1 = -0.5*TMath::Pi() - 0.05;
      rangeX2 = 0 + 0.05;
   } 

   canvas->Clear();
   //canvas->Divide(3,3);
   for(int t = 0; t < nTrials; t++){
      canvas->cd(t+1);   
      htemp2D[t][layer] = *hS1ResVsPhi[layer][0][t];
      htemp2D[t][layer].Add(hS1ResVsPhi[layer][1][t]);  
      htemp2D[t][layer].GetXaxis()->SetRangeUser(rangeX1,rangeX2);
      htemp2D[t][layer].GetYaxis()->SetRangeUser(rangeY1,rangeY2);          
      htemp2D[t][layer].Draw();
   }
}
void plotS2ResVsPhi(int layer = 0, int qr = 0, double rangeY1 = -0.2, double rangeY2 = 0.2){
   
   double rangeX1 = -TMath::Pi(); double rangeX2 = TMath::Pi();

   if(qr==1){
      rangeX1 = 0 - 0.05;
      rangeX2 = 0.5*TMath::Pi() + 0.05;
   } else if(qr==2) {
      rangeX1 = 0.5*TMath::Pi() - 0.05;
      rangeX2 = TMath::Pi(); 
   } else if(qr==3) {
      rangeX1 = -TMath::Pi();
      rangeX2 = -0.5*TMath::Pi() + 0.05;
   } else if(qr==4) {
      rangeX1 = -0.5*TMath::Pi() - 0.05;
      rangeX2 = 0 + 0.05;
   } 

   canvas->Clear();
   //canvas->Divide(3,3);
   for(int t = 0; t < nTrials; t++){
      canvas->cd(t+1);   
      htemp2D[t][layer] = *hS2ResVsPhi[layer][0][t];
      htemp2D[t][layer].Add(hS2ResVsPhi[layer][1][t]);  
      htemp2D[t][layer].GetXaxis()->SetRangeUser(rangeX1,rangeX2);
      htemp2D[t][layer].GetYaxis()->SetRangeUser(rangeY1,rangeY2);          
      htemp2D[t][layer].Draw();
   }
}


void plotS2ResVsPhi_Test(int layer = 0, int qr = 0, double rangeY1 = -0.2, double rangeY2 = 0.2){
   
   double rangeX1 = -TMath::Pi(); double rangeX2 = TMath::Pi();

   if(qr==1){
      rangeX1 = 0 - 0.05;
      rangeX2 = 0.5*TMath::Pi() + 0.05;
   } else if(qr==2) {
      rangeX1 = 0.5*TMath::Pi() - 0.05;
      rangeX2 = TMath::Pi(); 
   } else if(qr==3) {
      rangeX1 = -TMath::Pi();
      rangeX2 = -0.5*TMath::Pi() + 0.05;
   } else if(qr==4) {
      rangeX1 = -0.5*TMath::Pi() - 0.05;
      rangeX2 = 0 + 0.05;
   } 

   canvas->Clear();
   canvas->Divide(4,2);
   for(int t = 0; t < nTrials; t++){
      canvas->cd(t+1);   
      if(layer==0) {
         htemp2D[t][layer] = *hS2ResVsPhiL0[t];
         htemp2D[t][layer].GetXaxis()->SetRangeUser(rangeX1,rangeX2);
         htemp2D[t][layer].GetYaxis()->SetRangeUser(rangeY1,rangeY2);          
         htemp2D[t][layer].Draw();
      } else if(layer==2) {
         htemp2D[t][layer] = *hS2ResVsPhiL2[t];
         htemp2D[t][layer].GetXaxis()->SetRangeUser(rangeX1,rangeX2);
         htemp2D[t][layer].GetYaxis()->SetRangeUser(rangeY1,rangeY2);          
         htemp2D[t][layer].Draw();
      } else if(layer==1010000) {
         htemp2D[t][0] = *hS2ResVsPhiL0L2Sum[t];
         htemp2D[t][0].GetXaxis()->SetRangeUser(rangeX1,rangeX2);
         htemp2D[t][0].GetYaxis()->SetRangeUser(rangeY1,rangeY2);          
         htemp2D[t][0].Draw();  
      }

   }
}

void plotS2ResVsPhi_ALL(int layer = 0, int qr = 0, double rangeY1 = -0.2, double rangeY2 = 0.2){
   
   double rangeX1 = -TMath::Pi(); double rangeX2 = TMath::Pi();

   if(qr==1){
      rangeX1 = 0 - 0.05;
      rangeX2 = 0.5*TMath::Pi() + 0.05;
   } else if(qr==2) {
      rangeX1 = 0.5*TMath::Pi() - 0.05;
      rangeX2 = TMath::Pi(); 
   } else if(qr==3) {
      rangeX1 = -TMath::Pi();
      rangeX2 = -0.5*TMath::Pi() + 0.05;
   } else if(qr==4) {
      rangeX1 = -0.5*TMath::Pi() - 0.05;
      rangeX2 = 0 + 0.05;
   } 

   canvas->Clear();
   //canvas->Divide(3,3);
   for(int t = 0; t < nTrials; t++){
      canvas->cd(t+1);   
      htemp2D[t][layer] = *hS2ResVsPhiLAYERALL[t][layer];
      htemp2D[t][layer].GetXaxis()->SetRangeUser(rangeX1,rangeX2);
      htemp2D[t][layer].GetYaxis()->SetRangeUser(rangeY1,rangeY2);          
      htemp2D[t][layer].Draw();


   }
}

void plotS2ResVsPhi_by_pstv_by_Trial(int tr = -1, int pstv = -1, double rangeY1 = -0.2, double rangeY2 = 0.2){
   
   if(tr==-1) return;
   if(pstv==-1) return;
   double rangeX1 = -TMath::Pi(); double rangeX2 = TMath::Pi();

   canvas->Clear();
   canvas->Divide(4,2);
   
   for(int layer = 0; layer < 7; layer++){
      canvas->cd(layer+2);   
      gPad->SetGridx();
      gPad->SetGridy();
      htemp2D[tr][layer] = *hS2ResVsPhiLAYER[tr][layer][pstv]; 
      htemp2D[tr][layer].GetXaxis()->SetRangeUser(rangeX1,rangeX2);
      htemp2D[tr][layer].GetYaxis()->SetRangeUser(rangeY1,rangeY2);          
      htemp2D[tr][layer].Draw();
   }
}

void plotS2Res_by_pstv_by_Trial(int tr = -1, int pstv = -1, double rangeY1 = -0.2, double rangeY2 = 0.2){
   
   if(tr==-1) return;
   if(pstv==-1) return;
   double rangeX1 = -TMath::Pi(); double rangeX2 = TMath::Pi();

   canvas->Clear();
   canvas->Divide(4,2);
   
   for(int layer = 0; layer < 7; layer++){
      canvas->cd(layer+2);   
      gStyle->SetOptFit(111111);
      gPad->SetGridx();
      gPad->SetGridy();
      htemp1D[tr][layer] = *(hS2ResVsPhiLAYER[tr][layer][pstv]->ProjectionY()); 
      htemp1D[tr][layer].GetXaxis()->SetRangeUser(rangeY1,rangeY2);
      fS2ResLAYER[tr][layer][pstv]->SetRange(rangeY1,rangeY2);
      htemp1D[tr][layer].Fit(Form("fS2ResL%d_pstv%d_B%d",layer,pstv,trialCase[tr]),"rQ");
      htemp1D[tr][layer].Draw();
   }
}

void plot_pT_with_charge(double Xmax=10.0){
   canvas->Clear();
   //canvas->Divide(3,3);
   for(int t = 0; t < nTrials; t++){
      canvas->cd(t+1);
      gStyle->SetOptStat(0);
      hpT[1][t]->SetLineColor(kBlue);
      hpT[2][t]->SetLineColor(kRed);
      hpT[1][t]->GetYaxis()->SetRangeUser(0.5,1e+5);
      hpT[2][t]->GetYaxis()->SetRangeUser(0.5,1e+5);  
      hpT[1][t]->GetXaxis()->SetRangeUser(0,Xmax);
      hpT[2][t]->GetXaxis()->SetRangeUser(0,Xmax);            
      hpT[1][t]->Draw("");
      hpT[2][t]->Draw("same");

      tl_hpT_by_pstv[1][0] = new TLatex(Xmax*0.6, 1e+4, Form("N_{positive} : %d", (int)(hpT[1][t]->GetEntries())));
      tl_hpT_by_pstv[1][0]->SetTextSize(0.08);
      tl_hpT_by_pstv[1][0]->SetTextColor(kBlue);
      tl_hpT_by_pstv[1][0]->Draw("same");      
      tl_hpT_by_pstv[2][0] = new TLatex(Xmax*0.6, 1e+3, Form("N_{negative} : %d", (int)(hpT[2][t]->GetEntries())));
      tl_hpT_by_pstv[2][0]->SetTextSize(0.08);
      tl_hpT_by_pstv[2][0]->SetTextColor(kRed);
      tl_hpT_by_pstv[2][0]->Draw("same");            
      
      gPad->SetLogy();      
   }
}

void plot_pT_with_charge_by_pstv(int tr = -1, double textsize = 0.08, double Xmax=10.0){
   if(tr==-1) return;
   canvas->Clear();
   canvas->Divide(5,4,0.0001,0.0001);
   for(int pstv = 0; pstv < 20; pstv++){
      canvas->cd(pstv+1);
      gStyle->SetOptStat(0);
      hpT_by_pstv[1][tr][pstv]->SetLineColor(kBlue);
      hpT_by_pstv[2][tr][pstv]->SetLineColor(kRed);
      hpT_by_pstv[1][tr][pstv]->GetXaxis()->SetLabelSize(0.06);
      hpT_by_pstv[1][tr][pstv]->GetXaxis()->SetTitleSize(0.06);   
      hpT_by_pstv[1][tr][pstv]->GetXaxis()->SetTitleOffset(0.85);               
      hpT_by_pstv[1][tr][pstv]->GetYaxis()->SetLabelSize(0.06);    
      hpT_by_pstv[1][tr][pstv]->GetYaxis()->SetTitleSize(0.06);    
      hpT_by_pstv[1][tr][pstv]->GetYaxis()->SetTitleOffset(0.65); 
      hpT_by_pstv[2][tr][pstv]->GetXaxis()->SetLabelSize(0.06);
      hpT_by_pstv[2][tr][pstv]->GetXaxis()->SetTitleSize(0.06);   
      hpT_by_pstv[2][tr][pstv]->GetXaxis()->SetTitleOffset(0.85);               
      hpT_by_pstv[2][tr][pstv]->GetYaxis()->SetLabelSize(0.06);    
      hpT_by_pstv[2][tr][pstv]->GetYaxis()->SetTitleSize(0.06);    
      hpT_by_pstv[2][tr][pstv]->GetYaxis()->SetTitleOffset(0.65);                                 
      hpT_by_pstv[1][tr][pstv]->GetXaxis()->SetRangeUser(0,Xmax);
      hpT_by_pstv[2][tr][pstv]->GetXaxis()->SetRangeUser(0,Xmax);   
      hpT_by_pstv[1][tr][pstv]->GetYaxis()->SetRangeUser(0.5,2e+3);
      hpT_by_pstv[2][tr][pstv]->GetYaxis()->SetRangeUser(0.5,2e+3);       
       
      hpT_by_pstv[1][tr][pstv]->Draw("");
      hpT_by_pstv[2][tr][pstv]->Draw("same"); 

      tl_hpT_by_pstv[1][pstv] = new TLatex(Xmax*0.6, 5e+2, Form("N_{positive} : %d", (int)(hpT_by_pstv[1][tr][pstv]->GetEntries())));
      tl_hpT_by_pstv[1][pstv]->SetTextSize(textsize);
      tl_hpT_by_pstv[1][pstv]->SetTextColor(kBlue);
      tl_hpT_by_pstv[1][pstv]->Draw("same");      
      tl_hpT_by_pstv[2][pstv] = new TLatex(Xmax*0.6, 1e+2, Form("N_{negative} : %d", (int)(hpT_by_pstv[2][tr][pstv]->GetEntries())));
      tl_hpT_by_pstv[2][pstv]->SetTextSize(textsize);
      tl_hpT_by_pstv[2][pstv]->SetTextColor(kRed);
      tl_hpT_by_pstv[2][pstv]->Draw("same");       
      
      gPad->SetLogy();                 
   }
}

/*

hpTR1[0] = new TH1D(Form("hpTR1_positive"),Form("p_{T} Distribution; p_{T} (GeV); N"),200, 0, 10);
hpTR1[1] = new TH1D(Form("hpTR1_negative"),Form("p_{T} Distribution; p_{T} (GeV); N"),200, 0, 10);
hpTR2[0] = new TH1D(Form("hpTR2_positive"),Form("p_{T} Distribution; p_{T} (GeV); N"),200, 0, 10);
hpTR2[1] = new TH1D(Form("hpTR2_negative"),Form("p_{T} Distribution; p_{T} (GeV); N"),200, 0, 10);
hpTR3[0] = new TH1D(Form("hpTR3_positive"),Form("p_{T} Distribution; p_{T} (GeV); N"),200, 0, 10);
hpTR3[1] = new TH1D(Form("hpTR3_negative"),Form("p_{T} Distribution; p_{T} (GeV); N"),200, 0, 10);
hpTR4[0] = new TH1D(Form("hpTR4_positive"),Form("p_{T} Distribution; p_{T} (GeV); N"),200, 0, 10);
hpTR4[1] = new TH1D(Form("hpTR4_negative"),Form("p_{T} Distribution; p_{T} (GeV); N"),200, 0, 10);


tree_ResMonitor[3]->Draw(Form("pT>>hpTR1_positive"),"fgPhi[2]>-10&&std::abs(1/pT)<2&&(fgPhi[2]>-TMath::Pi()&&fgPhi[2]<=-0.5*TMath::Pi())&&(cuvR>0)","goff")
tree_ResMonitor[3]->Draw(Form("pT>>hpTR1_negative"),"fgPhi[2]>-10&&std::abs(1/pT)<2&&(fgPhi[2]>-TMath::Pi()&&fgPhi[2]<=-0.5*TMath::Pi())&&(cuvR<0)","goff")
tree_ResMonitor[3]->Draw(Form("pT>>hpTR2_positive"),"fgPhi[2]>-10&&std::abs(1/pT)<2&&(fgPhi[2]>-0.5*TMath::Pi()&&fgPhi[2]<=0*TMath::Pi())&&(cuvR>0)","goff")
tree_ResMonitor[3]->Draw(Form("pT>>hpTR2_negative"),"fgPhi[2]>-10&&std::abs(1/pT)<2&&(fgPhi[2]>-0.5*TMath::Pi()&&fgPhi[2]<=0*TMath::Pi())&&(cuvR<0)","goff")
tree_ResMonitor[3]->Draw(Form("pT>>hpTR3_positive"),"fgPhi[2]>-10&&std::abs(1/pT)<2&&(fgPhi[2]>0*TMath::Pi()&&fgPhi[2]<=0.5*TMath::Pi())&&(cuvR>0)","goff")
tree_ResMonitor[3]->Draw(Form("pT>>hpTR3_negative"),"fgPhi[2]>-10&&std::abs(1/pT)<2&&(fgPhi[2]>0*TMath::Pi()&&fgPhi[2]<=0.5*TMath::Pi())&&(cuvR<0)","goff")
tree_ResMonitor[3]->Draw(Form("pT>>hpTR4_positive"),"fgPhi[2]>-10&&std::abs(1/pT)<2&&(fgPhi[2]>0.5*TMath::Pi()&&fgPhi[2]<=TMath::Pi())&&(cuvR>0)","goff")
tree_ResMonitor[3]->Draw(Form("pT>>hpTR4_negative"),"fgPhi[2]>-10&&std::abs(1/pT)<2&&(fgPhi[2]>0.5*TMath::Pi()&&fgPhi[2]<=TMath::Pi())&&(cuvR<0)","goff")

hpTR1[0]->SetLineColor(kBlue);
hpTR1[1]->SetLineColor(kRed);
hpTR2[0]->SetLineColor(kBlue);
hpTR2[1]->SetLineColor(kRed);
hpTR3[0]->SetLineColor(kBlue);
hpTR3[1]->SetLineColor(kRed);
hpTR4[0]->SetLineColor(kBlue);
hpTR4[1]->SetLineColor(kRed);

*/
