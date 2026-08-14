void run_profile_beam(){

   TFile* file = new TFile(Form("XXXXinput.root"),"read");
   TTree* tree_DataInput = (TTree*) file->GetObjectUnchecked("DataInput");

   TH2D* hZA = new TH2D("hZA","hZA",500,-0.5,0.5,500,-0.5,0.5);
   TH2D* hZB = new TH2D("hZB","hZB",500,-0.5,0.5,500,-0.5,0.5);
   TH2D* hZC = new TH2D("hZC","hZC",500,-0.5,0.5,500,-0.5,0.5);
      
   tree_DataInput->Draw("X2:X1>>hZA","TMath::Abs(X1)<0.5&&TMath::Abs(X2)<0.5&&TMath::Abs(X3-7.5)<1.5","goff");
   tree_DataInput->Draw("X2:X1>>hZB","TMath::Abs(X1)<0.5&&TMath::Abs(X2)<0.5&&TMath::Abs(X3-0)<1.5","goff");
   tree_DataInput->Draw("X2:X1>>hZC","TMath::Abs(X1)<0.5&&TMath::Abs(X2)<0.5&&TMath::Abs(X3+7.5)<1.5","goff");

   TF1* fXA = new TF1("fXA","gaus",-0.5,0.5);
   TF1* fXB = new TF1("fXB","gaus",-0.5,0.5);
   TF1* fXC = new TF1("fXC","gaus",-0.5,0.5);
   hZA->ProjectionX()->Fit("fXA");
   hZB->ProjectionX()->Fit("fXB");
   hZC->ProjectionX()->Fit("fXC");
   double meanX[] = {fXC->GetParameter(1), fXB->GetParameter(1), fXA->GetParameter(1) };

   TF1* fYA = new TF1("fYA","gaus",-0.5,0.5);
   TF1* fYB = new TF1("fYB","gaus",-0.5,0.5);
   TF1* fYC = new TF1("fYC","gaus",-0.5,0.5);
   hZA->ProjectionY()->Fit("fYA");
   hZB->ProjectionY()->Fit("fYB");
   hZC->ProjectionY()->Fit("fYC");
   double meanY[] = {fYC->GetParameter(1), fYB->GetParameter(1), fYA->GetParameter(1) };

   double Z[] = {-7.5, 0, 7.5};

   TGraph* grXZ = new TGraph(3, Z, meanX);
   TGraph* grYZ = new TGraph(3, Z, meanY);
   TF1* fpolXZ = new TF1("fpolXZ","pol1");
   TF1* fpolYZ = new TF1("fpolYZ","pol1");
   grXZ->Fit("fpolXZ");
   grYZ->Fit("fpolYZ");

   double BeamProfileXZ0 = fpolXZ->GetParameter(0);
   double BeamProfileXZ1 = fpolXZ->GetParameter(1);
   double BeamProfileYZ0 = fpolYZ->GetParameter(0);	
   double BeamProfileYZ1 = fpolYZ->GetParameter(1);
   
   
   gSystem->Exec(Form("echo \"#define BeamProfileXZ0 %g\"  > YMLPBeamProfile.h",BeamProfileXZ0));
   gSystem->Exec(Form("echo \"#define BeamProfileXZ1 %g\" >> YMLPBeamProfile.h",BeamProfileXZ1));
   gSystem->Exec(Form("echo \"#define BeamProfileYZ0 %g\" >> YMLPBeamProfile.h",BeamProfileYZ0));
   gSystem->Exec(Form("echo \"#define BeamProfileYZ1 %g\" >> YMLPBeamProfile.h",BeamProfileYZ1));         

}
