#include <vector>
float PITCH_ROW =26.88e-4;
float PITCH_COL =29.24e-4;
float PITCH_CHIP=3+150e-4;
float ALPHA[3]={16.997, 17.504, 17.337};
float RMID[3] ={2.3490-500e-4,3.1586-500e-4,3.9341-500e-4};
float PERST[3]={30,22.5,18};
float OFFSET_CHIP_Y=+(1.5-PITCH_ROW*512)/2;

TVector3 l2g(int layer ,int stave, int chip, float lx, float ly);
TVector3 g2l(int layer, int stave, int chip, double gx, double gy, double gz);
TVector3 normalV(int layer ,int stave, int chip);
TVector3 DnormalV(int layer ,int stave, int chip, double ds1_0, double ds2_0, double *ds3);
TVector3 g2s(int layer, int stave, int chip, double gx, double gy, double gz);
TVector3 s2g(int layer, int stave, int chip, double s1, double s2, double s3);
TVector3 Func_D(int layer, int stave, int chip, float lx, float ly, double ds1_0, double ds2_0, double *ds3);

TVector3 l2g(int layer ,int stave, int chip, float lx, float ly){
  TVector3 v3(0,PITCH_ROW*(ly-256)+OFFSET_CHIP_Y   , PITCH_COL*(lx-512) + (chip-4)*PITCH_CHIP );
  v3.RotateZ(-TMath::DegToRad()*ALPHA[layer]);
  v3.SetX(v3.X()+RMID[layer]);
  v3.RotateZ(TMath::DegToRad()*ALPHA[layer]);
  v3.RotateZ(TMath::DegToRad()*stave*PERST[layer]);
  return v3;
}

TVector3 g2l(int layer, int stave, int chip, double gx, double gy, double gz){
  TVector3 v3(gx, gy, gz);
  v3.RotateZ(-TMath::DegToRad()*stave*PERST[layer]);
  v3.RotateZ(-TMath::DegToRad()*ALPHA[layer]);
  v3.SetX(v3.X()-RMID[layer]); 
  v3.RotateZ(TMath::DegToRad()*ALPHA[layer]);  
  float lx = ((v3.Z()-(chip-4)*PITCH_CHIP)/PITCH_COL)+512;
  float ly = ((v3.Y()-OFFSET_CHIP_Y)/PITCH_ROW)+256;
  //v3.SetXYZ(v3.X(),((v3.Y()-OFFSET_CHIP_Y)/PITCH_ROW)+256,((v3.Z()-(chip-4)*PITCH_CHIP)/PITCH_COL)+512);
  v3.SetXYZ(0,lx,ly);
  return v3;
}

TVector3 normalV(int layer, int stave, int chip){ //layer,stave,chip
  TVector3 x1(l2g(layer,stave,chip,512,0).X()-l2g(layer,stave,chip,512,256).X(),
              l2g(layer,stave,chip,512,0).Y()-l2g(layer,stave,chip,512,256).Y(),
              l2g(layer,stave,chip,512,0).Z()-l2g(layer,stave,chip,512,256).Z());
  TVector3 x3(0,0,1);
  TVector3 x3x1(x3.Cross(x1).X(),x3.Cross(x1).Y(),x3.Cross(x1).Z());
  double X3X1 = TMath::Sqrt((x3x1.X()*x3x1.X())+(x3x1.Y()*x3x1.Y())+(x3x1.Z()*x3x1.Z()));
  TVector3 x2(x3x1.X()/X3X1,x3x1.Y()/X3X1,x3x1.Z()/X3X1);
  return x2;
}

TVector3 DnormalV(int layer ,int stave, int chip, double ds1_0, double ds2_0, double *ds3){ //layer,stave,chip
  TVector3 x1(l2g(layer,stave,chip,512,0).X()-l2g(layer,stave,chip,512,256).X(),
              l2g(layer,stave,chip,512,0).Y()-l2g(layer,stave,chip,512,256).Y(),
              l2g(layer,stave,chip,512,0).Z()-l2g(layer,stave,chip,512,256).Z());
 // std::cout<<"DnormalV("<<layer<<") : "<<x1(0)<<" "<<x1(1)<<" "<<x1(2)<<std::endl;
  //std::cout<<"DnormalV("<<layer<<") Func_D : "<<Func_D(layer, stave, chip, 512, 0, ds1_0, ds2_0, ds3)(0)-Func_D(layer, stave, chip, 512, 256, ds1_0, ds2_0, ds3)(0)<<" "
   //                              <<Func_D(layer, stave, chip, 512, 0, ds1_0, ds2_0, ds3)(1)-Func_D(layer, stave, chip, 512, 256, ds1_0, ds2_0, ds3)(1)<<" "
   //                              <<Func_D(layer, stave, chip, 512, 0, ds1_0, ds2_0, ds3)(2)-Func_D(layer, stave, chip, 512, 256, ds1_0, ds2_0, ds3)(2)<<std::endl;
  x1.SetX(x1.X()+(Func_D(layer, stave, chip, 512, 0, ds1_0, ds2_0, ds3).X()-Func_D(layer, stave, chip, 512, 256, ds1_0, ds2_0, ds3).X()));      
  x1.SetY(x1.Y()+(Func_D(layer, stave, chip, 512, 0, ds1_0, ds2_0, ds3).Y()-Func_D(layer, stave, chip, 512, 256, ds1_0, ds2_0, ds3).Y()));      
  x1.SetZ(x1.Z()+(Func_D(layer, stave, chip, 512, 0, ds1_0, ds2_0, ds3).Z()-Func_D(layer, stave, chip, 512, 256, ds1_0, ds2_0, ds3).Z()));                  
  TVector3 x3(0,0,1);
  TVector3 x3x1(x3.Cross(x1).X(),x3.Cross(x1).Y(),x3.Cross(x1).Z());
  double X3X1 = TMath::Sqrt((x3x1.X()*x3x1.X())+(x3x1.Y()*x3x1.Y())+(x3x1.Z()*x3x1.Z()));
  TVector3 x2(x3x1.X()/X3X1,x3x1.Y()/X3X1,x3x1.Z()/X3X1);
  return x2;
}

TVector3 g2s(int layer, int stave, int chip, double gx, double gy, double gz){
  TVector3 x_gy(0,1,0);
  TVector3 x_sy(normalV(layer,stave, chip).X(),normalV(layer,stave, chip).Y(),normalV(layer,stave, chip).Z());
    
  //double beta = TMath::ATan2(x_sy.X(),x_sy.Y());
  double beta = -TMath::ATan2(x_sy.X(),x_sy.Y());	
  TMatrixD Mg2s(4,4);
  Mg2s[0] = { 	+TMath::Cos(beta),	+TMath::Sin(beta), 		0, 		0};
  Mg2s[1] = { 	-TMath::Sin(beta),	+TMath::Cos(beta),		0,		0};
  Mg2s[2] = {			0,			0,		1,		0};
  Mg2s[3] = {			0,			0,		0,		1};

  TMatrixD Mg(1,4), Ms(1,4);
  Mg[0] = {gx-l2g(layer,stave, chip,512,256).X(), gy-l2g(layer,stave, chip,512,256).Y(), gz, 1};
  //Mg[0] = {gy-l2g(layer,stave, chip,512,256).Y(), gx-l2g(layer,stave, chip,512,256).X(), gz, 1};
  Ms.T() = Mg2s * Mg.T();
  TVector3 output(Ms[2][0],Ms[0][0],Ms[1][0]);
  //y axis
  //TVector3 output(Ms[1][0],Ms[0][0],Ms[2][0]);
  return output;
  
}

TVector3 s2g(int layer, int stave, int chip, double s1, double s2, double s3){
  TVector3 x_gy(0,1,0);
  TVector3 x_sy(normalV(layer,stave, chip).X(),normalV(layer,stave, chip).Y(),normalV(layer,stave, chip).Z());
  
  //double beta = TMath::ATan2(x_sy.X(),x_sy.Y());
  double beta = -TMath::ATan2(x_sy.X(),x_sy.Y());
  TMatrixD Mg2s(4,4);
  Mg2s[0] = { 	+TMath::Cos(beta),	+TMath::Sin(beta), 		0, 		0};
  Mg2s[1] = { 	-TMath::Sin(beta),	+TMath::Cos(beta),		0,		0};
  Mg2s[2] = {			0,			0,		1,		0};
  Mg2s[3] = {			0,			0,		0,		1};
  
  TMatrixD Mg(1,4), Ms(1,4);
  Ms[0] = {s2, s3, s1, 1};
  Mg.T() = Mg2s.Invert() * Ms.T();
  
  double gamma = TMath::ATan2(x_sy.Y(), x_sy.X());
  //double gx_s3 = 0;//s3*TMath::Cos(gamma);
  //double gy_s3 = 0;//s3*TMath::Sin(gamma);
  //TVector3 output(Mg[1][0]+l2g(layer,stave, chip,512,256).Y(),Mg[0][0]+l2g(layer,stave, chip,512,256).X(),Mg[2][0]+l2g(layer,stave, chip,512,256).Z());
  TVector3 output(Mg[0][0]+l2g(layer,stave, chip,512,256).X(),Mg[1][0]+l2g(layer,stave, chip,512,256).Y(),Mg[2][0]);
  return output;
}

TVector3 Func_D(int layer, int stave, int chip, float lx, float ly, double ds1_0, double ds2_0, double *ds3){
  TVector3 func_D(0,0,0);
    
  //Calculate l2s = l2g * g2s
  TVector3 gX_Ideal = l2g(layer, stave, chip, lx, ly);
  TVector3 gS_Ideal = g2s(layer, stave, chip, gX_Ideal.X(), gX_Ideal.Y(), gX_Ideal.Z());
  TVector3 gS_Deformed; 

  //Calculate length

  double i1 =g2s(layer,stave,chip,l2g(layer,stave,chip,0,0)(0),l2g(layer,stave,chip,0,0)(1),l2g(layer,stave,chip,0,0)(2))(0);
  double i2 =g2s(layer,stave,chip,l2g(layer,stave,chip,0,0)(0),l2g(layer,stave,chip,0,0)(1),l2g(layer,stave,chip,0,0)(2))(1);
  double c1 =g2s(layer,stave,chip,l2g(layer,stave,chip,512,256)(0),l2g(layer,stave,chip,512,256)(1),l2g(layer,stave,chip,512,256)(2))(0);
  double c2 =g2s(layer,stave,chip,l2g(layer,stave,chip,512,256)(0),l2g(layer,stave,chip,512,256)(1),l2g(layer,stave,chip,512,256)(2))(1); 
  double f1 =g2s(layer,stave,chip,l2g(layer,stave,chip,1024,512)(0),l2g(layer,stave,chip,1024,512)(1),l2g(layer,stave,chip,1024,512)(2))(0);
  double f2 =g2s(layer,stave,chip,l2g(layer,stave,chip,1024,512)(0),l2g(layer,stave,chip,1024,512)(1),l2g(layer,stave,chip,1024,512)(2))(1);
  double length1 = TMath::Abs(f1-i1);
  double length2 = TMath::Abs(f2-i2);



  //0th order in s1, s2 and s3 (translation)
  gS_Deformed.SetX(gS_Ideal(0) + ds1_0);
  gS_Deformed.SetY(gS_Ideal(1) + ds2_0);
  gS_Deformed.SetZ(gS_Ideal(2) + ds3[0]);

  //1st order only in s3 (tilt)   //s3 -> trans (0) tilt (angle 1, angle 2, angle 3) 
  TMatrixD R11(3,3), R12(3,3), R13(3,3);
  
  double a11 = ds3[1];
  double a12 = ds3[2];
  double a13 = ds3[3];
  
  if(TMath::Abs(a11)>0.0001||TMath::Abs(a12)>0.0001||TMath::Abs(a13)>0.0001){
    R11[0] = {	1,		       0,		       0};  
    R11[1] = { 	0, 	+TMath::Cos(a11),	-TMath::Sin(a11)};
    R11[2] = { 	0,	+TMath::Sin(a11),	+TMath::Cos(a11)};

    R12[0] = { 	+TMath::Cos(a12),      0, 	+TMath::Sin(a12)};
    R12[1] = {		       0,      1,	               0};    
    R12[2] = { 	-TMath::Sin(a12),      0, 	+TMath::Cos(a12)}; 
  
    R13[0] = { 	+TMath::Cos(a13),      -TMath::Sin(a13),       0};
    R13[1] = { 	+TMath::Sin(a13),      +TMath::Cos(a13),       0}; 
    R13[2] = {		       0,      		      0,       1};    

    TMatrixD P_ds3(1,3);
    P_ds3[0] = {gS_Ideal(0)-c1, gS_Ideal(1)-c2, gS_Ideal(2)};
    //std::cout<<"FuncD : "<<P_ds3[0][0]<<" "<<P_ds3[0][1]<<" "<<P_ds3[0][2]<<std::endl;
    TMatrixD TP_ds3(1,3);  
    TP_ds3.T() = R11*R12*R13*P_ds3.T();

    double ds3_11 = TP_ds3[0][0] - P_ds3[0][0];
    double ds3_12 = TP_ds3[1][0] - P_ds3[1][0];
    double ds3_13 = TP_ds3[2][0] - P_ds3[2][0];

    gS_Deformed.SetX(gS_Deformed(0) + ds3_11);
    gS_Deformed.SetY(gS_Deformed(1) + ds3_12);
    gS_Deformed.SetZ(gS_Deformed(2) + ds3_13);
  }  


  //2nd order only in s3 (bend) angle 4 (s2 axis transl only)
  double htilt = ds3[4];  
  if(TMath::Abs(htilt)>0.0001){ //1 micron above
    double l1 =g2s(layer,stave,chip,l2g(layer,stave,chip,lx,ly)(0),l2g(layer,stave,chip,lx,ly)(1),l2g(layer,stave,chip,lx,ly)(2))(0);
    double value_R = 1/htilt;//((l1 - c1)*(l1 - c1) + htilt*htilt )/(2*htilt);
    double dl1 = value_R*TMath::Sin((l1 - c1)/value_R);
    //std::cout<<"2nd Order :: "<<htilt<<" "<<l1<<" "<<c1<<" "<<value_R<<" "<<dl1<<" "<<l1-c1<<std::endl;
    double ds3_21 = (dl1) - (l1-c1);
    double ds3_22 = 0;
    double ds3_23 = value_R - TMath::Sqrt(value_R*value_R - dl1*dl1);
  
    gS_Deformed.SetX(gS_Deformed(0) + ds3_21);
    gS_Deformed.SetY(gS_Deformed(1) + ds3_22);
    gS_Deformed.SetZ(gS_Deformed(2) + ds3_23);  
  }


  //unification  
  
  TVector3 gX_Deformed = s2g(layer, stave, chip, gS_Deformed(0), gS_Deformed(1), gS_Deformed(2));
  
  func_D.SetX(gX_Deformed.X() - gX_Ideal.X());
  func_D.SetY(gX_Deformed.Y() - gX_Ideal.Y());
  func_D.SetZ(gX_Deformed.Z() - gX_Ideal.Z());
  //std::cout<<"Output ("<<layer<<") Func_D : "<<func_D(0)<<" "<<func_D(1)<<" "<<func_D(2)<<std::endl;
  return func_D;
}

