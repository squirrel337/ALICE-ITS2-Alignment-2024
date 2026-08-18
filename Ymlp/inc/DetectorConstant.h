
#define nLAYER 		7
#define nSensors	24120//const int ChipBoundary[NLayer + 1] = { 0, 108, 252, 432, 3120, 6480, 14712, 24120 };	
#define LAYERTRAIN      1111111

#define nLAYERtot	7
#define nLAYERIB	3
#define nLAYEROB	4
#define nLAYEROB1	2
#define nLAYEROB2	2
#define nSTAVELAYER0 	12 
#define nSTAVELAYER1	16
#define nSTAVELAYER2 	20
#define nChips	 	9   


#define BeamQX		0	
#define BeamQY		0
#define BeamQZ		1
#define BeamPX		0
#define BeamPY		0 //.7469541
#define BeamPZ		0

//detector information
//#define NCols 1024 //column number in Alpide chip
//#define NRows 512  //row number in Alpide chip

#define nTrackMax 	8

#define FITMODEL	2 // 1 : Line , 2 : Circle

#define SIGMA_MEAS	1

#define DET_MAG 	-0.5 //T  signed: negative solenoid polarity for this run.
// The sign is used, not just the magnitude: det_mag_in_kG = -10*DET_MAG feeds
// YImpactParameter::getImpactParams, where rp4 = yP[4]*bz*B2C sets the curvature
// correction. GetSigma squares the field so it is unaffected, but anything taking a
// magnitude from DET_MAG must say so explicitly -- see RecMomentumT.

#define VERTEXFIT	kFALSE //TRUE FALSE

//Training strategy

#define RANGE_IMPACTPARAMS_R 	0.2
#define RANGE_IMPACTPARAMS_Z 	0.5

#define TrackRejection 		110.0
#define RANGE_CHI_IB		100.0
#define RANGE_CHI_OB		100.0
#define RANGE_CHI_IB_TRAINING	5.0
#define RANGE_CHI_OB_TRAINING	9.0

//Monitoring policy
#define MONITORHALFSTAVEUNIT
//#define MONITORSENSORUNITpT
//#define MONITORSENSORUNITprofile

//Update policy
#define MONITORONLYUPDATES
#define MONITORONLYUPDATES_MODE 1 //0 : generate, 1 : fixed, -1 : flexible(update by epoch)
#define Min_Cluster_by_Sensor 0

#define MONITOR_NbinsIB 400
#define MONITOR_NbinsOB 400
#define MONITOR_RangeIB 0.2
#define MONITOR_RangeOB 0.2

#define MONITOR_NbinsChi 50

#define Update_pTmin 0.40
#define Update_pTmax 2.00
