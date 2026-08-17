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
// YDetectorGeometry
//
// Two interchangeable backends behind one interface:
//
//   default        - a geometry cache produced once by tools/export_geometry_cache.C.
//                    Needs ROOT only, so the module runs without O2.
//   YGEOM_USE_O2   - the original, reading the geometry and alignment through
//                    o2::its::GeometryTGeo. This is the reference the cache is
//                    validated against; see tools/validate_geometry_cache.C.
//
// The public interface is identical either way, so no call site changes.
//
///////////////////////////////////////////////////////////////////////////

#include "../inc/YDetectorGeometry.h"

#ifndef YGEOM_USE_O2
#include <TFile.h>
#include <TTree.h>
#include <TNamed.h>
#include <TSystem.h>
#include <Riostream.h>
#endif

static void at_exit_of_YDetectorGeometry() {
   if (GEOM::Internal::yGEOMLocal)
      GEOM::Internal::yGEOMLocal->~YDetectorGeometry();
}

// This local static object initializes the GEOM system
namespace GEOM {
namespace Internal {
   class YDetectorGeometryAllocator {

      char fHolder[sizeof(YDetectorGeometry)];
   public:
      YDetectorGeometryAllocator() {
         new(&(fHolder[0])) YDetectorGeometry("geom","The GEOM of EVERYTHING");
      }

      ~YDetectorGeometryAllocator() {
         if (yGEOMLocal) {
            yGEOMLocal->~YDetectorGeometry();
         }
      }
   };

   extern YDetectorGeometry *yGEOMLocal;

   YDetectorGeometry *GetGEOM1() {
      if (yGEOMLocal)
         return yGEOMLocal;
      static YDetectorGeometryAllocator alloc;
      return yGEOMLocal;
   }

   YDetectorGeometry *GetGEOM2() {
      static Bool_t initInterpreter = kFALSE;
      if (!initInterpreter) {
         initInterpreter = kTRUE;
      }
      return yGEOMLocal;
   }
   typedef YDetectorGeometry *(*GetGEOMFun_t)();

   static GetGEOMFun_t yGetGEOM = &GetGEOM1;


} // end of Internal sub namespace
// back to GEOM namespace

   YDetectorGeometry *GetGEOM() {
      return (*Internal::yGetGEOM)();
   }
}

YDetectorGeometry *GEOM::Internal::yGEOMLocal = GEOM::GetGEOM();

ClassImp(YDetectorGeometry);

////////////////////////////////////////////////////////////////////////////////
/// Default Constructor YDetectorGeometry

YDetectorGeometry::YDetectorGeometry(const char *name, const char *title) {
   std::cout<<"Default Constructor YDetectorGeometry "<<std::endl;

#ifdef YGEOM_USE_O2
   o2::base::GeometryManager::loadGeometry("", false, false);

   TFile file("ITSAlignment.root");
   std::vector<o2::detectors::AlignParam>* aliPars;
   file.GetObject("ccdb_object", aliPars);
   o2::base::GeometryManager::applyAlignment(*aliPars);
   std::cout<<" LOAD ITSaligment constants"<< std::endl;
   for(int ich = 0; ich < (*aliPars).size(); ich ++) (*aliPars)[ich].print();

   geom = o2::its::GeometryTGeo::Instance();

   geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::L2G));
#else
   const char* env = gSystem->Getenv("YGEOM_CACHE");
   LoadCache(env && env[0] ? env : "geometry/its2_geom.root");
#endif

   GEOM::Internal::yGEOMLocal = this;
   GEOM::Internal::yGetGEOM = &GEOM::Internal::GetGEOM2;
}

#ifndef YGEOM_USE_O2

////////////////////////////////////////////////////////////////////////////////
/// LoadCache
///
/// Reads the per-chip transforms and addressing written by
/// tools/export_geometry_cache.C. Refuses to continue on a short or missing file
/// rather than silently running on a partial geometry.

void YDetectorGeometry::LoadCache(const char* filename)
{
   TFile* f = TFile::Open(filename, "read");
   if (!f || f->IsZombie()) {
      Fatal("YDetectorGeometry", "cannot open geometry cache '%s'. "
            "Produce it with tools/export_geometry_cache.C, or set YGEOM_CACHE.", filename);
      return;
   }
   TNamed* prov = (TNamed*)f->Get("provenance");
   if (prov) fProvenance = prov->GetTitle();

   TTree* t = (TTree*)f->Get("geom");
   if (!t) { Fatal("YDetectorGeometry", "no 'geom' tree in %s", filename); return; }

   Int_t chipID, layer, halfBarrel, stave, halfStave, module, chipInModule;
   Int_t chipInLayer, chipInStave, chipInHalfStave;
   Double_t R[9], T[3];
   t->SetBranchAddress("chipID", &chipID);
   t->SetBranchAddress("layer", &layer);
   t->SetBranchAddress("halfBarrel", &halfBarrel);
   t->SetBranchAddress("stave", &stave);
   t->SetBranchAddress("halfStave", &halfStave);
   t->SetBranchAddress("module", &module);
   t->SetBranchAddress("chipInModule", &chipInModule);
   t->SetBranchAddress("chipInLayer", &chipInLayer);
   t->SetBranchAddress("chipInStave", &chipInStave);
   t->SetBranchAddress("chipInHalfStave", &chipInHalfStave);
   t->SetBranchAddress("R", R);
   t->SetBranchAddress("T", T);

   const int nExpected = ChipBoundary[NLayer];
   fChip.resize(nExpected);
   std::vector<bool> seen(nExpected, false);

   const Long64_t n = t->GetEntries();
   for (Long64_t i = 0; i < n; ++i) {
      t->GetEntry(i);
      if (chipID < 0 || chipID >= nExpected) continue;
      ChipGeom& c = fChip[chipID];
      for (int k = 0; k < 9; ++k) c.R[k] = R[k];
      for (int k = 0; k < 3; ++k) c.T[k] = T[k];
      c.layer = layer; c.halfBarrel = halfBarrel; c.stave = stave;
      c.halfStave = halfStave; c.module = module; c.chipInModule = chipInModule;
      c.chipInLayer = chipInLayer; c.chipInStave = chipInStave;
      c.chipInHalfStave = chipInHalfStave;
      seen[chipID] = true;
   }
   int nMissing = 0;
   for (int i = 0; i < nExpected; ++i) if (!seen[i]) ++nMissing;
   if (nMissing) {
      Fatal("YDetectorGeometry", "geometry cache '%s' covers %d of %d chips",
            filename, nExpected - nMissing, nExpected);
      return;
   }
   std::cout<<" LOAD geometry cache "<<filename<<" : "<<nExpected<<" chips"<<std::endl;
   if (fProvenance != "") std::cout<<"   provenance : "<<fProvenance<<std::endl;
   f->Close();
}

////////////////////////////////////////////////////////////////////////////////
/// L2G / G2L : the cached rigid transform of one chip, and its exact inverse.

void YDetectorGeometry::L2G(int chipID, const double loc[3], double glo[3]) const
{
   const ChipGeom& c = fChip[chipID];
   for (int i = 0; i < 3; ++i)
      glo[i] = c.R[3*i]*loc[0] + c.R[3*i+1]*loc[1] + c.R[3*i+2]*loc[2] + c.T[i];
}

void YDetectorGeometry::G2L(int chipID, const double glo[3], double loc[3]) const
{
   const ChipGeom& c = fChip[chipID];
   const double d[3] = {glo[0]-c.T[0], glo[1]-c.T[1], glo[2]-c.T[2]};
   // R is orthonormal, so the inverse rotation is its transpose
   for (int i = 0; i < 3; ++i)
      loc[i] = c.R[i]*d[0] + c.R[3+i]*d[1] + c.R[6+i]*d[2];
}

////////////////////////////////////////////////////////////////////////////////
/// DetectorToLocal
///
/// Pixel (row, col) to the chip-local frame: X along rows, Y the sensor normal,
/// Z along columns. Chip-independent, so it is pure segmentation arithmetic.
///
/// This is the exact inverse of the local-to-pixel conversion already written out
/// in GToL below, so the two are consistent by construction rather than by
/// transcription of the O2 formula.

void YDetectorGeometry::DetectorToLocal(float row, float col, double loc[3])
{
   loc[0] = 0.5 * (ActiveMatrixSizeRows - PassiveEdgeTop + PassiveEdgeReadOut)
          - (double(row) + 0.5) * PitchRow;
   loc[1] = 0.0;
   loc[2] = (double(col) + 0.5) * PitchCol - 0.5 * ActiveMatrixSizeCols;
}

#endif // !YGEOM_USE_O2

////////////////////////////////////////////////////////////////////////////////
/// GetL2GComponents

void YDetectorGeometry::GetL2GComponents(int chipID,
                                         double &Rxx, double &Rxy, double &Rxz, double &Tdx,
                                         double &Ryx, double &Ryy, double &Ryz, double &Tdy,
                                         double &Rzx, double &Rzy, double &Rzz, double &Tdz) const
{
#ifdef YGEOM_USE_O2
   geom->getMatrixL2G(chipID).GetComponents(Rxx, Rxy, Rxz, Tdx, Ryx, Ryy, Ryz, Tdy, Rzx, Rzy, Rzz, Tdz);
#else
   const ChipGeom& c = fChip[chipID];
   Rxx = c.R[0]; Rxy = c.R[1]; Rxz = c.R[2]; Tdx = c.T[0];
   Ryx = c.R[3]; Ryy = c.R[4]; Ryz = c.R[5]; Tdy = c.T[1];
   Rzx = c.R[6]; Rzy = c.R[7]; Rzz = c.R[8]; Tdz = c.T[2];
#endif
}

////////////////////////////////////////////////////////////////////////////////
/// GToS

TVector3 YDetectorGeometry::GToS(int chipID, double gx, double gy, double gz)
{
   if(chipID<0){
      TVector3 v3(-9999,-9999,-9999);
      return v3;
   }

#ifdef YGEOM_USE_O2
   o2::math_utils::Point3D<float> gloC(gx, gy, gz);
   auto loc = geom->getMatrixL2G(chipID) ^ gloC; // convert global coordinates to local.
   float l1= loc.X(); //xrow s1
   float l2= loc.Y(); //s3
   float l3= loc.Z(); //zcol s2
#else
   const double glo[3] = {gx, gy, gz};
   double locd[3];
   G2L(chipID, glo, locd);
   // narrowed to float at the same point the O2 path does, via Point3D<float>
   float l1= (float)locd[0]; //xrow s1
   float l2= (float)locd[1]; //s3
   float l3= (float)locd[2]; //zcol s2
#endif

   int Layer = GetLayer(chipID);

   if(Layer<NLayerIB){
      TVector3 v3(-l3,-l1,l2);
      return v3;
   } else {
      //  for OB (chipID in Module)
      //   6  5  4  3  2  1  0    	 (+)
      //   7  8  9 10 11 12 13     	 (-)  -> Z

      //                  ///////////////////////
      //                  //                   //
      // 6  5  4  3  2  1 //         0         //
      //                  //                   //
      //                  /////////////////////## (0,0)

      //            (0,0) ##/////////////////////
      //                  //                   //
      // 7  8  9 10 11 12 //         13        //
      //                  //                   //
      //                  ///////////////////////
      short dir        = (GetChipIdInModule(chipID)<7) ? +1 : -1;
      TVector3 v3(-dir*l3,-dir*l1,l2);
      return v3;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// LToG

TVector3 YDetectorGeometry::LToG(int chipID, float row, float col)
{
   if(chipID<0){
      TVector3 v3(-9999,-9999,-9999);
      return v3;
   }

#ifdef YGEOM_USE_O2
   o2::math_utils::Point3D<float> locC;
   o2::itsmft::SegmentationAlpide::detectorToLocalUnchecked(row, col, locC); // local coordinates
   auto gloC = geom->getMatrixL2G(chipID) * locC;

   float gx = gloC.X();
   float gy = gloC.Y();
   float gz = gloC.Z();
#else
   double locd[3], glod[3];
   DetectorToLocal(row, col, locd);
   L2G(chipID, locd, glod);

   float gx = (float)glod[0];
   float gy = (float)glod[1];
   float gz = (float)glod[2];
#endif

   TVector3 v3(gx,gy,gz);
   return v3;
}

////////////////////////////////////////////////////////////////////////////////
/// SToL

TVector3 YDetectorGeometry::SToL(int chipID, double s1, double s2, double s3){

   if(chipID<0){
      TVector3 v3(-9999,-9999,-9999);
      return v3;
   }

   int Layer = GetLayer(chipID);

   if(Layer<NLayerIB){
      s1 = -s1;
      s2 = -s2;
   } else {
      short dir        = (GetChipIdInModule(chipID)<7) ? +1 : -1;
      s1 = -dir*s1;
      s2 = -dir*s2;
   }

   s2 = 0.5 * (ActiveMatrixSizeRows - PassiveEdgeTop + PassiveEdgeReadOut) - s2;   // coordinate wrt top edge of Active matrix
   s1 += 0.5 * ActiveMatrixSizeCols;                                               // coordinate wrt left edge of Active matrix
   float frow = float(s2 / PitchRow) - 0.5;
   float fcol = float(s1 / PitchCol) - 0.5;
   TVector3 v3(frow,fcol,0);
   return v3;
}

////////////////////////////////////////////////////////////////////////////////
/// SToG

TVector3 YDetectorGeometry::SToG(int chipID, double s1, double s2, double s3){

   if(chipID<0){
      TVector3 v3(-9999,-9999,-9999);
      return v3;
   }

   TVector3 stol = SToL(chipID, s1, s2, s3);
   float row = stol(0);
   float col = stol(1);

   TVector3 ltog = LToG(chipID, row, col);

   float gx = ltog.X();
   float gy = ltog.Y();
   float gz = ltog.Z();

   TVector3 normV = NormalVector(chipID);
   float dgx = s3*normV(0);
   float dgy = s3*normV(1);

   TVector3 v3(gx+dgx,gy+dgy,gz);
   return v3;
}

////////////////////////////////////////////////////////////////////////////////
/// GToL

TVector3 YDetectorGeometry::GToL(int chipID, double gx, double gy, double gz)
{
   if(chipID<0){
      TVector3 v3(-9999,-9999,-9999);
      return v3;
   }

   TVector3 gtos = GToS(chipID, gx, gy, gz);

   float l1= -gtos(1);
   float l2= gtos(2);
   float l3= -gtos(0);
   float frow, fcol;
   //
   // convert to row/col w/o over/underflow check
   l1 = 0.5 * (ActiveMatrixSizeRows - PassiveEdgeTop + PassiveEdgeReadOut) - l1; // coordinate wrt top edge of Active matrix
   l3 += 0.5 * ActiveMatrixSizeCols;                                               // coordinate wrt left edge of Active matrix
   frow = float(l1 / PitchRow) - 0.5;
   fcol = float(l3 / PitchCol) - 0.5;
   if (l1 < 0) {
     frow -= 1;
   }
   if (l3 < 0) {
     fcol -= 1;
   }
   //
   TVector3 v3(frow,fcol,0);
   return v3;
}

////////////////////////////////////////////////////////////////////////////////
/// NormalVector

TVector3 YDetectorGeometry::NormalVector(int chipID){				// Stave plane
   if(chipID<0){
      TVector3 e3(-9999,-9999,-9999);
      return e3;
   }

      //  for OB (chipID in Module)
      //   6  5  4  3  2  1  0    	 (+)
      //   7  8  9 10 11 12 13     	 (-)  -> Z

      //                  ///////////////////////
      //                  //                   //
      // 6  5  4  3  2  1 //   v1    0         //
      //                  //        v2         //
      //                  /////////////////////## (0,0)

      //            (0,0) ##/////////////////////
      //                  //         v2        //
      // 7  8  9 10 11 12 //         13  v1    //
      //                  //                   //
      //                  ///////////////////////

   TVector3 v1(LToG(chipID,256,512).X()-LToG(chipID,256,0).X(),
               LToG(chipID,256,512).Y()-LToG(chipID,256,0).Y(),
               LToG(chipID,256,512).Z()-LToG(chipID,256,0).Z());
   TVector3 v2(LToG(chipID,0,0).X()-LToG(chipID,256,0).X(),
               LToG(chipID,0,0).Y()-LToG(chipID,256,0).Y(),
               LToG(chipID,0,0).Z()-LToG(chipID,256,0).Z());
   TVector3 v2v1 = v2.Cross(v1);
   double m2m1 = TMath::Sqrt((v2v1.X()*v2v1.X())+(v2v1.Y()*v2v1.Y())+(v2v1.Z()*v2v1.Z()));
   TVector3 v3(v2v1.X()/m2m1,v2v1.Y()/m2m1,v2v1.Z()/m2m1);
   return v3;

}

////////////////////////////////////////////////////////////////////////////////
/// Chip addressing

#ifdef YGEOM_USE_O2
 #define YGEOM_ADDR(call, field) return geom->call(index);
#else
 #define YGEOM_ADDR(call, field) return fChip[index].field;
#endif

int YDetectorGeometry::GetLayer(int index) const {
   if(index<0) return -9999;
   YGEOM_ADDR(getLayer, layer)
}
int YDetectorGeometry::GetHalfBarrel(int index) const {
   if(index<0) return -9999;
   YGEOM_ADDR(getHalfBarrel, halfBarrel)
}
int YDetectorGeometry::GetStave(int index) const {
   if(index<0) return -9999;
   YGEOM_ADDR(getStave, stave)
}
int YDetectorGeometry::GetHalfStave(int index) const {
   if(index<0) return -9999;
   YGEOM_ADDR(getHalfStave, halfStave)
}
int YDetectorGeometry::GetModule(int index) const {
   if(index<0) return -9999;
   YGEOM_ADDR(getModule, module)
}
int YDetectorGeometry::GetChipIdInLayer(int index) const {
   if(index<0) return -9999;
   YGEOM_ADDR(getChipIdInLayer, chipInLayer)
}
int YDetectorGeometry::GetChipIdInStave(int index) const {
   if(index<0) return -9999;
   YGEOM_ADDR(getChipIdInStave, chipInStave)
}
int YDetectorGeometry::GetChipIdInHalfStave(int index) const {
   if(index<0) return -9999;
   YGEOM_ADDR(getChipIdInHalfStave, chipInHalfStave)
}
int YDetectorGeometry::GetChipIdInModule(int index) const {
   if(index<0) return -9999;
   YGEOM_ADDR(getChipIdInModule, chipInModule)
}

#undef YGEOM_ADDR
