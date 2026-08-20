// Export the per-chip geometry that YDetectorGeometry currently obtains from O2,
// so that training can run with ROOT alone.
//
//   root -l -b -q 'tools/export_geometry_cache.C'         # with O2 loaded
//
//   root -l -b -q 'tools/make_alignlib.C'                 # without O2: dictionary first
//   root -l -b -q 'tools/export_geometry_cache.C'
//
// Runs on plain ROOT. Both inputs are self-describing: o2sim_geometry.root holds a
// TGeoManager with the ITS alignable entries already registered (UID == ChipID for
// 0..24119), and ITSAlignment.root holds std::vector<o2::detectors::AlignParam> whose
// class is rebuilt by tools/make_alignlib.C from the StreamerInfo in the file itself.
//
// WHAT STILL NEEDS O2 TO CONFIRM: the alignment-delta convention in DeltaMatrix()
// below. Everything else is either read verbatim from the files or is plain TGeo
// navigation. See tools/validate_geometry_cache.C.
//
// REQUIRES ROOT'S GEOMETRY COMPONENT. This is the only macro in the tree that uses
// TGeo; the module itself never does, because the cache exists precisely so that
// training reads transforms from a file instead of navigating a TGeoManager. On an
// installation that packages geometry separately -- EPEL splits ROOT into root-core,
// root-tree, root-geom and so on -- the core headers resolve and the TGeo ones do not,
// which shows up as
//
//     fatal error: 'TGeoManager.h' file not found
//
// followed by a wall of "unknown type name 'TGeoManager'". Check with
//
//     root -l -b -q -e 'gSystem->Load("libGeom"); printf("%s\n", gSystem->Which(TROOT::GetIncludeDir(), "TGeoManager.h"))'
//
// A null answer means the headers are absent and the geometry package has to be
// installed; nothing in this macro can work around that.

// NOTE: AlignLib's headers are deliberately NOT included. o2::detectors::AlignParam
// keeps its fields private, so naming them compiles only against the MakeProject copy
// and fails the moment O2 is loaded and the real class wins:
//
//     error: 'mSymName' is a private member of 'o2::detectors::AlignParam'
//
// The fields are therefore read through ROOT reflection, which reports offsets for
// private members just as it does for public ones. That path works against either
// dictionary, so the exporter behaves the same with O2 loaded and without it.

#include "AlignFingerprint.h"

// Loaded at parse time, before the TGeo includes below are resolved. On a build with
// runtime C++ modules this is what makes those headers findable at all, so it is
// load-bearing rather than tidiness.
R__LOAD_LIBRARY(libGeom)

#include <TFile.h>
#include <TKey.h>
#include <TTree.h>
#include <TNamed.h>
#include <TDatime.h>
#include <TGeoManager.h>
#include <TGeoPhysicalNode.h>
#include <TGeoMatrix.h>
#include <TGeoVolume.h>
#include <TGeoNode.h>
#include <TSystem.h>
#include <TString.h>
#include <TObjArray.h>
#include <TObjString.h>
#include <TROOT.h>
#include <TClass.h>
#include <TDataMember.h>
#include <TVirtualCollectionProxy.h>
#include <string>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

namespace {

constexpr int kNChips = 24120;

// o2::itsmft::SegmentationAlpide, in cm. Their difference is the bias
// GeometryTGeo::extractMatrixSensor corrects for.
constexpr double kSensorLayerThickness    = 30.e-4;
constexpr double kSensorLayerThicknessEff = 28.e-4;
constexpr int kNLayer = 7;
const int kChipBoundary[kNLayer + 1] = {0, 108, 252, 432, 3120, 6480, 14712, 24120};
const int kNStaves[kNLayer]     = {12, 16, 20, 24, 30, 42, 48};
const int kNSubStave[kNLayer]   = {1, 1, 1, 2, 2, 2, 2};
const int kHicPerStave[kNLayer] = {1, 1, 1, 8, 8, 14, 14};
const int kChipsPerHic[kNLayer] = {9, 9, 9, 14, 14, 14, 14};

// ---------------------------------------------------------------------------
// Alignment delta convention.
//
// AlignParam stores a global delta: (mX,mY,mZ) in cm and (mPsi,mTheta,mPhi) in
// radians. The rotation below is AlignParam::anglesToMatrix copied verbatim from
// AliceO2, DataFormats/Detectors/Common/src/AlignParam.cxx -- "Euler angles in
// 'x y z' notation". Checked identical at tag nightly-20230501, the O2 version this
// module is run against, and on dev.
//
// This was previously reconstructed from the StreamerInfo comments as
// Rz(phi)Ry(theta)Rx(psi), which agreed with O2 in only two of the nine elements and
// is why the cache backend disagreed with the O2 backend. Do not re-derive it: if it
// ever needs changing, copy it from O2 again.
// ---------------------------------------------------------------------------
// All composition below is done on plain arrays. TGeoHMatrix is registered and
// managed by the geometry manager, so returning one by value or copying it into a
// local corrupts ROOT's matrix bookkeeping — it shows up as
// "Registered matrix was removed" and then a free() of a stack address from inside
// TGeoNodeCache. Plain doubles avoid the whole question; a TGeoHMatrix is built only
// at the Align() boundary.
void DeltaRT(double x, double y, double z, double psi, double theta, double phi,
             double R[9], double T[3])
{
   const double sinpsi = std::sin(psi),   cospsi = std::cos(psi);
   const double sinthe = std::sin(theta), costhe = std::cos(theta);
   const double sinphi = std::sin(phi),   cosphi = std::cos(phi);

   R[0] = costhe * cosphi;
   R[1] = -costhe * sinphi;
   R[2] = sinthe;
   R[3] = sinpsi * sinthe * cosphi + cospsi * sinphi;
   R[4] = -sinpsi * sinthe * sinphi + cospsi * cosphi;
   R[5] = -costhe * sinpsi;
   R[6] = -cospsi * sinthe * cosphi + sinpsi * sinphi;
   R[7] = cospsi * sinthe * sinphi + sinpsi * cosphi;
   R[8] = costhe * cospsi;
   T[0] = x; T[1] = y; T[2] = z;
}

// C = A * B for rigid transforms (rotation row-major, translation applied after)
void ComposeRT(const double Ar[9], const double At[3],
               const double Br[9], const double Bt[3],
               double Cr[9], double Ct[3])
{
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         Cr[3*i + j] = Ar[3*i]*Br[j] + Ar[3*i + 1]*Br[3 + j] + Ar[3*i + 2]*Br[6 + j];
   for (int i = 0; i < 3; ++i)
      Ct[i] = Ar[3*i]*Bt[0] + Ar[3*i + 1]*Bt[1] + Ar[3*i + 2]*Bt[2] + At[i];
}

// Inverse of a rigid transform: R^T | -R^T t
void InvertRT(const double R[9], const double T[3], double Ri[9], double Ti[3])
{
   Ri[0] = R[0]; Ri[1] = R[3]; Ri[2] = R[6];
   Ri[3] = R[1]; Ri[4] = R[4]; Ri[5] = R[7];
   Ri[6] = R[2]; Ri[7] = R[5]; Ri[8] = R[8];
   for (int i = 0; i < 3; ++i)
      Ti[i] = -(Ri[3*i]*T[0] + Ri[3*i + 1]*T[1] + Ri[3*i + 2]*T[2]);
}

// "ITS/ITSULayer3/ITSUHalfBarrel1/ITSUStave11/ITSUHalfStave1/ITSUModule3/ITSUChip13"
// Inner-barrel entries carry no HalfStave/Module level; those report 0, matching how
// O2 numbers a single-substave stave.
bool ParseSymName(const TString& s, int& layer, int& halfBarrel, int& stave,
                  int& halfStave, int& module, int& chipInModule)
{
   layer = halfBarrel = stave = halfStave = module = chipInModule = -1;
   TObjArray* parts = s.Tokenize("/");
   for (int i = 0; i < parts->GetEntries(); ++i) {
      TString p = ((TObjString*)parts->At(i))->GetString();
      // longer prefixes first: HalfStave before Stave, HalfBarrel before Barrel
      if      (p.BeginsWith("ITSULayer"))      layer        = TString(p(9,  p.Length())).Atoi();
      else if (p.BeginsWith("ITSUHalfBarrel")) halfBarrel   = TString(p(14, p.Length())).Atoi();
      else if (p.BeginsWith("ITSUHalfStave"))  halfStave    = TString(p(13, p.Length())).Atoi();
      else if (p.BeginsWith("ITSUStave"))      stave        = TString(p(9,  p.Length())).Atoi();
      else if (p.BeginsWith("ITSUModule"))     module       = TString(p(10, p.Length())).Atoi();
      else if (p.BeginsWith("ITSUChip"))       chipInModule = TString(p(8,  p.Length())).Atoi();
   }
   delete parts;
   if (halfStave < 0) halfStave = 0;
   if (module    < 0) module    = 0;
   return layer >= 0 && halfBarrel >= 0 && stave >= 0 && chipInModule >= 0;
}

// --- AlignParam access by reflection ---------------------------------------------
// Offsets are resolved once from whichever dictionary provided the class.
struct AlignAccess {
   TVirtualCollectionProxy* px = nullptr;
   Long_t oSym = -1, oX = -1, oY = -1, oZ = -1, oPsi = -1, oTheta = -1, oPhi = -1;

   bool init(TClass* vcl)
   {
      px = vcl ? vcl->GetCollectionProxy() : nullptr;
      if (!px) { ::Error("export", "no collection proxy for %s", vcl ? vcl->GetName() : "?"); return false; }
      TClass* ecl = px->GetValueClass();
      if (!ecl) { ::Error("export", "collection has no value class"); return false; }
      if (ecl->GetState() == TClass::kEmulated) {
         ::Error("export", "%s has no dictionary. Load O2, or build one with "
                 "tools/make_alignlib.C", ecl->GetName());
         return false;
      }
      struct { const char* n; Long_t* o; } want[] = {
         {"mSymName", &oSym}, {"mX", &oX}, {"mY", &oY}, {"mZ", &oZ},
         {"mPsi", &oPsi}, {"mTheta", &oTheta}, {"mPhi", &oPhi}, {nullptr, nullptr}
      };
      for (int i = 0; want[i].n; ++i) {
         TDataMember* dm = ecl->GetDataMember(want[i].n);
         if (!dm) { ::Error("export", "%s has no member %s", ecl->GetName(), want[i].n); return false; }
         *want[i].o = (Long_t)dm->GetOffset();
      }
      return true;
   }
   const std::string& sym(char* e) const { return *(std::string*)(e + oSym); }
   double x(char* e) const     { return *(double*)(e + oX); }
   double y(char* e) const     { return *(double*)(e + oY); }
   double z(char* e) const     { return *(double*)(e + oZ); }
   double psi(char* e) const   { return *(double*)(e + oPsi); }
   double theta(char* e) const { return *(double*)(e + oTheta); }
   double phi(char* e) const   { return *(double*)(e + oPhi); }
};

} // namespace

void export_geometry_cache(const char* geomFile  = "o2sim_geometry.root",
                           const char* alignFile = "ITSAlignment.root",
                           const char* outFile   = "geometry/its2_geom.root")
{
   // A dictionary for AlignParam has to come from somewhere: O2 brings its own, and
   // tools/make_alignlib.C builds one from the file's StreamerInfo when O2 is absent.
   // Loading AlignLib on top of O2's would be a duplicate definition, so only load it
   // when nothing else has supplied the class.
   {
      TClass* have = TClass::GetClass("o2::detectors::AlignParam");
      if (!have || have->GetState() == TClass::kEmulated) {
         if (gSystem->Load("tools/AlignLib/AlignLib.so") < 0)
            ::Warning("export", "could not load tools/AlignLib/AlignLib.so; "
                      "run tools/make_alignlib.C first, or load O2");
      } else {
         printf("[align] using the AlignParam dictionary already loaded\n");
      }
   }

   // ---- 1. geometry -------------------------------------------------------
   TFile* fg = TFile::Open(geomFile);
   if (!fg || fg->IsZombie()) { ::Error("export", "cannot open %s", geomFile); return; }
   TGeoManager* g = (TGeoManager*)fg->Get("ccdb_object");
   if (!g) { ::Error("export", "no TGeoManager under key ccdb_object in %s", geomFile); return; }
   printf("[geom ] %s : %d alignable entries\n", geomFile, g->GetNAlignable());

   // ---- 2. alignment ------------------------------------------------------
   TFile* fa = TFile::Open(alignFile);
   if (!fa || fa->IsZombie()) { ::Error("export", "cannot open %s", alignFile); return; }
   TKey* key = fa->GetKey("ccdb_object");
   if (!key) { ::Error("export", "no key ccdb_object in %s", alignFile); return; }

   TClass* vcl = TClass::GetClass(key->GetClassName());
   AlignAccess acc;
   if (!acc.init(vcl)) return;
   void* apv = key->ReadObjectAny(vcl);
   if (!apv) { ::Error("export", "cannot read AlignParam vector from %s", alignFile); return; }

   TVirtualCollectionProxy::TPushPop guard(acc.px, apv);
   const UInt_t nAlign = acc.px->Size();
   printf("[align] %s : %u AlignParam entries\n", alignFile, nAlign);

   // ---- 3. apply the deltas, parents before children ----------------------
   // Ordering by symname depth puts a stave delta before its chips'.
   std::vector<std::pair<int, UInt_t>> order;
   order.reserve(nAlign);
   for (UInt_t i = 0; i < nAlign; ++i)
      order.emplace_back(TString(acc.sym((char*)acc.px->At(i)).c_str()).CountChar('/'), i);
   std::sort(order.begin(), order.end());

   int nApplied = 0, nMissing = 0;
   for (auto& pr : order) {
      char* a = (char*)acc.px->At(pr.second);
      TGeoPNEntry* e = g->GetAlignableEntry(acc.sym(a).c_str());
      if (!e) { ++nMissing; continue; }
      TGeoPhysicalNode* pn = g->MakeAlignablePN(e);
      if (!pn) { ++nMissing; continue; }

      // newGlobal = delta * origGlobal ; newLocal = motherGlobal^-1 * newGlobal
      double dR[9], dT[3];
      DeltaRT(acc.x(a), acc.y(a), acc.z(a), acc.psi(a), acc.theta(a), acc.phi(a), dR, dT);

      const Double_t* oR = pn->GetMatrix()->GetRotationMatrix();
      const Double_t* oT = pn->GetMatrix()->GetTranslation();
      double gR[9], gT[3];
      ComposeRT(dR, dT, oR, oT, gR, gT);

      double lR[9], lT[3];
      const int lvl = pn->GetLevel();
      if (lvl > 0) {
         const TGeoHMatrix* mm = pn->GetMatrix(lvl - 1);       // mother global
         double mR[9], mT[3];
         InvertRT(mm->GetRotationMatrix(), mm->GetTranslation(), mR, mT);
         ComposeRT(mR, mT, gR, gT, lR, lT);
      } else {
         for (int k = 0; k < 9; ++k) lR[k] = gR[k];
         for (int k = 0; k < 3; ++k) lT[k] = gT[k];
      }

      // Heap-allocated: TGeoPhysicalNode::Align keeps the matrix it is given.
      TGeoHMatrix* local = new TGeoHMatrix();
      local->SetRotation(lR);
      local->SetTranslation(lT);
      pn->Align(local);
      ++nApplied;
   }
   printf("[align] applied %d, unmatched %d\n", nApplied, nMissing);

   // Computed here, while no output file is open: AlignFingerprint opens the
   // alignment file, and that changes gDirectory.
   const TString alignFP = AlignFingerprintString(alignFile);

   // ---- 4. harvest the per-chip cache -------------------------------------
   TString dir = gSystem->DirName(outFile);
   if (dir != "" && dir != ".") gSystem->mkdir(dir, kTRUE);
   TFile* fo = TFile::Open(outFile, "recreate");
   TTree* t = new TTree("geom", "ITS2 per-chip geometry cache");

   Int_t chipID, layer, halfBarrel, stave, halfStave, module, chipInModule;
   Int_t chipInLayer, chipInStave, chipInHalfStave;
   Double_t R[9], T[3];
   t->Branch("chipID", &chipID, "chipID/I");
   t->Branch("layer", &layer, "layer/I");
   t->Branch("halfBarrel", &halfBarrel, "halfBarrel/I");
   t->Branch("stave", &stave, "stave/I");
   t->Branch("halfStave", &halfStave, "halfStave/I");
   t->Branch("module", &module, "module/I");
   t->Branch("chipInModule", &chipInModule, "chipInModule/I");
   t->Branch("chipInLayer", &chipInLayer, "chipInLayer/I");
   t->Branch("chipInStave", &chipInStave, "chipInStave/I");
   t->Branch("chipInHalfStave", &chipInHalfStave, "chipInHalfStave/I");
   t->Branch("R", R, "R[9]/D");
   t->Branch("T", T, "T[3]/D");

   int nBad = 0;
   for (chipID = 0; chipID < kNChips; ++chipID) {
      TGeoPNEntry* e = g->GetAlignableEntryByUID(chipID);
      if (!e) { ++nBad; continue; }
      if (!ParseSymName(e->GetName(), layer, halfBarrel, stave, halfStave, module, chipInModule)) {
         ++nBad; continue;
      }
      TGeoPhysicalNode* pn = g->MakeAlignablePN(e);

      // The alignable entry stops at the CHIP volume, but getMatrixL2G returns the
      // SENSOR's transform -- GeometryTGeo::extractMatrixSensor navigates one level
      // further, to ITSUSensor<lay>_1, and then shifts by half the difference between
      // the physical and effective sensitive thickness. Both matter: the sensor sits
      // at local y = -5 um inside an IB chip and +20 um inside an OB one, and the
      // shift adds +1 um, which is exactly the -4 um / +21 um this used to be out by.
      TGeoHMatrix mSens = *pn->GetMatrix();          // chip -> global
      TGeoVolume* chipVol = pn->GetVolume();
      TGeoNode* sensNode = chipVol ? chipVol->FindNode(Form("ITSUSensor%d_1", layer)) : nullptr;
      if (!sensNode) { ++nBad; continue; }
      mSens.Multiply(sensNode->GetMatrix());         // -> sensor volume

      // o2::its::GeometryTGeo::extractMatrixSensor: matTmp *= TGeoTranslation(0, 0.5*delta, 0)
      const double delta = kSensorLayerThickness - kSensorLayerThicknessEff;
      TGeoTranslation tra(0., 0.5 * delta, 0.);
      mSens.Multiply(&tra);

      const Double_t* r  = mSens.GetRotationMatrix();
      const Double_t* tr = mSens.GetTranslation();
      for (int k = 0; k < 9; ++k) R[k] = r[k];
      for (int k = 0; k < 3; ++k) T[k] = tr[k];

      // The symname numbers staves within their half-barrel, so layer 0 reads
      // HalfBarrel0/Stave0..5 and HalfBarrel1/Stave0..5. O2's GetStave returns a
      // global index over the whole layer, which is what every call site assumes —
      // GetCost recovers the half-barrel-local index with stave%(NStaves/2).
      // Level 0 caught this as exactly half the expected stave count per layer.
      stave = halfBarrel * (kNStaves[layer] / 2) + stave;

      const int chipsPerHalfStave = kHicPerStave[layer] * kChipsPerHic[layer] / kNSubStave[layer];
      chipInLayer     = chipID - kChipBoundary[layer];
      chipInHalfStave = module * kChipsPerHic[layer] + chipInModule;
      chipInStave     = halfStave * chipsPerHalfStave + chipInHalfStave;
      t->Fill();
   }
   printf("[cache] filled %lld chips, %d skipped\n", t->GetEntries(), nBad);

   // The alignment is baked in here and never re-read by the cache backend, so the
   // cache carries a fingerprint of the file it was built from. Recording the name
   // alone is useless: ITSAlignment.root is the same name in every directory.
   // Computed above, before the output file existed, because opening the alignment
   // file moves gDirectory -- writing after it would land nowhere.
   fo->cd();
   TNamed afp("alignfingerprint", alignFP.Data());
   afp.Write();
   printf("[align] fingerprint %s\n", alignFP.Data());

   TNamed prov("provenance",
               Form("geometry=%s;alignment=%s;producer=ROOT-only;root=%s;"
                    "delta=O2 AlignParam::anglesToMatrix (nightly-20230501), newGlobal=delta*origGlobal;date=%s",
                    geomFile, alignFile, gROOT->GetVersion(), TDatime().AsSQLString()));
   prov.Write();
   t->Write();
   fo->Close();
   printf("[cache] wrote %s\n", outFile);
}
