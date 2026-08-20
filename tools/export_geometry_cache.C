// Export the per-chip geometry that YDetectorGeometry currently obtains from O2,
// so that training can run with ROOT alone.
//
//   root -l -b -q 'tools/make_alignlib.C'            # once
//   root -l -b -q 'tools/export_geometry_cache.C'    # then this
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

#include "AlignLib/AlignLibProjectHeaders.h"

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
#include <TSystem.h>
#include <TString.h>
#include <TObjArray.h>
#include <TObjString.h>
#include <TROOT.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

namespace {

constexpr int kNChips = 24120;
constexpr int kNLayer = 7;
const int kChipBoundary[kNLayer + 1] = {0, 108, 252, 432, 3120, 6480, 14712, 24120};
const int kNStaves[kNLayer]     = {12, 16, 20, 24, 30, 42, 48};
const int kNSubStave[kNLayer]   = {1, 1, 1, 2, 2, 2, 2};
const int kHicPerStave[kNLayer] = {1, 1, 1, 8, 8, 14, 14};
const int kChipsPerHic[kNLayer] = {9, 9, 9, 14, 14, 14, 14};

// ---------------------------------------------------------------------------
// Alignment delta convention. THE ONE THING HERE THAT IS NOT READ FROM A FILE.
//
// AlignParam stores a global delta: (mX,mY,mZ) in cm and (mPsi,mTheta,mPhi) in
// radians, described by the file's own StreamerInfo as "pitch" about the final X
// axis, "roll" about Y after the first rotation, and "yaw" about Z — an extrinsic
// Z-Y-X composition, Rz(phi)Ry(theta)Rx(psi).
//
// If validation against O2 disagrees, change this function and nothing else.
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
   const double cps = std::cos(psi),   sps = std::sin(psi);
   const double cth = std::cos(theta), sth = std::sin(theta);
   const double cph = std::cos(phi),   sph = std::sin(phi);

   R[0] =  cph * cth;
   R[1] =  cph * sth * sps - sph * cps;
   R[2] =  cph * sth * cps + sph * sps;
   R[3] =  sph * cth;
   R[4] =  sph * sth * sps + cph * cps;
   R[5] =  sph * sth * cps - cph * sps;
   R[6] = -sth;
   R[7] =  cth * sps;
   R[8] =  cth * cps;
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

} // namespace

void export_geometry_cache(const char* geomFile  = "o2sim_geometry.root",
                           const char* alignFile = "ITSAlignment.root",
                           const char* outFile   = "geometry/its2_geom.root")
{
   gSystem->Load("tools/AlignLib/AlignLib.so");

   // ---- 1. geometry -------------------------------------------------------
   TFile* fg = TFile::Open(geomFile);
   if (!fg || fg->IsZombie()) { ::Error("export", "cannot open %s", geomFile); return; }
   TGeoManager* g = (TGeoManager*)fg->Get("ccdb_object");
   if (!g) { ::Error("export", "no TGeoManager under key ccdb_object in %s", geomFile); return; }
   printf("[geom ] %s : %d alignable entries\n", geomFile, g->GetNAlignable());

   // ---- 2. alignment ------------------------------------------------------
   TFile* fa = TFile::Open(alignFile);
   TKey* key = fa->GetKey("ccdb_object");
   auto* ap = (std::vector<o2::detectors::AlignParam>*)
              key->ReadObjectAny(TClass::GetClass("vector<o2::detectors::AlignParam>"));
   if (!ap) { ::Error("export", "cannot read AlignParam vector from %s", alignFile); return; }
   printf("[align] %s : %zu AlignParam entries\n", alignFile, ap->size());

   // ---- 3. apply the deltas, parents before children ----------------------
   // Ordering by symname depth puts a stave delta before its chips'.
   std::vector<std::pair<int, size_t>> order;
   order.reserve(ap->size());
   for (size_t i = 0; i < ap->size(); ++i)
      order.emplace_back(TString((*ap)[i].mSymName.c_str()).CountChar('/'), i);
   std::sort(order.begin(), order.end());

   int nApplied = 0, nMissing = 0;
   for (auto& pr : order) {
      const auto& a = (*ap)[pr.second];
      TGeoPNEntry* e = g->GetAlignableEntry(a.mSymName.c_str());
      if (!e) { ++nMissing; continue; }
      TGeoPhysicalNode* pn = g->MakeAlignablePN(e);
      if (!pn) { ++nMissing; continue; }

      // newGlobal = delta * origGlobal ; newLocal = motherGlobal^-1 * newGlobal
      double dR[9], dT[3];
      DeltaRT(a.mX, a.mY, a.mZ, a.mPsi, a.mTheta, a.mPhi, dR, dT);

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
      const TGeoHMatrix* m = pn->GetMatrix();
      const Double_t* r  = m->GetRotationMatrix();
      const Double_t* tr = m->GetTranslation();
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

   TNamed prov("provenance",
               Form("geometry=%s;alignment=%s;producer=ROOT-only;root=%s;"
                    "delta=Rz(phi)Ry(theta)Rx(psi) global, newGlobal=delta*origGlobal;date=%s",
                    geomFile, alignFile, gROOT->GetVersion(), TDatime().AsSQLString()));
   prov.Write();
   t->Write();
   fo->Close();
   printf("[cache] wrote %s\n", outFile);
}
