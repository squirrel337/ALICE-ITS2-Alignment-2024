// ==========================================================================
//  dump_geometry_probe.C -- the composed transforms, from whichever backend
// ==========================================================================
//  The cache-vs-O2 matrix comparison checks getMatrixL2G. It does NOT check the
//  functions the module actually calls: LToG, GToS, SToG, NormalVector. Those
//  compose the matrix with the segmentation and with sign conventions, and a
//  transpose, an active/passive mix-up or a float-narrowing difference lives there,
//  not in the matrix.
//
//  This dumps them over a pixel grid that INCLUDES THE CORNERS. A rotation error
//  vanishes near the rotation centre and a pitch error vanishes near the origin, so
//  a centre-only probe passes with the convention inverted.
//
//  Run it once under each backend and diff the two files:
//
//      # O2 build
//      eval `alienv load -w $O2_DIR/sw O2/latest`
//      root -l -b -q 'tools/dump_geometry_probe.C("geometry/probe_o2.root", 1)'
//
//      # cache build
//      root -l -b -q 'tools/dump_geometry_probe.C("geometry/probe_cache.root", 0)'
//
//      root -l -b -q 'tools/compare_geometry_probe.C("geometry/probe_o2.root","geometry/probe_cache.root")'
//
//  The second argument only labels the provenance; which backend is actually used is
//  decided by YGEOM_USE_O2 when YDetectorGeometry.cxx is compiled.
// ==========================================================================

// The implementation is included, not .L'd, so yGEOM exists at parse time. Which
// backend this becomes is decided by YGEOM_USE_O2 in the header, exactly as it is for
// the training job -- so this probes the same build the job would run.
#include "../Ymlp/src/YDetectorGeometry.cxx"

#include <TFile.h>
#include <TTree.h>
#include <TNamed.h>
#include <TVector3.h>
#include <TSystem.h>
#include <TDatime.h>
#include <cstdio>

void dump_geometry_probe(const char* outFile = "geometry/probe.root", int isO2 = -1)
{
   constexpr int kNChips = 24120;
   // corners first, then the centre: the corners are what give the test its power
   const int kRow[5] = {0,   0, 511, 511, 255};
   const int kCol[5] = {0, 1023,  0, 1023, 511};

   gSystem->mkdir(gSystem->DirName(outFile), kTRUE);
   TFile fo(outFile, "recreate");
   TTree t("probe", "composed transforms per chip");

   Int_t chipID, ip;
   Double_t gx, gy, gz;          // LToG
   Double_t s1, s2, s3;          // GToS of that global point
   Double_t nx, ny, nz;          // NormalVector
   t.Branch("chipID", &chipID, "chipID/I");
   t.Branch("ip", &ip, "ip/I");
   t.Branch("gx", &gx, "gx/D"); t.Branch("gy", &gy, "gy/D"); t.Branch("gz", &gz, "gz/D");
   t.Branch("s1", &s1, "s1/D"); t.Branch("s2", &s2, "s2/D"); t.Branch("s3", &s3, "s3/D");
   t.Branch("nx", &nx, "nx/D"); t.Branch("ny", &ny, "ny/D"); t.Branch("nz", &nz, "nz/D");

   for (chipID = 0; chipID < kNChips; ++chipID) {
      TVector3 n = yGEOM->NormalVector(chipID);
      nx = n.X(); ny = n.Y(); nz = n.Z();
      for (ip = 0; ip < 5; ++ip) {
         TVector3 g = yGEOM->LToG(chipID, kRow[ip], kCol[ip]);
         gx = g.X(); gy = g.Y(); gz = g.Z();
         TVector3 s = yGEOM->GToS(chipID, gx, gy, gz);
         s1 = s.X(); s2 = s.Y(); s3 = s.Z();
         t.Fill();
      }
   }

   TNamed prov("provenance",
               Form("backend=%s; chips=%d; 5 pixels incl. corners; %s",
                    isO2 > 0 ? "O2" : (isO2 == 0 ? "cache" : "unlabelled"),
                    kNChips, TDatime().AsSQLString()));
   prov.Write();
   t.Write();
   fo.Close();
   printf("[probe] wrote %s : %d chips x 5 pixels\n", outFile, kNChips);
}
