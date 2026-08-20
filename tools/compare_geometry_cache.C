// ==========================================================================
//  compare_geometry_cache.C -- diff two per-chip geometry files, chip by chip
// ==========================================================================
//  Needs ROOT only, so it runs on either machine:
//
//      root -l -b -q 'tools/compare_geometry_cache.C("geometry/its2_geom_o2.root","geometry/its2_geom.root")'
//
//  Reports, per layer: the worst and mean sensor displacement, the worst rotation
//  element difference, and any addressing mismatch. Addressing is compared exactly --
//  a single wrong integer routes hits to the wrong sensor network and would leave a
//  plausible cost with a meaningless alignment, so it is not given a tolerance.
//
//  The two files must be the SAME geometry and the SAME starting alignment. Comparing
//  caches built from different inputs measures the inputs, not the code.
//
//  What a real match looks like, measured against an O2 dump on 24120 chips: rotation
//  bit-identical, worst displacement 6.3e-7 um. Anything at the micron level is a bug,
//  not rounding.
// ==========================================================================

#include <TFile.h>
#include <TTree.h>
#include <TNamed.h>
#include <cstdio>
#include <cmath>
#include <algorithm>

void compare_geometry_cache(const char* refFile = "geometry/its2_geom_o2.root",
                            const char* newFile = "geometry/its2_geom.root",
                            double tolT_um = 1.0, double tolR = 1e-6)
{
   TFile* fa = TFile::Open(refFile);
   TFile* fb = TFile::Open(newFile);
   if (!fa || fa->IsZombie()) { ::Error("cmp", "cannot open %s", refFile); return; }
   if (!fb || fb->IsZombie()) { ::Error("cmp", "cannot open %s", newFile); return; }

   for (int i = 0; i < 2; ++i) {
      TNamed* p = (TNamed*)(i ? fb : fa)->Get("provenance");
      printf("[%s] %s\n", i ? "new" : "ref", p ? p->GetTitle() : "(no provenance)");
   }

   TTree* ta = (TTree*)fa->Get("geom");
   TTree* tb = (TTree*)fb->Get("geom");
   if (!ta || !tb) { ::Error("cmp", "missing 'geom' tree"); return; }
   if (ta->GetEntries() != tb->GetEntries()) {
      ::Error("cmp", "entry counts differ: %lld vs %lld", ta->GetEntries(), tb->GetEntries());
      return;
   }

   const char* ints[] = {"chipID","layer","halfBarrel","stave","halfStave",
                         "module","chipInModule","chipInLayer","chipInStave","chipInHalfStave"};
   const int nInt = 10;
   Int_t va[nInt], vb[nInt];
   Double_t Ra[9], Ta[3], Rb[9], Tb[3];
   for (int i = 0; i < nInt; ++i) {
      ta->SetBranchAddress(ints[i], &va[i]);
      tb->SetBranchAddress(ints[i], &vb[i]);
   }
   ta->SetBranchAddress("R", Ra); ta->SetBranchAddress("T", Ta);
   tb->SetBranchAddress("R", Rb); tb->SetBranchAddress("T", Tb);

   const int nL = 7;
   double maxT[nL] = {0}, sumT[nL] = {0}, maxR[nL] = {0};
   long   nChip[nL] = {0}, badAddr[nL] = {0};
   Long64_t firstBad = -1;
   int    badField = -1;

   const Long64_t N = ta->GetEntries();
   for (Long64_t i = 0; i < N; ++i) {
      ta->GetEntry(i); tb->GetEntry(i);
      const int l = (va[1] >= 0 && va[1] < nL) ? va[1] : 0;

      for (int k = 0; k < nInt; ++k)
         if (va[k] != vb[k]) {
            ++badAddr[l];
            if (firstBad < 0) { firstBad = i; badField = k; }
            break;
         }

      const double d = std::sqrt(std::pow(Ta[0]-Tb[0],2) + std::pow(Ta[1]-Tb[1],2) +
                                 std::pow(Ta[2]-Tb[2],2));
      double r = 0;
      for (int k = 0; k < 9; ++k) r = std::max(r, std::fabs(Ra[k]-Rb[k]));

      maxT[l] = std::max(maxT[l], d); sumT[l] += d; maxR[l] = std::max(maxR[l], r);
      ++nChip[l];
   }

   printf("\n layer    chips   max|dT| um   mean|dT| um      max|dR|   addressing\n");
   double gT = 0, gR = 0; long gBad = 0;
   for (int l = 0; l < nL; ++l) {
      printf("   L%d   %7ld   %10.3f   %11.3f   %10.3e   %s\n", l, nChip[l],
             maxT[l]*1e4, nChip[l] ? sumT[l]/nChip[l]*1e4 : 0.0, maxR[l],
             badAddr[l] ? Form("%ld MISMATCH", badAddr[l]) : "ok");
      gT = std::max(gT, maxT[l]); gR = std::max(gR, maxR[l]); gBad += badAddr[l];
   }

   printf("\n worst displacement : %.3f um\n worst rotation elem: %.3e\n", gT*1e4, gR);
   if (gBad) {
      ta->GetEntry(firstBad); tb->GetEntry(firstBad);
      printf(" addressing         : %ld chips differ; first at entry %lld, field '%s' (%d vs %d)\n",
             gBad, firstBad, ints[badField], va[badField], vb[badField]);
   } else {
      printf(" addressing         : identical on all %lld chips\n", N);
   }

   const bool ok = (gBad == 0) && (gT*1e4 <= tolT_um) && (gR <= tolR);
   printf("\n VERDICT: %s  (tolerance %.3f um, %.1e on rotation)\n",
          ok ? "MATCH" : "DIFFER", tolT_um, tolR);
}
