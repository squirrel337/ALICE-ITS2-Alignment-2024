// ==========================================================================
//  compare_geometry_probe.C -- diff two composed-transform probes
// ==========================================================================
//  Pairs with tools/dump_geometry_probe.C. ROOT only, so it runs on either machine:
//
//      root -l -b -q 'tools/compare_geometry_probe.C("geometry/probe_o2.root","geometry/probe_cache.root")'
//
//  Reports the worst difference in the global point from LToG, in the sensor
//  coordinates from GToS, and in the normal vector -- separately, because they fail
//  for different reasons. A difference only in GToS with LToG clean points at the
//  inverse convention; a difference in the normal with both clean points at the
//  rotation.
//
//  Both backends round to float at several points, so a residual of order the float
//  ULP is expected: about 5e-6 cm at r = 40 cm. Anything at 1e-4 cm is a bug.
// ==========================================================================

#include <TFile.h>
#include <TTree.h>
#include <TNamed.h>
#include <cstdio>
#include <cmath>
#include <algorithm>

void compare_geometry_probe(const char* aFile = "geometry/probe_o2.root",
                            const char* bFile = "geometry/probe_cache.root")
{
   TFile* fa = TFile::Open(aFile);
   TFile* fb = TFile::Open(bFile);
   if (!fa || fa->IsZombie() || !fb || fb->IsZombie()) { ::Error("cmp", "cannot open inputs"); return; }
   for (int i = 0; i < 2; ++i) {
      TNamed* p = (TNamed*)(i ? fb : fa)->Get("provenance");
      printf("[%c] %s\n", i ? 'b' : 'a', p ? p->GetTitle() : "(none)");
   }

   TTree* ta = (TTree*)fa->Get("probe");
   TTree* tb = (TTree*)fb->Get("probe");
   if (!ta || !tb || ta->GetEntries() != tb->GetEntries()) {
      ::Error("cmp", "probe trees missing or of different length"); return;
   }

   Int_t ca, cb, ia, ib;
   Double_t ga[3], sa[3], na[3], gb[3], sb[3], nb[3];
   const char* gn[3] = {"gx","gy","gz"};
   const char* sn[3] = {"s1","s2","s3"};
   const char* nn[3] = {"nx","ny","nz"};
   ta->SetBranchAddress("chipID", &ca); tb->SetBranchAddress("chipID", &cb);
   ta->SetBranchAddress("ip", &ia);     tb->SetBranchAddress("ip", &ib);
   for (int k = 0; k < 3; ++k) {
      ta->SetBranchAddress(gn[k], &ga[k]); tb->SetBranchAddress(gn[k], &gb[k]);
      ta->SetBranchAddress(sn[k], &sa[k]); tb->SetBranchAddress(sn[k], &sb[k]);
      ta->SetBranchAddress(nn[k], &na[k]); tb->SetBranchAddress(nn[k], &nb[k]);
   }

   double mG = 0, mS = 0, mN = 0;
   Long64_t worstG = -1;
   const Long64_t N = ta->GetEntries();
   for (Long64_t i = 0; i < N; ++i) {
      ta->GetEntry(i); tb->GetEntry(i);
      if (ca != cb || ia != ib) { ::Error("cmp", "probes are not aligned at entry %lld", i); return; }
      double dg = 0, ds = 0, dn = 0;
      for (int k = 0; k < 3; ++k) {
         dg += (ga[k]-gb[k])*(ga[k]-gb[k]);
         ds += (sa[k]-sb[k])*(sa[k]-sb[k]);
         dn += (na[k]-nb[k])*(na[k]-nb[k]);
      }
      dg = std::sqrt(dg); ds = std::sqrt(ds); dn = std::sqrt(dn);
      if (dg > mG) { mG = dg; worstG = i; }
      mS = std::max(mS, ds); mN = std::max(mN, dn);
   }

   printf("\n points compared      : %lld  (%lld chips x 5 pixels)\n", N, N/5);
   printf(" worst |dLToG|        : %.4e cm   (%.4f um)\n", mG, mG*1e4);
   printf(" worst |dGToS|        : %.4e cm   (%.4f um)\n", mS, mS*1e4);
   printf(" worst |dNormalVector|: %.4e\n", mN);
   if (worstG >= 0) {
      ta->GetEntry(worstG); tb->GetEntry(worstG);
      printf(" worst LToG at chip %d pixel %d:\n   a = (%.9f, %.9f, %.9f)\n   b = (%.9f, %.9f, %.9f)\n",
             ca, ia, ga[0], ga[1], ga[2], gb[0], gb[1], gb[2]);
   }
   const double ulp = 5e-6;   // float ULP at r ~ 40 cm
   printf("\n VERDICT: %s  (float ULP at r=40cm is ~%.0e cm)\n",
          (mG <= 10*ulp && mS <= 10*ulp && mN <= 1e-6) ? "consistent with float rounding"
                                                       : "DIFFER beyond rounding", ulp);
}
