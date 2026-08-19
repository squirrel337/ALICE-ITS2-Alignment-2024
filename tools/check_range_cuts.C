// Replay YAlignment::PrepareData's coordinate step on real hits and report which of its
// three range cuts fire.
//
//   python3 tools/extract_hits.py 10 > hits.txt
//   root -l -b -q 'tools/check_range_cuts.C("hits.txt")'
//
// PrepareData turns each cluster into a sensor-frame point by
// GToS(chipID, LToG(chipID, row + 0.5, col + 0.5)) and then applies three cuts
// (YAlignment.cxx:630-647): s1 and s2 against the per-sensor box, and |s3| <= 1e-4 cm.
// The box itself is built at YAlignment.cxx:413-421 from LToG(iID, 0, 0) and
// LToG(iID, 511, 1023) -- bare pixel indices, while the hits carry +0.5 -- so this macro
// prints every out-of-range hit with its row/col, which is what makes that offset visible.
#include "../Ymlp/inc/DetectorConstant.h"
#include "../Ymlp/src/YDetectorGeometry.cxx"
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>

void check_range_cuts(const char* hits = "hits.txt")
{
   std::vector<double> imin0(nSensors), imax0(nSensors), imin1(nSensors), imax1(nSensors);
   for (int iID = 0; iID < nSensors; ++iID) {
      TVector3 a = yGEOM->LToG(iID, 0, 0);
      TVector3 b = yGEOM->LToG(iID, 511, 1023);
      double ip1 = yGEOM->GToS(iID, a(0), a(1), a(2))(0);
      double fp1 = yGEOM->GToS(iID, b(0), b(1), b(2))(0);
      double ip2 = yGEOM->GToS(iID, a(0), a(1), a(2))(1);
      double fp2 = yGEOM->GToS(iID, b(0), b(1), b(2))(1);
      imax0[iID] = std::max(ip1, fp1); imin0[iID] = std::min(ip1, fp1);
      imax1[iID] = std::max(ip2, fp2); imin1[iID] = std::min(ip2, fp2);
   }
   printf("[boundaries] built for %d sensors\n\n", nSensors);

   std::ifstream in(hits);
   if (!in) { ::Error("check_range_cuts", "cannot open %s", hits); return; }
   int ev, it, lay, cid; double row, col, gs1, gs2, gs3;
   long n = 0, cut_s1 = 0, cut_s2 = 0, cut_s3 = 0, cut_1000 = 0;
   double s3max = 0, s3sum = 0;
   std::vector<double> s3v;
   long perlayer[nLAYER] = {0}, s3fail[nLAYER] = {0};
   double s3maxL[nLAYER] = {0};
   while (in >> ev >> it >> lay >> cid >> row >> col >> gs1 >> gs2 >> gs3) {
      ++n; ++perlayer[lay];
      if (std::abs(gs1) > 1000 || std::abs(gs2) > 1000 || std::abs(gs3) > 1000) { ++cut_1000; continue; }
      TVector3 G = yGEOM->LToG(cid, row + 0.5, col + 0.5);
      TVector3 S = yGEOM->GToS(cid, G(0), G(1), G(2));
      double s3 = S(2);
      s3v.push_back(std::abs(s3)); s3sum += std::abs(s3);
      s3max = std::max(s3max, std::abs(s3));
      s3maxL[lay] = std::max(s3maxL[lay], std::abs(s3));
      bool f1 = (S(0) < imin0[cid] || S(0) > imax0[cid]);
      bool f2 = (S(1) < imin1[cid] || S(1) > imax1[cid]);
      if (f1) ++cut_s1;
      if (f2) ++cut_s2;
      if (f1 || f2)
         printf("  OUT  lay %d chip %5d  row %7.1f col %7.1f   s1 %+.6f [%+.6f,%+.6f]  s2 %+.6f [%+.6f,%+.6f]\n",
                lay, cid, row, col, S(0), imin0[cid], imax0[cid], S(1), imin1[cid], imax1[cid]);
      if (s3 < -1.0e-4 || s3 > +1.0e-4) { ++cut_s3; ++s3fail[lay]; }
   }
   if (s3v.empty()) { ::Error("check_range_cuts", "no hits read from %s", hits); return; }
   std::sort(s3v.begin(), s3v.end());
   printf("\nhits read              : %ld\n", n);
   printf("rejected by |g|>1000   : %ld\n", cut_1000);
   printf("fail s1 range          : %ld  (%.2f%%)\n", cut_s1, 100.0*cut_s1/n);
   printf("fail s2 range          : %ld  (%.2f%%)\n", cut_s2, 100.0*cut_s2/n);
   printf("fail s3 |s3|>1e-4 cm   : %ld  (%.2f%%)\n", cut_s3, 100.0*cut_s3/n);
   printf("\n|s3| mean %.3e  median %.3e  p99 %.3e  max %.3e cm   (cut at 1.0e-04)\n",
          s3sum/s3v.size(), s3v[s3v.size()/2], s3v[(size_t)(0.99*s3v.size())], s3max);
   printf("\n%-6s %8s %8s %14s\n", "layer", "hits", "s3 fail", "max |s3| (cm)");
   for (int l = 0; l < nLAYER; ++l)
      printf("%-6d %8ld %8ld %14.3e\n", l, perlayer[l], s3fail[l], s3maxL[l]);
}
