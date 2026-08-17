// Rebuild the o2::detectors::AlignParam class from the StreamerInfo that
// ITSAlignment.root carries, so the alignment file can be read without O2.
//
//   root -l -b -q 'tools/make_alignlib.C("ITSAlignment.root","tools/AlignLib")'
//
// Run once. It writes a compiled shared library that
// tools/export_geometry_cache.C includes directly — going through the
// interpreter field by field instead is ~158k ProcessLine calls and does not
// finish in reasonable time.

#include <TFile.h>
#include <TSystem.h>
#include <cstdio>

void make_alignlib(const char* alignFile = "ITSAlignment.root",
                   const char* outDir    = "tools/AlignLib")
{
   TFile* f = TFile::Open(alignFile);
   if (!f || f->IsZombie()) { ::Error("make_alignlib", "cannot open %s", alignFile); return; }
   printf("[alignlib] source   : %s\n", alignFile);
   printf("[alignlib] target   : %s\n", outDir);
   f->MakeProject(outDir, "*", "recreate++");
   delete f;
   printf("[alignlib] done. include %s/%sProjectHeaders.h and load %s/%s.so\n",
          outDir, gSystem->BaseName(outDir), outDir, gSystem->BaseName(outDir));
}
