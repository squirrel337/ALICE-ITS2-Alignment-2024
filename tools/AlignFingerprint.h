// ==========================================================================
//  AlignFingerprint.h -- a content fingerprint for an ITSAlignment.root
// ==========================================================================
//  The geometry cache BAKES THE ALIGNMENT IN at export time; the cache backend
//  never opens ITSAlignment.root again. The O2 backend reads it on every run. So a
//  cache built from one alignment file and a job running against another use
//  different geometry, silently -- the caches differ by whatever the two alignments
//  differ by, and nothing says so.
//
//  Recording the file's NAME is not enough: ITSAlignment.root is the same name in
//  every working directory. This hashes the values instead, so two files that
//  disagree anywhere produce different fingerprints.
//
//  Read through reflection for the same reason the exporter is: o2::detectors::
//  AlignParam keeps its fields private, so naming them compiles only against the
//  MakeProject copy and breaks the moment O2 is loaded.
// ==========================================================================

#ifndef ALIGN_FINGERPRINT_H
#define ALIGN_FINGERPRINT_H

#include <TFile.h>
#include <TKey.h>
#include <TClass.h>
#include <TDataMember.h>
#include <TVirtualCollectionProxy.h>
#include <TString.h>
#include <string>
#include <cstdio>
#include <cstdint>

// FNV-1a over the raw bytes. Not cryptographic; it only has to notice a difference.
inline void AlignFP_Feed(uint64_t& h, const void* p, size_t n)
{
   const unsigned char* b = (const unsigned char*)p;
   for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
}

// Returns false if the file cannot be read. On success nEntries and hash are set.
inline bool AlignFingerprint(const char* alignFile, Long64_t& nEntries, uint64_t& hash)
{
   nEntries = 0; hash = 1469598103934665603ULL;

   TFile* f = TFile::Open(alignFile, "read");
   if (!f || f->IsZombie()) { ::Error("AlignFingerprint", "cannot open %s", alignFile); return false; }
   TKey* k = f->GetKey("ccdb_object");
   if (!k) { ::Error("AlignFingerprint", "no key ccdb_object in %s", alignFile); delete f; return false; }

   TClass* vcl = TClass::GetClass(k->GetClassName());
   TVirtualCollectionProxy* px = vcl ? vcl->GetCollectionProxy() : nullptr;
   if (!px) { ::Error("AlignFingerprint", "no collection proxy"); delete f; return false; }
   TClass* ecl = px->GetValueClass();
   if (!ecl || ecl->GetState() == TClass::kEmulated) {
      ::Error("AlignFingerprint", "AlignParam has no dictionary. Load O2, or build one "
              "with tools/make_alignlib.C");
      delete f; return false;
   }

   const char* names[] = {"mX","mY","mZ","mPsi","mTheta","mPhi", nullptr};
   Long_t off[6];
   for (int i = 0; names[i]; ++i) {
      TDataMember* dm = ecl->GetDataMember(names[i]);
      if (!dm) { ::Error("AlignFingerprint", "no member %s", names[i]); delete f; return false; }
      off[i] = (Long_t)dm->GetOffset();
   }
   TDataMember* dmSym = ecl->GetDataMember("mSymName");
   if (!dmSym) { ::Error("AlignFingerprint", "no member mSymName"); delete f; return false; }
   const Long_t oSym = (Long_t)dmSym->GetOffset();

   void* vec = k->ReadObjectAny(vcl);
   if (!vec) { ::Error("AlignFingerprint", "cannot read ccdb_object"); delete f; return false; }

   TVirtualCollectionProxy::TPushPop guard(px, vec);
   nEntries = (Long64_t)px->Size();
   for (Long64_t i = 0; i < nEntries; ++i) {
      char* e = (char*)px->At(i);
      const std::string& s = *(std::string*)(e + oSym);
      AlignFP_Feed(hash, s.data(), s.size());
      for (int j = 0; j < 6; ++j) {
         const double v = *(double*)(e + off[j]);
         AlignFP_Feed(hash, &v, sizeof(v));
      }
   }
   f->Close();
   return true;
}

inline TString AlignFingerprintString(const char* alignFile)
{
   Long64_t n = 0; uint64_t h = 0;
   if (!AlignFingerprint(alignFile, n, h)) return TString("unreadable");
   return TString::Format("n=%lld,fp=%016llx", n, (unsigned long long)h);
}

#endif
