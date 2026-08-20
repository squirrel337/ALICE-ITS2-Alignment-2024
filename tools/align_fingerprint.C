// Print the content fingerprint of an ITSAlignment.root, so a cache can be checked
// against the alignment a job is actually going to run with. See AlignFingerprint.h.
//
//     root -l -b -q 'tools/align_fingerprint.C("ITSAlignment.root")'
//         -> ALIGN_FP n=26307,fp=....
#include "AlignFingerprint.h"
#include <TSystem.h>

void align_fingerprint(const char* alignFile = "ITSAlignment.root")
{
   TClass* have = TClass::GetClass("o2::detectors::AlignParam");
   if (!have || have->GetState() == TClass::kEmulated) gSystem->Load("tools/AlignLib/AlignLib.so");
   printf("ALIGN_FP %s\n", AlignFingerprintString(alignFile).Data());
}
