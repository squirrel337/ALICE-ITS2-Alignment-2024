// ==========================================================================
//  tree_entries.C -- how many entries a tree holds, for the run console
// ==========================================================================
//  This is what lets config/runconf.sh check an input file without Python.
//  ROOT is present on the target machine by definition, so the console needs
//  nothing else to read a .root file.
//
//     root -l -b -q 'tools/tree_entries.C("in.root","DataInput")'
//        -> RC_ENTRIES 131035
//     root -l -b -q 'tools/tree_entries.C("in.root","")'
//        -> RC_TREES DataInput OtherTree
//
//  Prints nothing on either line if the file or the tree cannot be read, so
//  the caller distinguishes "unreadable" from "zero entries" by presence.
// ==========================================================================

#include <TFile.h>
#include <TTree.h>
#include <TKey.h>
#include <TList.h>
#include <TString.h>
#include <TClass.h>
#include <iostream>

void tree_entries(const char *path, const char *tree = "")
{
   TFile *f = TFile::Open(path, "READ");
   if (!f || f->IsZombie()) { std::cout << "RC_ERROR cannot open " << path << std::endl; return; }

   if (tree && tree[0]) {
      TTree *t = 0;
      f->GetObject(tree, t);
      if (t) std::cout << "RC_ENTRIES " << t->GetEntries() << std::endl;
      else   std::cout << "RC_ERROR no tree '" << tree << "'" << std::endl;
   } else {
      TString names;
      TIter next(f->GetListOfKeys());
      TKey *k;
      while ((k = (TKey *)next())) {
         TClass *c = TClass::GetClass(k->GetClassName());
         if (c && c->InheritsFrom(TTree::Class())) {
            if (names.Length()) names += " ";
            names += k->GetName();
         }
      }
      std::cout << "RC_TREES " << names << std::endl;
   }
   f->Close();
}
