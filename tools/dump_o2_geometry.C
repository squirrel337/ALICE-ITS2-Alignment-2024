// ==========================================================================
//  dump_o2_geometry.C -- the reference dump, straight from O2
// ==========================================================================
//  RUN THIS UNDER O2. It writes the same tree the geometry cache uses, but the
//  numbers come from o2::its::GeometryTGeo instead of from
//  tools/export_geometry_cache.C, so the two files can be diffed chip by chip:
//
//      eval `alienv load -w $O2_DIR/sw O2/latest`
//      root -l -b -q tools/dump_o2_geometry.C            # -> geometry/its2_geom_o2.root
//      root -l -b -q 'tools/compare_geometry_cache.C("geometry/its2_geom_o2.root","geometry/its2_geom.root")'
//
//  This is the level-1 and level-2 validation the cache has never had: transform
//  equivalence and addressing equivalence, over all 24120 chips, against the
//  implementation the cache is meant to reproduce.
//
//  It reads the geometry exactly the way YDetectorGeometry's O2 backend does --
//  loadGeometry, applyAlignment from ITSAlignment.root, fillMatrixCache -- so a
//  disagreement is a disagreement in the cache, not in how it was queried.
// ==========================================================================

#include <DetectorsCommonDataFormats/DetID.h>
#include <DetectorsCommonDataFormats/AlignParam.h>
#include <DetectorsBase/GeometryManager.h>
#include <ITSBase/GeometryTGeo.h>

#include <TFile.h>
#include <TTree.h>
#include <TNamed.h>
#include <TSystem.h>
#include <TDatime.h>
#include <vector>
#include <cstdio>

void dump_o2_geometry(const char* alignFile = "ITSAlignment.root",
                      const char* outFile   = "geometry/its2_geom_o2.root")
{
   constexpr int kNChips = 24120;

   o2::base::GeometryManager::loadGeometry("", false, false);

   TFile fa(alignFile);
   std::vector<o2::detectors::AlignParam>* ap = nullptr;
   fa.GetObject("ccdb_object", ap);
   if (!ap) { ::Error("dump", "no ccdb_object in %s", alignFile); return; }
   o2::base::GeometryManager::applyAlignment(*ap);
   printf("[align] %s : %zu AlignParam entries applied\n", alignFile, ap->size());

   auto* geom = o2::its::GeometryTGeo::Instance();
   geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L,
                                                 o2::math_utils::TransformType::T2GRot,
                                                 o2::math_utils::TransformType::L2G));

   gSystem->mkdir(gSystem->DirName(outFile), kTRUE);
   TFile fo(outFile, "recreate");
   TTree t("geom", "ITS2 per-chip geometry, from O2");

   Int_t chipID, layer, halfBarrel, stave, halfStave, module, chipInModule;
   Int_t chipInLayer, chipInStave, chipInHalfStave;
   Double_t R[9], T[3];
   t.Branch("chipID", &chipID, "chipID/I");
   t.Branch("layer", &layer, "layer/I");
   t.Branch("halfBarrel", &halfBarrel, "halfBarrel/I");
   t.Branch("stave", &stave, "stave/I");
   t.Branch("halfStave", &halfStave, "halfStave/I");
   t.Branch("module", &module, "module/I");
   t.Branch("chipInModule", &chipInModule, "chipInModule/I");
   t.Branch("chipInLayer", &chipInLayer, "chipInLayer/I");
   t.Branch("chipInStave", &chipInStave, "chipInStave/I");
   t.Branch("chipInHalfStave", &chipInHalfStave, "chipInHalfStave/I");
   t.Branch("R", R, "R[9]/D");
   t.Branch("T", T, "T[3]/D");

   for (chipID = 0; chipID < kNChips; ++chipID) {
      layer           = geom->getLayer(chipID);
      halfBarrel      = geom->getHalfBarrel(chipID);
      stave           = geom->getStave(chipID);
      halfStave       = geom->getHalfStave(chipID);
      module          = geom->getModule(chipID);
      chipInModule    = geom->getChipIdInModule(chipID);
      chipInLayer     = geom->getChipIdInLayer(chipID);
      chipInStave     = geom->getChipIdInStave(chipID);
      chipInHalfStave = geom->getChipIdInHalfStave(chipID);

      geom->getMatrixL2G(chipID).GetComponents(R[0], R[1], R[2], T[0],
                                               R[3], R[4], R[5], T[1],
                                               R[6], R[7], R[8], T[2]);
      t.Fill();
   }

   TNamed prov("provenance",
               Form("from o2::its::GeometryTGeo; align=%s; O2 build; %s",
                    alignFile, TDatime().AsSQLString()));
   prov.Write();
   t.Write();
   fo.Close();
   printf("[dump ] wrote %s : %d chips\n", outFile, kNChips);
}
