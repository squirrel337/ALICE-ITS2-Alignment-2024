// ==========================================================================
//  RunConsoleUI.C -- one window to configure, check, compose and launch a run
// ==========================================================================
//  Run it from the repository root, with the environment loaded:
//
//     eval `alienv load -w $O2_DIR/sw O2/latest`     # if this tree needs O2
//     ./config/runctl.sh ui
//
//  or directly:
//
//     root -l 'tools/ConfigUI/RunConsoleUI.C("config/runconsole.conf")'
//
//  The window never writes the configuration file itself, and never composes
//  or launches anything on its own. Every button shells out to
//  config/runctl.sh, so the file format, the validation rules and the job
//  layout stay in one place and the GUI and the command line cannot drift
//  apart. Whatever you can do here you can do over ssh with no display.
//
//  Tested against ROOT 6. CentOS 7 needs no extra packages -- everything here
//  is ROOT's own GUI toolkit, and there is no Python anywhere in the console.
// ==========================================================================

#include "DirBrowser.h"

#include <TGFrame.h>
#include <TGTab.h>
#include <TGButton.h>
#include <TGLabel.h>
#include <TGTextEntry.h>
#include <TGNumberEntry.h>
#include <TGComboBox.h>
#include <TGTextView.h>
#include <TGFileDialog.h>
#include <TGMsgBox.h>
#include <TSystem.h>
#include <TString.h>
#include <TObjArray.h>
#include <TObjString.h>
#include <TApplication.h>

class AlignRunConsoleUI : public TGMainFrame {
private:
   TString fConf, fCtl, fRoot;

   // Inputs
   TGTextEntry   *fData, *fTree, *fGeom, *fAlign, *fParams;
   // Module
   TGTextEntry   *fModuleDir;
   TGNumberEntry *fFirstStep, *fNSteps;
   TGLabel       *fModuleInfo, *fStepInfo;
   // Job: size and model
   TGNumberEntry *fNData, *fNEpoch, *fNCore, *fJPar, *fNTrackMax, *fDetMag;
   TGComboBox    *fFitModel, *fVertexFit, *fMethod;
   TGLabel       *fJobSummary;
   // Job: learning
   TGNumberEntry *fEtaConst, *fEtaDetres, *fValidWindow;
   TGTextEntry   *fEtaScale;
   TGLabel       *fEtaLabel;
   // Job: selection
   TGNumberEntry *fPtMin, *fPtMax, *fChiIB, *fChiOB, *fChiIBTr, *fChiOBTr;
   TGNumberEntry *fTrackReject, *fIpR, *fIpZ, *fMinCluster;
   // Run
   TGTextEntry   *fOutDir, *fTag, *fRootSys;
   TGLabel       *fRunState;

   TGTextView    *fLog;

   TString Run(const char *args);
   TString Get(const char *key);
   static TString Quote(const TString &s);
   void Log(const TString &text);
   void LogCommand(const TString &title, const TString &output);

   TGTextEntry   *MakeRow(TGCompositeFrame *p, const char *label, const char *slot);
   TGNumberEntry *AddInt(TGCompositeFrame *tab, const char *label, Long_t lo, Long_t hi);
   TGNumberEntry *AddReal(TGCompositeFrame *tab, const char *label);
   TGComboBox    *AddCombo(TGCompositeFrame *tab, const char *label, const char *items);
   void           Note(TGCompositeFrame *tab, const char *text, Int_t padTop);

   void BuildInputs(TGCompositeFrame *tab);
   void BuildModule(TGCompositeFrame *tab);
   void BuildJob(TGCompositeFrame *tab);
   void BuildLearning(TGCompositeFrame *tab);
   void BuildSelection(TGCompositeFrame *tab);
   void BuildRun(TGCompositeFrame *tab);

   void PickFile(TGTextEntry *e, const char *filter);
   void PickDir(TGTextEntry *e);
   static void SelectByName(TGComboBox *c, const TString &name);
   static TString SelectedName(TGComboBox *c);

public:
   AlignRunConsoleUI(const TGWindow *p, const char *conf);
   virtual ~AlignRunConsoleUI() { Cleanup(); }

   void LoadAll();
   void UpdateSummary();

   void OnBrowseData();
   void OnBrowseGeom();
   void OnBrowseAlign();
   void OnBrowseParams();
   void OnBrowseModule();
   void OnBrowseOutput();
   void OnBrowseRootSys();

   void OnReload();
   void OnValidate();
   void OnDoctor();
   void OnSave();
   void OnCompose();
   void OnRun();
   void OnStatus();
   void OnTail();
   void OnStop();
   void OnOutputs();
   void OnQuit();

   ClassDef(AlignRunConsoleUI, 0)
};

// -------------------------------------------------------------- helpers ---

TString AlignRunConsoleUI::Quote(const TString &s)
{
   // Single-quote for the shell. A value containing a single quote is
   // rejected before we get here, in OnSave.
   if (s.Contains("'")) return TString("");
   return TString("'") + s + "'";
}

TString AlignRunConsoleUI::Run(const char *args)
{
   TString cmd = TString::Format("%s %s 2>&1", fCtl.Data(), args);
   return gSystem->GetFromPipe(cmd);
}

TString AlignRunConsoleUI::Get(const char *key)
{
   TString v = Run(TString::Format("get %s", key));
   return v.Strip(TString::kTrailing, '\n');
}

void AlignRunConsoleUI::Log(const TString &text)
{
   TObjArray *lines = text.Tokenize("\n");
   for (Int_t i = 0; i < lines->GetEntries(); ++i)
      fLog->AddLine(((TObjString *)lines->At(i))->GetString().Data());
   delete lines;
   fLog->ShowBottom();
}

void AlignRunConsoleUI::LogCommand(const TString &title, const TString &output)
{
   Log(TString("--- ") + title + " ---");
   Log(output);
}

TGTextEntry *AlignRunConsoleUI::MakeRow(TGCompositeFrame *p, const char *label, const char *slot)
{
   TGHorizontalFrame *row = new TGHorizontalFrame(p);
   TGLabel *l = new TGLabel(row, label);
   l->SetWidth(160); l->SetTextJustify(kTextRight);
   row->AddFrame(l, new TGLayoutHints(kLHintsCenterY, 4, 6, 3, 3));

   TGTextEntry *entry = new TGTextEntry(row, "");
   // Rows without a Browse button reserve its width on the right, so every entry in a
   // tab ends at the same place instead of the browseless ones running long.
   const Int_t rpad = (slot && slot[0]) ? 4 : 94;
   row->AddFrame(entry, new TGLayoutHints(kLHintsExpandX | kLHintsCenterY, 0, rpad, 3, 3));

   if (slot && slot[0]) {
      TGTextButton *browse = new TGTextButton(row, " Browse... ");
      browse->Connect("Clicked()", "AlignRunConsoleUI", this, slot);
      row->AddFrame(browse, new TGLayoutHints(kLHintsCenterY, 0, 4, 3, 3));
   }
   p->AddFrame(row, new TGLayoutHints(kLHintsExpandX, 2, 2, 1, 1));
   return entry;
}

TGNumberEntry *AlignRunConsoleUI::AddInt(TGCompositeFrame *tab, const char *label,
                                         Long_t lo, Long_t hi)
{
   TGHorizontalFrame *r = new TGHorizontalFrame(tab);
   TGLabel *rl = new TGLabel(r, label);
   rl->SetWidth(160); rl->SetTextJustify(kTextRight);
   r->AddFrame(rl, new TGLayoutHints(kLHintsCenterY, 4, 6, 3, 3));
   TGNumberEntry *n = new TGNumberEntry(r, 0, 9, -1, TGNumberFormat::kNESInteger,
                                        TGNumberFormat::kNEANonNegative,
                                        TGNumberFormat::kNELLimitMinMax, lo, hi);
   n->Connect("ValueSet(Long_t)", "AlignRunConsoleUI", this, "UpdateSummary()");
   r->AddFrame(n, new TGLayoutHints(kLHintsCenterY, 0, 4, 3, 3));
   tab->AddFrame(r, new TGLayoutHints(kLHintsExpandX, 2, 2, 1, 1));
   return n;
}

TGNumberEntry *AlignRunConsoleUI::AddReal(TGCompositeFrame *tab, const char *label)
{
   TGHorizontalFrame *r = new TGHorizontalFrame(tab);
   TGLabel *rl = new TGLabel(r, label);
   rl->SetWidth(160); rl->SetTextJustify(kTextRight);
   r->AddFrame(rl, new TGLayoutHints(kLHintsCenterY, 4, 6, 3, 3));
   // Any number: DET_MAG is signed, and its sign is the whole point.
   TGNumberEntry *n = new TGNumberEntry(r, 0, 9, -1, TGNumberFormat::kNESRealThree,
                                        TGNumberFormat::kNEAAnyNumber);
   n->Connect("ValueSet(Long_t)", "AlignRunConsoleUI", this, "UpdateSummary()");
   r->AddFrame(n, new TGLayoutHints(kLHintsCenterY, 0, 4, 3, 3));
   tab->AddFrame(r, new TGLayoutHints(kLHintsExpandX, 2, 2, 1, 1));
   return n;
}

TGComboBox *AlignRunConsoleUI::AddCombo(TGCompositeFrame *tab, const char *label,
                                        const char *items)
{
   TGHorizontalFrame *r = new TGHorizontalFrame(tab);
   TGLabel *rl = new TGLabel(r, label);
   rl->SetWidth(160); rl->SetTextJustify(kTextRight);
   r->AddFrame(rl, new TGLayoutHints(kLHintsCenterY, 4, 6, 3, 3));

   TGComboBox *c = new TGComboBox(r);
   TObjArray *parts = TString(items).Tokenize(" ");
   for (Int_t i = 0; i < parts->GetEntries(); ++i)
      c->AddEntry(((TObjString *)parts->At(i))->GetString().Data(), i);
   delete parts;
   c->Resize(220, 22);
   c->Connect("Selected(Int_t)", "AlignRunConsoleUI", this, "UpdateSummary()");
   r->AddFrame(c, new TGLayoutHints(kLHintsCenterY, 0, 4, 3, 3));
   tab->AddFrame(r, new TGLayoutHints(kLHintsExpandX, 2, 2, 1, 1));
   return c;
}

void AlignRunConsoleUI::Note(TGCompositeFrame *tab, const char *text, Int_t padTop)
{
   tab->AddFrame(new TGLabel(tab, text), new TGLayoutHints(kLHintsLeft, 168, 4, padTop, 0));
}

void AlignRunConsoleUI::SelectByName(TGComboBox *c, const TString &name)
{
   TGTextLBEntry *e;
   for (Int_t i = 0; (e = (TGTextLBEntry *)c->GetListBox()->GetEntry(i)); ++i) {
      if (name == e->GetText()->GetString()) { c->Select(i, kFALSE); return; }
   }
}

TString AlignRunConsoleUI::SelectedName(TGComboBox *c)
{
   TGTextLBEntry *e = (TGTextLBEntry *)c->GetSelectedEntry();
   return e ? TString(e->GetText()->GetString()) : TString("");
}

// ---------------------------------------------------------- construction ---

AlignRunConsoleUI::AlignRunConsoleUI(const TGWindow *p, const char *conf)
   : TGMainFrame(p, 940, 760)
{
   fConf = conf;
   if (fConf.IsNull()) fConf = "config/runconsole.conf";

   // The repository is two levels above this macro. DirName is called one
   // level at a time and its result copied: ROOT may hand back a buffer it
   // reuses on the next call.
   TString configui = gSystem->DirName(__FILE__);      // .../tools/ConfigUI
   TString tools    = gSystem->DirName(configui);      // .../tools
   fRoot            = gSystem->DirName(tools);         // repository root
   if (fRoot.IsNull() || fRoot == ".") fRoot = gSystem->WorkingDirectory();
   fCtl = fRoot + "/config/runctl.sh";

   SetWindowName("ITS2 Run Console");
   SetCleanup(kDeepCleanup);

   TGTab *tabs = new TGTab(this, 930, 470);
   BuildInputs   (tabs->AddTab("Inputs"));
   BuildModule   (tabs->AddTab("Module"));
   BuildJob      (tabs->AddTab("Job"));
   BuildLearning (tabs->AddTab("Learning"));
   BuildSelection(tabs->AddTab("Selection"));
   BuildRun      (tabs->AddTab("Run"));
   AddFrame(tabs, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY, 4, 4, 4, 2));

   TGHorizontalFrame *bar = new TGHorizontalFrame(this);
   struct { const char *text; const char *slot; } acts[] = {
      { " Reload ",        "OnReload()" },
      { " Validate ",      "OnValidate()" },
      { " Check machine ", "OnDoctor()" },
      { " Save ",          "OnSave()" },
      { " Compose ",       "OnCompose()" },
      { " Run ",           "OnRun()" },
      { " Status ",        "OnStatus()" },
      { " Log ",           "OnTail()" },
      { " Outputs ",       "OnOutputs()" },
      { 0, 0 }
   };
   for (Int_t i = 0; acts[i].text; ++i) {
      TGTextButton *b = new TGTextButton(bar, acts[i].text);
      b->Connect("Clicked()", "AlignRunConsoleUI", this, acts[i].slot);
      bar->AddFrame(b, new TGLayoutHints(kLHintsLeft, 3, 0, 4, 4));
   }
   TGTextButton *quit = new TGTextButton(bar, " Close ");
   quit->Connect("Clicked()", "AlignRunConsoleUI", this, "OnQuit()");
   bar->AddFrame(quit, new TGLayoutHints(kLHintsRight, 4, 4, 4, 4));
   TGTextButton *stop = new TGTextButton(bar, " Stop job ");
   stop->Connect("Clicked()", "AlignRunConsoleUI", this, "OnStop()");
   bar->AddFrame(stop, new TGLayoutHints(kLHintsRight, 4, 0, 4, 4));
   AddFrame(bar, new TGLayoutHints(kLHintsExpandX, 2, 2, 0, 0));

   fLog = new TGTextView(this, 930, 190);
   AddFrame(fLog, new TGLayoutHints(kLHintsExpandX, 4, 4, 2, 4));

   MapSubwindows();
   Resize(GetDefaultSize());
   MapWindow();

   if (gSystem->AccessPathName(fCtl, kExecutePermission)) {
      Log(TString("cannot execute ") + fCtl);
      Log("the window can only show; every action needs runctl.sh");
   } else {
      LoadAll();
   }
}

void AlignRunConsoleUI::BuildInputs(TGCompositeFrame *tab)
{
   fData   = MakeRow(tab, "Data file",       "OnBrowseData()");
   fTree   = MakeRow(tab, "Tree",            "");
   fGeom   = MakeRow(tab, "Geometry",        "OnBrowseGeom()");
   fAlign  = MakeRow(tab, "Start alignment", "OnBrowseAlign()");
   fParams = MakeRow(tab, "Seed parameters", "OnBrowseParams()");

   Note(tab, "The data file, geometry and alignment are linked into the job directory, not copied.", 14);
   Note(tab, "Seed parameters are unpacked into MLPTrain_Step<FIRST_STEP-1>/. Later steps need no seed:", 8);
   Note(tab, "step N reads MLPTrain_Step<N-1>/, which step N-1 has just written.", 0);
   Note(tab, "A seed without weightsDU.txt leaves the detector-unit normalisations uninitialised and", 8);
   Note(tab, "the cost comes out -nan. 'Check machine' says so before the run, not after.", 0);
}

void AlignRunConsoleUI::BuildModule(TGCompositeFrame *tab)
{
   fModuleDir = MakeRow(tab, "Module checkout", "OnBrowseModule()");
   fFirstStep = AddInt(tab, "First step", 1, 100000);
   fNSteps    = AddInt(tab, "Number of steps", 1, 1000);

   fStepInfo   = new TGLabel(tab, "");
   tab->AddFrame(fStepInfo, new TGLayoutHints(kLHintsLeft, 168, 4, 14, 2));
   fModuleInfo = new TGLabel(tab, "");
   tab->AddFrame(fModuleInfo, new TGLayoutHints(kLHintsLeft, 168, 4, 2, 4));

   Note(tab, "Leave the checkout empty to use this repository. Point it at another tree -- a 2025", 10);
   Note(tab, "checkout, say -- and the console reads that tree's own headers instead of assuming these.", 0);
   Note(tab, "First step must be at least 1: at step 0 the module hands LoadUpdateSensorList an empty name.", 8);
   Note(tab, "Steps run back to back in one detached job. If a step fails the run stops there.", 8);
}

void AlignRunConsoleUI::BuildJob(TGCompositeFrame *tab)
{
   fNData     = AddInt (tab, "nDATA",     1, 100000000);
   fNEpoch    = AddInt (tab, "nEPOCH",    0, 10000);
   fNCore     = AddInt (tab, "nCORE",     1, 256);
   fJPar      = AddInt (tab, "jparallel", 0, 256);
   fNTrackMax = AddInt (tab, "nTrackMax", 2, 200);
   fDetMag    = AddReal(tab, "DET_MAG (T)");
   fFitModel  = AddCombo(tab, "FITMODEL", "1 2");
   fVertexFit = AddCombo(tab, "VERTEXFIT", "kFALSE kTRUE");
   fMethod    = AddCombo(tab, "Learning method",
                         "kStochastic kBatch kBatchDetectorUnitUser kSteepestDescent "
                         "kRibierePolak kFletcherReeves kBFGS kOffsetTuneByMean");

   fJobSummary = new TGLabel(tab, "");
   tab->AddFrame(fJobSummary, new TGLayoutHints(kLHintsLeft, 168, 4, 14, 4));

   Note(tab, "nEPOCH 0 evaluates the epoch -1 baseline and stops; above 0 the weights move.", 10);
   Note(tab, "DET_MAG is signed. The sign reaches the impact parameter, not just the magnitude.", 0);
   Note(tab, "FITMODEL 1 = Line, 2 = Circle. Changing it changes the track model, not a threshold.", 8);
}

void AlignRunConsoleUI::BuildLearning(TGCompositeFrame *tab)
{
   fEtaConst    = AddReal(tab, "UpdateConstant");
   fEtaScale    = MakeRow(tab, "UpdateScale", "");
   fEtaDetres   = AddReal(tab, "DETRES (um)");
   fValidWindow = AddReal(tab, "ValidWindow (cm)");

   fEtaLabel = new TGLabel(tab, "");
   tab->AddFrame(fEtaLabel, new TGLayoutHints(kLHintsLeft, 168, 4, 14, 4));

   Note(tab, "eta = UpdateConstant * UpdateScale * (DETRES * 1e-4)^2", 10);
   Note(tab, "These are plain globals in YMultiLayerPerceptron.cxx, not macros, so the job's own", 8);
   Note(tab, "copy of that source is patched. Nothing is rebuilt: the driver includes the source.", 0);
   Note(tab, "THIS IS THE STARTING VALUE. The module backtracks on its own -- if the cost rises it", 10);
   Note(tab, "reloads the previous epoch and divides eta by sqrt(10), up to four times. The eta in", 0);
   Note(tab, "the log will differ from this one whenever that has happened.", 0);
   Note(tab, "ValidWindow clamps how much a single hit's residual may contribute to the update.", 10);
}

void AlignRunConsoleUI::BuildSelection(TGCompositeFrame *tab)
{
   fPtMin       = AddReal(tab, "Update_pTmin");
   fPtMax       = AddReal(tab, "Update_pTmax");
   fChiIB       = AddReal(tab, "chi IB  (cost)");
   fChiOB       = AddReal(tab, "chi OB  (cost)");
   fChiIBTr     = AddReal(tab, "chi IB  (update)");
   fChiOBTr     = AddReal(tab, "chi OB  (update)");
   fTrackReject = AddReal(tab, "TrackRejection");
   fIpR         = AddReal(tab, "ip range r (cm)");
   fIpZ         = AddReal(tab, "ip range z (cm)");
   fMinCluster  = AddInt (tab, "Min cluster/sensor", 0, 1000000);

   Note(tab, "The two chi pairs are deliberately different: the first governs which hits spoil the", 12);
   Note(tab, "COST, the second which hits spoil the WEIGHT UPDATE.", 0);
   Note(tab, "The impact-parameter range decides how much the layer-0 defect costs in this legacy", 10);
   Note(tab, "tree: a track whose reference point was never written lands far outside and is dropped", 0);
   Note(tab, "from both the cost and the update. Widening it is the only lever here, short of a patch.", 0);
}

void AlignRunConsoleUI::BuildRun(TGCompositeFrame *tab)
{
   fOutDir  = MakeRow(tab, "Output directory", "OnBrowseOutput()");
   fTag     = MakeRow(tab, "Job tag",          "");
   fRootSys = MakeRow(tab, "ROOTSYS override", "OnBrowseRootSys()");

   fRunState = new TGLabel(tab, "");
   tab->AddFrame(fRunState, new TGLayoutHints(kLHintsLeft, 168, 4, 14, 4));

   Note(tab, "Compose builds OUTPUT_DIR/JOB_TAG and patches the knobs into that job's own headers", 10);
   Note(tab, "and sources. The module checkout is only ever read.", 0);
   Note(tab, "Run detaches the job with setsid, so it survives this window and the shell that", 8);
   Note(tab, "started it. Status reports which step the run is on, even after reopening.", 0);
   Note(tab, "Each job holds about 8 GB resident. Two at once have OOM-killed each other.", 8);
}

// ------------------------------------------------------------------ load ---

void AlignRunConsoleUI::LoadAll()
{
   fData  ->SetText(Get("DATA_FILE"));
   fTree  ->SetText(Get("DATA_TREE"));
   fGeom  ->SetText(Get("GEOM_FILE"));
   fAlign ->SetText(Get("ALIGN_FILE"));
   fParams->SetText(Get("PARAMS_ARCHIVE"));

   fModuleDir->SetText(Get("MODULE_DIR"));
   fFirstStep->SetIntNumber(Get("FIRST_STEP").Atoll());
   fNSteps   ->SetIntNumber(Get("N_STEPS").Atoll());

   fNData    ->SetIntNumber(Get("JOB_NDATA").Atoll());
   fNEpoch   ->SetIntNumber(Get("JOB_NEPOCH").Atoll());
   fNCore    ->SetIntNumber(Get("JOB_NCORE").Atoll());
   fJPar     ->SetIntNumber(Get("JOB_JPARALLEL").Atoll());
   fNTrackMax->SetIntNumber(Get("JOB_NTRACKMAX").Atoll());
   fDetMag   ->SetNumber(Get("JOB_DET_MAG").Atof());
   SelectByName(fFitModel,  Get("JOB_FITMODEL"));
   SelectByName(fVertexFit, Get("JOB_VERTEXFIT"));
   SelectByName(fMethod,    Get("JOB_LEARNING_METHOD"));

   fEtaConst   ->SetNumber(Get("JOB_ETA_CONSTANT").Atof());
   fEtaScale   ->SetText(Get("JOB_ETA_SCALE"));
   fEtaDetres  ->SetNumber(Get("JOB_ETA_DETRES").Atof());
   fValidWindow->SetNumber(Get("JOB_VALID_WINDOW").Atof());

   fPtMin      ->SetNumber(Get("JOB_PT_MIN").Atof());
   fPtMax      ->SetNumber(Get("JOB_PT_MAX").Atof());
   fChiIB      ->SetNumber(Get("JOB_CHI_IB").Atof());
   fChiOB      ->SetNumber(Get("JOB_CHI_OB").Atof());
   fChiIBTr    ->SetNumber(Get("JOB_CHI_IB_TRAIN").Atof());
   fChiOBTr    ->SetNumber(Get("JOB_CHI_OB_TRAIN").Atof());
   fTrackReject->SetNumber(Get("JOB_TRACK_REJECT").Atof());
   fIpR        ->SetNumber(Get("JOB_IP_RANGE_R").Atof());
   fIpZ        ->SetNumber(Get("JOB_IP_RANGE_Z").Atof());
   fMinCluster ->SetIntNumber(Get("JOB_MIN_CLUSTER").Atoll());

   fOutDir ->SetText(Get("OUTPUT_DIR"));
   fTag    ->SetText(Get("JOB_TAG"));
   fRootSys->SetText(Get("ROOTSYS_OVERRIDE"));

   UpdateSummary();
}

void AlignRunConsoleUI::UpdateSummary()
{
   // Everything derived comes from runctl.sh rather than being recomputed
   // here, so there is one runtime model, one eta formula and one backend
   // detector -- not two that can disagree.
   TString est  = Get("RC_EST_MIN");
   TString per  = Get("RC_EST_STEP");
   TString eta  = Get("RC_ETA");
   TString o2   = Get("RC_O2_REQUIRED");

   Long64_t ntm = fNTrackMax ? fNTrackMax->GetIntNumber() : 0;
   fJobSummary->SetText(TString::Format(
      "ndf per event = 12n+1 = %lld        %s min total (%.1f h), %s min per step",
      12 * ntm + 1, est.Data(), est.Atof() / 60.0, per.Data()));

   fStepInfo->SetText(TString::Format(
      "steps %s .. %s        seed unpacks to MLPTrain_Step%s/",
      Get("FIRST_STEP").Data(), Get("RC_LAST_STEP").Data(), Get("RC_SEED_STEP").Data()));

   fModuleInfo->SetText(TString::Format(
      "%s        module holds nDATA=%s nEPOCH=%s nTrackMax=%s DET_MAG=%s",
      (o2 == "1") ? "needs O2 at runtime" : "cache-backed geometry, no O2 needed",
      Get("RC_MOD_NDATA").Data(), Get("RC_MOD_NEPOCH").Data(),
      Get("RC_MOD_NTRACKMAX").Data(), Get("RC_MOD_DET_MAG").Data()));

   fEtaLabel->SetText(TString("starting eta = ") + eta +
                      "        (saved values; press Save to recompute from the fields above)");

   fRunState->SetText(TString("job directory  ") + Get("RC_JOB_DIR"));

   // TGWindow -- what GetParent() returns -- has no Layout(); it is TGCompositeFrame
   // that does. Relayout from the main frame instead, which recurses into the tabs and
   // picks up all four labels in one call.
   Layout();
}

// ------------------------------------------------------------------ slots ---

void AlignRunConsoleUI::PickFile(TGTextEntry *e, const char *filter)
{
   static const char *anyfile[]  = { "All files", "*", 0, 0 };
   static const char *rootfile[] = { "ROOT files", "*.root", "All files", "*", 0, 0 };
   static const char *archive[]  = { "Archives", "*.tgz", "All files", "*", 0, 0 };

   TGFileInfo fi;
   TString f(filter);
   if      (f == "root")    fi.fFileTypes = rootfile;
   else if (f == "archive") fi.fFileTypes = archive;
   else                     fi.fFileTypes = anyfile;

   TString start = e->GetText();
   if (!start.IsNull()) fi.fIniDir = StrDup(gSystem->DirName(start));
   else                 fi.fIniDir = StrDup(fRoot.Data());

   new TGFileDialog(gClient->GetRoot(), this, kFDOpen, &fi);
   if (fi.fFilename) e->SetText(fi.fFilename);
}

void AlignRunConsoleUI::PickDir(TGTextEntry *e)
{
   TString start = e->GetText();
   if (start.IsNull()) start = fRoot;
   TString picked = DirBrowser::Pick(gClient->GetRoot(), start.Data());
   if (!picked.IsNull()) e->SetText(picked);
}

void AlignRunConsoleUI::OnBrowseData()    { PickFile(fData,   "root"); }
void AlignRunConsoleUI::OnBrowseGeom()    { PickFile(fGeom,   "root"); }
void AlignRunConsoleUI::OnBrowseAlign()   { PickFile(fAlign,  "root"); }
void AlignRunConsoleUI::OnBrowseParams()  { PickFile(fParams, "archive"); }
void AlignRunConsoleUI::OnBrowseModule()  { PickDir(fModuleDir); }
void AlignRunConsoleUI::OnBrowseOutput()  { PickDir(fOutDir); }
void AlignRunConsoleUI::OnBrowseRootSys() { PickDir(fRootSys); }

void AlignRunConsoleUI::OnReload()   { LoadAll(); Log("reloaded from the configuration file"); }
void AlignRunConsoleUI::OnValidate() { LogCommand("validate", Run("validate")); }
void AlignRunConsoleUI::OnDoctor()   { LogCommand("check machine", Run("doctor")); }

void AlignRunConsoleUI::OnSave()
{
   // Reject a single quote before building the command rather than after.
   // Quote() returns empty for such a value, which would otherwise be written
   // out as an empty setting -- silently losing it.
   const char *texts[] = { fData->GetText(), fTree->GetText(), fGeom->GetText(),
                           fAlign->GetText(), fParams->GetText(), fModuleDir->GetText(),
                           fEtaScale->GetText(), fOutDir->GetText(), fTag->GetText(),
                           fRootSys->GetText(), 0 };
   for (Int_t i = 0; texts[i]; ++i) {
      if (TString(texts[i]).Contains("'")) {
         Log("a value contains a single quote, which is not supported; remove it and retry");
         return;
      }
   }

   TString args = "set";
   args += " DATA_FILE="      + Quote(fData->GetText());
   args += " DATA_TREE="      + Quote(fTree->GetText());
   args += " GEOM_FILE="      + Quote(fGeom->GetText());
   args += " ALIGN_FILE="     + Quote(fAlign->GetText());
   args += " PARAMS_ARCHIVE=" + Quote(fParams->GetText());
   args += " MODULE_DIR="     + Quote(fModuleDir->GetText());
   args += TString::Format(" FIRST_STEP=%lld",    (Long64_t)fFirstStep->GetIntNumber());
   args += TString::Format(" N_STEPS=%lld",       (Long64_t)fNSteps->GetIntNumber());
   args += TString::Format(" JOB_NDATA=%lld",     (Long64_t)fNData->GetIntNumber());
   args += TString::Format(" JOB_NEPOCH=%lld",    (Long64_t)fNEpoch->GetIntNumber());
   args += TString::Format(" JOB_NCORE=%lld",     (Long64_t)fNCore->GetIntNumber());
   args += TString::Format(" JOB_JPARALLEL=%lld", (Long64_t)fJPar->GetIntNumber());
   args += TString::Format(" JOB_NTRACKMAX=%lld", (Long64_t)fNTrackMax->GetIntNumber());
   args += TString::Format(" JOB_DET_MAG=%.6g",   fDetMag->GetNumber());
   args += " JOB_FITMODEL="        + Quote(SelectedName(fFitModel));
   args += " JOB_VERTEXFIT="       + Quote(SelectedName(fVertexFit));
   args += " JOB_LEARNING_METHOD=" + Quote(SelectedName(fMethod));
   args += TString::Format(" JOB_ETA_CONSTANT=%.6g", fEtaConst->GetNumber());
   args += " JOB_ETA_SCALE="       + Quote(fEtaScale->GetText());
   args += TString::Format(" JOB_ETA_DETRES=%.6g",   fEtaDetres->GetNumber());
   args += TString::Format(" JOB_VALID_WINDOW=%.6g", fValidWindow->GetNumber());
   args += TString::Format(" JOB_PT_MIN=%.6g",       fPtMin->GetNumber());
   args += TString::Format(" JOB_PT_MAX=%.6g",       fPtMax->GetNumber());
   args += TString::Format(" JOB_CHI_IB=%.6g",       fChiIB->GetNumber());
   args += TString::Format(" JOB_CHI_OB=%.6g",       fChiOB->GetNumber());
   args += TString::Format(" JOB_CHI_IB_TRAIN=%.6g", fChiIBTr->GetNumber());
   args += TString::Format(" JOB_CHI_OB_TRAIN=%.6g", fChiOBTr->GetNumber());
   args += TString::Format(" JOB_TRACK_REJECT=%.6g", fTrackReject->GetNumber());
   args += TString::Format(" JOB_IP_RANGE_R=%.6g",   fIpR->GetNumber());
   args += TString::Format(" JOB_IP_RANGE_Z=%.6g",   fIpZ->GetNumber());
   args += TString::Format(" JOB_MIN_CLUSTER=%lld",  (Long64_t)fMinCluster->GetIntNumber());
   args += " OUTPUT_DIR="       + Quote(fOutDir->GetText());
   args += " JOB_TAG="          + Quote(fTag->GetText());
   args += " ROOTSYS_OVERRIDE=" + Quote(fRootSys->GetText());

   LogCommand("save", Run(args));
   UpdateSummary();
}

void AlignRunConsoleUI::OnCompose()
{
   LogCommand("compose", Run("compose"));
   UpdateSummary();
}

void AlignRunConsoleUI::OnRun()
{
   // Launching is the one action worth a confirmation: it is hours long, it
   // consumes the machine, and with several steps it is much longer still.
   Int_t answer = 0;
   new TGMsgBox(gClient->GetRoot(), this, "Launch this run",
                TString::Format("Start the job in %s?\n\n"
                                "steps %s..%s, nDATA %lld, nEPOCH %lld\n"
                                "about %s minutes in total.\n\n"
                                "It detaches, so closing this window will not stop it.",
                                Get("RC_JOB_DIR").Data(),
                                Get("FIRST_STEP").Data(), Get("RC_LAST_STEP").Data(),
                                (Long64_t)fNData->GetIntNumber(),
                                (Long64_t)fNEpoch->GetIntNumber(),
                                Get("RC_EST_MIN").Data()),
                kMBIconQuestion, kMBOk | kMBCancel, &answer);
   if (answer != kMBOk) { Log("launch cancelled"); return; }
   LogCommand("run", Run("run"));
}

void AlignRunConsoleUI::OnStatus()  { LogCommand("status", Run("status")); }
void AlignRunConsoleUI::OnTail()    { LogCommand("log", Run("log")); }
void AlignRunConsoleUI::OnOutputs() { LogCommand("outputs", Run("outputs")); }

void AlignRunConsoleUI::OnStop()
{
   Int_t answer = 0;
   new TGMsgBox(gClient->GetRoot(), this, "Stop the job",
                "Send TERM to the running job?\n\nSteps already finished are kept.",
                kMBIconExclamation, kMBOk | kMBCancel, &answer);
   if (answer != kMBOk) return;
   LogCommand("stop", Run("stop"));
}

void AlignRunConsoleUI::OnQuit()
{
   // Only the window closes. A launched job is detached and keeps running.
   UnmapWindow();
   CloseWindow();
   if (gApplication) gApplication->Terminate(0);
}

// ------------------------------------------------------------------ entry ---

void RunConsoleUI(const char *conf = "config/runconsole.conf")
{
   new AlignRunConsoleUI(gClient->GetRoot(), conf);
}
