// ==========================================================================
//  AlignRunConsoleUI.C -- one window to configure, check, compose and launch a run
// ==========================================================================
//  Run it from the repository root, with the environment loaded:
//
//     eval `alienv load -w $O2_DIR/sw O2/latest`     # if this tree needs O2
//     ./config/runctl.sh ui
//
//  or directly:
//
//     root -l 'tools/ConfigUI/AlignRunConsoleUI.C("config/runconsole.conf")'
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
   TString fConf;        // configuration file
   TString fCtl;         // config/runctl.sh
   TString fRoot;        // repository root

   TGTextEntry   *fData, *fTree, *fGeom, *fAlign, *fParams;
   TGTextEntry   *fModuleDir;
   TGNumberEntry *fStep;
   TGLabel       *fModuleInfo;

   TGNumberEntry *fNData, *fNEpoch, *fNCore, *fJPar;
   TGNumberEntry *fNTrackMax, *fDetMag, *fPtMin, *fPtMax;
   TGLabel       *fJobSummary;

   TGTextEntry   *fOutDir, *fTag, *fRootSys;
   TGLabel       *fRunState;

   TGTextView    *fLog;

   TString Run(const char *args);
   TString Get(const char *key);
   static TString Quote(const TString &s);
   void Log(const TString &text);
   void LogCommand(const TString &title, const TString &output);

   TGNumberEntry *MakeInt(TGCompositeFrame *p, Long_t min, Long_t max);
   TGNumberEntry *MakeReal(TGCompositeFrame *p);
   TGTextEntry   *MakeRow(TGCompositeFrame *p, const char *label, const char *slot);
   void           AddNumberRow(TGCompositeFrame *tab, const char *label,
                               TGNumberEntry **slot, Bool_t real, Long_t lo, Long_t hi);

   void BuildInputs(TGCompositeFrame *tab);
   void BuildModule(TGCompositeFrame *tab);
   void BuildJob(TGCompositeFrame *tab);
   void BuildRun(TGCompositeFrame *tab);

   void PickFile(TGTextEntry *e, const char *filter);
   void PickDir(TGTextEntry *e);

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
   // rejected rather than escaped, because none of these settings should
   // ever need one.
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
   v = v.Strip(TString::kTrailing, '\n');
   return v;
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

TGNumberEntry *AlignRunConsoleUI::MakeInt(TGCompositeFrame *p, Long_t min, Long_t max)
{
   TGNumberEntry *n = new TGNumberEntry(p, 0, 9, -1, TGNumberFormat::kNESInteger,
                                        TGNumberFormat::kNEANonNegative,
                                        TGNumberFormat::kNELLimitMinMax, min, max);
   return n;
}

TGNumberEntry *AlignRunConsoleUI::MakeReal(TGCompositeFrame *p)
{
   // Any number: DET_MAG is signed, and its sign is the whole point.
   return new TGNumberEntry(p, 0, 9, -1, TGNumberFormat::kNESRealTwo,
                            TGNumberFormat::kNEAAnyNumber);
}

TGTextEntry *AlignRunConsoleUI::MakeRow(TGCompositeFrame *p, const char *label, const char *slot)
{
   TGHorizontalFrame *row = new TGHorizontalFrame(p);
   TGLabel *l = new TGLabel(row, label);
   l->SetWidth(150); l->SetTextJustify(kTextRight);
   row->AddFrame(l, new TGLayoutHints(kLHintsCenterY, 4, 6, 3, 3));

   TGTextEntry *entry = new TGTextEntry(row, "");
   row->AddFrame(entry, new TGLayoutHints(kLHintsExpandX | kLHintsCenterY, 0, 4, 3, 3));

   if (slot && slot[0]) {
      TGTextButton *browse = new TGTextButton(row, " Browse... ");
      browse->Connect("Clicked()", "AlignRunConsoleUI", this, slot);
      row->AddFrame(browse, new TGLayoutHints(kLHintsCenterY, 0, 4, 3, 3));
   }
   p->AddFrame(row, new TGLayoutHints(kLHintsExpandX, 2, 2, 1, 1));
   return entry;
}

void AlignRunConsoleUI::AddNumberRow(TGCompositeFrame *tab, const char *label,
                                TGNumberEntry **slot, Bool_t real, Long_t lo, Long_t hi)
{
   TGHorizontalFrame *r = new TGHorizontalFrame(tab);
   TGLabel *rl = new TGLabel(r, label);
   rl->SetWidth(150); rl->SetTextJustify(kTextRight);
   r->AddFrame(rl, new TGLayoutHints(kLHintsCenterY, 4, 6, 3, 3));
   *slot = real ? MakeReal(r) : MakeInt(r, lo, hi);
   (*slot)->Connect("ValueSet(Long_t)", "AlignRunConsoleUI", this, "UpdateSummary()");
   r->AddFrame(*slot, new TGLayoutHints(kLHintsCenterY, 0, 4, 3, 3));
   tab->AddFrame(r, new TGLayoutHints(kLHintsExpandX, 2, 2, 1, 1));
}

// ---------------------------------------------------------- construction ---

AlignRunConsoleUI::AlignRunConsoleUI(const TGWindow *p, const char *conf)
   : TGMainFrame(p, 900, 700)
{
   fConf = conf;
   if (fConf.IsNull()) fConf = "config/runconsole.conf";

   // The repository is two levels above this macro, so the window works from
   // any working directory. DirName is called one level at a time and its
   // result copied: ROOT may hand back a buffer it reuses on the next call.
   TString configui = gSystem->DirName(__FILE__);      // .../tools/ConfigUI
   TString tools    = gSystem->DirName(configui);      // .../tools
   fRoot            = gSystem->DirName(tools);         // repository root
   if (fRoot.IsNull() || fRoot == ".") fRoot = gSystem->WorkingDirectory();
   fCtl = fRoot + "/config/runctl.sh";

   SetWindowName("ITS2 Run Console");
   SetCleanup(kDeepCleanup);

   TGTab *tabs = new TGTab(this, 890, 460);
   BuildInputs(tabs->AddTab("Inputs"));
   BuildModule(tabs->AddTab("Module"));
   BuildJob(tabs->AddTab("Job"));
   BuildRun(tabs->AddTab("Run"));
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

   fLog = new TGTextView(this, 890, 180);
   AddFrame(fLog, new TGLayoutHints(kLHintsExpandX, 4, 4, 2, 4));

   MapSubwindows();
   Resize(GetDefaultSize());
   MapWindow();

   if (gSystem->AccessPathName(fCtl, kExecutePermission)) {
      Log(TString("cannot execute ") + fCtl);
      Log("the window can only read and show; every action needs runctl.sh");
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

   tab->AddFrame(new TGLabel(tab,
      "The data file, geometry and alignment are linked into the job directory, not copied."),
      new TGLayoutHints(kLHintsLeft, 158, 4, 12, 2));
   tab->AddFrame(new TGLabel(tab,
      "Seed parameters are unpacked into MLPTrain_Step<STEP-1>/, which is where the module looks."),
      new TGLayoutHints(kLHintsLeft, 158, 4, 0, 2));
   tab->AddFrame(new TGLabel(tab,
      "A seed without weightsDU.txt leaves the detector-unit normalisations uninitialised"),
      new TGLayoutHints(kLHintsLeft, 158, 4, 8, 0));
   tab->AddFrame(new TGLabel(tab,
      "and the cost comes out -nan. 'Check machine' says so before the run, not after."),
      new TGLayoutHints(kLHintsLeft, 158, 4, 0, 2));
}

void AlignRunConsoleUI::BuildModule(TGCompositeFrame *tab)
{
   fModuleDir = MakeRow(tab, "Module checkout", "OnBrowseModule()");
   AddNumberRow(tab, "Step", &fStep, kFALSE, 1, 100000);

   fModuleInfo = new TGLabel(tab, "");
   tab->AddFrame(fModuleInfo, new TGLayoutHints(kLHintsLeft, 158, 4, 14, 4));

   tab->AddFrame(new TGLabel(tab,
      "Leave the checkout empty to use this repository. Point it at another tree -- a 2025"),
      new TGLayoutHints(kLHintsLeft, 158, 4, 10, 0));
   tab->AddFrame(new TGLabel(tab,
      "checkout, say -- and the console reads that tree's own headers instead of assuming these."),
      new TGLayoutHints(kLHintsLeft, 158, 4, 0, 2));
   tab->AddFrame(new TGLabel(tab,
      "Step must be at least 1: at step 0 the module hands LoadUpdateSensorList an empty name."),
      new TGLayoutHints(kLHintsLeft, 158, 4, 8, 2));
}

void AlignRunConsoleUI::BuildJob(TGCompositeFrame *tab)
{
   AddNumberRow(tab, "nDATA",     &fNData,     kFALSE, 1, 100000000);
   AddNumberRow(tab, "nEPOCH",    &fNEpoch,    kFALSE, 0, 10000);
   AddNumberRow(tab, "nCORE",     &fNCore,     kFALSE, 1, 256);
   AddNumberRow(tab, "jparallel", &fJPar,      kFALSE, 0, 256);
   AddNumberRow(tab, "nTrackMax", &fNTrackMax, kFALSE, 2, 200);
   AddNumberRow(tab, "DET_MAG",   &fDetMag,    kTRUE,  0, 0);
   AddNumberRow(tab, "pT min",    &fPtMin,     kTRUE,  0, 0);
   AddNumberRow(tab, "pT max",    &fPtMax,     kTRUE,  0, 0);

   fJobSummary = new TGLabel(tab, "");
   tab->AddFrame(fJobSummary, new TGLayoutHints(kLHintsLeft, 158, 4, 14, 4));

   tab->AddFrame(new TGLabel(tab,
      "nEPOCH 0 evaluates the epoch -1 baseline and stops; above 0 the weights move."),
      new TGLayoutHints(kLHintsLeft, 158, 4, 10, 0));
   tab->AddFrame(new TGLabel(tab,
      "DET_MAG is signed. The sign reaches the impact parameter, not just the magnitude."),
      new TGLayoutHints(kLHintsLeft, 158, 4, 0, 2));
}

void AlignRunConsoleUI::BuildRun(TGCompositeFrame *tab)
{
   fOutDir  = MakeRow(tab, "Output directory", "OnBrowseOutput()");
   fTag     = MakeRow(tab, "Job tag",          "");
   fRootSys = MakeRow(tab, "ROOTSYS override", "OnBrowseRootSys()");

   fRunState = new TGLabel(tab, "");
   tab->AddFrame(fRunState, new TGLayoutHints(kLHintsLeft, 158, 4, 14, 4));

   tab->AddFrame(new TGLabel(tab,
      "Compose builds OUTPUT_DIR/JOB_TAG and patches the knobs into that job's own headers."),
      new TGLayoutHints(kLHintsLeft, 158, 4, 10, 0));
   tab->AddFrame(new TGLabel(tab,
      "The module checkout is only ever read, so a tree meant to stay untouched stays untouched."),
      new TGLayoutHints(kLHintsLeft, 158, 4, 0, 2));
   tab->AddFrame(new TGLabel(tab,
      "Run detaches the job with setsid, so it survives this window and the shell that started it."),
      new TGLayoutHints(kLHintsLeft, 158, 4, 8, 2));
   tab->AddFrame(new TGLabel(tab,
      "Each job holds about 8 GB resident. Two at once have OOM-killed each other."),
      new TGLayoutHints(kLHintsLeft, 158, 4, 0, 2));
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
   fStep->SetIntNumber(Get("STEP").Atoll());

   fNData    ->SetIntNumber(Get("JOB_NDATA").Atoll());
   fNEpoch   ->SetIntNumber(Get("JOB_NEPOCH").Atoll());
   fNCore    ->SetIntNumber(Get("JOB_NCORE").Atoll());
   fJPar     ->SetIntNumber(Get("JOB_JPARALLEL").Atoll());
   fNTrackMax->SetIntNumber(Get("JOB_NTRACKMAX").Atoll());
   fDetMag   ->SetNumber(Get("JOB_DET_MAG").Atof());
   fPtMin    ->SetNumber(Get("JOB_PT_MIN").Atof());
   fPtMax    ->SetNumber(Get("JOB_PT_MAX").Atof());

   fOutDir ->SetText(Get("OUTPUT_DIR"));
   fTag    ->SetText(Get("JOB_TAG"));
   fRootSys->SetText(Get("ROOTSYS_OVERRIDE"));

   UpdateSummary();
}

void AlignRunConsoleUI::UpdateSummary()
{
   // The runtime estimate and the backend come from runctl.sh rather than
   // being recomputed here, so there is one model and one detector, not two.
   TString est  = Get("RC_EST_MIN");
   TString o2   = Get("RC_O2_REQUIRED");
   TString jd   = Get("RC_JOB_DIR");
   TString seed = Get("RC_SEED_STEP");

   Long64_t ntm = fNTrackMax ? fNTrackMax->GetIntNumber() : 0;
   fJobSummary->SetText(TString::Format(
      "ndf per event = 12n+1 = %lld       estimated runtime %s min (%.1f h)",
      12 * ntm + 1, est.Data(), est.Atof() / 60.0));

   fModuleInfo->SetText(TString::Format(
      "%s     module holds nDATA=%s nEPOCH=%s nTrackMax=%s DET_MAG=%s     seed -> MLPTrain_Step%s/",
      (o2 == "1") ? "needs O2 at runtime" : "cache-backed geometry, no O2 needed",
      Get("RC_MOD_NDATA").Data(), Get("RC_MOD_NEPOCH").Data(),
      Get("RC_MOD_NTRACKMAX").Data(), Get("RC_MOD_DET_MAG").Data(), seed.Data()));

   fRunState->SetText(TString("job directory  ") + jd);

   fJobSummary->GetParent()->Layout();
   fModuleInfo->GetParent()->Layout();
   fRunState->GetParent()->Layout();
}

// ------------------------------------------------------------------ slots ---

void AlignRunConsoleUI::PickFile(TGTextEntry *e, const char *filter)
{
   static const char *anyfile[] = { "All files", "*", 0, 0 };
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
                           fOutDir->GetText(), fTag->GetText(), fRootSys->GetText(), 0 };
   for (Int_t i = 0; texts[i]; ++i) {
      if (TString(texts[i]).Contains("'")) {
         Log("a value contains a single quote, which is not supported; remove it and retry");
         return;
      }
   }

   TString args = "set";
   args += " DATA_FILE="       + Quote(fData->GetText());
   args += " DATA_TREE="       + Quote(fTree->GetText());
   args += " GEOM_FILE="       + Quote(fGeom->GetText());
   args += " ALIGN_FILE="      + Quote(fAlign->GetText());
   args += " PARAMS_ARCHIVE="  + Quote(fParams->GetText());
   args += " MODULE_DIR="      + Quote(fModuleDir->GetText());
   args += TString::Format(" STEP=%lld",          (Long64_t)fStep->GetIntNumber());
   args += TString::Format(" JOB_NDATA=%lld",     (Long64_t)fNData->GetIntNumber());
   args += TString::Format(" JOB_NEPOCH=%lld",    (Long64_t)fNEpoch->GetIntNumber());
   args += TString::Format(" JOB_NCORE=%lld",     (Long64_t)fNCore->GetIntNumber());
   args += TString::Format(" JOB_JPARALLEL=%lld", (Long64_t)fJPar->GetIntNumber());
   args += TString::Format(" JOB_NTRACKMAX=%lld", (Long64_t)fNTrackMax->GetIntNumber());
   args += TString::Format(" JOB_DET_MAG=%.4g",   fDetMag->GetNumber());
   args += TString::Format(" JOB_PT_MIN=%.4g",    fPtMin->GetNumber());
   args += TString::Format(" JOB_PT_MAX=%.4g",    fPtMax->GetNumber());
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
   // Launching is the one action worth a confirmation: it is hours long and
   // it is what actually consumes the machine.
   Int_t answer = 0;
   new TGMsgBox(gClient->GetRoot(), this, "Launch this run",
                TString::Format("Start the job in %s?\n\n"
                                "nDATA %lld, nEPOCH %lld -- about %s minutes.\n"
                                "It detaches, so closing this window will not stop it.",
                                Get("RC_JOB_DIR").Data(),
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
                "Send TERM to the running job?", kMBIconExclamation,
                kMBOk | kMBCancel, &answer);
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
