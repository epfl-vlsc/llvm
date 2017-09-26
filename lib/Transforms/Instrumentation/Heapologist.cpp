//===-- Heapologist.cpp ---------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of Heapologist, a tool to detect inefficient heap
// usage patterns. This file was adapted from the EfficiencySanitizer
// instrumentation.
//
// The instrumentation phase is straightforward:
//   - Take action on every memory access: either inlined instrumentation,
//     or Inserted calls to our run-time library.
//   - Optimizations may apply to avoid instrumenting some of the accesses.
//   - Turn mem{set,cpy,move} instrinsics into library calls.
// The rest is handled by the run-time library.
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <fstream>
#include <sys/stat.h>
#include <cxxabi.h>
#include <llvm/IR/DebugInfoMetadata.h>

using namespace llvm;

#define DEBUG_TYPE "hplgst"

// The tool type must be just one of these ClTool* options, as the tools
// cannot be combined due to shadow memory constraints.

// Each new tool will get its own opt flag here.
// These are converted to HeapologistOptions for use
// in the code.

static cl::opt<bool> ClInstrumentLoadsAndStores(
    "hplgst-instrument-loads-and-stores", cl::init(true),
    cl::desc("Instrument loads and stores"), cl::Hidden);
static cl::opt<bool> ClInstrumentMemIntrinsics(
    "hplgst-instrument-memintrinsics", cl::init(true),
    cl::desc("Instrument memintrinsics (memset/memcpy/memmove)"), cl::Hidden);
/*static cl::opt<bool> ClInstrumentFastpath(
    "esan-instrument-fastpath", cl::init(true),
    cl::desc("Instrument fastpath"), cl::Hidden);
static cl::opt<bool> ClAuxFieldInfo(
    "esan-aux-field-info", cl::init(true),
    cl::desc("Generate binary with auxiliary struct field information"),
    cl::Hidden);*/

// Experiments show that the performance difference can be 2x or more,
// and accuracy loss is typically negligible, so we turn this on by default.
/*static cl::opt<bool> ClAssumeIntraCacheLine(
    "esan-assume-intra-cache-line", cl::init(true),
    cl::desc("Assume each memory access touches just one cache line, for "
             "better performance but with a potential loss of accuracy."),
    cl::Hidden);*/

STATISTIC(NumInstrumentedLoads, "Number of instrumented loads");
STATISTIC(NumInstrumentedStores, "Number of instrumented stores");
STATISTIC(NumFastpaths, "Number of instrumented fastpaths");
STATISTIC(NumAccessesWithIrregularSize,
          "Number of accesses with a size outside our targeted callout sizes");
STATISTIC(NumIgnoredStructs, "Number of ignored structs");
STATISTIC(NumIgnoredGEPs, "Number of ignored GEP instructions");
STATISTIC(NumInstrumentedGEPs, "Number of instrumented GEP instructions");
STATISTIC(NumAssumedIntraCacheLine,
          "Number of accesses assumed to be intra-cache-line");

static const uint64_t HplgstCtorAndDtorPriority = 0;
static const char *const HplgstModuleCtorName = "hplgst.module_ctor";
static const char *const HplgstModuleDtorName = "hplgst.module_dtor";
static const char *const HplgstInitName = "__hplgst_init";
static const char *const HplgstExitName = "__hplgst_exit";

// We need to specify the tool to the runtime earlier than
// the ctor is called in some cases, so we set a global variable.
static const char *const HplgstWhichToolName = "__hplgst_which_tool";


namespace {

static HeapologistOptions
OverrideOptionsFromCL(HeapologistOptions Options) {

  // no options ... yet

  return Options;
}

// Create a constant for Str so that we can pass it to the run-time lib.
static GlobalVariable *createPrivateGlobalForString(Module &M, StringRef Str,
                                                    bool AllowMerging) {
  Constant *StrConst = ConstantDataArray::getString(M.getContext(), Str);
  // We use private linkage for module-local strings. If they can be merged
  // with another one, we set the unnamed_addr attribute.
  GlobalVariable *GV =
    new GlobalVariable(M, StrConst->getType(), true,
                       GlobalValue::PrivateLinkage, StrConst, "");
  if (AllowMerging)
    GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  GV->setAlignment(1);  // Strings may not be merged w/o setting align 1.
  return GV;
}

/// Heapologist: instrument each module to find performance issues.
class Heapologist : public ModulePass {
public:
  Heapologist(
      const HeapologistOptions &Opts = HeapologistOptions())
      : ModulePass(ID), Options(OverrideOptionsFromCL(Opts)) {
  }
  ~Heapologist() {
    type_file.close();
  }
  StringRef getPassName() const override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnModule(Module &M) override;
  static char ID;

private:
  bool initOnModule(Module &M);
  void initializeCallbacks(Module &M);
  Constant *createHplgstInitToolInfoArg(Module &M, const DataLayout &DL);
  void createDestructor(Module &M, Constant *ToolInfoArg);
  bool runOnFunction(Function &F, Module &M);
  bool instrumentLoadOrStore(Instruction *I, const DataLayout &DL);
  bool instrumentMemIntrinsic(MemIntrinsic *MI);
  bool shouldIgnoreMemoryAccess(Instruction *I);
  int getMemoryAccessFuncIndex(Value *Addr, const DataLayout &DL);
  void maybeInstrumentMallocNew(CallInst *CI);
  void instrumentMallocNew(CallInst *CI, StringRef const& name);

  HeapologistOptions Options;
  LLVMContext *Ctx;
  Type *IntptrTy;
  // Our slowpath involves callouts to the runtime library.
  // Access sizes are powers of two: 1, 2, 4, 8, 16.
  static const size_t NumberOfAccessSizes = 5;
  Function *HplgstAlignedLoad[NumberOfAccessSizes];
  Function *HplgstAlignedStore[NumberOfAccessSizes];
  Function *HplgstUnalignedLoad[NumberOfAccessSizes];
  Function *HplgstUnalignedStore[NumberOfAccessSizes];
  // For irregular sizes of any alignment:
  Function *HplgstUnalignedLoadN, *HplgstUnalignedStoreN;
  Function *MemmoveFn, *MemcpyFn, *MemsetFn;
  Function *HplgstCtorFunction;
  Function *HplgstDtorFunction;
  // file we will dump alloc point type info to
  std::ofstream type_file;
};
} // namespace

char Heapologist::ID = 0;
INITIALIZE_PASS_BEGIN(
    Heapologist, "hplgst",
    "Heapologist: finds heap performance issues.", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(
    Heapologist, "hplgst",
    "Heapologist: finds heap performance issues.", false, false)

StringRef Heapologist::getPassName() const {
  return "Heapologist";
}

void Heapologist::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetLibraryInfoWrapperPass>();
}

ModulePass *
llvm::createHeapologistPass(const HeapologistOptions &Options) {
  return new Heapologist(Options);
}

void Heapologist::initializeCallbacks(Module &M) {
  IRBuilder<> IRB(M.getContext());
  // Initialize the callbacks.
  for (size_t Idx = 0; Idx < NumberOfAccessSizes; ++Idx) {
    const unsigned ByteSize = 1U << Idx;
    std::string ByteSizeStr = utostr(ByteSize);
    // We'll inline the most common (i.e., aligned and frequent sizes)
    // load + store instrumentation: these callouts are for the slowpath.
    SmallString<32> AlignedLoadName("__hplgst_aligned_load" + ByteSizeStr);
    HplgstAlignedLoad[Idx] =
        checkSanitizerInterfaceFunction(M.getOrInsertFunction(
            AlignedLoadName, IRB.getVoidTy(), IRB.getInt8PtrTy(), nullptr));
    SmallString<32> AlignedStoreName("__hplgst_aligned_store" + ByteSizeStr);
    HplgstAlignedStore[Idx] =
        checkSanitizerInterfaceFunction(M.getOrInsertFunction(
            AlignedStoreName, IRB.getVoidTy(), IRB.getInt8PtrTy(), nullptr));
    SmallString<32> UnalignedLoadName("__hplgst_unaligned_load" + ByteSizeStr);
    HplgstUnalignedLoad[Idx] =
        checkSanitizerInterfaceFunction(M.getOrInsertFunction(
            UnalignedLoadName, IRB.getVoidTy(), IRB.getInt8PtrTy(), nullptr));
    SmallString<32> UnalignedStoreName("__hplgst_unaligned_store" + ByteSizeStr);
    HplgstUnalignedStore[Idx] =
        checkSanitizerInterfaceFunction(M.getOrInsertFunction(
            UnalignedStoreName, IRB.getVoidTy(), IRB.getInt8PtrTy(), nullptr));
  }
  HplgstUnalignedLoadN = checkSanitizerInterfaceFunction(
      M.getOrInsertFunction("__hplgst_unaligned_loadN", IRB.getVoidTy(),
                            IRB.getInt8PtrTy(), IntptrTy, nullptr));
  HplgstUnalignedStoreN = checkSanitizerInterfaceFunction(
      M.getOrInsertFunction("__hplgst_unaligned_storeN", IRB.getVoidTy(),
                            IRB.getInt8PtrTy(), IntptrTy, nullptr));
  MemmoveFn = checkSanitizerInterfaceFunction(
      M.getOrInsertFunction("memmove", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
                            IRB.getInt8PtrTy(), IntptrTy, nullptr));
  MemcpyFn = checkSanitizerInterfaceFunction(
      M.getOrInsertFunction("memcpy", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
                            IRB.getInt8PtrTy(), IntptrTy, nullptr));
  MemsetFn = checkSanitizerInterfaceFunction(
      M.getOrInsertFunction("memset", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
                            IRB.getInt32Ty(), IntptrTy, nullptr));
}


// Create global variables with auxiliary information (e.g., struct field size,
// offset, and type name) for better user report.


// Create the tool-specific argument passed to HplgstInit and HplgstExit.
// TODO can probably get rid of this,
Constant *Heapologist::createHplgstInitToolInfoArg(Module &M,
                                                         const DataLayout &DL) {
  // This structure contains tool-specific information about each compilation
  // unit (module) and is passed to the runtime library.
  GlobalVariable *ToolInfoGV = nullptr;

  auto *Int8PtrTy = Type::getInt8PtrTy(*Ctx);
  // Compilation unit name.
  auto *UnitName = ConstantExpr::getPointerCast(
      createPrivateGlobalForString(M, M.getModuleIdentifier(), true),
      Int8PtrTy);


  if (ToolInfoGV != nullptr)
    return ConstantExpr::getPointerCast(ToolInfoGV, Int8PtrTy);

  // Create the null pointer if no tool-specific variable created.
  return ConstantPointerNull::get(Int8PtrTy);
}

void Heapologist::createDestructor(Module &M, Constant *ToolInfoArg) {
  PointerType *Int8PtrTy = Type::getInt8PtrTy(*Ctx);
  HplgstDtorFunction = Function::Create(FunctionType::get(Type::getVoidTy(*Ctx),
                                                        false),
                                      GlobalValue::InternalLinkage,
                                      HplgstModuleDtorName, &M);
  ReturnInst::Create(*Ctx, BasicBlock::Create(*Ctx, "", HplgstDtorFunction));
  IRBuilder<> IRB_Dtor(HplgstDtorFunction->getEntryBlock().getTerminator());
  Function *HplgstExit = checkSanitizerInterfaceFunction(
      M.getOrInsertFunction(HplgstExitName, IRB_Dtor.getVoidTy(),
                            Int8PtrTy, nullptr));
  HplgstExit->setLinkage(Function::ExternalLinkage);
  IRB_Dtor.CreateCall(HplgstExit, {ToolInfoArg});
  appendToGlobalDtors(M, HplgstDtorFunction, HplgstCtorAndDtorPriority);
}

bool Heapologist::initOnModule(Module &M) {


  mkdir("typefiles", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); // just needs to exist, dont care if it fails
  SmallString<64> dir("typefiles/");
  SmallString<4096> filename(M.getName().str() + ".types");
  // form a unique flattened name
  std::replace(filename.begin(), filename.end(), '/', '.');

  type_file.open((dir + filename).str(), std::ios::out);

  Ctx = &M.getContext();
  const DataLayout &DL = M.getDataLayout();
  IRBuilder<> IRB(M.getContext());
  IntegerType *OrdTy = IRB.getInt32Ty();
  PointerType *Int8PtrTy = Type::getInt8PtrTy(*Ctx);
  IntptrTy = DL.getIntPtrType(M.getContext());
  // Create the variable passed to HplgstInit and HplgstExit.
  Constant *ToolInfoArg = createHplgstInitToolInfoArg(M, DL);
  // Constructor
  // We specify the tool type both in the HplgstWhichToolName global
  // and as an arg to the init routine as a sanity check.
  std::tie(HplgstCtorFunction, std::ignore) = createSanitizerCtorAndInitFunctions(
      M, HplgstModuleCtorName, HplgstInitName, /*InitArgTypes=*/{OrdTy, Int8PtrTy},
      /*InitArgs=*/{
        ConstantInt::get(OrdTy, static_cast<int>(Options.test_op)),
        ToolInfoArg});
  appendToGlobalCtors(M, HplgstCtorFunction, HplgstCtorAndDtorPriority);

  createDestructor(M, ToolInfoArg);

  new GlobalVariable(M, OrdTy, true,
                     GlobalValue::WeakAnyLinkage,
                     ConstantInt::get(OrdTy,
                                      static_cast<int>(Options.test_op)),
                     HplgstWhichToolName);

  return true;
}


bool Heapologist::shouldIgnoreMemoryAccess(Instruction *I) {
  /*if (Options.ToolType == HeapologistOptions::ESAN_CacheFrag) {
    // We'd like to know about cache fragmentation in vtable accesses and
    // constant data references, so we do not currently ignore anything.
    return false;
  } else if (Options.ToolType == HeapologistOptions::ESAN_WorkingSet) {
    // TODO: the instrumentation disturbs the data layout on the stack, so we
    // may want to add an option to ignore stack references (if we can
    // distinguish them) to reduce overhead.
  }*/
  // don't ignore anything for now
  return false;
}

bool Heapologist::runOnModule(Module &M) {
  bool Res = initOnModule(M);
  initializeCallbacks(M);
  for (auto &F : M) {
    Res |= runOnFunction(F, M);
  }
  return Res;
}

void Heapologist::instrumentMallocNew(CallInst *CI, StringRef const& name) {

  auto& loc = CI->getDebugLoc();
  auto* diloc = loc.get();
  Type* t = nullptr;
  for (auto it = CI->user_begin(); it != CI->user_end(); it++) {
    if (isa<CastInst>(*it)) {
      if (!t)
        t = it->getType();
      if (t && it->getType()->isStructTy())
        t = it->getType();
    }
  }
  if (t) {
    // we can't use isStructTy here because they are all
    // pointer types
    std::string type_name;
    llvm::raw_string_ostream rso(type_name);
    t->print(rso);
    type_name = rso.str();
    size_t pos = type_name.find_first_of('.');
    if (pos != std::string::npos)
      type_name = type_name.substr(pos+1);
    // remove extraneous `"` sometimes added by LLVM
    type_name.erase(std::remove(type_name.begin(), type_name.end(), '\"'), type_name.end());
    if (diloc->getFilename()[0] == '/') {
      type_file << diloc->getFilename().str() << ":" << diloc->getLine() << ":" << diloc->getColumn() << "|" << type_name << "\n";
    } else {
      type_file << diloc->getDirectory().str() << "/" << diloc->getFilename().str() << ":" << diloc->getLine() << ":" << diloc->getColumn() << "|" << type_name << "\n";
    }
  } else {
    // no cast found means that its basically char*
    if (diloc->getFilename()[0] == '/') {
      type_file << diloc->getFilename().str() << ":" << diloc->getLine() << ":" << diloc->getColumn() << "|" << "i8*" << "\n";
    } else {
      type_file << diloc->getDirectory().str() << "/" << diloc->getFilename().str() << ":" << diloc->getLine() << ":" << diloc->getColumn() << "|" << "i8*" << "\n";
    }
  }
}

// OK technically not instrumenting, should probably change name of this method
void Heapologist::maybeInstrumentMallocNew(CallInst *CI) {
  // saying right now there is probably a better way to do this,
  // like get pointer to alloc functions and compare those?
  // but not sure how to do and this works :-P
  Function *F = CI->getCalledFunction();
  if (!F) {
    return;
  }
  int     status;
  char   *realname;
  realname = abi::__cxa_demangle(F->getName().str().c_str(), 0, 0, &status);
  //errs() << "hplgst found function named " << F->getName() << " with demangled name " << realname << "\n";
  if (F->getName().compare("malloc") == 0 || F->getName().compare("realloc") == 0
      || F->getName().compare("calloc") == 0
      || (realname && strcmp(realname, "operator new[](unsigned long)") == 0)
         || (realname && strcmp(realname, "operator new(unsigned long)") == 0)){
    //errs() << "hplgst found function named " << F->getName() << " with demangled name " << realname << "\n";
    if (realname) {
      StringRef name(realname);
      instrumentMallocNew(CI, name);
    } else {
      instrumentMallocNew(CI, F->getName());
    }
  }
  free(realname);
}

bool Heapologist::runOnFunction(Function &F, Module &M) {
  // This is required to prevent instrumenting the call to __hplgst_init from
  // within the module constructor.
  //errs() << "running heapologist instrumenter on function!\n";
  if (&F == HplgstCtorFunction)
    return false;
  SmallVector<Instruction *, 8> LoadsAndStores;
  SmallVector<Instruction *, 8> MemIntrinCalls;
  SmallVector<Instruction *, 8> GetElementPtrs;
  bool Res = false;
  const DataLayout &DL = M.getDataLayout();
  const TargetLibraryInfo *TLI =
      &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();

  for (auto &BB : F) {
    for (auto &Inst : BB) {
      if ((isa<LoadInst>(Inst) || isa<StoreInst>(Inst) ||
           isa<AtomicRMWInst>(Inst) || isa<AtomicCmpXchgInst>(Inst)) &&
          !shouldIgnoreMemoryAccess(&Inst))
        LoadsAndStores.push_back(&Inst);
      else if (isa<MemIntrinsic>(Inst))
        MemIntrinCalls.push_back(&Inst);
      else if (isa<GetElementPtrInst>(Inst))
        GetElementPtrs.push_back(&Inst);
      else if (CallInst *CI = dyn_cast<CallInst>(&Inst)) {
        maybeMarkSanitizerLibraryCallNoBuiltin(CI, TLI);
        maybeInstrumentMallocNew(CI);
      }
    }
  }

  if (ClInstrumentLoadsAndStores) {
    for (auto Inst : LoadsAndStores) {
      Res |= instrumentLoadOrStore(Inst, DL);
    }
  }

  if (ClInstrumentMemIntrinsics) {
    for (auto Inst : MemIntrinCalls) {
      Res |= instrumentMemIntrinsic(cast<MemIntrinsic>(Inst));
    }
  }

  return Res;
}

bool Heapologist::instrumentLoadOrStore(Instruction *I,
                                                const DataLayout &DL) {
  IRBuilder<> IRB(I);
  bool IsStore;
  Value *Addr;
  unsigned Alignment;
  if (LoadInst *Load = dyn_cast<LoadInst>(I)) {
    IsStore = false;
    Alignment = Load->getAlignment();
    Addr = Load->getPointerOperand();
  } else if (StoreInst *Store = dyn_cast<StoreInst>(I)) {
    IsStore = true;
    Alignment = Store->getAlignment();
    Addr = Store->getPointerOperand();
  } else if (AtomicRMWInst *RMW = dyn_cast<AtomicRMWInst>(I)) {
    IsStore = true;
    Alignment = 0;
    Addr = RMW->getPointerOperand();
  } else if (AtomicCmpXchgInst *Xchg = dyn_cast<AtomicCmpXchgInst>(I)) {
    IsStore = true;
    Alignment = 0;
    Addr = Xchg->getPointerOperand();
  } else
    llvm_unreachable("Unsupported mem access type");

  Type *OrigTy = cast<PointerType>(Addr->getType())->getElementType();
  const uint32_t TypeSizeBytes = DL.getTypeStoreSizeInBits(OrigTy) / 8;
  Value *OnAccessFunc = nullptr;

  // Convert 0 to the default alignment.
  if (Alignment == 0)
    Alignment = DL.getPrefTypeAlignment(OrigTy);

  if (IsStore)
    NumInstrumentedStores++;
  else
    NumInstrumentedLoads++;
  int Idx = getMemoryAccessFuncIndex(Addr, DL);
  if (Idx < 0) {
    OnAccessFunc = IsStore ? HplgstUnalignedStoreN : HplgstUnalignedLoadN;
    IRB.CreateCall(OnAccessFunc,
                   {IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy()),
                    ConstantInt::get(IntptrTy, TypeSizeBytes)});
  } else {
    if (Alignment == 0 || (Alignment % TypeSizeBytes) == 0)
      OnAccessFunc = IsStore ? HplgstAlignedStore[Idx] : HplgstAlignedLoad[Idx];
    else
      OnAccessFunc = IsStore ? HplgstUnalignedStore[Idx] : HplgstUnalignedLoad[Idx];
    IRB.CreateCall(OnAccessFunc,
                   IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy()));
  }
  return true;
}

// It's simplest to replace the memset/memmove/memcpy intrinsics with
// calls that the runtime library intercepts.
// Our pass is late enough that calls should not turn back into intrinsics.
bool Heapologist::instrumentMemIntrinsic(MemIntrinsic *MI) {
  IRBuilder<> IRB(MI);
  bool Res = false;
  if (isa<MemSetInst>(MI)) {
    IRB.CreateCall(
        MemsetFn,
        {IRB.CreatePointerCast(MI->getArgOperand(0), IRB.getInt8PtrTy()),
         IRB.CreateIntCast(MI->getArgOperand(1), IRB.getInt32Ty(), false),
         IRB.CreateIntCast(MI->getArgOperand(2), IntptrTy, false)});
    MI->eraseFromParent();
    Res = true;
  } else if (isa<MemTransferInst>(MI)) {
    IRB.CreateCall(
        isa<MemCpyInst>(MI) ? MemcpyFn : MemmoveFn,
        {IRB.CreatePointerCast(MI->getArgOperand(0), IRB.getInt8PtrTy()),
         IRB.CreatePointerCast(MI->getArgOperand(1), IRB.getInt8PtrTy()),
         IRB.CreateIntCast(MI->getArgOperand(2), IntptrTy, false)});
    MI->eraseFromParent();
    Res = true;
  } else
    llvm_unreachable("Unsupported mem intrinsic type");
  return Res;
}



int Heapologist::getMemoryAccessFuncIndex(Value *Addr,
                                                  const DataLayout &DL) {
  Type *OrigPtrTy = Addr->getType();
  Type *OrigTy = cast<PointerType>(OrigPtrTy)->getElementType();
  assert(OrigTy->isSized());
  // The size is always a multiple of 8.
  uint32_t TypeSizeBytes = DL.getTypeStoreSizeInBits(OrigTy) / 8;
  if (TypeSizeBytes != 1 && TypeSizeBytes != 2 && TypeSizeBytes != 4 &&
      TypeSizeBytes != 8 && TypeSizeBytes != 16) {
    // Irregular sizes do not have per-size call targets.
    NumAccessesWithIrregularSize++;
    return -1;
  }
  size_t Idx = countTrailingZeros(TypeSizeBytes);
  assert(Idx < NumberOfAccessSizes);
  return Idx;
}



