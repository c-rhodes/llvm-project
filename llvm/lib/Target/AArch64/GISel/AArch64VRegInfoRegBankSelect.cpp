//===- AArch64VRegInfoRegBankSelect.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Experimental AArch64 RegBankSelect replacement that assigns register banks
// from virtual-register type information only.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "GISel/AArch64RegisterBankInfo.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterBank.h"
#include "llvm/CodeGen/RegisterBankInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/CodeGenTypes/LowLevelType.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-vreginfo-regbankselect"

namespace {

class AArch64VRegInfoRegBankSelect : public MachineFunctionPass {
public:
  static char ID;

  AArch64VRegInfoRegBankSelect() : MachineFunctionPass(ID) {
    initializeAArch64VRegInfoRegBankSelectPass(
        *PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "AArch64 VRegInfo RegBankSelect";
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setIsSSA().setLegalized();
  }

  MachineFunctionProperties getSetProperties() const override {
    return MachineFunctionProperties().setRegBankSelected();
  }

  MachineFunctionProperties getClearedProperties() const override {
    return MachineFunctionProperties().setNoPHIs();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    getSelectionDAGFallbackAnalysisUsage(AU);
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  const RegisterBank *getBankForType(LLT Ty, const RegisterBank &GPRBank,
                                     const RegisterBank &FPRBank) const;
};

} // end anonymous namespace

char AArch64VRegInfoRegBankSelect::ID = 0;

INITIALIZE_PASS(AArch64VRegInfoRegBankSelect, DEBUG_TYPE,
                "AArch64 VRegInfo RegBankSelect", false, false)

const RegisterBank *AArch64VRegInfoRegBankSelect::getBankForType(
    LLT Ty, const RegisterBank &GPRBank, const RegisterBank &FPRBank) const {
  if (!Ty.isValid())
    return nullptr;

  if (Ty.isVector() || Ty.isFloatOrFloatVector() ||
      Ty.getSizeInBits().getKnownMinValue() > 64)
    return &FPRBank;

  if (Ty.isPointer() || Ty.isScalar())
    return &GPRBank;

  return nullptr;
}

bool AArch64VRegInfoRegBankSelect::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasFailedISel())
    return false;

  MachineRegisterInfo &MRI = MF.getRegInfo();
  const RegisterBankInfo &RBI = *MF.getSubtarget().getRegBankInfo();
  const RegisterBank &GPRBank = RBI.getRegBank(AArch64::GPRRegBankID);
  const RegisterBank &FPRBank = RBI.getRegBank(AArch64::FPRRegBankID);

  LLVM_DEBUG(dbgs() << "Assign register banks from VRegInfo for: "
                    << MF.getName() << '\n');

  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (MRI.getRegClassOrRegBank(Reg))
      continue;

    const RegisterBank *RB = getBankForType(MRI.getType(Reg), GPRBank, FPRBank);
    if (!RB)
      continue;

    MRI.setRegBank(Reg, *RB);
    LLVM_DEBUG(dbgs() << "  " << printReg(Reg) << " -> " << *RB << '\n');
  }

  return false;
}

FunctionPass *llvm::createAArch64VRegInfoRegBankSelectPass() {
  return new AArch64VRegInfoRegBankSelect();
}
