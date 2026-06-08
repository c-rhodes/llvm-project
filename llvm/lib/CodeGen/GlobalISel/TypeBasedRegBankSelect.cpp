//===- llvm/CodeGen/GlobalISel/TypeBasedRegBankSelect.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file implements a RegBankSelect replacement that assigns register
/// banks from virtual-register type information.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/TypeBasedRegBankSelect.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterBank.h"
#include "llvm/CodeGen/RegisterBankInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "type-based-regbankselect"

char TypeBasedRegBankSelect::ID = 0;

INITIALIZE_PASS(TypeBasedRegBankSelect, DEBUG_TYPE, "Type-based RegBankSelect",
                false, false)

TypeBasedRegBankSelect::TypeBasedRegBankSelect() : MachineFunctionPass(ID) {
  initializeTypeBasedRegBankSelectPass(*PassRegistry::getPassRegistry());
}

void TypeBasedRegBankSelect::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  getSelectionDAGFallbackAnalysisUsage(AU);
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool TypeBasedRegBankSelect::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasFailedISel())
    return false;

  MachineRegisterInfo &MRI = MF.getRegInfo();
  const RegisterBankInfo *RBI = MF.getSubtarget().getRegBankInfo();
  assert(RBI && "Cannot assign register banks without RegisterBankInfo");

  LLVM_DEBUG(dbgs() << "Assign register banks from type information for: "
                    << MF.getName() << '\n');

  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (MRI.getRegClassOrRegBank(Reg))
      continue;

    const RegisterBank *RB = RBI->getRegBankForType(MRI.getType(Reg));
    if (!RB)
      continue;

    MRI.setRegBank(Reg, *RB);
    LLVM_DEBUG(dbgs() << "  " << printReg(Reg) << " -> " << *RB << '\n');
  }

  return false;
}
