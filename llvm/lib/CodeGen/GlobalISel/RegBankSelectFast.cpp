//===- llvm/CodeGen/GlobalISel/RegBankSelectFast.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements a trivial fast register bank selector.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/RegBankSelectFast.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/RegBankSelect.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterBankInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "regbankselectfast"

using namespace llvm;

namespace {

class FallbackRegBankSelect final : public RegBankSelect {
public:
  bool runOnCurrentFunction(MachineFunction &MF) {
    init(MF);
    return assignRegisterBanks(MF);
  }
};

} // end anonymous namespace

char RegBankSelectFast::ID = 0;
INITIALIZE_PASS_BEGIN(RegBankSelectFast, DEBUG_TYPE,
                      "Fast register bank selection", false, false)
INITIALIZE_PASS_END(RegBankSelectFast, DEBUG_TYPE,
                    "Fast register bank selection", false, false)

void RegBankSelectFast::getAnalysisUsage(AnalysisUsage &AU) const {
  getSelectionDAGFallbackAnalysisUsage(AU);
  MachineFunctionPass::getAnalysisUsage(AU);
}

static bool assignInstr(MachineInstr &MI, MachineIRBuilder &MIRBuilder,
                        const RegisterBankInfo &RBI,
                        const TargetRegisterInfo &TRI,
                        MachineRegisterInfo &MRI) {
  LLVM_DEBUG(dbgs() << "Assign: " << MI);

  const RegisterBankInfo::InstructionMapping &Mapping = RBI.getInstrMapping(MI);
  for (unsigned OpIdx = 0, End = Mapping.getNumOperands(); OpIdx != End;
       ++OpIdx) {
    MachineOperand &MO = MI.getOperand(OpIdx);
    if (!MO.isReg())
      continue;
    Register Reg = MO.getReg();
    if (!Reg || Reg.isPhysical())
      continue;

    const RegisterBankInfo::ValueMapping &ValMapping =
        Mapping.getOperandMapping(OpIdx);
    if (!ValMapping.isValid() || ValMapping.NumBreakDowns != 1 ||
        !ValMapping.BreakDown[0].RegBank)
      return false;

    const RegisterBank *CurrentRB = RBI.getRegBank(Reg, MRI, TRI);
    const RegisterBank *DesiredRB = ValMapping.BreakDown[0].RegBank;
    // RegBankSelectFast only assigns previously-unbanked regs. If a default
    // mapping wants a different bank than one already assigned, fall back to
    // full RegBankSelect so it can repair the conflict.
    if (CurrentRB && CurrentRB != DesiredRB &&
        Mapping.getID() == RegisterBankInfo::DefaultMappingID)
      return false;
    if (!CurrentRB)
      MRI.setRegBank(Reg, *DesiredRB);
  }

  if (Mapping.getID() != RegisterBankInfo::DefaultMappingID) {
    RegisterBankInfo::OperandsMapper OpdMapper(MI, Mapping, MRI);
    RBI.applyMapping(MIRBuilder, OpdMapper);
  }

  return true;
}

bool RegBankSelectFast::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasFailedISel())
    return false;

  const RegisterBankInfo &RBI = *MF.getSubtarget().getRegBankInfo();
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();

  MachineRegisterInfo &MRI = MF.getRegInfo();
  MachineIRBuilder MIRBuilder(MF);

  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  for (MachineBasicBlock *MBB : RPOT) {
    MIRBuilder.setMBB(*MBB);
    SmallVector<MachineInstr *> WorkList(
        make_pointer_range(reverse(MBB->instrs())));

    while (!WorkList.empty()) {
      MachineInstr &MI = *WorkList.pop_back_val();

      // Ignore target-specific post-isel instructions: they should use proper
      // regclasses.
      if (isTargetSpecificOpcode(MI.getOpcode()) && !MI.isPreISelOpcode())
        continue;

      // Ignore inline asm instructions: they should use physical
      // registers/regclasses
      if (MI.isInlineAsm())
        continue;

      // Ignore IMPLICIT_DEF which must have a regclass.
      if (MI.isImplicitDef())
        continue;

      if (!assignInstr(MI, MIRBuilder, RBI, TRI, MRI)) {
        LLVM_DEBUG(dbgs() << "Falling back to full RegBankSelect for "
                          << MF.getName() << " after failing on: " << MI);
        FallbackRegBankSelect FallbackRBS;
        return FallbackRBS.runOnCurrentFunction(MF);
      }
    }
  }

  return true;
}
