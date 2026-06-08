//===- llvm/CodeGen/GlobalISel/TypeBasedRegBankSelect.h ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file describes a RegBankSelect replacement that assigns register banks
/// from virtual-register type information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_TYPEBASEDREGBANKSELECT_H
#define LLVM_CODEGEN_GLOBALISEL_TYPEBASEDREGBANKSELECT_H

#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {

class LLVM_ABI TypeBasedRegBankSelect : public MachineFunctionPass {
public:
  static char ID;

  TypeBasedRegBankSelect();

  StringRef getPassName() const override { return "TypeBasedRegBankSelect"; }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setIsSSA().setLegalized();
  }

  MachineFunctionProperties getSetProperties() const override {
    return MachineFunctionProperties().setRegBankSelected();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // end namespace llvm

#endif // LLVM_CODEGEN_GLOBALISEL_TYPEBASEDREGBANKSELECT_H
