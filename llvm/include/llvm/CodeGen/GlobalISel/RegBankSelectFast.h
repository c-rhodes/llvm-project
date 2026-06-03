//===- llvm/CodeGen/GlobalISel/RegBankSelectFast.h --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes a trivial fast register bank selector for GlobalISel.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_REGBANKSELECTFAST_H
#define LLVM_CODEGEN_GLOBALISEL_REGBANKSELECTFAST_H

#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class LLVM_ABI RegBankSelectFast : public MachineFunctionPass {
public:
  static char ID;

  RegBankSelectFast() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "RegBankSelectFast"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setIsSSA().setLegalized();
  }

  MachineFunctionProperties getSetProperties() const override {
    return MachineFunctionProperties().setRegBankSelected();
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // end namespace llvm

#endif
