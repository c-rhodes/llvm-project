//===- ArmSMETypeConverter.cpp - Convert builtin to LLVM dialect types ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/ArmSME/Utils/Utils.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

using namespace mlir;
arm_sme::ArmSMETypeConverter::ArmSMETypeConverter(
    MLIRContext *ctx, const LowerToLLVMOptions &options)
    : LLVMTypeConverter(ctx, options) {
  // Disable LLVM type conversion for vectors. This is to prevent 2-d scalable
  // vectors (common in the context of ArmSME), e.g.
  //    `vector<[16]x[16]xi8>`,
  // entering the LLVM Type converter. LLVM does not support arrays of scalable
  // vectors, but in the case of SME such types are effectively eliminated when
  // emitting ArmSME LLVM IR intrinsics.
  addConversion([&](VectorType type) -> Type {
    if (arm_sme::isValidSMETileVectorType(type)) {
      auto elemType = type.getElementType();
      if (elemType.isInteger(8))
        return LLVM::LLVMTargetExtType::get(type.getContext(), "aarch64.za.b",
                                            std::nullopt, std::nullopt);
      else if (elemType.isInteger(16) || elemType.isF16() || elemType.isBF16())
        return LLVM::LLVMTargetExtType::get(type.getContext(), "aarch64.za.h",
                                            std::nullopt, std::nullopt);
      else if (elemType.isInteger(32) || elemType.isF32())
        return LLVM::LLVMTargetExtType::get(type.getContext(), "aarch64.za.s",
                                            std::nullopt, std::nullopt);
      else if (elemType.isInteger(64) || elemType.isF64())
        return LLVM::LLVMTargetExtType::get(type.getContext(), "aarch64.za.d",
                                            std::nullopt, std::nullopt);
      else if (elemType.isInteger(128) || elemType.isF128())
        return LLVM::LLVMTargetExtType::get(type.getContext(), "aarch64.za.q",
                                            std::nullopt, std::nullopt);
      llvm::outs() << type << "\n";
      llvm_unreachable("unexpected type!");
    }
    return type;
  });
}
