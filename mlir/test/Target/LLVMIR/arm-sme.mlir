// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// CHECK-LABEL: @arm_sme_zero
llvm.func @arm_sme_zero() {
  %c0 = llvm.mlir.constant(0 : index) : i32
  // CHECK: call void @llvm.aarch64.sme.zero(i32 0)
  "arm_sme.intr.zero"(%c0) : (i32) -> ()
  llvm.return
}

// -----

// CHECK-LABEL: @arm_sme_fmopa
llvm.func @arm_sme_fmopa(%nxv2f64 : vector<[2]xf64>,
                         %nxv4f32 : vector<[4]xf32>,
                         %nxv8f16 : vector<[8]xf16>,
                         %nxv8bf16: vector<[8]xbf16>,
                         %nxv2i1  : vector<[2]xi1>,
                         %nxv4i1  : vector<[4]xi1>,
                         %nxv8i1  : vector<[8]xi1>) {
  %c0 = llvm.mlir.constant(0 : index) : i32
  // CHECK: call void @llvm.aarch64.sme.mopa.nxv2f64
  "arm_sme.intr.mopa"(%c0, %nxv2i1, %nxv2i1, %nxv2f64, %nxv2f64) :
    (i32, vector<[2]xi1>, vector<[2]xi1>, vector<[2]xf64>, vector<[2]xf64>) -> ()
  // CHECK: call void @llvm.aarch64.sme.mopa.nxv4f32
  "arm_sme.intr.mopa"(%c0, %nxv4i1, %nxv4i1, %nxv4f32, %nxv4f32) :
    (i32, vector<[4]xi1>, vector<[4]xi1>, vector<[4]xf32>, vector<[4]xf32>) -> ()
  // CHECK: call void @llvm.aarch64.sme.mopa.wide.nxv8f16
  "arm_sme.intr.mopa.wide"(%c0, %nxv8i1, %nxv8i1, %nxv8f16, %nxv8f16) :
    (i32, vector<[8]xi1>, vector<[8]xi1>, vector<[8]xf16>, vector<[8]xf16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.mopa.wide.nxv8bf16
  "arm_sme.intr.mopa.wide"(%c0, %nxv8i1, %nxv8i1, %nxv8bf16, %nxv8bf16) :
    (i32, vector<[8]xi1>, vector<[8]xi1>, vector<[8]xbf16>, vector<[8]xbf16>) -> ()
  llvm.return
}

// -----

// CHECK-LABEL: @arm_sme_imopa
llvm.func @arm_sme_imopa(%nxv8i16 : vector<[8]xi16>,
                         %nxv16i8 : vector<[16]xi8>,
                         %nxv8i1  : vector<[8]xi1>,
                         %nxv16i1 : vector<[16]xi1>) {
  %c0 = llvm.mlir.constant(0 : index) : i32
  // CHECK: call void @llvm.aarch64.sme.smopa.wide.nxv8i16
  "arm_sme.intr.smopa.wide"(%c0, %nxv8i1, %nxv8i1, %nxv8i16, %nxv8i16) :
    (i32, vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.umopa.wide.nxv8i16
  "arm_sme.intr.umopa.wide"(%c0, %nxv8i1, %nxv8i1, %nxv8i16, %nxv8i16) :
    (i32, vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.sumopa.wide.nxv8i16
  "arm_sme.intr.sumopa.wide"(%c0, %nxv8i1, %nxv8i1, %nxv8i16, %nxv8i16) :
    (i32, vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.usmopa.wide.nxv8i16
  "arm_sme.intr.usmopa.wide"(%c0, %nxv8i1, %nxv8i1, %nxv8i16, %nxv8i16) :
    (i32, vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.smopa.wide.nxv16i8
  "arm_sme.intr.smopa.wide"(%c0, %nxv16i1, %nxv16i1, %nxv16i8, %nxv16i8) :
    (i32, vector<[16]xi1>, vector<[16]xi1>, vector<[16]xi8>, vector<[16]xi8>) -> ()
  // CHECK: call void @llvm.aarch64.sme.umopa.wide.nxv16i8
  "arm_sme.intr.umopa.wide"(%c0, %nxv16i1, %nxv16i1, %nxv16i8, %nxv16i8) :
    (i32, vector<[16]xi1>, vector<[16]xi1>, vector<[16]xi8>, vector<[16]xi8>) -> ()
  // CHECK: call void @llvm.aarch64.sme.sumopa.wide.nxv16i8
  "arm_sme.intr.sumopa.wide"(%c0, %nxv16i1, %nxv16i1, %nxv16i8, %nxv16i8) :
    (i32, vector<[16]xi1>, vector<[16]xi1>, vector<[16]xi8>, vector<[16]xi8>) -> ()
  // CHECK: call void @llvm.aarch64.sme.usmopa.wide.nxv16i8
  "arm_sme.intr.usmopa.wide"(%c0, %nxv16i1, %nxv16i1, %nxv16i8, %nxv16i8) :
    (i32, vector<[16]xi1>, vector<[16]xi1>, vector<[16]xi8>, vector<[16]xi8>) -> ()
  llvm.return
}

// -----

// CHECK-LABEL: @arm_sme_fmops
llvm.func @arm_sme_fmops(%nxv2f64 : vector<[2]xf64>,
                         %nxv4f32 : vector<[4]xf32>,
                         %nxv8f16 : vector<[8]xf16>,
                         %nxv8bf16: vector<[8]xbf16>,
                         %nxv2i1  : vector<[2]xi1>,
                         %nxv4i1  : vector<[4]xi1>,
                         %nxv8i1  : vector<[8]xi1>) {
  %c0 = llvm.mlir.constant(0 : index) : i32
  // CHECK: call void @llvm.aarch64.sme.mops.nxv2f64
  "arm_sme.intr.mops"(%c0, %nxv2i1, %nxv2i1, %nxv2f64, %nxv2f64) :
    (i32, vector<[2]xi1>, vector<[2]xi1>, vector<[2]xf64>, vector<[2]xf64>) -> ()
  // CHECK: call void @llvm.aarch64.sme.mops.nxv4f32
  "arm_sme.intr.mops"(%c0, %nxv4i1, %nxv4i1, %nxv4f32, %nxv4f32) :
    (i32, vector<[4]xi1>, vector<[4]xi1>, vector<[4]xf32>, vector<[4]xf32>) -> ()
  // CHECK: call void @llvm.aarch64.sme.mops.wide.nxv8f16
  "arm_sme.intr.mops.wide"(%c0, %nxv8i1, %nxv8i1, %nxv8f16, %nxv8f16) :
    (i32, vector<[8]xi1>, vector<[8]xi1>, vector<[8]xf16>, vector<[8]xf16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.mops.wide.nxv8bf16
  "arm_sme.intr.mops.wide"(%c0, %nxv8i1, %nxv8i1, %nxv8bf16, %nxv8bf16) :
    (i32, vector<[8]xi1>, vector<[8]xi1>, vector<[8]xbf16>, vector<[8]xbf16>) -> ()
  llvm.return
}

// -----

// CHECK-LABEL: @arm_sme_imops
llvm.func @arm_sme_imops(%nxv8i16 : vector<[8]xi16>,
                         %nxv16i8 : vector<[16]xi8>,
                         %nxv8i1  : vector<[8]xi1>,
                         %nxv16i1 : vector<[16]xi1>) {
  %c0 = llvm.mlir.constant(0 : index) : i32
  // CHECK: call void @llvm.aarch64.sme.smops.wide.nxv8i16
  "arm_sme.intr.smops.wide"(%c0, %nxv8i1, %nxv8i1, %nxv8i16, %nxv8i16) :
    (i32, vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.umops.wide.nxv8i16
  "arm_sme.intr.umops.wide"(%c0, %nxv8i1, %nxv8i1, %nxv8i16, %nxv8i16) :
    (i32, vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.sumops.wide.nxv8i16
  "arm_sme.intr.sumops.wide"(%c0, %nxv8i1, %nxv8i1, %nxv8i16, %nxv8i16) :
    (i32, vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.usmops.wide.nxv8i16
  "arm_sme.intr.usmops.wide"(%c0, %nxv8i1, %nxv8i1, %nxv8i16, %nxv8i16) :
    (i32, vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.smops.wide.nxv16i8
  "arm_sme.intr.smops.wide"(%c0, %nxv16i1, %nxv16i1, %nxv16i8, %nxv16i8) :
    (i32, vector<[16]xi1>, vector<[16]xi1>, vector<[16]xi8>, vector<[16]xi8>) -> ()
  // CHECK: call void @llvm.aarch64.sme.umops.wide.nxv16i8
  "arm_sme.intr.umops.wide"(%c0, %nxv16i1, %nxv16i1, %nxv16i8, %nxv16i8) :
    (i32, vector<[16]xi1>, vector<[16]xi1>, vector<[16]xi8>, vector<[16]xi8>) -> ()
  // CHECK: call void @llvm.aarch64.sme.sumops.wide.nxv16i8
  "arm_sme.intr.sumops.wide"(%c0, %nxv16i1, %nxv16i1, %nxv16i8, %nxv16i8) :
    (i32, vector<[16]xi1>, vector<[16]xi1>, vector<[16]xi8>, vector<[16]xi8>) -> ()
  // CHECK: call void @llvm.aarch64.sme.usmops.wide.nxv16i8
  "arm_sme.intr.usmops.wide"(%c0, %nxv16i1, %nxv16i1, %nxv16i8, %nxv16i8) :
    (i32, vector<[16]xi1>, vector<[16]xi1>, vector<[16]xi8>, vector<[16]xi8>) -> ()
  llvm.return
}

// -----

// CHECK-LABEL: @arm_sme_load
llvm.func @arm_sme_load(%nxv1i1  : vector<[1]xi1>,
                        %nxv2i1  : vector<[2]xi1>,
                        %nxv4i1  : vector<[4]xi1>,
                        %nxv8i1  : vector<[8]xi1>,
                        %nxv16i1 : vector<[16]xi1>,
                        %tile_q : !llvm.target<"aarch64.za.q">,
                        %tile_d : !llvm.target<"aarch64.za.d">,
                        %tile_s : !llvm.target<"aarch64.za.s">,
                        %tile_h : !llvm.target<"aarch64.za.h">,
                        %tile_b : !llvm.target<"aarch64.za.b">,
                        %ptr    : !llvm.ptr) {
  %c0 = llvm.mlir.constant(0 : index) : i32
  // CHECK: call target("aarch64.za.q") @llvm.aarch64.sme.x.ld1q.horiz
  "arm_sme.intr.x.ld1q.horiz"(%tile_q, %nxv1i1, %ptr, %c0) :
    (!llvm.target<"aarch64.za.q">, vector<[1]xi1>, !llvm.ptr, i32) -> !llvm.target<"aarch64.za.q">
  // CHECK: call target("aarch64.za.d") @llvm.aarch64.sme.x.ld1d.horiz
  "arm_sme.intr.x.ld1d.horiz"(%tile_d, %nxv2i1, %ptr, %c0) :
    (!llvm.target<"aarch64.za.d">, vector<[2]xi1>, !llvm.ptr, i32) -> !llvm.target<"aarch64.za.d">
  // CHECK: call target("aarch64.za.s") @llvm.aarch64.sme.x.ld1w.horiz
  "arm_sme.intr.x.ld1w.horiz"(%tile_s, %nxv4i1, %ptr, %c0) :
    (!llvm.target<"aarch64.za.s">, vector<[4]xi1>, !llvm.ptr, i32) -> !llvm.target<"aarch64.za.s">
  // CHECK: call target("aarch64.za.h") @llvm.aarch64.sme.x.ld1h.horiz
  "arm_sme.intr.x.ld1h.horiz"(%tile_h, %nxv8i1, %ptr, %c0) :
    (!llvm.target<"aarch64.za.h">, vector<[8]xi1>, !llvm.ptr, i32) -> !llvm.target<"aarch64.za.h">
  // CHECK: call target("aarch64.za.b") @llvm.aarch64.sme.x.ld1b.horiz
  "arm_sme.intr.x.ld1b.horiz"(%tile_b, %nxv16i1, %ptr, %c0) :
    (!llvm.target<"aarch64.za.b">, vector<[16]xi1>, !llvm.ptr, i32) -> !llvm.target<"aarch64.za.b">
  // CHECK: call target("aarch64.za.q") @llvm.aarch64.sme.x.ld1q.vert
  "arm_sme.intr.x.ld1q.vert"(%tile_q, %nxv1i1, %ptr, %c0) :
    (!llvm.target<"aarch64.za.q">, vector<[1]xi1>, !llvm.ptr, i32) -> !llvm.target<"aarch64.za.q">
  // CHECK: call target("aarch64.za.d") @llvm.aarch64.sme.x.ld1d.vert
  "arm_sme.intr.x.ld1d.vert"(%tile_d, %nxv2i1, %ptr, %c0) :
    (!llvm.target<"aarch64.za.d">, vector<[2]xi1>, !llvm.ptr, i32) -> !llvm.target<"aarch64.za.d">
  // CHECK: call target("aarch64.za.s") @llvm.aarch64.sme.x.ld1w.vert
  "arm_sme.intr.x.ld1w.vert"(%tile_s, %nxv4i1, %ptr, %c0) :
    (!llvm.target<"aarch64.za.s">, vector<[4]xi1>, !llvm.ptr, i32) -> !llvm.target<"aarch64.za.s">
  // CHECK: call target("aarch64.za.h") @llvm.aarch64.sme.x.ld1h.vert
  "arm_sme.intr.x.ld1h.vert"(%tile_h, %nxv8i1, %ptr, %c0) :
    (!llvm.target<"aarch64.za.h">, vector<[8]xi1>, !llvm.ptr, i32) -> !llvm.target<"aarch64.za.h">
  // CHECK: call target("aarch64.za.b") @llvm.aarch64.sme.x.ld1b.vert
  "arm_sme.intr.x.ld1b.vert"(%tile_b, %nxv16i1, %ptr, %c0) :
    (!llvm.target<"aarch64.za.b">, vector<[16]xi1>, !llvm.ptr, i32) -> !llvm.target<"aarch64.za.b">
  llvm.return
}

// -----

// CHECK-LABEL: @arm_sme_store
llvm.func @arm_sme_store(%nxv1i1  : vector<[1]xi1>,
                         %nxv2i1  : vector<[2]xi1>,
                         %nxv4i1  : vector<[4]xi1>,
                         %nxv8i1  : vector<[8]xi1>,
                         %nxv16i1 : vector<[16]xi1>,
                         %tile_q : !llvm.target<"aarch64.za.q">,
                         %tile_d : !llvm.target<"aarch64.za.d">,
                         %tile_s : !llvm.target<"aarch64.za.s">,
                         %tile_h : !llvm.target<"aarch64.za.h">,
                         %tile_b : !llvm.target<"aarch64.za.b">,
                         %ptr    : !llvm.ptr) {
  %c0 = llvm.mlir.constant(0 : index) : i32
  // CHECK: call void @llvm.aarch64.sme.x.st1q.horiz
  "arm_sme.intr.x.st1q.horiz"(%nxv1i1, %ptr, %tile_q, %c0) :
              (vector<[1]xi1>, !llvm.ptr, !llvm.target<"aarch64.za.q">, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.x.st1d.horiz
  "arm_sme.intr.x.st1d.horiz"(%nxv2i1, %ptr, %tile_d, %c0) :
              (vector<[2]xi1>, !llvm.ptr, !llvm.target<"aarch64.za.d">, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.x.st1w.horiz
  "arm_sme.intr.x.st1w.horiz"(%nxv4i1, %ptr, %tile_s, %c0) :
              (vector<[4]xi1>, !llvm.ptr, !llvm.target<"aarch64.za.s">, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.x.st1h.horiz
  "arm_sme.intr.x.st1h.horiz"(%nxv8i1, %ptr, %tile_h, %c0) :
              (vector<[8]xi1>, !llvm.ptr, !llvm.target<"aarch64.za.h">, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.x.st1b.horiz
  "arm_sme.intr.x.st1b.horiz"(%nxv16i1, %ptr, %tile_b, %c0) :
              (vector<[16]xi1>, !llvm.ptr, !llvm.target<"aarch64.za.b">, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.x.st1q.vert
  "arm_sme.intr.x.st1q.vert"(%nxv1i1, %ptr, %tile_q, %c0) :
              (vector<[1]xi1>, !llvm.ptr, !llvm.target<"aarch64.za.q">, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.x.st1d.vert
  "arm_sme.intr.x.st1d.vert"(%nxv2i1, %ptr, %tile_d, %c0) :
              (vector<[2]xi1>, !llvm.ptr, !llvm.target<"aarch64.za.d">, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.x.st1w.vert
  "arm_sme.intr.x.st1w.vert"(%nxv4i1, %ptr, %tile_s, %c0) :
              (vector<[4]xi1>, !llvm.ptr, !llvm.target<"aarch64.za.s">, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.x.st1h.vert
  "arm_sme.intr.x.st1h.vert"(%nxv8i1, %ptr, %tile_h, %c0) :
              (vector<[8]xi1>, !llvm.ptr, !llvm.target<"aarch64.za.h">, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.x.st1b.vert
  "arm_sme.intr.x.st1b.vert"(%nxv16i1, %ptr, %tile_b, %c0) :
              (vector<[16]xi1>, !llvm.ptr, !llvm.target<"aarch64.za.b">, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.str
  "arm_sme.intr.str"(%c0, %ptr, %c0) : (i32, !llvm.ptr, i32) -> ()
  llvm.return
}

// -----

// CHECK-LABEL: @arm_sme_vector_to_tile_horiz
llvm.func @arm_sme_vector_to_tile_horiz(%tileslice : i32,
                                        %nxv16i1 : vector<[16]xi1>,
                                        %nxv8i1 : vector<[8]xi1>,
                                        %nxv4i1 : vector<[4]xi1>,
                                        %nxv2i1 : vector<[2]xi1>,
                                        %nxv1i1 : vector<[1]xi1>,
                                        %nxv16i8 : vector<[16]xi8>,
                                        %nxv8i16 : vector<[8]xi16>,
                                        %nxv4i32 : vector<[4]xi32>,
                                        %nxv2i64 : vector<[2]xi64>,
                                        %nxv1i128 : vector<[1]xi128>,
                                        %nxv8f16 : vector<[8]xf16>,
                                        %nxv8bf16 : vector<[8]xbf16>,
                                        %nxv4f32 : vector<[4]xf32>,
                                        %nxv2f64 : vector<[2]xf64>,
                                        %tile_b : !llvm.target<"aarch64.za.b">,
                                        %tile_h : !llvm.target<"aarch64.za.h">,
                                        %tile_s : !llvm.target<"aarch64.za.s">,
                                        %tile_d : !llvm.target<"aarch64.za.d">,
                                        %tile_q : !llvm.target<"aarch64.za.q">) {
  // CHECK: call target("aarch64.za.b") @llvm.aarch64.sme.x.write.horiz.{{.*}}.nxv16i8
  "arm_sme.intr.x.write.horiz"(%tile_b, %tileslice, %nxv16i1, %nxv16i8) :
      (!llvm.target<"aarch64.za.b">, i32, vector<[16]xi1>, vector<[16]xi8>) -> !llvm.target<"aarch64.za.b">
  // CHECK: call target("aarch64.za.h") @llvm.aarch64.sme.x.write.horiz.{{.*}}.nxv8i16
  "arm_sme.intr.x.write.horiz"(%tile_h, %tileslice, %nxv8i1, %nxv8i16) :
      (!llvm.target<"aarch64.za.h">, i32, vector<[8]xi1>, vector<[8]xi16>) -> !llvm.target<"aarch64.za.h">
  // CHECK: call target("aarch64.za.s") @llvm.aarch64.sme.x.write.horiz.{{.*}}.nxv4i32
  "arm_sme.intr.x.write.horiz"(%tile_s, %tileslice, %nxv4i1, %nxv4i32) :
      (!llvm.target<"aarch64.za.s">, i32, vector<[4]xi1>, vector<[4]xi32>) -> !llvm.target<"aarch64.za.s">
  // CHECK: call target("aarch64.za.d") @llvm.aarch64.sme.x.write.horiz.{{.*}}.nxv2i64
  "arm_sme.intr.x.write.horiz"(%tile_d, %tileslice, %nxv2i1, %nxv2i64) :
      (!llvm.target<"aarch64.za.d">, i32, vector<[2]xi1>, vector<[2]xi64>) -> !llvm.target<"aarch64.za.d">
  // CHECK: call target("aarch64.za.q") @llvm.aarch64.sme.x.write.horiz.{{.*}}.nxv1i128
  "arm_sme.intr.x.write.horiz"(%tile_q, %tileslice, %nxv1i1, %nxv1i128) :
      (!llvm.target<"aarch64.za.q">, i32, vector<[1]xi1>, vector<[1]xi128>) -> !llvm.target<"aarch64.za.q">
  // CHECK: call target("aarch64.za.h") @llvm.aarch64.sme.x.write.horiz.{{.*}}.nxv8f16
  "arm_sme.intr.x.write.horiz"(%tile_h, %tileslice, %nxv8i1, %nxv8f16) :
      (!llvm.target<"aarch64.za.h">, i32, vector<[8]xi1>, vector<[8]xf16>) -> !llvm.target<"aarch64.za.h">
  // CHECK: call target("aarch64.za.h") @llvm.aarch64.sme.x.write.horiz.{{.*}}.nxv8bf16
  "arm_sme.intr.x.write.horiz"(%tile_h, %tileslice, %nxv8i1, %nxv8bf16) :
      (!llvm.target<"aarch64.za.h">, i32, vector<[8]xi1>, vector<[8]xbf16>) -> !llvm.target<"aarch64.za.h">
  // CHECK: call target("aarch64.za.s") @llvm.aarch64.sme.x.write.horiz.{{.*}}.nxv4f32
  "arm_sme.intr.x.write.horiz"(%tile_s, %tileslice, %nxv4i1, %nxv4f32) :
      (!llvm.target<"aarch64.za.s">, i32, vector<[4]xi1>, vector<[4]xf32>) -> !llvm.target<"aarch64.za.s">
  // CHECK: call target("aarch64.za.d") @llvm.aarch64.sme.x.write.horiz.{{.*}}.nxv2f64
  "arm_sme.intr.x.write.horiz"(%tile_d, %tileslice, %nxv2i1, %nxv2f64) :
      (!llvm.target<"aarch64.za.d">, i32, vector<[2]xi1>, vector<[2]xf64>) -> !llvm.target<"aarch64.za.d">
  llvm.return
}

// -----

// CHECK-LABEL: @arm_sme_vector_to_tile_vert
llvm.func @arm_sme_vector_to_tile_vert(%tileslice : i32,
                                       %nxv16i1 : vector<[16]xi1>,
                                       %nxv8i1 : vector<[8]xi1>,
                                       %nxv4i1 : vector<[4]xi1>,
                                       %nxv2i1 : vector<[2]xi1>,
                                       %nxv1i1 : vector<[1]xi1>,
                                       %nxv16i8 : vector<[16]xi8>,
                                       %nxv8i16 : vector<[8]xi16>,
                                       %nxv4i32 : vector<[4]xi32>,
                                       %nxv2i64 : vector<[2]xi64>,
                                       %nxv1i128 : vector<[1]xi128>,
                                       %nxv8f16 : vector<[8]xf16>,
                                       %nxv8bf16 : vector<[8]xbf16>,
                                       %nxv4f32 : vector<[4]xf32>,
                                       %nxv2f64 : vector<[2]xf64>,
                                       %tile_b : !llvm.target<"aarch64.za.b">,
                                       %tile_h : !llvm.target<"aarch64.za.h">,
                                       %tile_s : !llvm.target<"aarch64.za.s">,
                                       %tile_d : !llvm.target<"aarch64.za.d">,
                                       %tile_q : !llvm.target<"aarch64.za.q">) {
  // CHECK: call target("aarch64.za.b") @llvm.aarch64.sme.x.write.vert.{{.*}}.nxv16i8
  "arm_sme.intr.x.write.vert"(%tile_b, %tileslice, %nxv16i1, %nxv16i8) :
      (!llvm.target<"aarch64.za.b">, i32, vector<[16]xi1>, vector<[16]xi8>) -> !llvm.target<"aarch64.za.b">
  // CHECK: call target("aarch64.za.h") @llvm.aarch64.sme.x.write.vert.{{.*}}.nxv8i16
  "arm_sme.intr.x.write.vert"(%tile_h, %tileslice, %nxv8i1, %nxv8i16) :
      (!llvm.target<"aarch64.za.h">, i32, vector<[8]xi1>, vector<[8]xi16>) -> !llvm.target<"aarch64.za.h">
  // CHECK: call target("aarch64.za.s") @llvm.aarch64.sme.x.write.vert.{{.*}}.nxv4i32
  "arm_sme.intr.x.write.vert"(%tile_s, %tileslice, %nxv4i1, %nxv4i32) :
      (!llvm.target<"aarch64.za.s">, i32, vector<[4]xi1>, vector<[4]xi32>) -> !llvm.target<"aarch64.za.s">
  // CHECK: call target("aarch64.za.d") @llvm.aarch64.sme.x.write.vert.{{.*}}.nxv2i64
  "arm_sme.intr.x.write.vert"(%tile_d, %tileslice, %nxv2i1, %nxv2i64) :
      (!llvm.target<"aarch64.za.d">, i32, vector<[2]xi1>, vector<[2]xi64>) -> !llvm.target<"aarch64.za.d">
  // CHECK: call target("aarch64.za.q") @llvm.aarch64.sme.x.write.vert.{{.*}}.nxv1i128
  "arm_sme.intr.x.write.vert"(%tile_q, %tileslice, %nxv1i1, %nxv1i128) :
      (!llvm.target<"aarch64.za.q">, i32, vector<[1]xi1>, vector<[1]xi128>) -> !llvm.target<"aarch64.za.q">
  // CHECK: call target("aarch64.za.h") @llvm.aarch64.sme.x.write.vert.{{.*}}.nxv8f16
  "arm_sme.intr.x.write.vert"(%tile_h, %tileslice, %nxv8i1, %nxv8f16) :
      (!llvm.target<"aarch64.za.h">, i32, vector<[8]xi1>, vector<[8]xf16>) -> !llvm.target<"aarch64.za.h">
  // CHECK: call target("aarch64.za.h") @llvm.aarch64.sme.x.write.vert.{{.*}}.nxv8bf16
  "arm_sme.intr.x.write.vert"(%tile_h, %tileslice, %nxv8i1, %nxv8bf16) :
      (!llvm.target<"aarch64.za.h">, i32, vector<[8]xi1>, vector<[8]xbf16>) -> !llvm.target<"aarch64.za.h">
  // CHECK: call target("aarch64.za.s") @llvm.aarch64.sme.x.write.vert.{{.*}}.nxv4f32
  "arm_sme.intr.x.write.vert"(%tile_s, %tileslice, %nxv4i1, %nxv4f32) :
      (!llvm.target<"aarch64.za.s">, i32, vector<[4]xi1>, vector<[4]xf32>) -> !llvm.target<"aarch64.za.s">
  // CHECK: call target("aarch64.za.d") @llvm.aarch64.sme.x.write.vert.{{.*}}.nxv2f64
  "arm_sme.intr.x.write.vert"(%tile_d, %tileslice, %nxv2i1, %nxv2f64) :
      (!llvm.target<"aarch64.za.d">, i32, vector<[2]xi1>, vector<[2]xf64>) -> !llvm.target<"aarch64.za.d">
  llvm.return
}

// -----

llvm.func @arm_sme_tile_slice_to_vector_horiz(%tileslice : i32,
                                              %nxv16i1   : vector<[16]xi1>,
                                              %nxv8i1    : vector<[8]xi1>,
                                              %nxv4i1    : vector<[4]xi1>,
                                              %nxv2i1    : vector<[2]xi1>,
                                              %nxv1i1    : vector<[1]xi1>,
                                              %nxv16i8   : vector<[16]xi8>,
                                              %nxv8i16   : vector<[8]xi16>,
                                              %nxv4i32   : vector<[4]xi32>,
                                              %nxv2i64   : vector<[2]xi64>,
                                              %nxv1i128  : vector<[1]xi128>,
                                              %nxv8f16   : vector<[8]xf16>,
                                              %nxv8bf16  : vector<[8]xbf16>,
                                              %nxv4f32   : vector<[4]xf32>,
                                              %nxv2f64   : vector<[2]xf64>,
                                              %tile_b : !llvm.target<"aarch64.za.b">,
                                              %tile_h : !llvm.target<"aarch64.za.h">,
                                              %tile_s : !llvm.target<"aarch64.za.s">,
                                              %tile_d : !llvm.target<"aarch64.za.d">,
                                              %tile_q : !llvm.target<"aarch64.za.q">) {
  // CHECK: call <vscale x 16 x i8> @llvm.aarch64.sme.x.read.horiz.nxv16i8
  %res0 = "arm_sme.intr.x.read.horiz"(%nxv16i8, %nxv16i1, %tile_b, %tileslice)
    : (vector<[16]xi8>, vector<[16]xi1>, !llvm.target<"aarch64.za.b">, i32) -> vector<[16]xi8>
  // CHECK: call <vscale x 8 x i16> @llvm.aarch64.sme.x.read.horiz.nxv8i16
  %res1 = "arm_sme.intr.x.read.horiz"(%nxv8i16, %nxv8i1, %tile_h, %tileslice)
    : (vector<[8]xi16>, vector<[8]xi1>, !llvm.target<"aarch64.za.h">, i32) -> vector<[8]xi16>
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sme.x.read.horiz.nxv4i32
  %res2 = "arm_sme.intr.x.read.horiz"(%nxv4i32, %nxv4i1, %tile_s, %tileslice)
    : (vector<[4]xi32>, vector<[4]xi1>, !llvm.target<"aarch64.za.s">, i32) -> vector<[4]xi32>
  // CHECK: call <vscale x 2 x i64> @llvm.aarch64.sme.x.read.horiz.nxv2i64
  %res3 = "arm_sme.intr.x.read.horiz"(%nxv2i64, %nxv2i1, %tile_d, %tileslice)
    : (vector<[2]xi64>, vector<[2]xi1>, !llvm.target<"aarch64.za.d">, i32) -> vector<[2]xi64>
  // CHECK: call <vscale x 1 x i128> @llvm.aarch64.sme.x.read.horiz.nxv1i128
  %res4 = "arm_sme.intr.x.read.horiz"(%nxv1i128, %nxv1i1, %tile_q, %tileslice)
    : (vector<[1]xi128>, vector<[1]xi1>, !llvm.target<"aarch64.za.q">, i32) -> vector<[1]xi128>
  // CHECK: call <vscale x 8 x half> @llvm.aarch64.sme.x.read.horiz.nxv8f16
  %res5 = "arm_sme.intr.x.read.horiz"(%nxv8f16, %nxv8i1, %tile_h, %tileslice)
    : (vector<[8]xf16>, vector<[8]xi1>, !llvm.target<"aarch64.za.h">, i32) -> vector<[8]xf16>
  // CHECK: call <vscale x 8 x bfloat> @llvm.aarch64.sme.x.read.horiz.nxv8bf16
  %res6 = "arm_sme.intr.x.read.horiz"(%nxv8bf16, %nxv8i1, %tile_h, %tileslice)
    : (vector<[8]xbf16>, vector<[8]xi1>, !llvm.target<"aarch64.za.h">, i32) -> vector<[8]xbf16>
  // CHECK: call <vscale x 4 x float> @llvm.aarch64.sme.x.read.horiz.nxv4f32
  %res7 = "arm_sme.intr.x.read.horiz"(%nxv4f32, %nxv4i1, %tile_s, %tileslice)
    : (vector<[4]xf32>, vector<[4]xi1>, !llvm.target<"aarch64.za.s">, i32) -> vector<[4]xf32>
  // CHECK: call <vscale x 2 x double> @llvm.aarch64.sme.x.read.horiz.nxv2f64
  %res8 = "arm_sme.intr.x.read.horiz"(%nxv2f64, %nxv2i1, %tile_d, %tileslice)
    : (vector<[2]xf64>, vector<[2]xi1>, !llvm.target<"aarch64.za.d">, i32) -> vector<[2]xf64>
  llvm.return
}

// -----

llvm.func @arm_sme_tile_slice_to_vector_vert(%tileslice : i32,
                                             %nxv16i1   : vector<[16]xi1>,
                                             %nxv8i1    : vector<[8]xi1>,
                                             %nxv4i1    : vector<[4]xi1>,
                                             %nxv2i1    : vector<[2]xi1>,
                                             %nxv1i1    : vector<[1]xi1>,
                                             %nxv16i8   : vector<[16]xi8>,
                                             %nxv8i16   : vector<[8]xi16>,
                                             %nxv4i32   : vector<[4]xi32>,
                                             %nxv2i64   : vector<[2]xi64>,
                                             %nxv1i128  : vector<[1]xi128>,
                                             %nxv8f16   : vector<[8]xf16>,
                                             %nxv8bf16  : vector<[8]xbf16>,
                                             %nxv4f32   : vector<[4]xf32>,
                                             %nxv2f64   : vector<[2]xf64>,
                                             %tile_b : !llvm.target<"aarch64.za.b">,
                                             %tile_h : !llvm.target<"aarch64.za.h">,
                                             %tile_s : !llvm.target<"aarch64.za.s">,
                                             %tile_d : !llvm.target<"aarch64.za.d">,
                                             %tile_q : !llvm.target<"aarch64.za.q">) {
  // CHECK: call <vscale x 16 x i8> @llvm.aarch64.sme.x.read.vert.nxv16i8
  %res0 = "arm_sme.intr.x.read.vert"(%nxv16i8, %nxv16i1, %tile_b, %tileslice)
    : (vector<[16]xi8>, vector<[16]xi1>, !llvm.target<"aarch64.za.b">, i32) -> vector<[16]xi8>
  // CHECK: call <vscale x 8 x i16> @llvm.aarch64.sme.x.read.vert.nxv8i16
  %res1 = "arm_sme.intr.x.read.vert"(%nxv8i16, %nxv8i1, %tile_h, %tileslice)
    : (vector<[8]xi16>, vector<[8]xi1>, !llvm.target<"aarch64.za.h">, i32) -> vector<[8]xi16>
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sme.x.read.vert.nxv4i32
  %res2 = "arm_sme.intr.x.read.vert"(%nxv4i32, %nxv4i1, %tile_s, %tileslice)
    : (vector<[4]xi32>, vector<[4]xi1>, !llvm.target<"aarch64.za.s">, i32) -> vector<[4]xi32>
  // CHECK: call <vscale x 2 x i64> @llvm.aarch64.sme.x.read.vert.nxv2i64
  %res3 = "arm_sme.intr.x.read.vert"(%nxv2i64, %nxv2i1, %tile_d, %tileslice)
    : (vector<[2]xi64>, vector<[2]xi1>, !llvm.target<"aarch64.za.d">, i32) -> vector<[2]xi64>
  // CHECK: call <vscale x 1 x i128> @llvm.aarch64.sme.x.read.vert.nxv1i128
  %res4 = "arm_sme.intr.x.read.vert"(%nxv1i128, %nxv1i1, %tile_q, %tileslice)
    : (vector<[1]xi128>, vector<[1]xi1>, !llvm.target<"aarch64.za.q">, i32) -> vector<[1]xi128>
  // CHECK: call <vscale x 8 x half> @llvm.aarch64.sme.x.read.vert.nxv8f16
  %res5 = "arm_sme.intr.x.read.vert"(%nxv8f16, %nxv8i1, %tile_h, %tileslice)
    : (vector<[8]xf16>, vector<[8]xi1>, !llvm.target<"aarch64.za.h">, i32) -> vector<[8]xf16>
  // CHECK: call <vscale x 8 x bfloat> @llvm.aarch64.sme.x.read.vert.nxv8bf16
  %res6 = "arm_sme.intr.x.read.vert"(%nxv8bf16, %nxv8i1, %tile_h, %tileslice)
    : (vector<[8]xbf16>, vector<[8]xi1>, !llvm.target<"aarch64.za.h">, i32) -> vector<[8]xbf16>
  // CHECK: call <vscale x 4 x float> @llvm.aarch64.sme.x.read.vert.nxv4f32
  %res7 = "arm_sme.intr.x.read.vert"(%nxv4f32, %nxv4i1, %tile_s, %tileslice)
    : (vector<[4]xf32>, vector<[4]xi1>, !llvm.target<"aarch64.za.s">, i32) -> vector<[4]xf32>
  // CHECK: call <vscale x 2 x double> @llvm.aarch64.sme.x.read.vert.nxv2f64
  %res8 = "arm_sme.intr.x.read.vert"(%nxv2f64, %nxv2i1, %tile_d, %tileslice)
    : (vector<[2]xf64>, vector<[2]xi1>, !llvm.target<"aarch64.za.d">, i32) -> vector<[2]xf64>
  llvm.return
}

