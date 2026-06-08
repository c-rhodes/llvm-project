; RUN: llc -mtriple=aarch64 -O0 -global-isel -global-isel-abort=1 -verify-machineinstrs -stop-after=instruction-select %s -o - | FileCheck %s

define fp128 @xor_fp128_signbit(fp128 %x) {
  ; CHECK-LABEL: name: xor_fp128_signbit
  ; CHECK: [[SRC:%[0-9]+]]:fpr128 = COPY $q0
  ; CHECK: [[LO_FPR:%[0-9]+]]:fpr64 = COPY [[SRC]].dsub
  ; CHECK: [[LO:%[0-9]+]]:gpr64 = FMOVDXr [[LO_FPR]]
  ; CHECK: [[HI:%[0-9]+]]:gpr64 = FMOVDXHighr [[SRC]], 1
  ; CHECK: EORXrr [[LO]]
  ; CHECK: EORXri [[HI]], 4160
  ; CHECK-NOT: G_UNMERGE_VALUES
  ; CHECK: RET_ReallyLR
entry:
  %bits = bitcast fp128 %x to i128
  %xor = xor i128 %bits, -170141183460469231731687303715884105728
  %ret = bitcast i128 %xor to fp128
  ret fp128 %ret
}
