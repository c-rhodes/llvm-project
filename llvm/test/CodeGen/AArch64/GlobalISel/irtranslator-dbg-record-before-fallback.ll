; RUN: llc -mtriple=aarch64 -mattr=+sve -global-isel -global-isel-abort=0 -aarch64-enable-gisel-sve=0 -stop-after=irtranslator %s -o - | FileCheck %s

; Check that IRTranslator does not translate debug records attached to an
; instruction that is going to fall back to SelectionDAG.

define void @dbg_before_fallback_alloca() !dbg !2 {
; CHECK-LABEL: name: dbg_before_fallback_alloca
; CHECK: failedISel: true
; CHECK-NOT: debug-info-variable
    #dbg_declare(ptr %alloca, !3, !DIExpression(), !4)
  %alloca = alloca <vscale x 16 x i8>, align 16
  ret void
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C,
                             file: !DIFile(filename: "test.c", directory: ""))
!2 = distinct !DISubprogram(type: !DISubroutineType(types: !{}), unit: !1)
!3 = !DILocalVariable(scope: !2)
!4 = !DILocation(scope: !2)
