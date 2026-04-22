; RUN: split-file %s %t

;--- duplicate.ll
define i32 @baz(i32 %x) {
entry:
  %same = add i32 %x, 2
  ret i32 %same
}

;--- unique.ll
define i32 @only_here(i32 %x) {
entry:
  %different = mul i32 %x, 3
  ret i32 %different
}
