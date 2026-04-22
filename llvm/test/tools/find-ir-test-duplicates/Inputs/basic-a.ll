declare i32 @unique_a_helper(i32)

define i32 @foo(i32 %arg) {
entry:
  %tmp = add i32 %arg, 2
  ret i32 %tmp
}

define i32 @unique_a(i32 %arg) {
entry:
  %tmp = call i32 @unique_a_helper(i32 %arg)
  ret i32 %tmp
}
