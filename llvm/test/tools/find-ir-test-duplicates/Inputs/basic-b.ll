declare i32 @unique_b_helper(i32)

define i32 @bar(i32 %value) {
entry:
  %result = add i32 2, %value
  ret i32 %result
}

define i32 @unique_b(i32 %value) {
entry:
  %result = call i32 @unique_b_helper(i32 %value)
  ret i32 %result
}
