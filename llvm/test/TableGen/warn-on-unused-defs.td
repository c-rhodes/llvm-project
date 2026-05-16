// RUN: llvm-tblgen --null-backend %s 2>&1 | FileCheck %s --check-prefix=QUIET --allow-empty --implicit-check-not="warning:"
// RUN: llvm-tblgen --null-backend --warn-on-unused-defs %s 2>&1 | FileCheck %s --check-prefix=WARN

class Helper {
  int Value = ?;
}

def Used : Helper {
  let Value = 7;
}

def Unused : Helper {
  let Value = 9;
}

class Trigger {
  Helper H = !cast<Helper>("Used");
}

def : Trigger;

// QUIET-NOT: warning:

// WARN: warning: def 'Unused' appears to be unused
// WARN-NOT: def 'Used' appears to be unused
