# Check exact duplicate test source detection.
#
# RUN: %{lit} %{inputs}/show-test-duplicates --show-test-duplicates > %t.out
# RUN: FileCheck %s < %t.out
#
# RUN: %{lit} -r %{inputs}/show-test-duplicates --show-test-duplicates > %t.relative.out
# RUN: FileCheck %s --check-prefix=RELATIVE < %t.relative.out
#
# RUN: %{lit} %{inputs}/show-test-duplicates/dup-a.txt \
# RUN:        %{inputs}/show-test-duplicates/dup-a.txt \
# RUN:        --show-test-duplicates > %t.none.out
# RUN: FileCheck %s --check-prefix=NONE < %t.none.out

# CHECK: -- Exact Duplicate Test Sources --
# CHECK-NEXT:   Group 1: 2 files, 2 tests
# CHECK-NEXT:     show-test-duplicates :: dup-a.txt
# CHECK-NEXT:     show-test-duplicates :: dup-b.txt
# CHECK-NOT: unique.txt

# RELATIVE: -- Exact Duplicate Test Sources --
# RELATIVE-NEXT:   Group 1: 2 files, 2 tests
# RELATIVE-NEXT:     Inputs{{[/\\]}}show-test-duplicates{{[/\\]}}dup-a.txt
# RELATIVE-NEXT:     Inputs{{[/\\]}}show-test-duplicates{{[/\\]}}dup-b.txt

# NONE: -- Exact Duplicate Test Sources --
# NONE-NEXT:   No exact duplicate test sources found.
