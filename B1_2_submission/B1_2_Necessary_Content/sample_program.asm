addi $t1, $zero, 1      // $t1 = 1
addi $t2, $zero, 3      // $t2 = 3
add $t0, $t1, $t2       // $t0 = 4
add $t3, $t0, $t2       // $t3 = 7
add $t4, $t0, $t1       // $t4 = 5
sw $t1, 3($t2)          // mem[$t2+3] = mem[6] = 1
sll $t1, $t1, 2         // $t1 = 4
beq $t0, $t1, label1    // branching will happen
j end
label1: sub $t4, $t3, $t0   // $t4 = 3
subi $t3, $t3, 1            // $t3 = 6
srl $t3, $t3, 1             // $t3 = 3
lw $t1, 3($t2)               // $t1 = mem[6] = 1
and $t0, $t1, $t3            // $t0 = 1
or $t1, $t3, $t4             // $t1 = 3
j label2
label3: push $t0             // mem[FF] = 1
push $t1                     // mem[FE] = 3
ori $t0, $t0, 4              // $t0 = 5
pop $t0                      // $t0 = 3
andi $t2, $t2, 0             // $t2 = 0
pop $t2                      // $t2 = 1
nor $t2, $t2, $t2            // $t2 = -2
j end
label2: bneq $t0, $t2, label3 // branching will be executed
end:
