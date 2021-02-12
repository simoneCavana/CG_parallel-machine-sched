\ Model Parallel identical processor scheduling with weighted completion times as set-covering formulation
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  609 x_s[0] + 641 x_s[1] + 621 x_s[2] + 629 x_s[3] + 596 x_s[4]
   + 654 x_s[5] + 608 x_s[6] + 642 x_s[7] + 596 x_s[8] + 654 x_s[9]
   + 641 x_s[10] + 609 x_s[11] + 609 x_s[12] + 641 x_s[13] + 614 x_s[14]
   + 636 x_s[15] + 642 x_s[16] + 608 x_s[17] + 635 x_s[18] + 615 x_s[19]
Subject To
 R0: x_s[0] + x_s[1] + x_s[2] + x_s[3] + x_s[4] + x_s[5] + x_s[6] + x_s[7]
   + x_s[8] + x_s[9] + x_s[10] + x_s[11] + x_s[12] + x_s[13] + x_s[14]
   + x_s[15] + x_s[16] + x_s[17] + x_s[18] + x_s[19] = 2
 R1: x_s[0] + x_s[3] + x_s[4] + x_s[7] + x_s[8] + x_s[11] + x_s[12]
   + x_s[15] + x_s[17] + x_s[19] = 1
 R2: x_s[0] + x_s[2] + x_s[5] + x_s[7] + x_s[9] + x_s[11] + x_s[12]
   + x_s[14] + x_s[16] + x_s[18] = 1
 R3: x_s[1] + x_s[3] + x_s[4] + x_s[6] + x_s[8] + x_s[10] + x_s[13]
   + x_s[15] + x_s[17] + x_s[19] = 1
 R4: x_s[1] + x_s[3] + x_s[5] + x_s[7] + x_s[9] + x_s[10] + x_s[13]
   + x_s[14] + x_s[17] + x_s[18] = 1
 R5: x_s[0] + x_s[2] + x_s[4] + x_s[6] + x_s[8] + x_s[11] + x_s[12]
   + x_s[15] + x_s[16] + x_s[19] = 1
 R6: x_s[1] + x_s[2] + x_s[5] + x_s[6] + x_s[9] + x_s[10] + x_s[13]
   + x_s[15] + x_s[16] + x_s[19] = 1
 R7: x_s[0] + x_s[3] + x_s[4] + x_s[7] + x_s[8] + x_s[11] + x_s[12]
   + x_s[14] + x_s[17] + x_s[18] = 1
 R8: x_s[1] + x_s[3] + x_s[4] + x_s[6] + x_s[8] + x_s[10] + x_s[13]
   + x_s[15] + x_s[16] + x_s[18] = 1
 R9: x_s[1] + x_s[3] + x_s[4] + x_s[6] + x_s[8] + x_s[10] + x_s[13]
   + x_s[15] + x_s[16] + x_s[18] = 1
 R10: x_s[0] + x_s[2] + x_s[5] + x_s[7] + x_s[9] + x_s[11] + x_s[12]
   + x_s[14] + x_s[17] + x_s[19] = 1
Bounds
Binaries
 x_s[0] x_s[1] x_s[2] x_s[3] x_s[4] x_s[5] x_s[6] x_s[7] x_s[8] x_s[9]
 x_s[10] x_s[11] x_s[12] x_s[13] x_s[14] x_s[15] x_s[16] x_s[17] x_s[18]
 x_s[19]
End
