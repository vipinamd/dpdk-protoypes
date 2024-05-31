#include <x86intrin.h>

void avxIntelReference(){
//https://networkrndhub.sec.samsung.net/confluence/display/PHILIPPEC/1.+Matrix+Multiplication

  __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

  float matrix1[64] __attribute__((aligned(32)));
  float matrix2[64] __attribute__((aligned(32)));
  float out[64] __attribute__((aligned(32)));

  int num = 100000;

  for(int y = 0; y < 8; y++)
    for(int x = 0; x < 8; x++)
      matrix1[y*8+x] = (float) (y*8 + x + 1);

  for(int y = 0; y < 8; y++)
    for(int x = 0; x < 8; x++)
      matrix2[y*8+x] = (float) (y*8 + x + 1);

  //Initialize some pointers
  float* pMatrix2 = matrix1;
  float* pIn = matrix2;
  float* pOut = out;

  for(int n = 0; n < num; n++) {
    //Read the eight rows of Matrix B into ymm registers
    ymm8 = _mm256_load_ps((float *) (pMatrix2));
    ymm9 = _mm256_load_ps((float *) (pMatrix2 + 1*8));
    ymm10 = _mm256_load_ps((float *) (pMatrix2 + 2*8));
    ymm11 = _mm256_load_ps((float *) (pMatrix2 + 3*8));
    ymm12 = _mm256_load_ps((float *) (pMatrix2 + 4*8));
    ymm13 = _mm256_load_ps((float *) (pMatrix2 + 5*8));
    ymm14 = _mm256_load_ps((float *) (pMatrix2 + 6*8));
    ymm15 = _mm256_load_ps((float *) (pMatrix2 + 7*8));

    //Broadcast each element of Matrix A Row 1 into a ymm register
    ymm0 = _mm256_broadcast_ss(pIn);
    ymm1 = _mm256_broadcast_ss(pIn + 1);
    ymm2 = _mm256_broadcast_ss(pIn + 2);
    ymm3 = _mm256_broadcast_ss(pIn + 3);
    ymm4 = _mm256_broadcast_ss(pIn + 4);
    ymm5 = _mm256_broadcast_ss(pIn + 5);
    ymm6 = _mm256_broadcast_ss(pIn + 6);
    ymm7 = _mm256_broadcast_ss(pIn + 7);
    ymm0 = _mm256_mul_ps(ymm0, ymm8);
    ymm1 = _mm256_mul_ps(ymm1, ymm9);
    ymm0 = _mm256_add_ps(ymm0, ymm1);
    ymm2 = _mm256_mul_ps(ymm2, ymm10);
    ymm3 = _mm256_mul_ps(ymm3, ymm11);
    ymm2 = _mm256_add_ps(ymm2, ymm3);
    ymm4 = _mm256_mul_ps(ymm4, ymm12);
    ymm5 = _mm256_mul_ps(ymm5, ymm13);
    ymm4 = _mm256_add_ps(ymm4, ymm5);
    ymm6 = _mm256_mul_ps(ymm6, ymm14);
    ymm7 = _mm256_mul_ps(ymm7, ymm15);
    ymm6 = _mm256_add_ps(ymm6, ymm7);
    ymm0 = _mm256_add_ps(ymm0, ymm2);
    ymm4 = _mm256_add_ps(ymm4, ymm6);
    ymm0 = _mm256_add_ps(ymm0, ymm4);
    _mm256_store_ps((float *) (pOut), ymm0);

    //Repeat using Matrix A Row 2
    ymm0 = _mm256_broadcast_ss(pIn + 1*8);
    ymm1 = _mm256_broadcast_ss(pIn + 1*8 + 1);
    ymm2 = _mm256_broadcast_ss(pIn + 1*8 + 2);
    ymm3 = _mm256_broadcast_ss(pIn + 1*8 + 3);
    ymm4 = _mm256_broadcast_ss(pIn + 1*8 + 4);
    ymm5 = _mm256_broadcast_ss(pIn + 1*8 + 5);
    ymm6 = _mm256_broadcast_ss(pIn + 1*8 + 6);
    ymm7 = _mm256_broadcast_ss(pIn + 1*8 + 7);
    ymm0 = _mm256_mul_ps(ymm0, ymm8);
    ymm1 = _mm256_mul_ps(ymm1, ymm9);
    ymm0 = _mm256_add_ps(ymm0, ymm1);
    ymm2 = _mm256_mul_ps(ymm2, ymm10);
    ymm3 = _mm256_mul_ps(ymm3, ymm11);
    ymm2 = _mm256_add_ps(ymm2, ymm3);
    ymm4 = _mm256_mul_ps(ymm4, ymm12);
    ymm5 = _mm256_mul_ps(ymm5, ymm13);
    ymm4 = _mm256_add_ps(ymm4, ymm5);
    ymm6 = _mm256_mul_ps(ymm6, ymm14);
    ymm7 = _mm256_mul_ps(ymm7, ymm15);
    ymm6 = _mm256_add_ps(ymm6, ymm7);
    ymm0 = _mm256_add_ps(ymm0, ymm2);
    ymm4 = _mm256_add_ps(ymm4, ymm6);
    ymm0 = _mm256_add_ps(ymm0, ymm4);
    _mm256_store_ps((float *) (pOut + 1*8), ymm0);


    //Repeat using Matrix A Row 3
    ymm0 = _mm256_broadcast_ss(pIn + 2*8);
    ymm1 = _mm256_broadcast_ss(pIn + 2*8 + 1);
    ymm2 = _mm256_broadcast_ss(pIn + 2*8 + 2);
    ymm3 = _mm256_broadcast_ss(pIn + 2*8 + 3);
    ymm4 = _mm256_broadcast_ss(pIn + 2*8 + 4);
    ymm5 = _mm256_broadcast_ss(pIn + 2*8 + 5);
    ymm6 = _mm256_broadcast_ss(pIn + 2*8 + 6);
    ymm7 = _mm256_broadcast_ss(pIn + 2*8 + 7);
    ymm0 = _mm256_mul_ps(ymm0, ymm8);
    ymm1 = _mm256_mul_ps(ymm1, ymm9);
    ymm0 = _mm256_add_ps(ymm0, ymm1);
    ymm2 = _mm256_mul_ps(ymm2, ymm10);
    ymm3 = _mm256_mul_ps(ymm3, ymm11);
    ymm2 = _mm256_add_ps(ymm2, ymm3);
    ymm4 = _mm256_mul_ps(ymm4, ymm12);
    ymm5 = _mm256_mul_ps(ymm5, ymm13);
    ymm4 = _mm256_add_ps(ymm4, ymm5);
    ymm6 = _mm256_mul_ps(ymm6, ymm14);
    ymm7 = _mm256_mul_ps(ymm7, ymm15);
    ymm6 = _mm256_add_ps(ymm6, ymm7);
    ymm0 = _mm256_add_ps(ymm0, ymm2);
    ymm4 = _mm256_add_ps(ymm4, ymm6);
    ymm0 = _mm256_add_ps(ymm0, ymm4);
    _mm256_store_ps((float *) (pOut + 2*8), ymm0);

    //Repeat using Matrix A Row 4
    ymm0 = _mm256_broadcast_ss(pIn + 3*8);
    ymm1 = _mm256_broadcast_ss(pIn + 3*8 + 1);
    ymm2 = _mm256_broadcast_ss(pIn + 3*8 + 2);
    ymm3 = _mm256_broadcast_ss(pIn + 3*8 + 3);
    ymm4 = _mm256_broadcast_ss(pIn + 3*8 + 4);
    ymm5 = _mm256_broadcast_ss(pIn + 3*8 + 5);
    ymm6 = _mm256_broadcast_ss(pIn + 3*8 + 6);
    ymm7 = _mm256_broadcast_ss(pIn + 3*8 + 7);
    ymm0 = _mm256_mul_ps(ymm0, ymm8);
    ymm1 = _mm256_mul_ps(ymm1, ymm9);
    ymm0 = _mm256_add_ps(ymm0, ymm1);
    ymm2 = _mm256_mul_ps(ymm2, ymm10);
    ymm3 = _mm256_mul_ps(ymm3, ymm11);
    ymm2 = _mm256_add_ps(ymm2, ymm3);
    ymm4 = _mm256_mul_ps(ymm4, ymm12);
    ymm5 = _mm256_mul_ps(ymm5, ymm13);
    ymm4 = _mm256_add_ps(ymm4, ymm5);
    ymm6 = _mm256_mul_ps(ymm6, ymm14);
    ymm7 = _mm256_mul_ps(ymm7, ymm15);
    ymm6 = _mm256_add_ps(ymm6, ymm7);
    ymm0 = _mm256_add_ps(ymm0, ymm2);
    ymm4 = _mm256_add_ps(ymm4, ymm6);
    ymm0 = _mm256_add_ps(ymm0, ymm4);
    _mm256_store_ps((float *) (pOut + 3*8), ymm0);

    //Repeat using Matrix A Row 5
    ymm0 = _mm256_broadcast_ss(pIn + 4*8);
    ymm1 = _mm256_broadcast_ss(pIn + 4*8 + 1);
    ymm2 = _mm256_broadcast_ss(pIn + 4*8 + 2);
    ymm3 = _mm256_broadcast_ss(pIn + 4*8 + 3);
    ymm4 = _mm256_broadcast_ss(pIn + 4*8 + 4);
    ymm5 = _mm256_broadcast_ss(pIn + 4*8 + 5);
    ymm6 = _mm256_broadcast_ss(pIn + 4*8 + 6);
    ymm7 = _mm256_broadcast_ss(pIn + 4*8 + 7);
    ymm0 = _mm256_mul_ps(ymm0, ymm8);
    ymm1 = _mm256_mul_ps(ymm1, ymm9);
    ymm0 = _mm256_add_ps(ymm0, ymm1);
    ymm2 = _mm256_mul_ps(ymm2, ymm10);
    ymm3 = _mm256_mul_ps(ymm3, ymm11);
    ymm2 = _mm256_add_ps(ymm2, ymm3);
    ymm4 = _mm256_mul_ps(ymm4, ymm12);
    ymm5 = _mm256_mul_ps(ymm5, ymm13);
    ymm4 = _mm256_add_ps(ymm4, ymm5);
    ymm6 = _mm256_mul_ps(ymm6, ymm14);
    ymm7 = _mm256_mul_ps(ymm7, ymm15);
    ymm6 = _mm256_add_ps(ymm6, ymm7);
    ymm0 = _mm256_add_ps(ymm0, ymm2);
    ymm4 = _mm256_add_ps(ymm4, ymm6);
    ymm0 = _mm256_add_ps(ymm0, ymm4);
    _mm256_store_ps((float *) (pOut + 4*8), ymm0);

    //Repeat using Matrix A Row 6
    ymm0 = _mm256_broadcast_ss(pIn + 5*8);
    ymm1 = _mm256_broadcast_ss(pIn + 5*8 + 1);
    ymm2 = _mm256_broadcast_ss(pIn + 5*8 + 2);
    ymm3 = _mm256_broadcast_ss(pIn + 5*8 + 3);
    ymm4 = _mm256_broadcast_ss(pIn + 5*8 + 4);
    ymm5 = _mm256_broadcast_ss(pIn + 5*8 + 5);
    ymm6 = _mm256_broadcast_ss(pIn + 5*8 + 6);
    ymm7 = _mm256_broadcast_ss(pIn + 5*8 + 7);
    ymm0 = _mm256_mul_ps(ymm0, ymm8);
    ymm1 = _mm256_mul_ps(ymm1, ymm9);
    ymm0 = _mm256_add_ps(ymm0, ymm1);
    ymm2 = _mm256_mul_ps(ymm2, ymm10);
    ymm3 = _mm256_mul_ps(ymm3, ymm11);
    ymm2 = _mm256_add_ps(ymm2, ymm3);
    ymm4 = _mm256_mul_ps(ymm4, ymm12);
    ymm5 = _mm256_mul_ps(ymm5, ymm13);
    ymm4 = _mm256_add_ps(ymm4, ymm5);
    ymm6 = _mm256_mul_ps(ymm6, ymm14);
    ymm7 = _mm256_mul_ps(ymm7, ymm15);
    ymm6 = _mm256_add_ps(ymm6, ymm7);
    ymm0 = _mm256_add_ps(ymm0, ymm2);
    ymm4 = _mm256_add_ps(ymm4, ymm6);
    ymm0 = _mm256_add_ps(ymm0, ymm4);
    _mm256_store_ps((float *) (pOut + 5*8), ymm0);


    //Repeat using Matrix A Row 7
    ymm0 = _mm256_broadcast_ss(pIn + 6*8);
    ymm1 = _mm256_broadcast_ss(pIn + 6*8 + 1);
    ymm2 = _mm256_broadcast_ss(pIn + 6*8 + 2);
    ymm3 = _mm256_broadcast_ss(pIn + 6*8 + 3);
    ymm4 = _mm256_broadcast_ss(pIn + 6*8 + 4);
    ymm5 = _mm256_broadcast_ss(pIn + 6*8 + 5);
    ymm6 = _mm256_broadcast_ss(pIn + 6*8 + 6);
    ymm7 = _mm256_broadcast_ss(pIn + 6*8 + 7);
    ymm0 = _mm256_mul_ps(ymm0, ymm8);
    ymm1 = _mm256_mul_ps(ymm1, ymm9);
    ymm0 = _mm256_add_ps(ymm0, ymm1);
    ymm2 = _mm256_mul_ps(ymm2, ymm10);
    ymm3 = _mm256_mul_ps(ymm3, ymm11);
    ymm2 = _mm256_add_ps(ymm2, ymm3);
    ymm4 = _mm256_mul_ps(ymm4, ymm12);
    ymm5 = _mm256_mul_ps(ymm5, ymm13);
    ymm4 = _mm256_add_ps(ymm4, ymm5);
    ymm6 = _mm256_mul_ps(ymm6, ymm14);
    ymm7 = _mm256_mul_ps(ymm7, ymm15);
    ymm6 = _mm256_add_ps(ymm6, ymm7);
    ymm0 = _mm256_add_ps(ymm0, ymm2);
    ymm4 = _mm256_add_ps(ymm4, ymm6);
    ymm0 = _mm256_add_ps(ymm0, ymm4);
    _mm256_store_ps((float *) (pOut + 6*8), ymm0);

    //Repeat using Matrix A Row 8
    ymm0 = _mm256_broadcast_ss(pIn + 7*8);
    ymm1 = _mm256_broadcast_ss(pIn + 7*8 + 1);
    ymm2 = _mm256_broadcast_ss(pIn + 7*8 + 2);
    ymm3 = _mm256_broadcast_ss(pIn + 7*8 + 3);
    ymm4 = _mm256_broadcast_ss(pIn + 7*8 + 4);
    ymm5 = _mm256_broadcast_ss(pIn + 7*8 + 5);
    ymm6 = _mm256_broadcast_ss(pIn + 7*8 + 6);
    ymm7 = _mm256_broadcast_ss(pIn + 7*8 + 7);
    ymm0 = _mm256_mul_ps(ymm0, ymm8);
    ymm1 = _mm256_mul_ps(ymm1, ymm9);
    ymm0 = _mm256_add_ps(ymm0, ymm1);
    ymm2 = _mm256_mul_ps(ymm2, ymm10);
    ymm3 = _mm256_mul_ps(ymm3, ymm11);
    ymm2 = _mm256_add_ps(ymm2, ymm3);
    ymm4 = _mm256_mul_ps(ymm4, ymm12);
    ymm5 = _mm256_mul_ps(ymm5, ymm13);
    ymm4 = _mm256_add_ps(ymm4, ymm5);
    ymm6 = _mm256_mul_ps(ymm6, ymm14);
    ymm7 = _mm256_mul_ps(ymm7, ymm15);
    ymm6 = _mm256_add_ps(ymm6, ymm7);
    ymm0 = _mm256_add_ps(ymm0, ymm2);
    ymm4 = _mm256_add_ps(ymm4, ymm6);
    ymm0 = _mm256_add_ps(ymm0, ymm4);
    _mm256_store_ps((float *) (pOut + 7*8), ymm0);
  }
/* block printf
  for(int y = 0; y < 8; y++) {
    for(int x = 0; x < 8; x++)
      printf("%8.1f ", out[y*8+x]);
    printf("\n");
  }
*/
}


float test256() {
  float matrix1[8][8] __attribute__((aligned(32)));
  float matrix2[8][8] __attribute__((aligned(32)));
  float out[8][8] __attribute__((aligned(32)));
  __m256 mmv[16];
  int num = 100000;
  int index = 0;
  for (int y = 0; y < 8; y++)
    for (int x = 0; x < 8; x++) matrix1[y][x] = (float) (y * 8.0 + x + 1.0);

  for (int y = 0; y < 8; y++)
    for (int x = 0; x < 8; x++) matrix2[y][x] = (float) (y * 8.0 + x + 1.0);

  for (int n = 0; n < num; n++) {
    for (int i = 0; i < 8; i++) mmv[i + 8] = _mm256_load_ps( matrix2[i]);

    for (int j = 0; j < 8; j++) {
      for (int i = 0; i < 8; i++) mmv[i] = _mm256_broadcast_ss((float *) matrix1[j] + i);

      for (int i = 0; i < 8; i++) mmv[i] = _mm256_mul_ps(mmv[i], mmv[i + 8]);

      for (int k = 2; k >= 0; k--)
        for (int i = 0; i < (1 << k); i++)
          mmv[(8 >> k) * i] = _mm256_add_ps(mmv[(8 >> k) * i], mmv[(8 >> k) * i + (4 >> k)]);

      _mm256_store_ps(out[j], mmv[0]);
      index++;
    }
  }

  float sum = 0;
  for(int k = 0; k < 8; k ++){
    for(int l = 0; l < 8; l++){
      sum += out[k][l];
    }
  }
#if withPrint
  printf("%f\n", sum);
#endif

  return (sum + (float)index);
}


float test512() {
  // matrixA[16][16] * matrixB[16][16] = matrixC[16][16]
  float matrixA[16][16] __attribute__((aligned(64)));
  float matrixB[16][16] __attribute__((aligned(64)));
  float matrixC[16][16] __attribute__((aligned(64)));
  __m512 mmv[32];
  int num = 100000;
  int index = 0;

  // matrixA initialize
  for (int y = 0; y < 16; y++)
    for (int x = 0; x < 16; x++)
      matrixA[y][x] = (float) (y * 16 + x + 1);

  // matrixB initialize
  for (int y = 0; y < 16; y++)
    for (int x = 0; x < 16; x++)
      matrixB[y][x] = (float) (y * 16 + x + 1);

  // matrix multiplication
  for (int n = 0; n < num; n++) {
    for (int i = 0; i < 16; i++)
      mmv[i + 16] = _mm512_load_ps(matrixB[i]);

    for (int j = 0; j < 16; j++) {
      for (int i = 0; i < 16; i++)
        mmv[i] = _mm512_set1_ps(matrixA[j][i]);

      for (int i = 0; i < 16; i++)
        mmv[i] = _mm512_mul_ps(mmv[i], mmv[i + 16]);

      __m512 mmv01 = mmv[0] + mmv[1];
      __m512 mmv23 = mmv[2] + mmv[3];
      __m512 mmv45 = mmv[4] + mmv[5];
      __m512 mmv67 = mmv[6] + mmv[7];
      __m512 mmv89 = mmv[8] + mmv[9];
      __m512 mmv1011 = mmv[10] + mmv[11];
      __m512 mmv1213 = mmv[12] + mmv[13];
      __m512 mmv1415 = mmv[14] + mmv[15];

      __m512 mmv0123 = mmv01 + mmv23;
      __m512 mmv4567 = mmv45 + mmv67;
      __m512 mmv891011 = mmv89 + mmv1011;
      __m512 mmv12131415 = mmv1213 + mmv1415;

      __m512 mmv01234567 = mmv0123 + mmv4567;
      __m512 mmv89101112131415 = mmv891011 + mmv12131415;


      mmv[0] = mmv01234567 + mmv89101112131415;

      _mm512_store_ps(matrixC[j], mmv[0]);
      index++;
    }
  }

  float sum = 0;
  for(int k = 0; k < 16; k ++){
    for(int l = 0; l < 16; l++){
      sum += matrixC[k][l];
    }
  }
#if withPrint
  printf("%f\n", sum);
#endif
  return (sum + (float)index);
}


#if 0
    minValue_512 = std::min(minValue_512, currValue_512);
    maxValue_512 = std::max(maxValue_512, currValue_512);
    totalValue_512 += currValue_512;
  avgValue_512 = (totalValue_512 / (double)count);

  printf("avxResult512(%f)\n", avxResult);
  printf("[Core%3d] test256(%lf,%lf,%lf) test512(%lf,%lf,%lf)\n", coreID
          ,minValue_256, maxValue_256, avgValue_256
          ,minValue_512, maxValue_512, avgValue_512);
 #endif

