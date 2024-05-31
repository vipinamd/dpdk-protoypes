rm testUncore
gcc avx.c test.c -march=x86_64-v4 $(pkg-config --static --libs --cflags libdpdk) -I. -L /opt/e-sms/e_smi/lib/ -o testUncore
