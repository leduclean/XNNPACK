KERNELS := scalar-p2 rvv-u1v rvv-u2v rvv-u2v-off rvv-stride-u2v rvv-u2v-inv rvv-u2v-dev
KDIR    := ../src/f32-ibilinear-chw/gen
OBJS    := $(patsubst %,%.o,$(KERNELS))
CC      := riscv64-linux-gnu-gcc-14
CXX     := riscv64-linux-gnu-g++-14
FLAGS   := -O2 -march=rv64gcv -I.. -DNDEBUG

bench_ibilinear: bench_ibilinear_gem5.cc $(OBJS)
	$(CXX) $(FLAGS) -static $^ -o $@

%.o: $(KDIR)/f32-ibilinear-chw-%.c
	$(CC) $(FLAGS) -c $< -o $@

clean: 
	rm -f *.o bench_ibilinear
