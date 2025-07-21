CC = mpicc
CFLAGS = -O3 -march=native -Wall -Wextra -Wpedantic -Werror -lm -fast -Minfo=all -v -Mneginfo -gpu=cc80 -target=gpu -gpu=nomanaged -mp=multicore,gpu 
UTILS_OBJ = utils.o

# Targets for each communication type
DIRS = one-side-communication one-side-comm-complete-parall two-side-communication

.PHONY: all clean

all: $(DIRS)

$(UTILS_OBJ): utils.c utils.h
	$(CC) $(CFLAGS) -c utils.c -o $(UTILS_OBJ)

one-side-communication/executable: one-side-communication/source.c $(UTILS_OBJ)
	$(CC) $(CFLAGS) $< $(UTILS_OBJ) -o $@

one-side-comm-complete-parall/executable: one-side-comm-complete-parall/source.c $(UTILS_OBJ)
	$(CC) $(CFLAGS) $< $(UTILS_OBJ) -o $@

two-side-communication/executable: two-side-communication/source.c $(UTILS_OBJ)
	$(CC) $(CFLAGS) $< $(UTILS_OBJ) -o $@

all: one-side-communication/executable one-side-comm-complete-parall/executable two-side-communication/executable


clean:
	rm -f $(UTILS_OBJ)
	rm -f one-side-communication/executable one-side-comm-complete-parall/executable two-side-communication/executable
