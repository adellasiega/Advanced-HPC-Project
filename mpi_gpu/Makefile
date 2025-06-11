CC = mpicc
CFLAGS = -fopenmp -Wall -O3
UTILS_OBJ = utils.o

# Targets for each communication type
DIRS = one-side-communication two-side-communication

.PHONY: all clean

all: $(DIRS)

$(UTILS_OBJ): utils.c utils.h
	$(CC) $(CFLAGS) -c utils.c -o $(UTILS_OBJ)

one-side-communication: $(UTILS_OBJ)
	$(CC) $(CFLAGS) one-side-communication/source.c $(UTILS_OBJ) -o one-side-communication/executable

two-side-communication: $(UTILS_OBJ)
	$(CC) $(CFLAGS) two-side-communication/source.c $(UTILS_OBJ) -o two-side-communication/executable

clean:
	rm -f $(UTILS_OBJ)
	rm -f one-side-communication/executable two-side-communication/executable
