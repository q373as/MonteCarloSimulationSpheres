FFTW_INCLUDE = /home/alexander/fsl/envs/fftw_env/include
FFTW_LIB = /home/alexander/fsl/envs/fftw_env/lib

PYTHON_INCLUDE = /home/alexander/fsl/envs/fftw_env/include/python3.13  # Pfad zu den Python-Headern
PYTHON_LIB = /home/alexander/fsl/envs/fftw_env/lib  # Pfad zu den Python-Bibliotheken

NUMPY_INCLUDE = /home/alexander/fsl/envs/fftw_env/lib/python3.13/site-packages/numpy/_core/include  # Pfad zu den NumPy-Headern

CXX = g++
CXXFLAGS = -fopenmp -I$(FFTW_INCLUDE) -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE)
LDFLAGS = -L$(FFTW_LIB) -lfftw3 -L$(PYTHON_LIB) -lpython3.11 -fopenmp

SRC = src.cpp
OBJ = src.o
OUT = main

all: $(OUT)

$(OUT): $(OBJ)
	$(CXX) $(OBJ) -o $(OUT) $(LDFLAGS)

$(OBJ): $(SRC)
	$(CXX) $(CXXFLAGS) -c $(SRC)

clean:
	rm -f $(OBJ) $(OUT)

run: $(OUT)
	./$(OUT)

