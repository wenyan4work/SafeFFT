CXX=icpc

# for mkl and intel compiler
FFTINC= -mkl -DFFTW3_MKL -I$(MKLROOT)/include/fftw
FFTLIB= -mkl

# for FFTW3
#FFTINC= -I/path_to_fftw3.h
#FFTLIB= -L/path_to_libfftw3 -lfftw3_omp -lfftw3 # or -lfftw3_threads -lfftw3  

CXXFLAGS= $(FFTINC) -std=c++11 -qopenmp -O3 -xcore-avx2 -DNDEBUG
LDLIBS= $(FFTLIB) $(CXXFLAGS)

RM = rm -f
MKDIRS = mkdir -p

BINDIR = ./bin
SRCDIR = ./test
OBJDIR = ./obj
INCDIR = ./include

INC = \
	$(INCDIR)/AlignedMemory.hpp \
	$(INCDIR)/SafeFFT.hpp

TARGET_BIN = \
       $(BINDIR)/TestForwardBackward \
       $(BINDIR)/Test1DSmall \
       $(BINDIR)/Test1DLarge \
       $(BINDIR)/Test2DSmall \
       $(BINDIR)/Test2DLarge \
       $(BINDIR)/Test3DSmall

all : $(TARGET_BIN)

$(BINDIR)/%: $(OBJDIR)/%.o
	-@$(MKDIRS) $(dir $@)
	$(CXX) $(CXXFLAGS) $^ $(LDLIBS) -o $@
#	$(CXX) $^ $(LDLIBS) -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -c $^ -o $@

clean:
	$(RM) -r $(BINDIR)/* $(OBJDIR)/*
	$(RM) *~ */*~

