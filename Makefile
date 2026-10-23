CXX = g++
CXXFLAGS = -Wall -O3 -std=c++0x -march=native
PYTHON ?= python3

# comment the following flags if you do not want to SSE instructions
DFLAG += -DUSESSE

# comment the following flags if you do not want to use OpenMP
DFLAG += -DUSEOMP
CXXFLAGS += -fopenmp

.PHONY: all test clean

all: ffm-train ffm-predict

ffm-train: ffm-train.cpp ffm.o timer.o
	$(CXX) $(CXXFLAGS) $(DFLAG) -o $@ $^

ffm-predict: ffm-predict.cpp ffm.o timer.o
	$(CXX) $(CXXFLAGS) $(DFLAG) -o $@ $^

ffm.o: ffm.cpp ffm.h timer.o
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<

timer.o: timer.cpp timer.h
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<

tests/early_stopping_api: tests/early_stopping_api.cpp ffm.h ffm.o timer.o
	$(CXX) $(CXXFLAGS) $(DFLAG) -o $@ $< ffm.o timer.o

test: all tests/early_stopping_api
	$(PYTHON) tests/test_early_stopping.py --trainer ./ffm-train --predictor ./ffm-predict --api ./tests/early_stopping_api

clean:
	rm -f ffm-train ffm-predict ffm.o timer.o tests/early_stopping_api
