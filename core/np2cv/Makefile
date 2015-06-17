CXX         = g++
CXXFLAGS    = -Wall -arch x86_64
OPENCV_INCS = $$(pkg-config --cflags opencv)
OPENCV_LIBS = $$(pkg-config --libs opencv)
BOOST_INCS  = 
BOOST_LIBS  = -lboost_python
PYTHON_INCS = $$(pkg-config --cflags python) -I/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include
PYTHON_LIBS = $$(pkg-config --libs python)
INCLUDES    = $(OPENCV_INCS) $(BOOST_INCS) $(PYTHON_INCS)
LIBS        = $(OPENCV_LIBS) $(BOOST_LIBS) $(PYTHON_LIBS)

TARGET   = $(shell basename $(shell pwd))

all: $(TARGET).so

$(TARGET).so: $(TARGET).o
	$(CXX) -shared $(LIBS) $(TARGET).o -o $(TARGET).so

$(TARGET).o: $(TARGET).cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(TARGET).cpp

.PHONY: clean
clean: 
	rm -f *.o *.so
