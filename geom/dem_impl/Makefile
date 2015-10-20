################################################################################
CXX         = g++
CXXFLAGS    = -Wall -Wno-unused

OPENCV_INCS = $$(pkg-config --cflags opencv)
OPENCV_LIBS = $$(pkg-config --libs opencv)
BOOST_INCS  = 
BOOST_LIBS  = -lboost_python
NUMPY_INCS  = -I/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include
PYTHON_INCS = $$(pkg-config --cflags python) $(NUMPY_INCS)
PYTHON_LIBS = $$(pkg-config --libs python)
NP2CV_DIR   = $(shell pwd)/../../core/np2cv
NP2CV_INCS  = -I$(NP2CV_DIR)
NP2CV_LIBS  = $(NP2CV_DIR)/np2cv.so
INCLUDES    = $(OPENCV_INCS) $(BOOST_INCS) $(PYTHON_INCS) $(NP2CV_INCS)
LIBS        = $(OPENCV_LIBS) $(BOOST_LIBS) $(PYTHON_LIBS) $(NP2CV_LIBS)

TARGET      = $(shell basename $(shell pwd))

################################################################################

all: lib

test: $(TARGET).out

lib: $(TARGET).so

install:
	if [ ! -f ../$(TARGET).so ]; then cd .. && ln -s $(TARGET)/$(TARGET).so; fi

.PHONY: clean
clean: 
	rm -f *.o *.so *.out

################################################################################

$(TARGET).out: $(TARGET).cpp
	cd $(NP2CV_DIR) && make
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(TARGET).cpp -DUNITTEST
	$(CXX) $(TARGET).o $(LIBS) -o $@
	#install_name_tool -change np2cv.so $(NP2CV_DIR)/np2cv.so $@

$(TARGET).so: $(TARGET).cpp
	brew link python
	cd $(NP2CV_DIR) && make
	$(CXX) $(CXXFLAGS) $(INCLUDES) -fPIC -c $(TARGET).cpp
	$(CXX) -shared $(TARGET).o $(LIBS) -o $@
	install_name_tool -change np2cv.so $(NP2CV_DIR)/np2cv.so $@
	brew unlink python

################################################################################

