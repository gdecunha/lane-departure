# Makefile
CXX = g++
CXXFLAGS = -std=c++17 `pkg-config --cflags --libs opencv4`

all: still video

still: still.cpp
	$(CXX) still.cpp -o still $(CXXFLAGS)

video: video.cpp
	$(CXX) video.cpp -o video $(CXXFLAGS)


clean:
	rm -f still video
