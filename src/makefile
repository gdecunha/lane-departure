# Makefile
CXX = g++
CXXFLAGS = -std=c++17 `pkg-config --cflags --libs opencv4`

all: still video

still: still.cpp
	$(CXX) -g still.cpp -o still $(CXXFLAGS)

video: video.cpp
	$(CXX) -g video.cpp -o video $(CXXFLAGS)

detector: LaneDetector.cpp
	$(CXX) -g LaneDetector.cpp -o detector $(CXXFLAGS)

clean:
	rm -f still video detector
