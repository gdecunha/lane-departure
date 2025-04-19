# Makefile
CXX = g++
CXXFLAGS = -std=c++17 `pkg-config --cflags --libs opencv4`
TARGET = tester

all: $(TARGET)

$(TARGET): tester.cpp
	$(CXX) tester.cpp -o $(TARGET) $(CXXFLAGS)

clean:
	rm -f $(TARGET)
