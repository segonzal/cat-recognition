generate_descriptors: generate_descriptors.cpp
	g++ generate_descriptors.cpp -O3 -o generate_descriptors -lopencv_core -lopencv_highgui -lopencv_features2d

kmeans: kmeans.cpp
	g++ kmeans.cpp -O3 -o kmeans -lopencv_core -lopencv_highgui -lopencv_features2d

histograms: histograms.cpp
	g++  histograms.cpp -O3  -o histograms -lopencv_core -lopencv_highgui -lopencv_features2d -lopencv_imgproc

segmentation: segmentation.cpp
	g++  segmentation.cpp -O3  -o segmentation -lopencv_core -lopencv_highgui -lopencv_features2d -lopencv_imgproc

histograms2: histograms.cpp
	g++ histograms.cpp -O3  -o histograms2 -lopencv_core -lopencv_highgui -lopencv_features2d -lopencv_imgproc
clean:
	rm kmeans generate_descriptors 
