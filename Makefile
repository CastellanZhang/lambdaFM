all:
	g++ -O3 lambdafm_train.cpp src/Frame/pc_frame.cpp src/Utils/utils.cpp -I . -std=c++0x -o bin/lambdafm_train -lpthread
	g++ -O3 lambdafm_predict.cpp src/Frame/pc_frame.cpp src/Utils/utils.cpp -I . -std=c++0x -o bin/lambdafm_predict -lpthread
