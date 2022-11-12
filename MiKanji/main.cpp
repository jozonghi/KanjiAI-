/*
* Joseph Zonghi
* High Performance Architecture
* Final Project: Kanji Character Dataset Creator
* December 3 2021
*/

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <string>
using namespace std;
#include "common.h"
#include <Windows.h>
#include "utf8.h"
#include <sys/stat.h>
#include <conio.h>
#include <sys/types.h>
#include <stdio.h>

/*Unused
Used for printing out the contents of a given matrix*/
void printOut(unsigned char* M, int num_images, int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size*num_images; j++) {
			cout << +M[i * size * num_images + j];
		}
		cout << "\n";
	}
}

int main() {
	//Set up output console to handle unicode
	SetConsoleOutputCP(65001);
	setvbuf(stdout, nullptr, _IOFBF, 1000);

	string kanjilist;
	clock_t startt, endt;
	float time;

	string fname;

	int size = 64;
	int num_images = 300;
	unsigned char* output = new unsigned char[num_images * size * size];
	
	//Read Kanji input file
	ifstream infile("joyo.txt");

	//Iterate over each character in file
	while (infile >> noskipws >> kanjilist) {
		//cout << kanjilist;
	}

	char* str = (char*)kanjilist.c_str();
	char* str_i = str;
	char* end = str + strlen(str) + 1;
	int i = 0;

	startt = clock();

	do {


		uint32_t code = utf8::next(str_i, end);
		if (code == 0) continue;

		unsigned char symbol[5] = { 0 };

		//Use utf8 package to handle utf8 characters (Kanji)
		utf8::append(code, symbol);

		//Load the base Kanji image (black character on white background)
		fname = "E:/MiKanji/BaseImages/"+std::to_string(i)+".png";
		//read it as grayscale
		cv::Mat img = cv::imread(fname, cv::IMREAD_GRAYSCALE);
		if (img.empty()) {
			cout << "Base Image Not Found" << endl;
			return 1;
		}
		//convert image to char array
		unsigned char* grayscale = img.data;

		//call GPU function
		bool result = MakeDataset(num_images, size, grayscale, output);
		if (!result) {
			cout << "Error when calling the GPU Function\n";
			return 1;
		}

		//Convert GPU output from char to opencv Mat
		cv::Mat img_out = cv::Mat(size, size * num_images, CV_8UC1, output);
		std::stringstream fname;
		fname << "E:/MiKanji/GPUWrite/Classes/";
		cv::Mat sub_img;

		//choose whether or not to split the mega-image into sub-images
		int split = 1;
		if (split) {
			for (int j = 0; j < num_images; j++) {
				std::stringstream fname;
				fname << "E:/MiKanji/GPUWrite/Classes/";
				fname << symbol << "/" << j << ".png";
				//Get the rows and columns to crop from the mega image
				cv::Range rows(0, size - 1);
				cv::Range cols(j * size, j * size + size - 1);
				//crop mega image
				sub_img = img_out(rows, cols);
				//write file
				cv::imwrite(fname.str(), sub_img);
			}
		}
		else {
			string large_file;
			large_file = "E:/MiKanji/GPUWrite/" + to_string(i) + ".png";
			//write mega image as is
			cv::imwrite(large_file, img_out);

		}

		cout << "Kanji " << symbol << ": " << i << " of " << "2445" << " done!\n";

		i++;

		//iterate through each Kanji in the string
	} while (str_i < end);

	endt = clock();
	time = (float)(endt - startt) * 1000 / (float)CLOCKS_PER_SEC;
	//report the time taken for all operations
	cout << "Time Taken for " << num_images << " images: " << time << " ms";


	return 0;
}