/*
* Joseph Zonghi
* High Performance Architecture
* Final Project
* December 3 2021
*/
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
using namespace std;

#include "common.h"

/*
* Function used to test and print the values of a and c (cpu and gpu)
* This is not used in the current main version, but I included it for posterity.
*/
/*
__global__ void printOut(unsigned char* M, int rows, int cols) {
	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++)
			printf("%f ", M[row * cols + col]);
		printf("> ");
		printf("\n");
	}
}*/

//Check if the given CUDA function returns an error
bool checkForError(cudaError_t error) {
	//If no error, do nothing and return true
	if (error == cudaSuccess) {
		return true;
	}
	//If error, print name of error and associated description
	else {
		cout << cudaGetErrorName(error) << ": ";
		cout << cudaGetErrorString(error) << endl;
		return false;
	}
}

//Kernel for coloring the sub-image 
/*
* Pd: output array
* colors: array holding the colors per sub image
* num_images: number of images per kanji
* size: size (NxN) of sub image
*/
__global__ void colorKernel(unsigned char* Pd, unsigned char* colors, int num_images, int size) {

	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	Pd[ty * size * num_images + tx] = colors[(int)floorf(tx/size)];

}

/*Kernel for drawing the circles on the image
* Pd: output image
* circles: array holding number of circles to plot per sub image
* num_images: number of images per Kanji
* size: size (NxN) of sub image
* circle_colors: array holding the color of the circles per sub image
* radius: radius of the circles
* x_coords: 2D array of x coordinates for all circles
* y_coords: 2D array of y coordinates for all circles
*/
__global__ void circleKernel(unsigned char* Pd, unsigned char* circles, int num_images, int size, unsigned char* circle_colors, int radius,  int* x_coords,  int* y_coords) {

	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	int circle_num = circles[(int)floorf(tx / size)];

	//Loop through number of circles to draw
	for (int i = 0; i < circle_num; i++) {
		//Get circle i's center coordinates
		int x = x_coords[(int)floorf(tx / size) * 20 + i];
		int y = y_coords[(int)floorf(tx / size) * 20 + i];

		//get thread's x position in sub image
		int blockx = 0;
		if (blockIdx.x % 2 == 1) {
			 blockx = threadIdx.x + blockDim.x;
		}
		else {
			 blockx = threadIdx.x;
		}
		
		//if thread's (x,y) satisfy the circle equation, plot that point
		//circle equation: (x - a) ^ 2 + (y - b) ^ 2 <= radius ^ 2
		if( (blockx - x)*(blockx- x) + (ty - y)*(ty - y) <= radius*radius){
			Pd[ty * size * num_images + tx] = circle_colors[(int)floorf(tx / size)];
		}
	}

}

/* Draw outline of character (before calling the centerKernel to draw the center)
* Pd: output array
* Md: base image array to use as reference
* num_images: number of images per kanji
* size: size (NxN) of sub image
* x_offset, y_offset: offset of base image array from its original position
*/
__global__ void outlineKernel(unsigned char* Pd, unsigned char* Md, int num_images, int size,  int* x_offset,  int* y_offset) {

	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	//get character's x and y offset values (positive or negative)
	int x_off = x_offset[(int)floorf(tx/size)];
	int y_off = y_offset[(int)floorf(tx / size)];

	//get thread's x position in sub image
	int blockx = 0;
	if (blockIdx.x % 2 == 1) {
		blockx = threadIdx.x + blockDim.x;
	}
	else {
		blockx = threadIdx.x;
	}

	//check boundaries
	if (blockx + - 5 > 0 && ty + - 5 > 0 && blockx + 5 < size && ty + 4 < size) {
		//If the original image has a black pixel corresponding to the current thread, plot that pixel shifted in many directions
		if (Md[ty * size + blockx] != 255) {
			Pd[(ty + y_off) * size * num_images + (tx + x_off)] = 0;
			Pd[(ty + y_off) * size * num_images + (tx + x_off + 1)] = 0;
			Pd[(ty + y_off) * size * num_images + (tx + x_off - 1)] = 0;
			Pd[(ty + y_off) * size * num_images + (tx + x_off + 2)] = 0;
			Pd[(ty + y_off) * size * num_images + (tx + x_off - 2)] = 0;

			Pd[(ty + y_off + 1) * size * num_images + (tx + x_off + 1)] = 0;
			Pd[(ty + y_off - 1) * size * num_images + (tx + x_off - 1)] = 0;
			Pd[(ty + y_off + 2) * size * num_images + (tx + x_off + 1)] = 0;
			Pd[(ty + y_off - 2) * size * num_images + (tx + x_off - 1)] = 0;
			Pd[(ty + y_off + 1) * size * num_images + (tx + x_off + 2)] = 0;
			Pd[(ty + y_off - 1) * size * num_images + (tx + x_off - 2)] = 0;
			Pd[(ty + y_off + 2) * size * num_images + (tx + x_off + 2)] = 0;
			Pd[(ty + y_off - 2) * size * num_images + (tx + x_off - 2)] = 0;

			Pd[(ty + y_off + 1) * size * num_images + (tx + x_off)] = 0;
			Pd[(ty + y_off - 1) * size * num_images + (tx + x_off)] = 0;
			Pd[(ty + y_off + 2) * size * num_images + (tx + x_off)] = 0;
			Pd[(ty + y_off - 2) * size * num_images + (tx + x_off)] = 0;

			Pd[(ty + y_off + 1) * size * num_images + (tx + x_off - 1)] = 0;
			Pd[(ty + y_off - 1) * size * num_images + (tx + x_off + 1)] = 0;
			Pd[(ty + y_off + 1) * size * num_images + (tx + x_off - 2)] = 0;
			Pd[(ty + y_off - 1) * size * num_images + (tx + x_off + 2)] = 0;
			Pd[(ty + y_off + 2) * size * num_images + (tx + x_off - 2)] = 0;
			Pd[(ty + y_off - 2) * size * num_images + (tx + x_off + 2)] = 0;
			Pd[(ty + y_off + 2) * size * num_images + (tx + x_off - 1)] = 0;
			Pd[(ty + y_off - 2) * size * num_images + (tx + x_off + 1)] = 0;

			Pd[(ty + y_off - 1) * size * num_images + (tx + x_off + 1)] = 0;
			Pd[(ty + y_off + 1) * size * num_images + (tx + x_off - 1)] = 0;
			Pd[(ty + y_off - 1) * size * num_images + (tx + x_off + 2)] = 0;
			Pd[(ty + y_off + 1) * size * num_images + (tx + x_off - 2)] = 0;
			Pd[(ty + y_off - 2) * size * num_images + (tx + x_off + 1)] = 0;
			Pd[(ty + y_off + 2) * size * num_images + (tx + x_off - 1)] = 0;
			Pd[(ty + y_off - 2) * size * num_images + (tx + x_off + 2)] = 0;
			Pd[(ty + y_off + 2) * size * num_images + (tx + x_off - 2)] = 0;
		}
	}

}

/* Draw white center of Kanji (assumed after outline is done)
* Pd: output array
* Md: base image array to use as reference
* num_images: number of images per kanji
* size: size (NxN) of sub image
* x_offset, y_offset: offset of base image array from its original position
*/
__global__ void centerKernel(unsigned char* Pd, unsigned char* Md, int num_images, int size, int* x_offset, int* y_offset) {

	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	//get character's x and y offset
	int x_off = x_offset[(int)floorf(tx / size)];
	int y_off = y_offset[(int)floorf(tx / size)];

	//get thread's x position in sub image
	int blockx = 0;
	if (blockIdx.x % 2 == 1) {
		blockx = threadIdx.x + blockDim.x;
	}
	else {
		blockx = threadIdx.x;
	}

	//boundary checking
	if (blockx + -5 > 0 && ty + -5 > 0 && blockx + 5 < size && ty + 5 < size) {
		//draw white center of character offset from the base example image
		if (Md[ty * size + blockx] != 255) {
			Pd[(ty + y_off) * size * num_images + (tx + x_off)] = 255;
		}
	}

}


bool MakeDataset(int num_images, int size, unsigned char* img, unsigned char* output) {

	//designate block tile size
	int TILE_SIZE = 32;
	//size of sub image
	int mid_bytes = size * size * sizeof(unsigned char);
	//size of mega image
	int total_bytes = size * size * num_images * sizeof(unsigned char);

	//specify randomness properties and ranges
	int radius = 6;
	int max_circles = 20;
	int min_circles = 5;
	int max_offset = 2;
	int min_offset = -2;
	int circle_num = 0;

	unsigned char* Md;
	unsigned char* Pd;

	//create arrays used for handling various values per sub image
	unsigned char* colors = new unsigned char[num_images];
	unsigned char* circle_colors = new unsigned char[num_images];
	unsigned char* circles = new unsigned char[num_images];

	int* x_coords = new int[num_images * max_circles];
	int* y_coords = new int[num_images * max_circles];

	int* x_offset = new int[num_images];
	int* y_offset = new int[num_images];

	//Create device pointers
	unsigned char* colors_d;
	unsigned char* circle_colors_d;
	unsigned char* circles_d;
	int* x_coords_d;
	int* y_coords_d;
	int* x_offset_d;
	int* y_offset_d;

	//fill reference arrays with mostly random numbers 
	for (int i = 0; i < num_images; i++) {
		//any color background
		colors[i] = (unsigned char) rand() % 255 + 1;
		//any color circles
		circle_colors[i] = (unsigned char) rand() % 255 + 1;
		//random number of circles in range(min_circles, max_circles)
		circle_num = (unsigned char)rand() % (max_circles - min_circles + 1) + min_circles;
		circles[i] = circle_num;
		//random circle center coordinates
		for (int j = 0; j < circle_num; j++) {
			x_coords[i * circle_num + j] = (int) rand() % (54 - 6 + 1) + 6;
			y_coords[i * circle_num + j] = (int) rand() % (54 - 6 + 1) + 6;
		}
		//random x,y offset of character from center
		x_offset[i] = (unsigned char)rand() % (max_offset - min_offset + 1) + (min_offset);
		y_offset[i] = (unsigned char)rand() % (max_offset - min_offset + 1) + (min_offset);
	}

	//allocate all device memory
	if (!checkForError(cudaMalloc((void**)&Md, mid_bytes)))return false;
	if (!checkForError(cudaMalloc((void**)&Pd, total_bytes)))return false;
	if (!checkForError(cudaMalloc((void**)&colors_d, num_images * sizeof(unsigned char))))return false;
	if (!checkForError(cudaMalloc((void**)&circle_colors_d, num_images * sizeof(unsigned char))))return false;
	if (!checkForError(cudaMalloc((void**)&circles_d, num_images * sizeof(unsigned char))))return false;
	if (!checkForError(cudaMalloc((void**)&x_coords_d, num_images * max_circles * sizeof(int))))return false;
	if (!checkForError(cudaMalloc((void**)&y_coords_d, num_images * max_circles * sizeof(int))))return false;
	if (!checkForError(cudaMalloc((void**)&x_offset_d, num_images * sizeof(int))))return false;
	if (!checkForError(cudaMalloc((void**)&y_offset_d, num_images * sizeof(int))))return false;
	
	//copy base image and random parameters into their corresponding device memory
	if (!checkForError(cudaMemcpy(Md, img, mid_bytes, cudaMemcpyHostToDevice))) return false;
	if (!checkForError(cudaMemcpy(colors_d, colors, num_images * sizeof(unsigned char), cudaMemcpyHostToDevice))) return false;
	if (!checkForError(cudaMemcpy(circle_colors_d, circle_colors, num_images * sizeof(unsigned char), cudaMemcpyHostToDevice))) return false;
	if (!checkForError(cudaMemcpy(circles_d, circles, num_images * sizeof(unsigned char), cudaMemcpyHostToDevice))) return false;
	if (!checkForError(cudaMemcpy(x_coords_d, x_coords, num_images * max_circles * sizeof(int), cudaMemcpyHostToDevice))) return false;
	if (!checkForError(cudaMemcpy(y_coords_d, y_coords, num_images * max_circles * sizeof(int), cudaMemcpyHostToDevice))) return false;
	if (!checkForError(cudaMemcpy(x_offset_d, x_offset, num_images * sizeof(int), cudaMemcpyHostToDevice))) return false;
	if (!checkForError(cudaMemcpy(y_offset_d, y_offset, num_images * sizeof(int), cudaMemcpyHostToDevice))) return false;
	

	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	dim3 dimGrid((int)ceil((float)size*num_images / (float)TILE_SIZE),
		(int)ceil((float)size / (float)TILE_SIZE));

	//color background
	colorKernel << <dimGrid, dimBlock >> > (Pd, colors_d, num_images, size);
	if (!checkForError(cudaThreadSynchronize()))return false;

	//draw circles
	circleKernel << <dimGrid, dimBlock >> > (Pd, circles_d, num_images, size, circle_colors_d, radius, x_coords_d, y_coords_d);
	if (!checkForError(cudaThreadSynchronize()))return false;

	//draw outline
	outlineKernel << <dimGrid, dimBlock >> > (Pd, Md, num_images, size, x_offset_d, y_offset_d);
	if (!checkForError(cudaThreadSynchronize()))return false;

	//draw white center
	centerKernel << <dimGrid, dimBlock >> > (Pd, Md, num_images, size, x_offset_d, y_offset_d);
	if (!checkForError(cudaThreadSynchronize()))return false;

	//copy resulting modified mega image array into host memory
	if (!checkForError(cudaMemcpy(output, Pd, total_bytes, cudaMemcpyDeviceToHost))) return false;
	

	//free used memory
	if (!checkForError(cudaFree(Pd))) return false;
	if (!checkForError(cudaFree(Md))) return false;
	if (!checkForError(cudaFree(colors_d))) return false;
	if (!checkForError(cudaFree(circle_colors_d))) return false;
	if (!checkForError(cudaFree(x_coords_d))) return false;
	if (!checkForError(cudaFree(y_coords_d))) return false;
	if (!checkForError(cudaFree(x_offset_d))) return false;
	if (!checkForError(cudaFree(y_offset_d))) return false;
	delete[] colors; delete[] circle_colors; delete[] circles;
	delete[] x_coords; delete[] y_coords;
	delete[] x_offset; delete[] y_offset;

	return true;
}