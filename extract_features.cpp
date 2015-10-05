#include <iostream>
#include <vector>
#include <sstream>

#include <png++/png.hpp>

const int BACKGROUND_DEPTH = 750;
const int OFFSET = 4500;

using namespace std;

int main() {
	string filename;
	int block_size;
	cin >> filename >> block_size;
	png::image<png::gray_pixel_16> image(filename);

	int w = image.get_width();
	int h = image.get_height();

	png::image<png::gray_pixel_16> scaled_image(w/block_size,h/block_size);

	for (int i = 0; i < w; i+=block_size) {
		for (int j = 0; j < h; j+=block_size) {	
			int depth = 0;		
			for (int k = i; k < i+block_size; k++) {
				for (int l = j; l < j+block_size; l++) {
					depth += image.get_pixel(k,l);
				}
			}
			scaled_image.set_pixel(i/block_size,j/block_size,depth/(block_size*block_size));
		}
	}
	
	w = scaled_image.get_width();
	h = scaled_image.get_height();
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < h; j++) {
			int depth = scaled_image.get_pixel(i,j);
			int local_offset = (OFFSET/block_size)/depth;
	
			int left = (i-local_offset > 0)? scaled_image.get_pixel(i-local_offset,j) : BACKGROUND_DEPTH;
			int right = (i+local_offset < w)? scaled_image.get_pixel(i+local_offset,j) : BACKGROUND_DEPTH;

			// I'm not sure where 0,0 is located though, up and down could actually be inverted (doesn't matter anyway)
			int up = (j-local_offset > 0)? scaled_image.get_pixel(i,j-local_offset) : BACKGROUND_DEPTH;
			int down = (j+local_offset < h)? scaled_image.get_pixel(i,j+local_offset) : BACKGROUND_DEPTH;
	
			cout << left-depth << " " << up-depth << " " << 
				right-depth << " " << down-depth << " ";
		}
	}
	cout << endl;
}
