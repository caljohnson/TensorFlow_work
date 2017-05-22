#displayData.py
#Week 4 Exercise 3
#Andrew Ng's coursera ML course
#Carter Johnson

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

def displayData(data, example_width):
	#displays 2D data in a nice grid

	colormap(gray)

	#compute rows, cols
	[m, n] = np.size(data)
	example_height = n/example_width

	#compute number of items to display
	display_rows = floor(np.sqrt(m))
	display_cols = ceil(m/display_rows)

	#between images padding
	pad = 1

	#setup blank display
	display_array = -np.ones(pad + display_rows * (example_height+pad),
							 pad + display_cols*(example_width+pad) )

	#copy each example into a patch on the display array
	curr_Ex = 1
	for j in range(display_rows):
		for i in range(display_cols):
			if curr_Ex > m:
				break
			#copy the patch
			#get max value of patch
			max_val = max(abs(data[curr_Ex, :]))
			display_array[pad + (j-1)*(example_height+pad) + (1:example_height),
							pad + (i-1)*(example_height+pad) + (1:example_width)] = 
							np.reshape(data[curr_Ex,:] example_height, example_width)/max_val			
			curr_Ex += 1

			if curr_Ex> n:
				break

	#display image

	return
