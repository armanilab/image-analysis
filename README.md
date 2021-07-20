# image-analysis
Masterprogram

This program does the main image analysis for diffraction images. There are two different types of graphs / data it outputs. The first one is a maximum vs y - axis graph, where the program scans each row of pixels to find the maximum grayscale value. The maximum values will be graphed on the y -axis, while the x axis will be the height of the image, ranging from 1 pixel to however tall the image is. The other graph / data it outputs is an average vs y - axis graph, which is essentially the same concept as the maximum vs y-axis graph, except it takes the average grayscale value of a particular row. 

Wavefront

This program simply scans the whole picture for the brightest sections, and then it measures the longest continuous rows / columns with grayscale values above some specific value (basically it measure bright spots). 

Measurement

This program creates a circle centered at a given value. This circle is then applied to different images, where any section outside of that circle will become "masked" (basically black). The program then sums up all the grayscale values of the pixels inside that circle.
