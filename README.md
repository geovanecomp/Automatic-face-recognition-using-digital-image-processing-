# Automatic Face Recognition Using Digital Image Processing.
The objective of this project is apply image processing methods to improve the image quality, distributing uniformly the image intensity, or remove noises. To make that tasks easier, was implemented the design pattern Chain of Responsibility, making the addition of new methods or the combination between they extremely faster and simple. Using these methods, was created four image databases to apply two face recognitions methods, the Normalized Correlation and Eigenfaces method.

The image processing methods implemented was: Histogram Equalization, Laplacian Filter and Suavization Filter. Using the design pattern Chain of Responsibility, were used four combinations of these methods to create new database images to apply to the face recognition, they are: Histogram Equalization, Laplacian Filter, Laplacian Filter with Suavization Filter and finally, all these filters in the same order that were apresented.

The success rate of these recognition methods applied in four image database each one with one combination of image processing depends of the source images. Images naturally perfect (no noises and good luminocity), hardly will be improved by image processing.

Using the FEI database (http://fei.edu.br/~cet/facedatabase.html), that is a database image with high quality and controlled environment, and using the Correlation and the Eigenfaces method to differents people and differents image dimensions, the average results was:


                                  Correlation (%)       Eigenfaces (%)
	Normal Database                   82,50                 75,00
	Laplacian Filter                  85,83                 70,83
	Laplacian and Suavization Filter  86,66                 73,33
	Histogram Equalization            77,50                 80,00
	All Methods                       72,50                 80,00
								  
					 

In this case, the bests images processing methods utilized were the Laplacian with Suavization Filter and Histogram Equalization.


This is a very short resume of my final project for course of Computing Engineering of the Rio de Janeiro State University (UERJ). Feel free to send me an email (geovane.pacheco99@gmail.com) for questions, critiques or compliments.
