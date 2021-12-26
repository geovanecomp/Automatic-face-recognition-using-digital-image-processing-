# Automatic Face Recognition Using Digital Image Processing.
The objective of this project is to apply Image Processing algorithms to improve the image quality, distributing uniformly the image intensity, and removing noises. Then, analyze how Facial Recognition Methods would recognize faces in different noise conditions.

The image sources are very important and are directly related to the success rate of these recognition methods. Images that are naturally perfect, which means no noises and good luminosity, tend to have better results. However, noisy images need to be fixed or prepared before being applied in the recognition methods to achieve a better success rate.

The following example shows how important image processing is when applied into noisy images:
<p align="center">
  <img width="900" src="https://user-images.githubusercontent.com/3878792/147414221-ffc5e949-1a76-4953-a936-41eec7fb5636.png">
</p>
    

As you can see, the image on the right after being processed by the `Histogram Equalization` is clearer than before.


Because of that, multiple image processing methods were implemented, and due to that, would be very difficult to create distinct combinations between them. So, to make it easier, was implemented the **Design Pattern Chain of Responsibility**, making the addition of new methods or the combination between them extremely faster and simple. 

Through this Design Pattern,  four image databases were created based on the combination of the Image Processing Algorithms: Histogram Equalization, Laplacian Filter, Laplacian Filter with Suavization Filter, and finally, all these filters in the same order that were presented.

Each processed database was submitted to two face recognitions methods, the Normalized Correlation and Eigenfaces.

The new databases were created using the FEI database (http://fei.edu.br/~cet/facedatabase.html) as original source.
FEI database is composed of images with high quality and controlled environment, considering several faces in different positions and luminosity conditions.
![examples_variations](https://user-images.githubusercontent.com/3878792/147413993-99310483-72d4-4c09-be18-e55ddab0982e.jpeg)

As result of the facial methods and each combination of image processing, the average success rate are:


                                  Correlation (%)       Eigenfaces (%)
	Normal Database                   82,50                 75,00
	Histogram Equalization            77,50                 80,00
	Laplacian Filter                  85,83                 70,83
	Laplacian and Suavization Filter  86,66                 73,33
	All Methods                       72,50                 80,00
								  
					 

In this case, the bests images processing methods utilized were the Laplacian with Suavization Filter and Histogram Equalization.


This is a very short resume of my final Computer Engineer Degree project from Rio de Janeiro State University (UERJ). Feel free to send me an email (geovane.pacheco99@gmail.com) for questions, critiques, or compliments.
