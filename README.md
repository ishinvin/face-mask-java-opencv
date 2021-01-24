# Face Mask Detection with OpenCV

A simple face mask detection application with OpenCV.

We use 2 pre-trained models of face detection and face-mask classification.

Then, we use OpenCV to run the model inference.



### Installing

download the project
``` shell script
git clone https://github.com/ishin-pie/face-mask-java-opencv.git
```

move into project directory
``` shell script
cd face-mask-java-opencv
```

build jar (for Linux/Mac user)
``` shell script
./mvnw clean package
```
build jar (for Windows user)
``` shell script
mvnw.cmd clean package
```

### Running demo

Image detection
``` shell script
java -jar target/face-mask-java-opencv-1.0.jar demo/image.jpg
```

### Result
![face-mask-detection](demo/demo.png)



### Acknowledgments

* https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/
* https://github.com/chandrikadeb7/Face-Mask-Detection.git
