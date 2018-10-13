#/bin/bash

download() {
	curl https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$1.zip --output $1.zip
	unzip $1.zip -d ./datasets
	rm $1.zip
}

download iphone2dslr_flower
download summer2winter_yosemite
