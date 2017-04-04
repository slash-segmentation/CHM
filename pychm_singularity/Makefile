.PHONY: clean singularity

help:
	@echo "clean - remove all build and test artifacts"
	@echo "singularity - Creates singularity image"

clean:
	rm -fr build/

singularity: clean
	@echo 'Creating Singularity image'
	mkdir -p build
	@echo "Creating sdist for python chm"
	cd ../python;python setup.py sdist
	cp -f ../python/dist/chm-0.1.tar.gz build/.
	@echo "Downloading segtools from git"
	cd build/;git clone git@github.com:slash-segmentation/segtools.git
	@echo "Creating sdist for segtools"
	cd build/segtools;python setup.py sdist
	imgfile=`echo build/pychm.img` ; \
	sudo singularity create -s 4096 $$imgfile ; \
	sudo singularity bootstrap $$imgfile centos.def; \
	echo 'Singularity image created $imgfile'
	ls -l build
