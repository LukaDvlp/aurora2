#
# Top-level Makefile
#
# Author: Kyohei Otsu <kyon@ac.jaxa.jp>
# Date:   2015-07-21
#

all:
	cd geom/dense_stereo;  make lib && make install
    #cd nongeom/rockdetect; make lib && make install
	cd loc/libviso2; make lib && make install

