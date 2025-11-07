# Makefile
all: thermo_driver.ko libthermo.so

thermo_driver.ko: thermo_driver.c
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

libthermo.so: thermo_cuda.cu
	nvcc -o libthermo.so --compiler-options -fPIC -shared thermo_cuda.cu -lcuda

install: thermo_driver.ko
	sudo insmod thermo_driver.ko
	sudo mknod /dev/thermo0 c $(shell cat /proc/devices | grep thermo0 | awk '{print $$1}') 0

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
	rm -f libthermo.so
