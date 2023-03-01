#!/bin/bash

set -x

docker pull fedora:36
CONTAINER=init-builder
docker stop $CONTAINER
docker rm -f $CONTAINER
SRCDIR=./.initrd
mkdir -p ${SRCDIR}
pushd ${SRCDIR}
if [ ! -d init-tools ]; then
	git clone https://github.com/unikernelLinux/init-tools.git
fi
popd

ABSINIT=`readlink -f ${SRCDIR}`
docker run --rm --privileged --name=${CONTAINER} -v ${ABSINIT}:/src -dit fedora:36 /bin/bash
docker exec -it $CONTAINER dnf -y update
docker exec -it $CONTAINER dnf -y install sed elfutils-libelf-devel bc hostname perl dropbear \
	msr-tools wget dnf-plugins-core bzip2 curl xz cpio shadow-utils
docker exec -w /src/init-tools/ -it $CONTAINER rm -rf ukl-initrd
docker exec -w /src/init-tools/ -it $CONTAINER ./set-passwd.sh
docker exec -w /src/init-tools/ -it $CONTAINER ./buildinitrd.sh ukl-initrd
docker exec -w /src/init-tools/ -it $CONTAINER rm -rf ukl-initrd
mv ${SRCDIR}/init-tools/ukl-initrd.cpio.xz .
docker stop $CONTAINER
docker rm -f $CONTAINER
