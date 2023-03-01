# EuroSys 23 Artifact Evaluation for ukl-eurosy23-artifacts

Unikernel Linux (UKL) is a small patch to Linux and glibc which allows
you to build many programs, unmodified, as unikernels.  That means
they are linked with the Linux kernel into a final `vmlinuz` and run
in kernel space.  You can boot these kernels on baremetal or inside a
virtual machine.  Almost all features and drivers in Linux are
available for use by the unikernel.

## Requirements

* autoconf & automake
* GCC or Clang including C++ support
* GNU make
* GNU sed
* supermin (https://github.com/libguestfs/supermin)
* qemu, if you want to test boot in a virtual machine
* git
* make
* ncurses-devel
* bc
* bison
* flex
* elfutils-libelf-devel
* openssl-devel
* wget
* bzip2

On Fedora the following line should bring in what you need:

`dnf install autoconf automake gcc g++ make sed supermin qemu git make ncurses-devel bc bison flex elfutils-libelf-devel openssl-devel wget bzip2`

## Artifact specific instructions (see the rest of the steps below)

### Building UKL Images

Note: reproducing the figures will require that you be able to boot a physical machine from the produced kernel and init ramdisk. In our lab we use a TFTP boot setup, but installing the kernel on a machine and having grub boot it works as well.

Each of these kernels should be started with the following command line (with ${IP} replaced with your chosen IP address:

`console=ttyS0 net.ifnames=0 biosdevname=0 nowatchdog nopti nosmap nosmep ip=${IP}:::255.255.255.0::eth0:none nokaslr selinux=0 root=/dev/ram0 init=/init`

To reproduce Figures 1 and 2 in the paper, you will need to create three images: Linux, UKL, and UKL_BYP.

1. To create the Linux image, use the saveconfig file and build Linux from the submodule provided in the ukl-eurosy23-artifacts repository.
2. To create the UKL image, follow the instructions in the README file.
3. To create the UKL_BYP image, follow the instructions in the README file, but run the configure script with the option `--enable-bypass`.
4. All of the images need to be configured with new_lebench program, i.e., `./configure --with-program=new_lebench`

So to build all the kernels needed:
```
./configure --with-program=new_lebench
make -j`nproc`
cp vmlinuz vmlinuz.ukl
make clean
./configure --with-program=new_lebench --enable-bypass
make -j`nproc`
cp vmlinuz vmlinuz.ukl_byp
cp saveconfig linux/.config
cd linux
make clean
make -j`nproc`
cp arch/x86/boot/bzImage ../vmlinuz.linux
cd new_lebench/new_lebench
gcc -o new_lebench -UUSE_VMALLOC -UBYPASS -UUSE_MALLOC -DREF_TEST -USEND_TEST -URECV_TEST -DTHREAD_TEST -UFORK_TEST -DWRITE_TEST -DREAD_TEST -DPF_TEST -DST_PF_TEST -UDEBUG new_lebench.c
cp new_lebench ../../
```

For figures 1, 2, and 4 you will need to boot a machine with the produced kernel and init ramdisk built with `./create-initrd.sh`. The results are dropped in the `/` directory and are csv files. The ramdisk includes an ssh server with root user password of `root` so you can retrieve the results. These CSV files will be fed to the approrpiate graphing scripts.

To reproduce Figure 4 in the paper, you will need three additional images: UKL_PF_DF, UKL_PF_SS, and UKL_RET_PF_DF.

1. To create the UKL_PF_DF image, configure UKL without any special flags.
2. To create the UKL_PF_SS image, configure UKL with the `--enable-use-ist-pf` option.
3. To create the UKL_RET_PF_DF image, configure UKL with the `--enable-use-ret` option.
4. This also needs to be configured with new_lebench program, i.e., `./configure --with-program=new_lebench`

To reproduce Figure 5 in the paper, you will also need to create the UKL_RET_BYP (shortcut) image.
1. To create the UKL_RET_BYP (shortcut) image, use the `ukl-main-5.14-sc` branch of the Linux submodule in the ukl-eurosy23-artifacts repository, the `redis-ukl-sc` branch under `redis/redis` in the ukl-eurosy23-artifacts repository, and configure UKL with the `--enable-bypass` and `--enable-use-ret` option.
2. All of the images need to be configured with redis program, i.e., `./configure --with-program=redis`

You will need to use the kernel produced and the init ramdisk that can be built with `./create-initrd.sh` to boot a machine (not a VM) and then drive that with another machine on the same top of rack switch with the memtier_benchmark line below. The output of the benchmark will include a tab delemited histogram which is the input for the fig4.py script.

To reproduce Figure 6 in the paper, the configurations will be as above but the program to be configured with will be memcached. The same memtier invocation can be used by switching the `--protocol=memcached` otherwise the run setup is the same.

Similarly, fio can also be configured.

### Deploying UKL Images
All images need to be deployed bare metal, except for figure 6, which can be deployed on Qemu-KVM

The outputs will be produced in the root directory, and need to be copied out so that the graphing scripts can generate outputs

### Driving the KV experiments
We used memtier_benchmark to drive the Redis and Memcached experiments. For the ones run on bare metal, this will require a second Linux machine attached to the same top of rack switch.

The memteir invocation that should be used is as follows:

`memtier_benchmark --server=${UKL_IP} --protocol=redis --out-file=normal --hdr-file-prefix=Linux --print-percentiles 25,50,75,90,99 --requests=100000 --clients=100 --threads=3 --pipeline=1`

### Graphs
The graphing scripts are given in graphs directory and they take the input generated by UKL images.



## Building the included programs

```
git clone https://github.com/unikernelLinux/ukl-eurosy23-artifacts
cd ukl-eurosy23-artifacts
git submodule update --init
autoreconf -i
./configure --with-program=new_lebench
make -j`nproc`
```

To test it (requires qemu):

```
make boot
```

If the program requires incoming network connections, use this target
instead.  Note this runs qemu with `sudo`:

```
make boot-priv
```

`new_lebench` is a simple new_lebench world example.  You can try one of the other
programs (see subdirectories in the source) by adjusting
`./configure --with-program=...`

Currently you must `make clean -C linux` if you change the program.
(This is a bug which we should fix.)

## Configuration options

Some additional options are available to turn on and off features of
UKL:

```
$ ./configure --help
...
  --enable-bypass         enable glibc bypass (UKL_BP) [default=no]
  --enable-same-stack     enable same stack (CONFIG_UKL_SAME_STACK)
                          [default=no]
  --enable-use-ret        use ret instead of iret (CONFIG_UKL_USE_RET)
                          [default=no]
  --enable-use-ist-pf     use IST for PF instead of DF (CONFIG_UKL_USE_IST_PF)
                          [default=no]
  --enable-afterspace     enable afterspace (CONFIG_UKL_CREATE_AFTERSPACE)
                          [default=no]
```

## Building into a separate build directory

If you want to build different configurations of UKL from the same
source tree, you can do this by creating separate build directories,
eg:

```
mkdir build-new_lebench
pushd build-new_lebench
../configure --with-program=new_lebench
make -j`nproc`
popd

mkdir build-redis
pushd build-redis
../configure --with-program=redis
make -j`nproc`
popd
```

## Building a simple initramfs

If you need a simple initramfs for booting a UKL kernel one can be produced as part of the build
(ukl-initrd.cpio.xz) or with the `./create-initrd.sh` script. This ramdisk has a very limited environment
but it does include an ssh sever for retrieving results. If you use this ramdisk the root password is set
to 'root' so don't leave it exposed to the wider internet.

## Building your own program

We would strongly recommend looking at the example new_lebench world program
in the `new_lebench/` subdirectory.

1. You need to build it (not link) with two flags: `-mno-red-zone -mcmodel=kernel`
2. Then you need to do a partial link with the required libraries, glibc, libgcc etc.
3. Your partially linked application binary should be named UKL.a and needs to be copied to the top build directory for the final kernel link stage.
