[singularity]: http://singularity.lbl.gov/
[sudo]: https://www.sudo.ws/
# chm_singularity

Generates [Singularity][singularity] image for PyCHM.


# Run requirements

* [Singularity 2.2][singularity]
* Linux

# Build requirements 

* [Singularity 2.2][singularity]
* Make
* Linux
* Bash
* [sudo][sudo] superuser access (required by [Singularity][singularity] to build images)

# To build

Run the following command to create the singularity image which will
be put under the **build/** directory

```Bash
make singularity
```

# To test

The image is built with a self test mode. To run the self test issue the following command after building:

```Bash
cd build/
./pychm.img verify /tmp
```

### Expected output from above command

```Bash
Running segtools check

Module       Required  Installed
Python       v2.7      v2.7.5     +
numpy        v1.7      v1.12.1    +
scipy        v0.12     v0.19.0    +

Optional:
cython       v0.19     v0.25.2    + (for some optimized libraries)
pillow       v2.0      v4.1.0     + (for loading common image formats)
h5py         v2.0      v2.7.0     + (for loading MATLAB v7.3 files)
psutil       v2.0      v5.2.2     + (for tasks)
subprocess32 v3.2.6    v3.2.7     + (for tasks on POSIX systems)

Running small PyCHM train job which writes output to /tmp/foo.out

2017-04-17 21:56:06 Training stage 1 level 0...
2017-04-17 21:56:06   Extracting features...
2017-04-17 21:56:12   Learning...
2017-04-17 21:56:12     Number of training samples = 19000
2017-04-17 21:56:12     Calculating initial weights...
2017-04-17 21:56:12       Initial error: 0.196449
2017-04-17 21:56:12     Gradient descent...
2017-04-17 21:56:17       Iteration #1 error: 0.164705
2017-04-17 21:56:21       Iteration #2 error: 0.117123
2017-04-17 21:56:26       Iteration #3 error: 0.094622
2017-04-17 21:56:30       Iteration #4 error: 0.086059
2017-04-17 21:56:35       Iteration #5 error: 0.081151
2017-04-17 21:56:39       Iteration #6 error: 0.077454
2017-04-17 21:56:44       Iteration #7 error: 0.074346
2017-04-17 21:56:48       Iteration #8 error: 0.071533
2017-04-17 21:56:53       Iteration #9 error: 0.069018
2017-04-17 21:56:58       Iteration #10 error: 0.066881
2017-04-17 21:57:02       Iteration #11 error: 0.064858
2017-04-17 21:57:07       Iteration #12 error: 0.063110
2017-04-17 21:57:11       Iteration #13 error: 0.061463
2017-04-17 21:57:16       Iteration #14 error: 0.060004
2017-04-17 21:57:20       Iteration #15 error: 0.058629
2017-04-17 21:57:20       Final error: 0.057734
2017-04-17 21:57:20   Generating outputs...
2017-04-17 21:57:21   Accuracy: 0.791368, F-value: 0.338009, G-mean: 0.882967
2017-04-17 21:57:21 Training stage 1 level 1...
2017-04-17 21:57:21   Extracting features...
2017-04-17 21:57:23   Learning...
2017-04-17 21:57:23     Number of training samples = 4800
2017-04-17 21:57:23     Calculating initial weights...
2017-04-17 21:57:23       Initial error: 0.158943
2017-04-17 21:57:23     Gradient descent...
2017-04-17 21:57:24       Iteration #1 error=0.135571
2017-04-17 21:57:25       Iteration #2 error=0.127412
2017-04-17 21:57:25       Iteration #3 error=0.123697
2017-04-17 21:57:26       Iteration #4 error=0.122194
2017-04-17 21:57:27       Iteration #5 error=0.118571
2017-04-17 21:57:27       Iteration #6 error=0.118447
2017-04-17 21:57:28       Iteration #7 error=0.117820
2017-04-17 21:57:29       Iteration #8 error=0.115000
2017-04-17 21:57:30       Iteration #9 error=0.114584
2017-04-17 21:57:30       Iteration #10 error=0.113756
2017-04-17 21:57:31       Iteration #11 error=0.113535
2017-04-17 21:57:32       Iteration #12 error=0.113147
2017-04-17 21:57:32       Iteration #13 error=0.112030
2017-04-17 21:57:33       Iteration #14 error=0.110919
2017-04-17 21:57:34       Iteration #15 error=0.110884
2017-04-17 21:57:34       Final error: 0.114680
2017-04-17 21:57:34   Generating outputs...
2017-04-17 21:57:34   Accuracy: 0.868333, F-value: 0.480263, G-mean: 0.927257
2017-04-17 21:57:34 Training stage 2 level 0...
2017-04-17 21:57:34   Extracting features...
2017-04-17 21:57:35   Learning...
2017-04-17 21:57:35     Number of training samples = 19000
2017-04-17 21:57:35     Calculating initial weights...
2017-04-17 21:57:35       Initial error: 0.109457
2017-04-17 21:57:35     Gradient descent...
2017-04-17 21:57:41       Iteration #1 error: 0.107923
2017-04-17 21:57:46       Iteration #2 error: 0.106695
2017-04-17 21:57:52       Iteration #3 error: 0.106100
2017-04-17 21:57:58       Iteration #4 error: 0.105597
2017-04-17 21:58:03       Iteration #5 error: 0.105171
2017-04-17 21:58:09       Iteration #6 error: 0.104835
2017-04-17 21:58:09       Final error: 0.104489
2017-04-17 21:58:10   Generating outputs...
2017-04-17 21:58:10   Accuracy: 0.980105, F-value: 0.842631, G-mean: 0.989437
2017-04-17 21:58:10 Complete!

```

# To run train

Invoke pychm.img as seen below with first argument set to **train** 
followed by any arguments accepted by pychm.

```Bash
cd build/
./pychm.img train 
```

Example with training data png files in images/ and labels/

```Bash
cd build/
./pychm.img train ./foo "[ images/*.png ]" "[ labels/*.png ]" -S 2 -L 1
```


# To run test

Invoke pychm.img as seen below with first argument set to **test**
followed by any arguments accepted by pychm.

```Bash
cd build/
./pychm.img test
```
 
