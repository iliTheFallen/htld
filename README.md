<h1>H-TLD Library and its Application</h1>

    Copyright 2013-2014 METU, Middle East Technical University, Informatics Institute
    
    This file is part of H-TLD.
   
    H-TLD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
   
    H-TLD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
   
    You should have received a copy of the GNU General Public License along 
    with the one from Surrey University for TLD Algorithm developed by Zdenek Kalal,
    and from Austrian Institute of Technology for C++ port coded by Georg Nebehay.
    If not, see <http://www.gnu.org/licenses/>. Please contact Alptekin TEMIZEL for more 
    info about licensing atemizel@metu.edu.tr, and/or Ilker GURCAN via github issues section.

<h2>Introduction</h2>

<p>Tracking objects in a video stream is an important problem in robot learning (learning an object’s visual features from different perspectives as it moves, rotates, scales, and is subjected to some morphological changes such as erosion), defense, public security and many other various domains. In this thesis, we focus on a recently proposed tracking framework called TLD (Tracking-Learning-Detection). While having promising tracking results, the algorithm has high computational cost. The computational cost of the algorithm prevents running it at higher resolutions as well as running multiple instances of the algorithm to track multiple objects on CPU. In this thesis, we analyzed this framework with an aim to optimize it computationally on a CPU-GPU hybrid setting and developed a solution via using GP-GPU (General Purpose GPU) programming using Open-MP and CUDA. Our results show that 2.82 times speed-up at 480x270 resolution can be achieved. The speed-ups are higher at higher resolutions as expected in a massively parallel GPU platform, increasing to 10.25 times speed-up at 1920x1080 resolution. The resulting performance of the algorithm enables the algorithm to track multiple objects at higher frame rates in real-time and improving detection and tracking quality by allowing selection of configuration parameters requiring higher processing power.</p>

<p>Purpose of this initial version is to show how H-TLD library works and affects the performance of TLD Algorithm. A revised version for multi-object detection and performance improvements will be released soon and made public on this web site.</p>

<h2>System Requirements</h2>

<p> Although your requirements may vary (due to many reasons such as the quality of video frame you use), following table illustrates the platform H-TLD had been tested. Running it on a Platform with "System Requirements" equal to the one given in the table below or higher is recommended. Only important thing that should be taken into consideration is that Kepler Specific instructions and keywords are used in H-TLD; hence  using a CUDA-powered GPU with version of 2.1+ is encouraged.</p>

|  Id  |  Requirement  | Test Platform |
|:----:|:-------------:|:-------------:|
| 1    | OS            | Windows 7 x64 |
| 2    | CPU           | Intel i7 4770K 3.5 GHz<br/> |
| 3    | GPU           | NVIDIA Tesla K40c Compute Capability 3.5,<br/> 15 SMs 192 Cores per SM, 2 Async Copy Engine,<br/> Hyper-Q Enabled |
| 4    | RAM           | 32GB DDR3 |
| 5    | Serial Computer <br/>Expansion Bus| PCIe 2.1 |
| 6    | CUDA Toolkit  | 6.0 |
| 7    | CUDA Driver   | 6.0 |
| 8    | CUDA Runtime  | 6.0 |
| 9    | NPP           | 6.0 |
| 10   | OpenCV        | 2.4.9 |
| 11   | Open-MP       | 2.0 |

<h2>Installation Details</h2>

<p>
There is no specific installer for now. All you require (except the ones defined in "Systems Requirements" section) are Visual Studio, CUDA Toolkit and OpenCV for generating dll and executable files. All project specific files in uploaded solution were created by Visual Studio 2012. Please ensure that you installed NVIDIA Nsight, right after you'd installed Visual Studio IDE; otherwise your projects will not be detected as CUDA projects.<br/> 
There are two separate projects in the solution : <br/>
<ul>
 <li>One for H-TLD which produces dll and lib files.</li>
 <li>One from Georg Nebehay for the Application which produces an executable file.</li>
</ul>

</p>

<p>
There are three additional 3rd party libraries and they are located under the folder "3rdParties" : 
<ul>
 <li><a href="http://www.hyperrealm.com/libconfig/">libconfig++</a></li>
 <li><a href="https://code.google.com/p/cvblob/">cvBlob</a></li>
 <li><a href="http://nvlabs.github.io/cub/">cub from NVIDIA Research Lab</a></li>
</ul>

</p>

<p>
Paths to their header and library files are defined with respect to the project's root folder (including the one for H-TLD in order to run the application). However, for OpenCV repeat the steps given below for both projects :
<ul>
 <li>Select the Project.</li>
 <li>Click on Project->Properties->Configuration Properties->C/C++->General->Additional Include Directories.</li>
 <li>Specify where the include folder for OpenCV Library is.</li>
 <li>Click on Project->Properties->Configuration Properties->Linker->General->Additional Library Directories.</li>
 <li>Specify where lib folder for OpenCV Library is. All Required library files are already defined in Input sub-section.</li>
</ul>
</p>

<p>
 As for CUDA support : 
 <ul>
  <li>Select the Project for H-TLD.</li>
  <li>Click on Project->Properties->Configuration Properties->CUDA C/C++->Common->Additional Include Directories.</li>
  <li>You should specify where the include folder of CUDA Samples is located. For instance, in our case it is equal to "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v6.0\common\inc" by default.</li>
  <li>Click on Project->Properties->Configuration Properties->CUDA C/C++->Device->Code Generation.</li>
  <li>You should specify compute capability and SM architecture versions to optimize your code in accord with the device you have. For us, they were both equal to 3.5</li>
 </ul>
</p>

<h2>Usage</h2>

<p>In order to realize the difference between serial and heterogeneous implementations, the original serial code was not modified (except one improvement mentioned in "Notes" section). Only macros are defined to enable/disable H-TLD usage. If you want to harness H-TLD library; before you compile you should uncomment the line on where "#define USE_HTLD" is located in "TLDGlobals.h" file (in fact, this is the default option, when you download the solution).</p>

<p>In order to find out how the application is used; please see this link : https://github.com/gnebehay/OpenTLD.
After you compile and link both projects (i.e. first build H-TLD, then the application), in the build folder you should see the dll and lib files for H-TLD and exe file for the application. When you run the executable along with the path to the configuration file(via command line), the tracking should commence.</p>

<p>Two different data sets are uploaded for testing. Folders are as follows : <br/>
<ul>
 <li>water_bottle_low : Set of Frames for Low Resolution Video, 480x270.</li>
 <li>water_bottle_medium : Set of Frames for Medium Resolution Video, 960x540.</li>
</ul>
</p>

<p>
Configuration File in "$(SolutionDir)\x64\Release" folder(config-sample.cfg) is set for medium resolution video frames by default.<br/>
To run the application : 
<ul>
 <li>Open a Command Line.</li>
 <li>Change Directory to $(SolutionDir)\x64\Release.</li>
 <li>Type "open_tld_app.exe .\config-sample.cfg".</li>
 <li>Press on Enter.</li>
</ul>
</p>

<h2>Notes</h2>
<p>
<ul>
 <li>Never Remove the "libconfig++.dll" file in x64\Release folder. It is not removed by Visual Studio automatically upon Clean; therefore no need to worry about whether it is going to be removed by the IDE.</li>
 <li>Keep in mind that a Tesla card was used during the test phase for it eliminates the overhead brought by display drivers like WDDM. This is valid for Windows Platforms only in the case you run H-TLD on Windows Vista and beyond. There might be some temporal solutions proposed on NVIDIA's web site to breach WDDM like drivers.</li>
 <li>As of yet, Qt4 support is not included.</li>
 <li>At the application level, only difference from Mr. Nebehay's serial implementation is that the image-warping right before training random fern classifier was added like it is in Zdenek Kalal's original Matlab implementation.</li>
</ul>
</p>
