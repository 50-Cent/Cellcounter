# Neurocounter
Neurocounter localizes neuron cell bodies in a stained slice of rodent brain. It includes stain-specific configurations of Neurocounter.  

There are two user manuals - Localization (Configurations already provided) and Training (To create a new configuration from scratch) - provided with this package.

Brief: Neurocounter is a Deep learning model (CNN and an attention module), designed to detect neurons in a stained brain image slice. The images are procured from a confocal microscope (Nikon) and each image slice has several 'Z' depths. So, it is a multi-channel Z-stack, which is compressed in .nd2 file system. We extracted each image in the 'Z' stack in .tif format by using ImageJ. Each of this images is uint16 encoded. We provided a MATLAB script to scale each image to uint8 format. After the scaling conversion is finished, the maximum intensity projection (MIP) is performed. So, each Z-stack produces one MIP image and matrix containing 'Z' depth. We provided a MATLAB script to carry out the MIP projection. In addition, Neurocounter (written in Python) is also equipped with a module to compute MIP. User can use anyone of these (MATLAB/Python) modules. 

Neurocounter uses Pytorch to build deep learning models. It takes (MIP image with 'Z' depth/ MIP image without Z depth / image (not necessarily MIP) / Z-stack). All the images have to be gray-valued in .tif format. While running the code, it asks for which configuration to use. For example, for NeuN images, users are requested type the number that corresponds to 'NeuN'. 
The code will finally give a colored image with red-dots marking the neuronal cell bodies. To access the packages, please click the 'master' branch instead of 'main'. You can see the word 'main' on this page. There is a downward arrow right next to 'main'. Once you click, you can access the 'master'. 

The codes are zipped for Windows and Linux separately. 


