#include "CudaDeviceProperties.h"
#include "HandleError.h"
#include <iostream>

using namespace std;

void properties () {
  int count;
  CHECK_CUDA (cudaGetDeviceCount (&count));
  cout << "count = " << count << endl;

  cudaDeviceProp prop;
  for (int i = 0; i < count; i++) {
    CHECK_CUDA(cudaGetDeviceProperties (&prop, i));
    cout << prop.name << endl;

    cout << " totalGlobalMem " << prop.totalGlobalMem << endl;//print bytes
    cout << " sharedMemPerBlock " << prop.sharedMemPerBlock << endl;
    cout << " regsPerBlock " << prop.regsPerBlock << endl;
    cout << " warpSize " << prop.warpSize << endl;
    cout << " memPitch " << prop.memPitch << endl;
    cout << " maxThreadsPerBlock " << prop.maxThreadsPerBlock << endl;
    cout << " maxThreadsDim[3] " << prop.maxThreadsDim[1] << ' ' << prop.maxThreadsDim[2] << endl;
    cout << " maxGridSize[3] " << prop.maxGridSize[1] << ' ' << prop.maxGridSize[2] << endl;
    cout << " totalConstMem " << prop.totalConstMem << endl;
    cout << " major " << prop.major << endl;
    cout << " minor " << prop.minor << endl;
    cout << " clockRate " << prop.clockRate << endl;
    cout << " textureAlignment " << prop.textureAlignment << endl;
    cout << " deviceOverlap " << prop.deviceOverlap << endl;
    cout << " multiProcessorCount " << prop.multiProcessorCount << endl;
    cout << " kernelExecTimeoutEnabled " << prop.kernelExecTimeoutEnabled << endl;
    cout << " integrated " << prop.integrated << endl;
    cout << " canMapHostMemory " << prop.canMapHostMemory << endl;
    cout << " computeMode " << prop.computeMode << endl;
    cout << " concurrentKernels " << prop.concurrentKernels << endl;
    cout << " ECCEnabled " << prop.ECCEnabled << endl;
    cout << " pciBusID " << prop.pciBusID << endl;
    cout << " pciDeviceID " << prop.pciDeviceID << endl;
    cout << " tccDriver " << prop.tccDriver << endl;
    cout << endl;
    cout << " asyncEngineCount " << prop.asyncEngineCount << endl;
    //cout << " concurrentManagedAccess " << prop.concurrentManagedAccess << endl;//Device can coherently access managed memory concurrently with the CPU
    cout << " globalL1CacheSupported " << prop.globalL1CacheSupported << endl;
    //cout << " hostNativeAtomicSupported " << prop.hostNativeAtomicSupported << endl;//Link between the device and the host supports native atomic operations
    cout << " isMultiGpuBoard " << prop.isMultiGpuBoard << endl;
    cout << " l2CacheSize " << prop.l2CacheSize << endl;
    cout << " managedMemory " << prop.managedMemory << endl;
    //cout << " maxSurface1D[2] " << prop.maxSurface1D[1] << prop.maxSurface1D[2] << endl;//Maximum 1D surface size
    cout << " maxSurface1DLayered[2] " << prop.maxSurface1DLayered[1] << endl;
    cout << " maxSurface2DLayered[3] " << prop.maxSurface2DLayered[1] << ' ' << prop.maxSurface2DLayered[2] << endl;
    cout << " maxSurface3D[3] " << prop.maxSurface3D[1] << ' ' << prop.maxSurface3D[2] << endl;
    cout << " maxSurfaceCubemap " << prop.maxSurfaceCubemap << endl;
    cout << " maxSurfaceCubemapLayered[2] " << prop.maxSurfaceCubemapLayered[1] << endl;
    cout << " maxTexture1D " << prop.maxTexture1D << endl;
    cout << " maxTexture1DLayered[2] " << prop.maxTexture1DLayered[1] << endl;
    cout << " maxTexture1DLinear " << prop.maxTexture1DLinear << endl;
    cout << " maxTexture1DMipmap " << prop.maxTexture1DMipmap << endl;
    cout << " maxTexture2D[2] " << prop.maxTexture2D[1] << endl;
    cout << " maxTexture2DGather[2] " << prop.maxTexture2DGather[1] << endl;
    cout << " maxTexture2DLayered[3] " << prop.maxTexture2DLayered[1] << ' ' << prop.maxTexture2DLayered[2] << endl;
    cout << " maxTexture2DLinear[3] " << prop.maxTexture2DLinear[1] << ' ' << prop.maxTexture2DLinear[2]  << endl;
    cout << " maxTexture2DMipmap[2] " << prop.maxTexture2DMipmap[1] << endl;
    //cout << " maxTexture3D[3] " << prop.maxTexture3D[1] << maxTexture3D[2] << prop.maxTexture3D[3] << endl; //Maximum 3D texture dimensions
    //cout << " maxTexture3DAlt[3] " << propmaxTexture3DAlt[1] << prop.maxTexture3DAlt[2] << prop.maxTexture3DAlt[3] << endl; //Maximum alternate 3D texture dimensions
    cout << " maxTextureCubemap " << prop.maxTextureCubemap << endl;
    cout << " maxTextureCubemapLayered[2] " << prop.maxTextureCubemapLayered[1] << endl;
    cout << " maxThreadsPerMultiProcessor " << prop.maxThreadsPerMultiProcessor << endl;
    cout << " memoryBusWidth " << prop.memoryBusWidth << endl;
    cout << " memoryClockRate " << prop.memoryClockRate << endl;
    cout << " multiGpuBoardGroupID " << prop.multiGpuBoardGroupID << endl;
    //cout << " pageableMemoryAccess " << prop.pageableMemoryAccess << endl; //Device supports coherently accessing pageable memory without calling cudaHostRegister on it
    cout << " pciDomainID " << prop.pciDomainID << endl;
    cout << " regsPerMultiprocessor " << prop.regsPerMultiprocessor << endl;
    cout << " sharedMemPerMultiprocessor " << prop.sharedMemPerMultiprocessor << endl;
    //cout << " singleToDoublePrecisionPerfRatio " << prop.singleToDoublePrecisionPerfRatio << endl; //Ratio of single precision performance (in floating-point operations per second) to double precision performance
    cout << " streamPrioritiesSupported " << prop.streamPrioritiesSupported << endl;
    cout << " surfaceAlignment " << prop.surfaceAlignment << endl;
    cout << " texturePitchAlignment " << prop.texturePitchAlignment << endl;
    cout << " unifiedAddressing " << prop.unifiedAddressing << endl;

  }
}
