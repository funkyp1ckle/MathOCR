cmake_minimum_required(VERSION 3.25)
project(MathOCR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

set(CAFFE2_USE_CUDNN 1)

find_package(OpenCV 4.6.0 REQUIRED)
find_package(Boost 1.82.0 COMPONENTS filesystem REQUIRED) # try to remove
find_package(Torch REQUIRED)
list(APPEND CMAKE_CXX_FLAGS ${TORCH_CXX_FLAGS})

set(GHOSTSCRIPT_ROOT ${CMAKE_SOURCE_DIR}/ghostscript)
set(WKHTMLTOX_ROOT ${CMAKE_SOURCE_DIR}/wkhtmltox)
set(V8_ROOT ${CMAKE_SOURCE_DIR}/v8)
set(TORCH_ROOT ${Torch_DIR}/../../..)

set(WKHTMLTOX_LIBRARIES ${WKHTMLTOX_ROOT}/lib/libwkhtmltox.a)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(V8_LINK_LIBRARIES ${V8_ROOT}/lib/debug/v8.dll.lib
            ${V8_ROOT}/lib/debug/v8_libbase.dll.lib
            ${V8_ROOT}/lib/debug/v8_libplatform.dll.lib)
    set(V8_RUNTIME_LIBRARIES ${V8_ROOT}/lib/debug/v8.dll
            ${V8_ROOT}/lib/debug/v8_libbase.dll
            ${V8_ROOT}/lib/debug/v8_libplatform.dll
            ${V8_ROOT}/lib/debug/icuuc.dll
            ${V8_ROOT}/lib/debug/icui18n.dll
            ${V8_ROOT}/lib/debug/zlib.dll)
    set(OpenCV_LINK_LIBRARIES ${OpenCV_DIR}/opencv_core460d.lib
            ${OpenCV_DIR}/opencv_imgcodecs460d.lib
            ${OpenCV_DIR}/opencv_imgproc460d.lib
            ${OpenCV_DIR}/opencv_dnn460d.lib
            ${OpenCV_DIR}/opencv_cudaarithm460d.lib
            ${OpenCV_DIR}/opencv_cudafilters460d.lib
            ${OpenCV_DIR}/opencv_cudaimgproc460d.lib
            ${OpenCV_DIR}/opencv_cudawarping460d.lib)
    set(OpenCV_RUNTIME_LIBRARIES ${OpenCV_DIR}/../bin/opencv_core460d.dll
            ${OpenCV_DIR}/../bin/opencv_imgcodecs460d.dll
            ${OpenCV_DIR}/../bin/opencv_imgproc460d.dll
            ${OpenCV_DIR}/../bin/opencv_dnn460d.dll
            ${OpenCV_DIR}/../bin/opencv_cudaarithm460d.dll
            ${OpenCV_DIR}/../bin/opencv_cudafilters460d.dll
            ${OpenCV_DIR}/../bin/opencv_cudaimgproc460d.dll
            ${OpenCV_DIR}/../bin/opencv_cudawarping460d.dll)
    set(TORCH_RUNTIME_LIBRARIES ${TORCH_ROOT}/lib/debug/asmjit.dll ${TORCH_ROOT}/lib/debug/c10.dll ${TORCH_ROOT}/lib/debug/c10_cuda.dll
            ${TORCH_ROOT}/lib/debug/caffe2_nvrtc.dll ${TORCH_ROOT}/lib/debug/cublas64_11.dll ${TORCH_ROOT}/lib/debug/cublasLt64_11.dll
            ${TORCH_ROOT}/lib/debug/cudart64_110.dll ${TORCH_ROOT}/lib/debug/cudnn64_8.dll ${TORCH_ROOT}/lib/debug/cudnn_adv_infer64_8.dll
            ${TORCH_ROOT}/lib/debug/cudnn_adv_train64_8.dll ${TORCH_ROOT}/lib/debug/cudnn_cnn_infer64_8.dll
            ${TORCH_ROOT}/lib/debug/cudnn_cnn_train64_8.dll ${TORCH_ROOT}/lib/debug/cudnn_ops_infer64_8.dll
            ${TORCH_ROOT}/lib/debug/cudnn_ops_train64_8.dll ${TORCH_ROOT}/lib/debug/cufft64_10.dll ${TORCH_ROOT}/lib/debug/cufftw64_10.dll
            ${TORCH_ROOT}/lib/debug/cupti64_2022.3.0.dll ${TORCH_ROOT}/lib/debug/curand64_10.dll ${TORCH_ROOT}/lib/debug/cusolver64_11.dll
            ${TORCH_ROOT}/lib/debug/cusolverMg64_11.dll ${TORCH_ROOT}/lib/debug/cusparse64_11.dll ${TORCH_ROOT}/lib/debug/fbgemm.dll
            ${TORCH_ROOT}/lib/debug/fbjni.dll ${TORCH_ROOT}/lib/debug/libiomp5md.dll ${TORCH_ROOT}/lib/debug/libiompstubs5md.dll
            ${TORCH_ROOT}/lib/debug/nvfuser_codegen.dll ${TORCH_ROOT}/lib/debug/nvrtc-builtins64_118.dll
            ${TORCH_ROOT}/lib/debug/nvrtc64_112_0.dll ${TORCH_ROOT}/lib/debug/nvToolsExt64_1.dll ${TORCH_ROOT}/lib/debug/pytorch_jni.dll
            ${TORCH_ROOT}/lib/debug/torch.dll ${TORCH_ROOT}/lib/debug/torch_cpu.dll ${TORCH_ROOT}/lib/debug/torch_cuda.dll
            ${TORCH_ROOT}/lib/debug/torch_global_deps.dll ${TORCH_ROOT}/lib/debug/uv.dll ${TORCH_ROOT}/lib/debug/zlibwapi.dll
            ${TORCH_ROOT}/lib/debug/torchtrt.dll ${TORCH_ROOT}/lib/debug/nvinfer.dll ${TORCH_ROOT}/lib/debug/nvinfer_builder_resource.dll)
    list(APPEND TORCH_LIBRARIES ${TORCH_ROOT}/lib/debug/torchtrt.lib)
else()
    set(V8_LINK_LIBRARIES ${V8_ROOT}/lib/release/v8.dll.lib
            ${V8_ROOT}/lib/release/v8_libbase.dll.lib
            ${V8_ROOT}/lib/release/v8_libplatform.dll.lib)
    set(V8_RUNTIME_LIBRARIES ${V8_ROOT}/lib/release/v8.dll
            ${V8_ROOT}/lib/release/v8_libbase.dll
            ${V8_ROOT}/lib/release/v8_libplatform.dll
            ${V8_ROOT}/lib/release/icuuc.dll
            ${V8_ROOT}/lib/release/icui18n.dll
            ${V8_ROOT}/lib/release/zlib.dll)
    set(OpenCV_LINK_LIBRARIES ${OpenCV_DIR}/opencv_core460.lib
            ${OpenCV_DIR}/opencv_imgcodecs460.lib
            ${OpenCV_DIR}/opencv_imgproc460.lib
            ${OpenCV_DIR}/opencv_dnn460.lib
            ${OpenCV_DIR}/opencv_cudaarithm460.lib
            ${OpenCV_DIR}/opencv_cudafilters460.lib
            ${OpenCV_DIR}/opencv_cudaimgproc460.lib
            ${OpenCV_DIR}/opencv_cudawarping460.lib)
    set(OpenCV_RUNTIME_LIBRARIES ${OpenCV_DIR}/../bin/opencv_core460.dll
            ${OpenCV_DIR}/../bin/opencv_imgcodecs460.dll
            ${OpenCV_DIR}/../bin/opencv_imgproc460.dll
            ${OpenCV_DIR}/../bin/opencv_dnn460.dll
            ${OpenCV_DIR}/../bin/opencv_cudaarithm460.dll
            ${OpenCV_DIR}/../bin/opencv_cudafilters460.dll
            ${OpenCV_DIR}/../bin/opencv_cudaimgproc460.dll
            ${OpenCV_DIR}/../bin/opencv_cudawarping460.dll)
    set(TORCH_RUNTIME_LIBRARIES ${TORCH_ROOT}/lib/release/asmjit.dll ${TORCH_ROOT}/lib/release/c10.dll ${TORCH_ROOT}/lib/release/c10_cuda.dll
            ${TORCH_ROOT}/lib/release/caffe2_nvrtc.dll ${TORCH_ROOT}/lib/release/cublas64_11.dll ${TORCH_ROOT}/lib/release/cublasLt64_11.dll
            ${TORCH_ROOT}/lib/release/cudart64_110.dll ${TORCH_ROOT}/lib/release/cudnn64_8.dll ${TORCH_ROOT}/lib/release/cudnn_adv_infer64_8.dll
            ${TORCH_ROOT}/lib/release/cudnn_adv_train64_8.dll ${TORCH_ROOT}/lib/release/cudnn_cnn_infer64_8.dll
            ${TORCH_ROOT}/lib/release/cudnn_cnn_train64_8.dll ${TORCH_ROOT}/lib/release/cudnn_ops_infer64_8.dll
            ${TORCH_ROOT}/lib/release/cudnn_ops_train64_8.dll ${TORCH_ROOT}/lib/release/cufft64_10.dll ${TORCH_ROOT}/lib/release/cufftw64_10.dll
            ${TORCH_ROOT}/lib/release/cupti64_2022.3.0.dll ${TORCH_ROOT}/lib/release/curand64_10.dll ${TORCH_ROOT}/lib/release/cusolver64_11.dll
            ${TORCH_ROOT}/lib/release/cusolverMg64_11.dll ${TORCH_ROOT}/lib/release/cusparse64_11.dll ${TORCH_ROOT}/lib/release/fbgemm.dll
            ${TORCH_ROOT}/lib/release/fbjni.dll ${TORCH_ROOT}/lib/release/libiomp5md.dll ${TORCH_ROOT}/lib/release/libiompstubs5md.dll
            ${TORCH_ROOT}/lib/release/nnapi_backend.dll ${TORCH_ROOT}/lib/release/nvfuser_codegen.dll ${TORCH_ROOT}/lib/release/nvrtc-builtins64_118.dll
            ${TORCH_ROOT}/lib/release/nvrtc64_112_0.dll ${TORCH_ROOT}/lib/release/nvToolsExt64_1.dll ${TORCH_ROOT}/lib/release/pytorch_jni.dll
            ${TORCH_ROOT}/lib/release/shm.dll ${TORCH_ROOT}/lib/release/torch.dll ${TORCH_ROOT}/lib/release/torch_cpu.dll ${TORCH_ROOT}/lib/release/torch_cuda.dll
            ${TORCH_ROOT}/lib/release/torch_global_deps.dll ${TORCH_ROOT}/lib/release/torch_python.dll ${TORCH_ROOT}/lib/release/uv.dll
            ${TORCH_ROOT}/lib/release/zlibwapi.dll ${TORCH_ROOT}/lib/release/torchtrt.dll ${TORCH_ROOT}/lib/release/nvinfer.dll
            ${TORCH_ROOT}/lib/release/nvinfer_builder_resource.dll)
    list(APPEND TORCH_LIBRARIES ${TORCH_ROOT}/lib/release/torchtrt.lib)
endif()

add_executable(MathOCR main.cpp latexpp/latex.cpp utils.cpp OCREngine.cpp)
target_include_directories(MathOCR PUBLIC ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${V8_ROOT} ${V8_ROOT}/include ${WKHTMLTOX_ROOT}/include)
target_compile_definitions(MathOCR PUBLIC V8_ENABLE_SANDBOX V8_COMPRESS_POINTERS _WIN32_WINNT=0x0601)
target_link_libraries(MathOCR PUBLIC ${TORCH_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LINK_LIBRARIES} ${V8_LINK_LIBRARIES} ${WKHTMLTOX_LIBRARIES})

add_custom_command(TARGET MathOCR POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${OpenCV_RUNTIME_LIBRARIES}
        ${V8_RUNTIME_LIBRARIES}
        ${TORCH_RUNTIME_LIBRARIES}
        ${WKHTMLTOX_ROOT}/lib/wkhtmltox.dll
        ${GHOSTSCRIPT_ROOT}/ghostscript.exe ${GHOSTSCRIPT_ROOT}/gsdll64.dll
        ${CMAKE_SOURCE_DIR}/models/classify.torchscript
        $<TARGET_FILE_DIR:MathOCR>)

