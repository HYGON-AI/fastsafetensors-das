WORK_SPACE=$1
MANYLINUX="yes"
if [[ "${MANYLINUX}" == "yes" ]];then
    echo ">>>>>>>>>>manylinux<<<<<<<<<<"
    echo "obtain the path of the torch"
    TORCH_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null)
    echo "${TORCH_PATH}"
    cd ${WORK_SPACE}
    export LD_LIBRARY_PATH=${TORCH_PATH}/lib:$LD_LIBRARY_PATH
    # export LD_LIBRARY_PATH=/usr/local/lib/python${PYTHON_VERSION}/site-packages/torch/lib:$LD_LIBRARY_PATH
    # export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64_lin/:$LD_LIBRARY_PATH   ###
    pip install patchelf "auditwheel>=6.1"
    export PLATFORM=$(ldd --version | awk '{print $NF}' | head -n1 | awk -F "." '{print "manylinux_"$1"_"$2"_x86_64"}')
    auditwheel repair --plat ${PLATFORM} --strip \
    --exclude 'libgalaxyhip.so*' \
    --exclude 'libMIOpen.so*' \
    --exclude 'librccl.so*' \
    --exclude 'libhipblas.so*' \
    --exclude 'libhipfft.so*' \
    --exclude 'libhiprand.so*' \
    --exclude 'libhipsolver.so*' \
    --exclude 'libhipsparse.so*' \
    --exclude 'libhipnn.so*' \
    --exclude 'librocblas.so*' \
    --exclude 'librocsolver.so*' \
    --exclude 'librocfft.so*' \
    --exclude 'librocrand.so*' \
    --exclude 'librocsparse.so*' \
    --exclude 'librocm_smi64.so*' \
    --exclude 'librocfft-device-0.so*' \
    --exclude 'librocfft-device-1.so*' \
    --exclude 'librocfft-device-2.so*' \
    --exclude 'librocfft-device-3.so*' \
    --exclude 'libc10.so*' \
    --exclude 'libc10_hip.so*' \
    --exclude 'libtorch_cpu.so*' \
    --exclude 'libtorch_hip.so*' \
    --exclude 'libtorch_python.so*' \
    --exclude 'libtorch.so*' \
    --exclude 'libtorchaudio_ffmpeg4.so*' \
    --exclude 'libtorchaudio_ffmpeg5.so*' \
    --exclude 'libtorchaudio_ffmpeg6.so*' \
    --exclude 'libtorchaudio.so*' \
    --exclude 'libtorchaudio_sox.so*' \
    --exclude '_torchaudio_ffmpeg4.so*' \
    --exclude '_torchaudio_ffmpeg5.so*' \
    --exclude '_torchaudio_ffmpeg6.so*' \
    --exclude '_torchaudio.so*' \
    --exclude '_torchaudio_sox.so*' \
    --exclude 'libavcodec.so*' \
    --exclude 'libavdevice.so*' \
    --exclude 'libavfilter.so*' \
    --exclude 'libavformat.so*' \
    --exclude 'libavutil.so*' \
    --exclude 'libsox.so*' \
    --exclude 'libomp.so*' \
    --exclude 'libhipblaslt.so*' \
    -w ${WORK_SPACE}/dist/ \
    ${WORK_SPACE}/dist/fast*.whl
    cd ${WORK_SPACE}/dist/
    pack=`ls |grep manylinux_2_27`
    wheel unpack ${pack}
    base=$(ls -t|head -n 1)
    cd "${base}"
    dir=$(ls | grep info)
    sed -i '/manylinux_2_27/d' "${dir}/WHEEL"
    cd ../
    wheel pack "${base}"
    rm -rf "${base}"
    rm -rf "${pack}"
    echo ">>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<"
fi
