#!/usr/bin/env bash
# Paper 9 native-Linux replication of the RQ2 smoke battery.
# Resolves two flagged caveats:
#   (1) does the CUDA >=12.6 SEGFAULT reproduce on native Linux, or is it a WSL2
#       (cudaMallocManaged/UVM) artifact?
#   (2) does the admission CLIFF move with the host driver (multi-driver point)?
#
# Runs on a native-Linux vast VM. Installs docker + nvidia-container-toolkit,
# then for each nvidia/cuda image: tries to start it (toolkit gate), and if it
# starts, compiles+runs a tiny vecadd (does the runtime segfault?). For images
# above the host's ceiling it also retries with NVIDIA_DISABLE_REQUIRE=1.
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "== host =="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "systemd-detect-virt: $(systemd-detect-virt)  (must be kvm/native, not wsl)"

echo "== install docker + nvidia-container-toolkit =="
apt-get update -qq
apt-get install -yqq ca-certificates curl gnupg >/dev/null
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg 2>/dev/null
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu jammy stable" > /etc/apt/sources.list.d/docker.list
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -qq
apt-get install -yqq docker-ce docker-ce-cli containerd.io nvidia-container-toolkit >/dev/null
nvidia-ctk runtime configure --runtime=docker >/dev/null 2>&1
# start dockerd (VM may lack systemd docker unit ready)
(dockerd >/var/log/dockerd.log 2>&1 &) ; sleep 8
docker info >/dev/null 2>&1 && echo "docker up" || { echo "docker FAILED"; tail -5 /var/log/dockerd.log; }
echo "toolkit: $(nvidia-ctk --version 2>&1 | head -1)"

# tiny CUDA vecadd (runtime API) to trigger the runtime-init segfault if present
cat > /root/vec.cu <<'CU'
#include <cstdio>
__global__ void add(float*a){a[threadIdx.x]+=1.0f;}
int main(){float*d;cudaMalloc(&d,64*sizeof(float));add<<<1,64>>>(d);
cudaError_t e=cudaDeviceSynchronize();
printf("sync=%s\n",cudaGetErrorString(e));cudaFree(d);return e?1:0;}
CU

OUT=/root/native_cuda_smoke.csv
echo "driver,cuda_image,require_bypass,start,nvcc_compile,run_result" > "$OUT"

# images spanning the host's ceiling (host cuda_max ~12.8)
IMAGES="12.8.1-devel-ubuntu24.04 13.0.1-devel-ubuntu24.04 13.1.1-devel-ubuntu24.04 13.2.1-devel-ubuntu24.04 13.3.0-devel-ubuntu24.04"

test_image() {  # $1=cuda tag  $2=bypass(0/1)
  local TAG="$1" BYP="$2" IMG="nvidia/cuda:$1" ENV=""
  [ "$BYP" = "1" ] && ENV="-e NVIDIA_DISABLE_REQUIRE=1"
  # start test: nvidia-smi inside
  if ! docker run --rm --gpus all $ENV "$IMG" nvidia-smi -L >/tmp/smi.log 2>&1; then
    local why=$(grep -oiE "unsatisfied condition: cuda>=[0-9.]+|requirement error" /tmp/smi.log | head -1)
    echo "$DRIVER,$TAG,$BYP,GATED,-,-  (${why:-start-fail})" | tee -a "$OUT"
    return
  fi
  # compile + run vecadd
  local R=$(docker run --rm --gpus all $ENV -v /root/vec.cu:/vec.cu "$IMG" \
    bash -c "nvcc -o /t /vec.cu 2>/tmp/nv.e && /t 2>&1; echo rc=\$?" 2>/tmp/run.log)
  local NVCC=$(grep -q "sync=\|rc=" <<<"$R" && echo ok || echo "compile-fail")
  local RUN=$(echo "$R" | tr '\n' ' ')
  # detect segfault: rc=139 or empty sync + nonzero
  echo "$DRIVER,$TAG,$BYP,STARTED,$NVCC,${RUN}" | tee -a "$OUT"
}

for T in $IMAGES; do
  echo "== $T (normal) =="; test_image "$T" 0
done
# gate-bypass retry only for images that were GATED
echo "== gate-bypass leg for gated images =="
for T in $IMAGES; do
  if grep -q ",$T,0,GATED" "$OUT"; then
    echo "== $T (bypass) =="; test_image "$T" 1
  fi
done

echo "SMOKE DONE"; echo "=== RESULT ==="; cat "$OUT"
