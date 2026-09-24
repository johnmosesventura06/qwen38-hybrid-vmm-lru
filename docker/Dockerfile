FROM vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8

# lane deviation (2026-09-11): the pinned base already ships
# humming-kernels==0.1.12 (Required-by: vllm) and nvidia-nccl-cu13 2.30.7.
# The upstream pip step re-resolves against PyPI metadata and DOWNGRADES NCCL
# to 2.29.7 — a divergence from the vendor-pinned runtime that was never part
# of any published measurement (the Sept-5 candidate ran "existing
# dependencies, not a fresh Docker build"). Step removed; base versions kept.
# RUN python3 -m pip install --no-cache-dir humming-kernels==0.1.12

COPY runtime/install_overlay.py /opt/qwen38/runtime/install_overlay.py
COPY runtime/vllm-overlay /opt/qwen38/runtime/vllm-overlay
RUN python3 /opt/qwen38/runtime/install_overlay.py /opt/qwen38/runtime/vllm-overlay

COPY configs/2x3090-128gb.env /opt/qwen38/configs/2x3090-128gb.env
COPY configs/static_hot_cache_rankings.json /workspace/static_hot_cache_rankings.json
COPY scripts/serve-container.sh /usr/local/bin/qwen38-serve
COPY scripts/check_custom_all_reduce.py /opt/qwen38/scripts/check_custom_all_reduce.py
COPY tests/test_qsa_exact_topk_cpu.py /opt/qwen38/tests/test_qsa_exact_topk_cpu.py
RUN chmod 0755 /usr/local/bin/qwen38-serve

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/qwen38-serve"]
