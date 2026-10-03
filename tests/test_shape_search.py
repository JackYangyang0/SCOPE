import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from SCOPE.specialization.shape_search import archive, load_families, run_search, shared_bytes, render


TILE = dict(BM=32, BN=32, BK=8, WM=32, WN=32, TM=2, TN=2,
            WMITER=8, WNITER=16, threads_per_block=32, warps_per_block=1)


def state():
    return {"problem": {"M":33,"N":35,"K":17,"dtype_A":"fp32","dtype_B":"fp32"},
            "hardware": {"gpu_name":"test-gpu","warp_size":32,"sm_count":8,
                         "max_shared_memory_per_block_bytes":49152}}


def verified(ir, gflops=10):
    result = copy.deepcopy(ir)
    result["verification"] = {"accepted":True, **{k:{"status":"pass"} for k in ("compile","correctness","runtime_safety")}}
    result["performance"] = {"gflops":gflops,"cublas_gflops":100}
    return result


def seed(registry):
    return archive({"verified_ir":verified(state()), "source_snapshot":{
        "cuda_kernel.cuh":"// test seed", "main.cpp":"// harness", "kernel.h":"// header"}}, registry)


class ShapeSearchTests(unittest.TestCase):
    def test_registry_rejects_mismatch_and_tampered_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            family_id = seed(tmp)
            self.assertEqual(len(load_families(tmp,state())),1)
            different = state()
            different["hardware"]["gpu_name"] = "other"
            self.assertEqual(load_families(tmp,different),[])
            (Path(tmp)/family_id/"cuda_kernel.cuh").write_text("modified")
            self.assertEqual(load_families(tmp,state()),[])

    def test_no_registry_requests_llm_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_search(state(), {}, Path(tmp)/"registry", Path(tmp)/"output", {})
            self.assertTrue(report["fallback_required"])
            self.assertEqual(report["results"],[])

    @patch("SCOPE.specialization.shape_search.make_diverse_tiling_pool", return_value=[TILE])
    @patch("SCOPE.specialization.shape_search.build_joint_tiling_candidates", return_value=[TILE])
    def test_three_rounds_retest_new_shape_and_keep_seed_immutable(self, *_):
        with tempfile.TemporaryDirectory() as tmp:
            registry, out = Path(tmp)/"registry", Path(tmp)/"out"
            fid = seed(registry)
            calls = []
            new = state()
            new["problem"].update(M=65,N=67,K=129)
            def verifier(ir, **kwargs):
                calls.append(copy.deepcopy(ir))
                return verified(ir)
            report = run_search(new,{},registry,out,{"candidates_per_round":12}, verify=verifier)
            self.assertEqual(len(report["rounds"]),3)
            self.assertTrue(report["fallback_required"])
            self.assertTrue(all(c["problem"]["M"] == 65 for c in calls))
            self.assertTrue(any(c.get("specialization",{}).get("split_k",1)>1 for c in calls))
            self.assertEqual((registry/fid/"cuda_kernel.cuh").read_text(),"// test seed")
            self.assertTrue(all((Path(r["code_dir"])/"verified_ir.json").exists() for r in report["results"]))

    @patch("SCOPE.specialization.shape_search.make_diverse_tiling_pool", return_value=[TILE])
    @patch("SCOPE.specialization.shape_search.build_joint_tiling_candidates", return_value=[TILE])
    def test_failure_continues_and_target_stops_llm_fallback(self, *_):
        with tempfile.TemporaryDirectory() as tmp:
            seed(Path(tmp)/"registry")
            calls = []
            def verifier(ir, **kwargs):
                calls.append(ir)
                if len(calls) == 1:
                    raise RuntimeError("test compile failure")
                return verified(ir,95)
            report = run_search(state(),{},Path(tmp)/"registry",Path(tmp)/"out",{},verify=verifier)
            self.assertFalse(report["fallback_required"])
            self.assertEqual(len(report["rounds"]),1)
            self.assertFalse(report["results"][0]["accepted"])

    def test_layout_accounting_and_splitk_are_explicit(self):
        self.assertEqual(shared_bytes(TILE,False,0),2048)
        self.assertEqual(shared_bytes(TILE,True,1),2112)
        source = render(TILE,True,1,8,4)
        self.assertIn("#define SCOPE_SPLIT_K 4",source)
        self.assertIn("reduce<<<",source)
        self.assertIn("cudaMalloc",source)


@unittest.skipUnless(os.environ.get("SCOPE_TEST_CUDA_SPECIALIZATION") == "1", "opt-in CUDA compile/run test")
class CudaSpecializationTests(unittest.TestCase):
    def test_direct_layout_and_splitk_against_reference(self):
        from SCOPE.verification.build_run_verifier import compile_gemm, run_command, DEFAULT_VCVARS64
        smoke = r'''
#include "cuda_kernel.cuh"
#include <vector>
#include <cmath>
int main() {
  for (int K : {1, 8, 35}) for (float beta : {0.0f, 0.5f}) {
    const int M=33, N=35; const float alpha=0.75f;
    std::vector<float> a(M*K), b(K*N), c(M*N), out(M*N);
    for (int i=0;i<M*K;++i) a[i]=(i%13-6)*0.125f;
    for (int i=0;i<K*N;++i) b[i]=(i%7-3)*0.25f;
    for (int i=0;i<M*N;++i) c[i]=beta==0 ? NAN : 0.3f;
    float *A,*B,*C;
    using scope_specialization::check;
    check(cudaMalloc((void**)&A,a.size()*4)); check(cudaMalloc((void**)&B,b.size()*4));
    check(cudaMalloc((void**)&C,c.size()*4));
    check(cudaMemcpy(A,a.data(),a.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(B,b.data(),b.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(C,c.data(),c.size()*4,cudaMemcpyHostToDevice));
    cuda_gemm(M,N,K,alpha,A,B,beta,C);
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(out.data(),C,c.size()*4,cudaMemcpyDeviceToHost));
    for (int m=0;m<M;++m) for (int n=0;n<N;++n) {
      double ref=0;
      for (int k=0;k<K;++k) ref+=double(a[m*K+k])*b[k*N+n];
      ref=alpha*ref+(beta==0 ? 0.0 : beta*c[m*N+n]);
      if (!std::isfinite(out[m*N+n]) || std::abs(out[m*N+n]-ref)>1e-4) return 2;
    }
    check(cudaFree(A)); check(cudaFree(B)); check(cudaFree(C));
  }
  return 0;
}
'''
        platform = "windows" if os.name == "nt" else "linux"
        for transpose, padding, split, stages, async_copy in ((False,0,1,1,False),(True,1,1,2,False),(True,1,4,1,False),(True,1,1,2,True),(True,1,4,2,True)):
            with self.subTest(transpose=transpose,padding=padding,split=split), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                multiwarp = {**TILE, "BM":64, "BN":64, "TM":4, "TN":4, "WMITER":16, "WNITER":32}
                (folder/"cuda_kernel.cuh").write_text(render(multiwarp,transpose,padding,8,split,stages,async_copy))
                (folder/"main.cu").write_text(smoke)
                exe = folder/("smoke.exe" if platform == "windows" else "smoke")
                ir = state()
                ir["hardware"]["compute_capability"] = "8.9"
                built = compile_gemm(ir,folder,exe,180,DEFAULT_VCVARS64,platform)
                self.assertEqual(built["status"],"pass",built)
                ran = run_command([str(exe)],cwd=folder,timeout_seconds=60)
                self.assertEqual(ran["status"],"pass",ran)


if __name__ == "__main__":
    unittest.main()
