#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <numeric>
#include <torch/torch.h>
#include <cstring>
#define HALF
#ifdef HALF
#include "half.hpp"
#endif

#include <opus/opus.hpp>

#define LOCAL_SCRATCH 0
#define RAND_INT 0
#define CUSTOMIZED_INT 1

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);            \
    }                       \
} while(0)

#define ABS(x) ((x) > 0 ? (x) : -(x))

using fp32_t = float;
using fp16_t = _Float16;
using float16 = half_float::half; // cpu type

using fp16x2_t = fp16_t __attribute__((ext_vector_type(2)));
using fp16x4_t = fp16_t __attribute__((ext_vector_type(4)));
using fp16x8_t = fp16_t __attribute__((ext_vector_type(8)));
using fp16x16_t = fp16_t __attribute__((ext_vector_type(16)));
using fp32x4_t = fp32_t __attribute__((ext_vector_type(4)));
using fp32x16_t = fp32_t __attribute__((ext_vector_type(16)));

using int32x4_t = int32_t __attribute__((ext_vector_type(4)));
#define BUFFER_LOAD_DWORD3 0x00020000   // This is valid for 
struct buffer_resource {
    const void * ptr;
    uint32_t range;
    uint32_t config;
};

__device__ int32x4_t make_buffer_resource(const void * ptr, uint32_t size = 0xffffffff)
{
    buffer_resource res {ptr, size, BUFFER_LOAD_DWORD3};
    return __builtin_bit_cast(int32x4_t, res);
}

#ifdef RAND_INT
#define PER_PIXEL_CHECK
#endif

static inline bool valid_vector( const float* ref, const float16* pred, int n, double nrms = 1e-3 )
{    
    double s0=0.0;
    double s1=0.0;
#ifdef PER_PIXEL_CHECK
    int pp_err = 0;
#endif
    int i_start = 0, i_end=n;
    
    for( int i=i_start; i<i_end; ++i ){
        double ri=(double)ref[i];
        double pi=(double)pred[i];
        double d=ri-pi;
        double dd=d*d;
        double rr=2.0*ri*ri;
        s0+=dd;
        s1+=rr;
        
#ifdef PER_PIXEL_CHECK
        double delta = ABS(ri-pi)/ri;
        if(delta>1e-3){
            if(pp_err<100)
                printf("diff at %4d, ref:%lf, pred:%lf(0x%04x), d:%lf\n",i,ri,pi,((uint16_t*)pred)[i],delta);
            pp_err++;
        }
#endif
    }
    // int i_num = i_end - i_start;
    // printf("pp_crr:%d, pp_err:%d, crr_ratio:%.3f, nrms:%lf, s0:%lf, s1:%lf\n",i_num-pp_err, pp_err, (float)(i_num-pp_err)/(float)i_num, sqrt(s0/s1),s0,s1);

    return (sqrt(s0/s1)<nrms)
#ifdef PER_PIXEL_CHECK
        && (pp_err==0)
#endif
    ;
}

template<int BLOCK_SIZE, int BLOCK_M, int BLOCK_N, int BLOCK_K, int TILE_M, int TILE_N, int TILE_K, int WAVE_M, int WAVE_N, int WAVE_K>
__global__ void matrix_core_kernel_block_v2(const void* __restrict__ ptr_a,
                                         const void* __restrict__ ptr_b,
                                         void* __restrict__ ptr_c,
                                         int k,
                                         int stride_a, // stride in unit of pixel
                                         int stride_b,
                                         int stride_c)
{
    using opus::operator""_I;
    constexpr int W_M = WAVE_M;
    constexpr int W_N = WAVE_N;
    constexpr int W_K = WAVE_K;

    constexpr int T_M = TILE_M;
    constexpr int T_N = TILE_N;
    constexpr int T_K = TILE_K;

    constexpr int E_M = BLOCK_M / (W_M * T_M);
    constexpr int E_N = BLOCK_N / (W_N * T_N);
    constexpr int E_K = BLOCK_K / (W_K * T_K);
    static_assert(E_K == 1);

    using d_a = opus::fp16_t;
    using d_b = opus::fp16_t;
    using d_c = opus::fp32_t;

    int lane_id = threadIdx.x % opus::get_warp_size();
    int wave_id = threadIdx.x / opus::get_warp_size();
    int g_im = blockIdx.x * BLOCK_M;
    int g_in = blockIdx.y * BLOCK_N;

    // NOTE: the shape merge is per-dim
    //
    // A:[(expd_m<y>, tile_m<p>), (expd_k<y>, tile_k<p>)] * [(grpm_a<p>), (rept_a<y>, grpk_a<p>, pack_a<y>)]
    // B:[(expd_n<y>, tile_n<p>), (expd_k<y>, tile_k<p>)] * [(grpn_b<p>), (rept_b<y>, grpk_b<p>, pack_b<y>)]
    // C:[(expd_m<y>, tile_m<p>), (expd_n<y>, tile_n<p>)] * [(grpn_c<p>), (rept_c<y>, grpm_c<p>, pack_c<y>)]
    //
    // A:[(expd_m<y>, tile_m<p>, grpm_a<p>), (expd_k<y>, tile_k<p>, rept_a<y>, grpk_a<p>, pack_a<y>)]
    // B:[(expd_n<y>, tile_n<p>, grpn_b<p>), (expd_k<y>, tile_k<p>, rept_b<y>, grpk_b<p>, pack_b<y>)]
    // C:[(expd_m<y>, tile_m<p>, grpn_c<p>), (expd_n<y>, tile_n<p>, rept_c<y>, grpm_c<p>, pack_c<y>)]
    //
    auto mma  = opus::make_tiled_mma<d_a, d_b, d_c>(opus::seq<E_M, E_N, E_K>{}, opus::seq<T_M, T_N, T_K>{}, opus::seq<W_M, W_N, W_K>{}, opus::mfma_adaptor_swap_ab{});

    auto u_a = opus::partition_layout_a(mma, opus::make_tuple(stride_a, 1_I), opus::make_tuple(wave_id / 2, lane_id % mma.grpm_a, 0_I, lane_id / mma.grpm_a) /*tile_m<p>, grpm_a<p>, tile_k<p>, grpk_a<p>*/);
    auto u_b = opus::partition_layout_b(mma, opus::make_tuple(stride_b, 1_I), opus::make_tuple(wave_id % 2, lane_id % mma.grpn_b, 0_I, lane_id / mma.grpn_b) /*tile_n<p>, grpn_b<p>, tile_k<p>, grpk_b<p>*/);
    auto u_c = opus::partition_layout_c(mma, opus::make_tuple(stride_c, 1_I), opus::make_tuple(wave_id / 2, lane_id % mma.grpn_c, wave_id % 2, lane_id / mma.grpn_c) /*tile_m<p>, grpn_c<p> tile_n<p>, grpm_c<p>*/);
    auto g_a = opus::make_gmem(reinterpret_cast<const d_a*>(ptr_a) + g_im * stride_a);
    auto g_b = opus::make_gmem(reinterpret_cast<const d_b*>(ptr_b) + g_in * stride_b);
    auto g_c = opus::make_gmem(reinterpret_cast<opus::fp16_t*>(ptr_c) + g_im * stride_c + g_in);

    // start of kernel
    int loops = (k + BLOCK_K - 1) / BLOCK_K;
#if 1
    typename decltype(mma)::vtype_c v_c;
    opus::clear(v_c);

    for(auto i = 0; i < loops; i++ ) {
        auto v_a = g_a.load<4>(u_a);  u_a += BLOCK_K;
        auto v_b = g_b.load<4>(u_b);  u_b += BLOCK_K;
        v_c = mma(v_a, v_b, v_c);
    }

    auto v_c_f16 = opus::cast<fp16_t>(v_c);
    g_c.store<4>(v_c_f16, u_c);
#else
    auto v_a = g_a.load<4>(u_a);  u_a += BLOCK_K;
    auto v_b = g_b.load<4>(u_b);  u_b += BLOCK_K;
    auto v_c = mma(v_a, v_b);   // first time, C is always zero

    for(auto i = 0; i < loops - 1; i++ ) {
        v_a = g_a.load<4>(u_a);  u_a += BLOCK_K;
        v_b = g_b.load<4>(u_b);  u_b += BLOCK_K;
        v_c = mma(v_a, v_b, v_c);
    }

    auto v_c_f16 = opus::cast<fp16_t>(v_c);
    g_c.store<4>(v_c_f16, u_c);
#endif
}

// 通用 Conv1d 函数
void casual_conv1d_rcr(float* input, float* output,
                float* weight, float* bias,
                int N, int C_in, int C_out, int L,
                int kernel_size, int pad, int stride = 1, int groups = 1) {
    // 1. 定义 Conv1d
    torch::nn::Conv1d conv(
        torch::nn::Conv1dOptions(C_in, C_out, kernel_size)
            .stride(stride)
            .padding(pad)
            .groups(groups)
    );
 
    // 2. 设置权重和偏置
    torch::Tensor w_tensor = torch::from_blob(weight, {C_out, C_in / groups, kernel_size}, torch::kFloat).clone();
    torch::Tensor b_tensor = torch::from_blob(bias, {C_out}, torch::kFloat).clone();
    conv->weight = w_tensor;
    conv->bias   = b_tensor;
 
    // 3. 输入张量
    torch::Tensor input_tensor = torch::from_blob(input, {N, C_in, L}, torch::kFloat).clone();
 
    // 4. 前向计算
    torch::Tensor output_tensor = conv->forward(input_tensor);
 
    // 5. 拷贝结果到一维指针
    int out_size = 1;
    for (auto s : output_tensor.sizes()) out_size *= s;
    std::memcpy(output, output_tensor.data_ptr<float>(), out_size * sizeof(float));
 
    // 打印输出形状
    std::cout << "Output shape: " << output_tensor.sizes() << std::endl;
}

void customized_vector_2d_in(float* v, int row, int col, int ld){
    int r,c;
    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }
    for(r=0;r<row;r++){
        for(c=0;c<col;c++){
            v[r*ld+c] = ((float)(r*ld+c)/10);
        }
    }
}

void customized_vector_2d_weight(float* v, int row, int col, int ld){
    int r,c;
    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }
    for(r=0;r<row;r++){
        for(c=0;c<col;c++){
            v[r*ld+c] = ((float)(r*ld+c)/10);
        }
    }
}

void gemm_rcr(
    const float*  __restrict__ ptr_a,
    const float*  __restrict__ ptr_b,
    float*  ptr_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc)
{
    for(auto i_m = 0 ; i_m < m; i_m++) {
        for(auto i_n = 0; i_n < n; i_n++) {
            float acc = 0;
            for(auto i_k = 0; i_k < k; i_k++) {
                acc += ptr_a[i_m * lda + i_k] * ptr_b[i_n * ldb + i_k];
            }
            ptr_c[i_m * ldc + i_n] = acc;
        }
    }
}

void casual_conv1d_block_run()
{
    int batch = 1;
    int hi = 32;
    int ci = 32;
    int hk = 4;
    int pad = hk - 1;

    // define gemm inputs a,b,c
    int ho = hi;
    int m = ho;
    int n = ci;
    int k = hk * ci;

    int lda = k;
    int ldb = k;
    int ldc = n;

    // init fp32 input and weight
    float *host_in, *host_w, *host_c, *host_bias;

    //fp32 input[hi, ci] and weight[ci, hi] on host
    host_in = (float*)malloc(batch*hi*ci*sizeof(float));
    host_w = (float*)malloc(batch*ci*hk*sizeof(float));
    host_bias = (float*)malloc(batch*ci*sizeof(float));
    host_c = (float*)malloc(batch*ldc*m*sizeof(float));
    int ld_in = ci;
    int ld_w = hk;

// #ifdef RAND_INT
//     rand_vector_2d_int(host_in, hi, ci, ld_in);
//     rand_vector_2d_int(host_w, ci, hk, ld_w);
// #ifdef CUSTOMIZED_INT
    customized_vector_2d_in(host_in, hi, ci, ld_in);
    customized_vector_2d_weight(host_w, ci, hk, ld_w);
    for (int i = 0; i < batch*ci; i++) host_bias[i] = 0.0f;
// #else
//     rand_vector_2d(host_in, hi, ci, ld_in, 0.0, 1.0);
//     rand_vector_2d(host_w, ci, hk, ld_w, -0.5, 0.5);
// #endif
    // for(int i=0; i<hi*ci; i++) {if (i<100) {printf("in[%d], %f \n", i, host_in[i]);}}
    // for(int i=0; i<ci*hk; i++) {printf("w[%d], %f \n", i, host_w[i]);}
    float16 *fp16_in, *fp16_w;
    //convert fp32 input into fp16 on host
    fp16_in = (float16*)malloc((hi*ci)*sizeof(float16));
    fp16_w = (float16*)malloc((ci*hk)*sizeof(float16));
    for(int i=0; i<hi*ci; i++)fp16_in[i]=__float2half_rn(host_in[i]);
    for(int i=0; i<ci*hk; i++)fp16_w[i]=__float2half_rn(host_w[i]);
    // for(int i=0; i<ci*hk; i++) {printf("fp16_w[%d], %f \n", i, (float)(fp16_w[i]));}

    // float *host_a, *host_b, *host_c;
    float16 *fp16_in_pad, *fp16_a, *fp16_b, *fp16_c, *dev_a, *dev_b, *dev_c;

    //preprocess input and weight to fp16 gemm inputs on host
    fp16_in_pad = (float16*)malloc(((hi+pad) * ci)*sizeof(float16));
    fp16_a = (float16*)malloc(lda*m*sizeof(float16));
    fp16_b = (float16*)malloc(ldb*n*sizeof(float16));
    fp16_c = (float16*)malloc(ldc*m*sizeof(float16));
    // //convert fp32 a and b into fp16 on host
    // for(int i=0; i<lda*m; i++)fp16_a[i]=__float2half_rn(host_a[i]);
    // for(int i=0; i<ldb*n; i++)fp16_b[i]=__float2half_rn(host_b[i]);

    // add pad for input
    for(int i = 0; i < pad * ci; i++) {
        fp16_in_pad[i] = 0;
    }
    for(int i = 0; i < hi * ci; i++) {
        fp16_in_pad[i + pad * ci] = fp16_in[i];
    }
    // for(int i=0; i<(hi+pad)*ci; i++) {
    //     if (i<200) {printf("fp16_in_pad[%d], %f \n", i, (float)(fp16_in_pad[i]));}
    // }
    // input with pad, img2col
    for(int i=0; i < ho; i++) {
        for (int j = 0; j < hk * ci; j++) {
            fp16_a[i * hk *ci + j] = fp16_in_pad[i * ci + j];
        }
    }
    // for(int i=0; i<ho * hk * ci; i++) {
    //     if (i<800) {printf("fp16_a[%d], %f \n", i, (float)(fp16_a[i]));}
    // }
    //convert dw weight to common weight
    // initialization
    for(int i=0; i < ci * hk * ci; i++) {
        fp16_b[i] = 0;
    }
    for(int i=0; i < ci; i++) {
        for (int j = 0; j < hk*ci; j++) {
            if ((j % ci) == i) {
                fp16_b[i * hk *ci + j] = fp16_w[i * hk + int(j / ci)];
                // printf("fp16_w[%d, %d], %f \n", i, int(j / ci), (float)(fp16_w[i * hk + int(j / ci)]));
            }
        }
    }
    // for(int i=0; i < ci; i++) {
    //     for (int j = 0; j < hk*ci; j++) {
    //         // if (i<4) {printf("fp16_b[%d, %d], %f ", i, j, (float)(fp16_b[i*hk*ci +j]));}
    //         if (i<4) {printf("%f ", (float)(fp16_b[i*hk*ci +j]));}
    //     }
    //     if (i<4) {printf("/n");}
    // }

    HIP_CALL(hipMalloc(&dev_a, lda*m*sizeof(float16)));
    HIP_CALL(hipMalloc(&dev_b, ldb*n*sizeof(float16)));
    HIP_CALL(hipMalloc(&dev_c, ldc*m*sizeof(float16)));
    //fp16 cpy to device
    HIP_CALL(hipMemcpy(dev_a, fp16_a, lda*m*sizeof(float16), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dev_b, fp16_b, ldb*n*sizeof(float16), hipMemcpyHostToDevice));

    printf("m:%d,n:%d,k:%d,lda:%d,ldb:%d,ldc:%d\n",  m, n, k, lda, ldb, ldc); fflush(stdout);
    // gemm_rcr(host_a, host_b, host_c, m,n,k,lda,ldb,ldc);
    // casual_conv1d_rcr(host_in, host_w, host_c, n, ci, hi, hk, pad);
    casual_conv1d_rcr(host_in, host_c, host_w, host_bias, batch, ci, ci, hi, hk, pad, 1, ci);
    // run_conv1d(host_in, host_c, host_w, host_bias, batch, ci, ci, hi, hk, pad, 1, ci);
    {
        constexpr int BLOCK_M = 32;
        constexpr int BLOCK_N = 32;
        constexpr int BLOCK_K = 16;
        constexpr int TILE_M = 2;
        constexpr int TILE_N = 2;
        constexpr int TILE_K = 1;
        constexpr int WAVE_M = 16;
        constexpr int WAVE_N = 16;
        constexpr int WAVE_K = 16;

        auto gdim = dim3(m / BLOCK_M, n / BLOCK_N);
        auto kernel = matrix_core_kernel_block_v2<256, BLOCK_M, BLOCK_N, BLOCK_K, TILE_M, TILE_N, TILE_K, WAVE_M, WAVE_N, WAVE_K>;
        kernel<<<gdim, 256>>>(dev_a, dev_b, dev_c, k, lda, ldb, ldc);

        HIP_CALL(hipMemcpy(fp16_c, dev_c, ldc*m*sizeof(float16), hipMemcpyDeviceToHost));
#if 1
        bool res = valid_vector(host_c, fp16_c, m*n, 1e-3);
        printf("[%dx%dx%d, block_gemm_%dx%dx%d_%dx%dx%d_%dx%dx%d], %s", m, n, k,
            BLOCK_M, BLOCK_N, BLOCK_K, TILE_M, TILE_N, TILE_K, WAVE_M, WAVE_N, WAVE_K,
            res?"valid":"fail");fflush(stdout);
        printf("\n"); fflush(stdout);
#endif
    }


    // free(host_a);
    // free(host_b);
    free(host_in);
    free(host_w);
    free(host_c);
    free(host_bias);
    free(fp16_in);
    free(fp16_w);
    free(fp16_a);
    free(fp16_b);
    free(fp16_c);
    
    HIP_CALL(hipFree(dev_a));
    HIP_CALL(hipFree(dev_b));
    HIP_CALL(hipFree(dev_c));
}
 
int main() {
    casual_conv1d_block_run();
    // int N = 1, C_in = 3, C_out = 3, L = 10, kernel_size = 4, pad = 2;
 
    // // 输入数据
    // float* input = new float[N * C_in * L];
    // for (int i = 0; i < N * C_in * L; i++) input[i] = i % 5;
 
    // // 权重和偏置
    // float* weight = new float[C_out * 1 * kernel_size];
    // for (int i = 0; i < C_out * kernel_size; i++) weight[i] = 1.0f;
    // float* bias = new float[C_out];
    // for (int i = 0; i < C_out; i++) bias[i] = 0.0f;
 
    // // 输出数组
    // int out_len = L + 2 * pad - kernel_size + 1; // 卷积输出长度公式
    // float* output = new float[N * C_out * out_len];
 
    // // 调用函数
    // run_conv1d(input, output, weight, bias, N, C_in, C_out, L, kernel_size, pad, 1, C_in);
 
    // // 打印部分结果
    // std::cout << "Output[0..9]: ";
    // for (int i = 0; i < std::min(N * C_out * out_len, 10); i++) {
    //     std::cout << output[i] << " ";
    // }
    // std::cout << std::endl;
 
    // delete[] input;
    // delete[] weight;
    // delete[] bias;
    // delete[] output;
    // return 0;
}