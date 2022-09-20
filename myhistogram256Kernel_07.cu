/*导入主机相关库*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define total_data 512*1024*1024 //数据总量为512M
#define length_bin 256

/*用于处理内核函数调用的错误*/
__host__ void cuda_error_check(const char * prefix,const char * postfix) //前缀，后缀
{
	if(cudaPeekAtLastError() != cudaSuccess)
	{
		printf("\n%s%s%s",prefix,cudaGetErrorString(cudaGetLastError()),postfix);
		cudaDeviceReset();
		//wait_exit();//等待用户的任何输入，然后退出
		exit(1);
	}
}

__shared__ unsigned int d_bin_data_shared[256];//为线程块申请一块共享内存

/*内核函数*/
__global__ void myhistogramKernel_07(
		const unsigned int * d_hist_data,
		unsigned int * d_bin_data,
		unsigned int N)
{
	/*计算出线程id*/
	const unsigned int idx = (blockIdx.x*(blockDim.x*N))+threadIdx.x;
	const unsigned int idy = (blockIdx.y*blockDim.y)+threadIdx.y;
	const unsigned int tid = blockDim.x*N*gridDim.x*idy+idx;
	/*清理共享内存*/
	d_bin_data_shared[threadIdx.x]=0;
	/*等待线程块中所有的线程清理完共享内存*/
	__syncthreads();
	/* 读取数据并更新共享内存
	 * 每个线程一次读取32字节的数据是为了利用GPU硬件的读合并的优势，减少对全局内存的读取
	 * 一次读取N块数据是为了减少写全局内存的带宽*/
	for(unsigned int i=0,tid_offset=0;i<N;i++,tid_offset+=256)
	{
		const unsigned int value_u32 = d_hist_data[tid+tid_offset];//读取数据
		atomicAdd(&d_bin_data_shared[((value_u32 & 0xff000000)>>24)],1);//取出最高位的数据
		atomicAdd(&d_bin_data_shared[((value_u32 & 0x00ff0000)>>16)],1);//
		atomicAdd(&d_bin_data_shared[((value_u32 & 0x0000ff00)>>8)],1);
		atomicAdd(&d_bin_data_shared[value_u32 & 0x000000ff],1);//取出最后一位
	}
	/*对线程块中的所有线程进行同步，等待线程块中所有的线程全部对d_b——datain数组更新完毕*/
	__syncthreads();
	atomicAdd(&d_bin_data[threadIdx.x],d_bin_data_shared[threadIdx.x]);
}

/*主机主程序*/
int main(void)
{
	srand(time(NULL));//先种种子

	unsigned int i,s=0;
	unsigned char j;

	//用于接收cuda调用的错误码
	cudaError_t err = cudaSuccess;

	//输出待处理数据的长度
	printf("准备处理长为%d字节的数据\n",total_data);

	//为待处理数据分配主机端内存
	unsigned char * h_data = (unsigned char *)malloc(total_data);//malloc函数以字节为单位申请totall——data字节的内存

	//验证主机端内存分配是否成功
	if (h_data == NULL) {
	    //fprintf(stderr, "Failed to allocate host vectors!\n");
		printf("为输入数据分配主机端内存失败");
	    exit(EXIT_FAILURE);
	  }
	unsigned int *h_result = (unsigned int *)malloc(length_bin*sizeof(unsigned int));
	if (h_result == NULL) {
		//fprintf(stderr, "Failed to allocate host vectors!\n");
		printf("为输入数据分配主机端内存失败");
		exit(EXIT_FAILURE);
	}

	/*写文件*/
	/*
	fp = fopen("/home/guet-chou/eclipse-workspace/data.odt","a");//在指定目录下创建.odt文件
	for(i = 0 ; i < total_data; i ++) //产生255以内的随机数
	{
		j = rand()%255 ;
		//printf("j:%d ",j);
		fprintf(fp,"%03d",j); //把随机数写进文件
	}
	//printf("\n");
	fclose(fp); //关闭文件
	*/

	/*从文件中读取待处理数据到主内存中*/
	FILE *fp = NULL;
	unsigned int * h_data2;
	fp = fopen("/home/guet-chou/eclipse-workspace/data.odt","r");
	if(fp == NULL)
	{
	    printf("文件读取无效.\n");
	    return -1;
	}
	for(i = 0; !feof(fp); i++)
	{
		fscanf(fp, "%3d", &h_data[i]);
		if(i==536870912)
			printf("error");
	}

	fclose(fp); //关闭文件
	h_data2 = (unsigned int *)h_data;//为了充分利用GPU硬件合并读取的功能，把数据类型强制转换为整形

	/*
	 *
	 //测试数据是否正确
	unsigned int test_data;
	for(i=0;i<10;i++)
	{
		test_data = h_data2[i];
		printf("%d ",(test_data & 0xff000000)>>24);
		printf("%d ",(test_data & 0x00ff0000)>>16);
		printf("%d ",(test_data & 0x0000ff00)>>8);
		printf("%d \n",(test_data & 0x000000ff));
	}
	*/

	//在设备（即显存）中分配用于存储输入数据的内存
	unsigned int * d_data = NULL;
	err = cudaMalloc((void **)&d_data,total_data);
	if (err != cudaSuccess) {
	    fprintf(stderr,
	            "Failed to Call kernel functions (error code %s)!\n",
	            cudaGetErrorString(err));
	    exit(EXIT_FAILURE);
	  }
	//在设备端分配用于存储每个数出现频率的内存
	unsigned int * d_bin_data = NULL;
	err = cudaMalloc((void **)&d_bin_data,length_bin*sizeof(unsigned int));
	if (err != cudaSuccess) {
		    fprintf(stderr,
		            "Failed to Call kernel functions (error code %s)!\n",
		            cudaGetErrorString(err));
		    exit(EXIT_FAILURE);
		  }

	//把存储在主机内存中的数据复制到设备内存中
	printf("把数据放到设备内存中\n");
	err = cudaMemcpy(d_data, h_data2, total_data, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		    fprintf(stderr,
		            "Failed to Call kernel functions (error code %s)!\n",
		            cudaGetErrorString(err));
		    exit(EXIT_FAILURE);
		  }

	//计算内核运行花了多少时间
	clock_t start_time, end_time;
	start_time = clock();//开始时间

	//启动内核函数
	unsigned int threadsPerBlock = 256;
	int N = 64;//每个线程块处理N个直方图
	unsigned int blocksPerGrid = (total_data/4 + threadsPerBlock - 1) / (threadsPerBlock*N);
	printf("使用每块线程块包含%d个线程的%d线程快启动内核\n",threadsPerBlock,blocksPerGrid);
	myhistogramKernel_07<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_bin_data, N);

	/*
	err = cudaGetLastError();
	if (err != cudaSuccess) {
	  fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
	          cudaGetErrorString(err));
	  exit(EXIT_FAILURE);
	}
	*/

	//主机和设备同步
	cudaDeviceSynchronize();

	//结束时间
	end_time = clock();//结束时间
	printf("N=%d时花了%lf秒\n ", N,(double)(end_time - start_time) / CLOCKS_PER_SEC);//输出花了多少时间

	cuda_error_check("Error ","Returned from gmem runtime kernel");

	//把结果传回主机内存
	printf("结果传回内存\n");
	err = cudaMemcpy(h_result, d_bin_data, length_bin*sizeof(unsigned int), cudaMemcpyDeviceToHost);

	if (err != cudaSuccess) {
	  fprintf(stderr,
	          "结果复制到主机内存失败 (error code %s)!\n",
	          cudaGetErrorString(err));
	  exit(EXIT_FAILURE);
	}

	/*打印结果*/
	for(i=0;i<10;i++)
		printf("%d:%d\n",i,h_result[i]);

	/*释放内存*/
	// Free device global memory
	err = cudaFree(d_data);

	if (err != cudaSuccess) {
	  fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
	          cudaGetErrorString(err));
	  exit(EXIT_FAILURE);
	}

	err = cudaFree(d_bin_data);

	if (err != cudaSuccess) {
	  fprintf(stderr, "Failed to free device vector B (error code %s)!\n",
	          cudaGetErrorString(err));
	  exit(EXIT_FAILURE);
	}

	// Free host memory
	free(h_data);
	free(h_result);

	printf("Done\n");

	return 0;
}
