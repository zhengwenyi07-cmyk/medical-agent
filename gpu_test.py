import torch
import time
import numpy as np



import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)

def test_gpu_basic():
    """基础GPU检测"""
    print("=" * 50)
    print("GPU基础检测")
    print("=" * 50)
    
    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {cuda_available}")
    
    if not cuda_available:
        print("❌ CUDA不可用，请检查驱动和CUDA安装")
        return False
    
    # 获取GPU数量和信息
    device_count = torch.cuda.device_count()
    print(f"检测到GPU数量: {device_count}")
    
    for i in range(device_count):
        print(f"\nGPU {i} 信息:")
        print(f"  设备名称: {torch.cuda.get_device_name(i)}")
        print(f"  计算能力: {torch.cuda.get_device_capability(i)}")
        print(f"  总显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"  当前显存使用: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        print(f"  保留显存: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
    
    # 检查PyTorch和CUDA版本
    print(f"\nPyTorch版本: {torch.__version__}")
    print(f"CUDA版本: {torch.version.cuda}")
    
    return True

def test_gpu_computation():
    """GPU计算能力测试"""
    print("\n" + "=" * 50)
    print("GPU计算能力测试")
    print("=" * 50)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试矩阵乘法（GPU优势明显）
    size = 5000  # 矩阵大小
    print(f"测试 {size}x{size} 矩阵乘法...")
    
    # 在GPU上创建随机矩阵
    start_time = time.time()
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    creation_time = time.time() - start_time
    print(f"矩阵创建时间: {creation_time:.4f}秒")
    
    # 同步GPU确保准确计时
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 矩阵乘法测试
    start_time = time.time()
    c = torch.matmul(a, b)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    computation_time = time.time() - start_time
    print(f"矩阵乘法时间: {computation_time:.4f}秒")
    
    # 验证计算结果
    result_norm = torch.norm(c).item()
    print(f"计算结果范数: {result_norm:.4f}")
    
    # 清理显存
    del a, b, c
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return computation_time

def test_gpu_memory():
    """GPU显存测试"""
    print("\n" + "=" * 50)
    print("GPU显存测试")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        return False
    
    try:
        # 测试显存分配
        block_size = 100  # MB
        blocks = []
        
        print("测试显存分配...")
        for i in range(20):  # 尝试分配最多2GB
            try:
                # 分配100MB的显存
                size = block_size * 1024 * 1024  # 100MB
                block = torch.cuda.ByteTensor(size)
                blocks.append(block)
                current_memory = (i + 1) * block_size
                print(f"成功分配 {current_memory}MB 显存")
                
                # 短暂延迟以便观察
                time.sleep(0.1)
                
            except RuntimeError as e:
                print(f"显存分配在 {(i + 1) * block_size}MB 时失败: {e}")
                break
        
        # 清理显存
        for block in blocks:
            del block
        torch.cuda.empty_cache()
        
        print("显存测试完成")
        return True
        
    except Exception as e:
        print(f"显存测试出错: {e}")
        return False

def test_gpu_transfer():
    """GPU数据传输测试"""
    print("\n" + "=" * 50)
    print("GPU数据传输测试")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        return False
    
    # 测试CPU到GPU的数据传输速度
    size = 1000  # 1D数组大小
    iterations = 100
    
    # 在CPU上创建数据
    cpu_data = torch.randn(size, size)
    
    start_time = time.time()
    for _ in range(iterations):
        gpu_data = cpu_data.cuda()  # 传输到GPU
        # 立即同步以确保传输完成
        torch.cuda.synchronize()
    
    transfer_time = time.time() - start_time
    avg_transfer_time = transfer_time / iterations
    print(f"平均CPU->GPU传输时间: {avg_transfer_time:.6f}秒")
    
    # 测试GPU到CPU的数据传输
    gpu_data = torch.randn(size, size).cuda()
    
    start_time = time.time()
    for _ in range(iterations):
        cpu_data = gpu_data.cpu()  # 传输回CPU
        torch.cuda.synchronize()
    
    transfer_time = time.time() - start_time
    avg_transfer_time = transfer_time / iterations
    print(f"平均GPU->CPU传输时间: {avg_transfer_time:.6f}秒")
    
    return True

def main():
    """主测试函数"""
    print("开始GPU全面测试")
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA版本: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print()
    
    try:
        # 运行各项测试
        basic_ok = test_gpu_basic()
        
        if basic_ok:
            # 计算测试
            computation_time = test_gpu_computation()
            
            # 显存测试
            memory_ok = test_gpu_memory()
            
            # 数据传输测试
            transfer_ok = test_gpu_transfer()
            
            print("\n" + "=" * 50)
            print("测试总结")
            print("=" * 50)
            print("✅ GPU测试完成！")
            
            # 性能评估
            if computation_time < 1.0:
                print("🚀 GPU计算性能: 优秀")
            elif computation_time < 3.0:
                print("👍 GPU计算性能: 良好")
            else:
                print("⚠️ GPU计算性能: 一般，建议检查配置")
                
            print(f"\n建议:")
            print("- 如果所有测试通过，您的GPU配置正确")
            print("- 可以开始使用PyTorch进行GPU加速计算")
            print("- 对于深度学习任务，建议监控显存使用情况")
            
        else:
            print("\n❌ GPU基础检测失败，请检查以下内容:")
            print("1. NVIDIA驱动是否正确安装")
            print("2. CUDA工具包是否安装且版本匹配")
            print("3. PyTorch是否为GPU版本")
            print("4. 环境变量PATH是否包含CUDA路径")
            
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        print("请检查CUDA和PyTorch安装配置")

if __name__ == "__main__":
    main()