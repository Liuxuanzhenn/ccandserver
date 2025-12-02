@echo off
chcp 65001 >nul
echo ========================================
echo ResNet50 单个测试命令（扁平化命名）
echo ========================================
echo.
echo 请选择要执行的测试：
echo.
echo === 量化 ===
echo 1. 检测模型能力
echo 2. FP16量化
echo 3. INT8量化（自动选择静态/动态）
echo 4. INT8动态量化
echo 5. INT8静态量化（需校准数据）
echo 6. 自动量化
echo.
echo === 剪枝 ===
echo 7. 结构化剪枝（默认30%%稀疏度）
echo 8. 结构化剪枝（自定义稀疏度55%%）
echo 9. 非结构化剪枝（默认30%%稀疏度）
echo 10. 非结构化剪枝（自定义稀疏度45%%）
echo 11. 自动剪枝
echo.
echo === 蒸馏 ===
echo 12. 自动蒸馏（默认参数）
echo 13. 自动蒸馏（自定义：温度5.0, alpha0.8, epochs25）
echo.
echo === 组合操作 ===
echo 14. 组合：自动量化 + 结构化剪枝
echo 15. 组合：FP16量化 + 自动剪枝
echo 16. 组合：量化 + 剪枝 + 蒸馏（全部自定义参数）
echo.
echo === 格式转换 ===
echo 17. 转换为 ONNX
echo 18. 转换为 TorchScript
echo.
set /p choice=请输入测试编号（1-18）: 

set BASE_DIR=D:/项目空间/image_classification_inceptionv4/模型转换和部署功能/artifacts/m_test_21515/v_20251119_165333
set API_URL=http://localhost:5000

if "%choice%"=="1" (
    echo [测试1] 检测模型能力
    curl -X POST %API_URL%/detect-capabilities -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\"}"
)
if "%choice%"=="2" (
    echo [测试2] FP16量化
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": \"quantize_fp16\"}"
)
if "%choice%"=="3" (
    echo [测试3] INT8量化（自动选择静态/动态）
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"extra_dir\": \"%BASE_DIR%/extra\", \"method\": \"quantize_int8\"}"
)
if "%choice%"=="4" (
    echo [测试4] INT8动态量化
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": \"quantize_int8_dynamic\"}"
)
if "%choice%"=="5" (
    echo [测试5] INT8静态量化（需校准数据）
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"extra_dir\": \"%BASE_DIR%/extra\", \"method\": \"quantize_int8_static\"}"
)
if "%choice%"=="6" (
    echo [测试6] 自动量化
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": \"quantize_auto\"}"
)
if "%choice%"=="7" (
    echo [测试7] 结构化剪枝（默认30%%稀疏度）
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": \"prune_structured\"}"
)
if "%choice%"=="8" (
    echo [测试8] 结构化剪枝（自定义稀疏度55%%）
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": \"prune_structured\", \"method_params\": {\"prune_structured\": {\"target_sparsity\": 0.55}}}"
)
if "%choice%"=="9" (
    echo [测试9] 非结构化剪枝（默认30%%稀疏度）
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": \"prune_unstructured\"}"
)
if "%choice%"=="10" (
    echo [测试10] 非结构化剪枝（自定义稀疏度45%%）
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": \"prune_unstructured\", \"method_params\": {\"prune_unstructured\": {\"target_sparsity\": 0.45}}}"
)
if "%choice%"=="11" (
    echo [测试11] 自动剪枝
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": \"prune_auto\"}"
)
if "%choice%"=="12" (
    echo [测试12] 自动蒸馏（默认参数）
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"extra_dir\": \"%BASE_DIR%/extra\", \"method\": \"distill_auto\"}"
)
if "%choice%"=="13" (
    echo [测试13] 自动蒸馏（自定义参数）
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"extra_dir\": \"%BASE_DIR%/extra\", \"method\": \"distill_auto\", \"method_params\": {\"distill_auto\": {\"temperature\": 5.0, \"alpha\": 0.8, \"epochs\": 2}}}"
)
if "%choice%"=="14" (
    echo [测试14] 组合：自动量化 + 结构化剪枝
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": [\"quantize_auto\", \"prune_structured\"]}"
)
if "%choice%"=="15" (
    echo [测试15] 组合：FP16量化 + 自动剪枝
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": [\"quantize_fp16\", \"prune_auto\"]}"
)
if "%choice%"=="16" (
    echo [测试16] 组合：量化 + 剪枝 + 蒸馏（全部自定义参数）
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"extra_dir\": \"%BASE_DIR%/extra\", \"method\": [\"quantize_auto\", \"prune_structured\", \"distill_auto\"], \"method_params\": {\"prune_structured\": {\"target_sparsity\": 0.5}, \"distill_auto\": {\"temperature\": 5.0, \"alpha\": 0.8, \"epochs\": 25}}}"
)
if "%choice%"=="17" (
    echo [测试17] 转换为 ONNX
    curl -X POST %API_URL%/convert-format -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"target_formats\": [\"onnx\"]}"
)
if "%choice%"=="18" (
    echo [测试18] 转换为 TorchScript
    curl -X POST %API_URL%/convert-format -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"target_formats\": [\"torchscript\"]}"
)

echo.
echo.
pause
