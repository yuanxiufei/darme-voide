# 阶段1：将剩余模型硬链接迁移到 ComfyUI Desktop 共享库
# 用法：powershell -NoProfile -ExecutionPolicy Bypass -File scripts/migrate_models.ps1
$ErrorActionPreference = 'Stop'

$files = @(
    @{ Src = 'D:\code\ComfyUI\ComfyUI\models\unet\flux1-dev.safetensors'; Dst = 'D:\Comfy-Desktop\ComfyUI-Shared\models\unet\flux1-dev.safetensors' },
    @{ Src = 'D:\code\ComfyUI\ComfyUI\models\unet\z_image_turbo_bf16.safetensors'; Dst = 'D:\Comfy-Desktop\ComfyUI-Shared\models\unet\z_image_turbo_bf16.safetensors' },
    @{ Src = 'D:\code\ComfyUI\ComfyUI\models\vae\ae.safetensors'; Dst = 'D:\Comfy-Desktop\ComfyUI-Shared\models\vae\ae.safetensors' }
)

function Link-Directory {
    param([string]$Src, [string]$Dst)
    New-Item -ItemType Directory -Force -Path $Dst | Out-Null
    Get-ChildItem -LiteralPath $Src -Recurse -File |
        Where-Object { $_.FullName -notmatch '[\\/]\.cache[\\/]' } |
        ForEach-Object {
            $rel = $_.FullName.Substring($Src.Length).TrimStart('\')
            $t = Join-Path $Dst $rel
            New-Item -ItemType Directory -Force -Path (Split-Path $t -Parent) | Out-Null
            if (Test-Path $t) { Remove-Item $t -Force }
            New-Item -ItemType HardLink -Path $t -Target $_.FullName | Out-Null
        }
}

foreach ($f in $files) {
    if (-not (Test-Path $f.Src)) { Write-Host "SKIP (missing src) $($f.Src)"; continue }
    if (Test-Path $f.Dst) { Remove-Item $f.Dst -Force }
    New-Item -ItemType HardLink -Path $f.Dst -Target $f.Src | Out-Null
    Write-Host "OK $($f.Dst)"
}

Link-Directory 'D:\code\ComfyUI\ComfyUI\models\vae\flux_vae' 'D:\Comfy-Desktop\ComfyUI-Shared\models\vae\flux_vae'
Link-Directory 'D:\code\ComfyUI\ComfyUI\models\LLM\qwen3-30b-a3b-gptq' 'D:\Comfy-Desktop\ComfyUI-Shared\models\llm\qwen3-30b-a3b-gptq'

Write-Host 'ALL DONE'
