param(
  [string]$LocalRoot = "C:\\Users\\sunno\\projects\\financial_agent_local",
  [string]$Catalog = "uc_cmifi_dev",
  [string]$Schema = "fin_agent",
  [string]$Volume = "annual_reports",
  [string]$SourceName = "parts_holding_europe",
  [int]$StartYear = 2011,
  [int]$EndYear = 2023
)

$ErrorActionPreference = "Stop"

$base = Join-Path $LocalRoot "Volumes\\$Catalog\\$Schema\\$Volume\\source=$SourceName"
New-Item -ItemType Directory -Force -Path $base | Out-Null

for ($y = $StartYear; $y -le $EndYear; $y++) {
  New-Item -ItemType Directory -Force -Path (Join-Path $base $y) | Out-Null
}

Write-Host "Created local staging structure:"
Write-Host $base

