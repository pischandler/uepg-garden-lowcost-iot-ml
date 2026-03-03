$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$hooksDir = Join-Path $repo ".git\hooks"
if (!(Test-Path $hooksDir)) {
  Write-Error "Cannot find .git/hooks at $hooksDir"
  exit 1
}
$src = Join-Path $PSScriptRoot "..\.githooks\pre-commit"
$dst = Join-Path $hooksDir "pre-commit"
Copy-Item $src $dst -Force
Write-Output "Installed pre-commit hook at $dst"
