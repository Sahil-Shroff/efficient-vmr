param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

if ($Clean) {
    latexmk -C main.tex
    exit $LASTEXITCODE
}

latexmk main.tex
exit $LASTEXITCODE
